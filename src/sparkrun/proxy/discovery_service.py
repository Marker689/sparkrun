"""Private discovery sessions shared by the process and daemon backends.

The generation is a fencing identity, not a credential. The descriptor never
crosses HTTP: SSH settings stay in the private proxy directory. A single lock
serializes owner changes and applications, while gateway process state has a
separate lock so a managed restart can complete during an application.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
import uuid

from sparkrun.core.process_identity import alive, process_identity
from sparkrun.utils.fs import atomic_private_write, file_lock


class DiscoveryOwnershipError(RuntimeError):
    pass


class DiscoverySession:
    def __init__(self, state_dir):
        self.directory = Path(state_dir)
        self.path = self.directory / "discovery.json"

    @contextmanager
    def lock(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with file_lock(self.directory / ".discovery.lock"):
            yield

    def read(self):
        try:
            return json.loads(self.path.read_text())
        except FileNotFoundError:
            return None

    def write(self, value):
        atomic_private_write(self.path, json.dumps(value))

    def prepare(self, descriptor):
        """Claim an immutable scope; active/uncertain owners are never replaced."""
        with self.lock():
            previous = self.read()
            if previous and previous["state"] not in {"stopped", "disabled"}:
                raise DiscoveryOwnershipError("Discovery already has an owner; stop it before replacing its scope")
            record = {
                "generation": uuid.uuid4().hex,
                "descriptor": descriptor,
                "state": "starting",
                "process": None,
                "job_id": None,
                "last_attempt": None,
                "last_success": None,
                "last_application": None,
                "last_error": None,
                "created": time.time(),
            }
            self.write(record)
            return record

    def matching(self, generation):
        record = self.read()
        if not record or record["generation"] != generation or record["state"] in {"stopped", "disabled"}:
            raise DiscoveryOwnershipError("Discovery session was stopped or replaced")
        return record

    def attach(self, generation, *, job_id=None):
        with self.lock():
            record = self.matching(generation)
            current = process_identity()
            if record["process"] and record["process"] != current:
                raise DiscoveryOwnershipError("Discovery worker already claimed this session; reconcile before retrying")
            record.update(process=current, job_id=job_id, state="running")
            self.write(record)

    def update(self, generation, **fields):
        with self.lock():
            record = self.matching(generation)
            record.update(fields)
            self.write(record)

    @contextmanager
    def applying(self, generation):
        with self.lock():
            record = self.matching(generation)
            from sparkrun.proxy._supervisor import GatewayState

            state = GatewayState(self.directory).get_state() or {}
            descriptor = record["descriptor"]
            if state.get("stop_requested") or state.get("instance_id") != descriptor.get("instance_id"):
                raise DiscoveryOwnershipError("Gateway stopped or was replaced during discovery")
            yield

    def stop(self, *, generation=None):
        with self.lock():
            record = self.read()
            if not record or (generation and record["generation"] != generation):
                return None
            record["state"] = "stopped"
            self.write(record)
            return record

    def status(self):
        record = self.read()
        if not record:
            return {"backend": None, "state": "disabled"}
        state = record["state"]
        if state not in {"stopped", "disabled"} and record["process"] and not alive(record["process"]):
            state = "recovery_required"
        descriptor = record["descriptor"]
        return {
            "backend": descriptor["backend"],
            "state": state,
            "generation": record["generation"],
            "job_id": record["job_id"],
            "pid": (record["process"] or {}).get("pid"),
            "scope": {
                "hosts": descriptor.get("host_list") or [],
                "cluster": descriptor.get("cluster"),
                "source": descriptor.get("scope_source", "session"),
            },
            "interval": descriptor["interval"],
            "removal_grace_sweeps": descriptor["removal_grace_sweeps"],
            "last_attempt": record["last_attempt"],
            "last_success": record["last_success"],
            "last_application": record["last_application"],
            "last_error": record["last_error"],
        }


def session_for(sctx):
    from sparkrun.api.proxy._ops import _running_engine

    return DiscoverySession(_running_engine(sctx).state_dir)


def run_session(session, generation, sctx, *, cancelled, wait, job_id=None):
    """Cancellable common loop. Failed coverage/application never advances grace."""
    from sparkrun.proxy.autodiscover import _EndpointRemovalGrace, sync_discovered
    from sparkrun.proxy.discovery import discover_snapshot
    from sparkrun.proxy._supervisor import GatewayState
    from sparkrun.core.operation import CURRENT

    session.attach(generation, job_id=job_id)
    descriptor = session.matching(generation)["descriptor"]
    grace = _EndpointRemovalGrace(descriptor["removal_grace_sweeps"])
    seeds = descriptor.get("initial_endpoints", [])
    if seeds:
        from sparkrun import api
        from sparkrun.proxy.discovery import DiscoveredEndpoint

        credentials = {job.cluster_id: job.metadata.get("api_key") for job in api.list_jobs(sctx=sctx)}
        initial = []
        for seed in seeds:
            seed = dict(seed)
            required = seed.pop("credential_required", False)
            credential = credentials.get(seed["cluster_id"])
            if required and not credential:
                raise ValueError("Initial endpoint credentials unavailable in workload metadata")
            initial.append(DiscoveredEndpoint(**seed, api_key=credential))
        grace.reconcile(initial)
    state = GatewayState(session.directory)
    cluster = sctx.cluster_manager.get(descriptor["cluster"]) if descriptor.get("cluster") else None
    interval = max(1, int(descriptor["interval"]))
    failures = 0
    previous_event = None
    try:
        while not cancelled():
            try:
                session.matching(generation)
                current = state.get_state() or {}
                if current.get("stop_requested") or current.get("instance_id") != descriptor.get("instance_id"):
                    return {"stopped": "gateway stopped or replaced"}
                if not current.get("restarting") and not alive(current.get("process")):
                    return {"stopped": "gateway exited"}
                session.update(generation, last_attempt=time.time())
                observation = discover_snapshot(
                    host_list=descriptor.get("host_list"),
                    host_filter=descriptor.get("host_list"),
                    ssh_kwargs=descriptor.get("ssh_kwargs"),
                    cache_dir=descriptor.get("cache_dir"),
                    cluster_def=cluster,
                    sctx=sctx,
                )
                result, deferred = sync_discovered(
                    observation.endpoints,
                    grace,
                    sctx,
                    covered_hosts=set(observation.covered_hosts),
                    discovery_token=(session, generation),
                )
                if not result.proxy_running:
                    return {"stopped": "gateway stopped"}
                health = "degraded" if observation.errors else "running"
                session.update(
                    generation,
                    state=health,
                    last_application=time.time(),
                    **({"last_success": observation.observed_at} if observation.complete else {}),
                    last_error="Some hosts or endpoints were not observed" if observation.errors else None,
                )
                event = (health, result.added, result.removed, deferred)
                operation = CURRENT.get()
                if operation and (event != previous_event or result.added or result.removed):
                    operation.emit("discovery", state=health, added=result.added, removed=result.removed, deferred_removals=deferred)
                previous_event = event
                failures = 0
            except DiscoveryOwnershipError:
                return {"stopped": "ownership released"}
            except (KeyboardInterrupt, InterruptedError):
                raise
            except Exception as exc:
                failures += 1
                # Exception arguments can contain credentials; report only the
                # category here, while operation log redaction handles details.
                try:
                    session.update(generation, state="degraded", last_error=type(exc).__name__)
                except DiscoveryOwnershipError:
                    return {"stopped": "ownership released"}
                operation = CURRENT.get()
                if operation and previous_event != ("error", type(exc).__name__):
                    operation.emit("discovery", state="degraded", error=type(exc).__name__)
                previous_event = ("error", type(exc).__name__)
            wait(min(300, interval * 2 ** min(failures, 4)))
        return {"stopped": "cancelled"}
    finally:
        session.stop(generation=generation)
