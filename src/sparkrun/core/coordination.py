"""Cross-process coordination shared by library, CLI, and daemon callers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import sqlite3
import threading

from sparkrun.core.process_identity import process_identity
from sparkrun.utils.fs import file_lock

_HELD = ContextVar("sparkrun_resources", default=frozenset())
_DOCUMENTS = threading.local()


class ResourceBusy(RuntimeError):
    pass


@contextmanager
def document_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = str((root / ".documents.lock").resolve())
    held = getattr(_DOCUMENTS, "held", set())
    if path in held:
        yield
        return
    with file_lock(path):
        _DOCUMENTS.held = held | {path}
        try:
            yield
        finally:
            _DOCUMENTS.held = held


def locked_document(function):
    @wraps(function)
    def invoke(self, *args, **kwargs):
        with document_lock(self.config_root):
            return function(self, *args, **kwargs)

    return invoke


@contextmanager
def reserve_resources(root, keys):
    """Atomically reserve destinations. Uncertain dead-owner claims stay visible."""
    import json
    import uuid

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    keys = frozenset(keys or {"*"})
    held = _HELD.get()
    if keys <= held or "*" in held:
        yield
        return
    owner = uuid.uuid4().hex
    path = root / "operations.sqlite3"
    db = sqlite3.connect(path, timeout=10, isolation_level=None)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS reservations (owner TEXT, resource TEXT UNIQUE, process TEXT)")
        path.chmod(0o600)
        db.execute("BEGIN IMMEDIATE")
        existing = {row[0] for row in db.execute("SELECT resource FROM reservations")}
        if "*" in existing or ("*" in keys and existing) or existing.intersection(keys - held):
            raise ResourceBusy("Execution resources are busy or require reconciliation after an interrupted operation")
        for key in sorted(keys - held):
            db.execute("INSERT INTO reservations VALUES(?,?,?)", (owner, key, json.dumps(process_identity())))
        db.commit()
        token = _HELD.set(held | keys)
        try:
            yield
        finally:
            _HELD.reset(token)
            db.execute("DELETE FROM reservations WHERE owner=?", (owner,))
    finally:
        db.close()


def operation_boundary(*, mutation=False):
    """Bind operation state and destination locks around a public API call."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def invoke(*args, **kwargs):
            from sparkrun.core.config import SparkrunConfig
            from sparkrun.core.operation import operation_scope

            bound = signature.bind(*args, **kwargs)
            sctx = bound.arguments.get("sctx")
            try:
                config = sctx.config if sctx is not None else SparkrunConfig()
            except Exception:
                # Preserve the operation's typed initialization/recovery errors.
                return function(*bound.args, **bound.kwargs)
            with operation_scope(sctx):
                if not mutation or config is None:
                    return function(*bound.args, **bound.kwargs)
                options = bound.arguments.get("options")
                if getattr(options, "dry_run", False) or bound.arguments.get("dry_run", False):
                    return function(*bound.args, **bound.kwargs)
                hosts = getattr(options, "hosts", None) or bound.arguments.get("hosts")
                cluster = getattr(options, "cluster", None) or bound.arguments.get("cluster")
                plan = bound.arguments.get("plan")
                if plan is not None:
                    hosts = getattr(plan, "host_list", hosts)
                    cluster = getattr(plan, "cluster", cluster)
                if not hosts:
                    try:
                        from sparkrun.core.cluster_manager import ClusterManager

                        if isinstance(cluster, str):
                            cluster = ClusterManager(config.config_path.parent).get(cluster)
                        if cluster is not None:
                            hosts = cluster.hosts
                        elif bound.arguments.get("cluster_id"):
                            from sparkrun.orchestration.job_metadata import load_job_metadata

                            meta = load_job_metadata(bound.arguments["cluster_id"], cache_dir=str(config.cache_dir))
                            hosts = (meta or {}).get("hosts")
                        else:
                            hosts = config.default_hosts
                    except Exception:  # Resolution errors belong to the API body, not lock inference.
                        hosts = None
                # Host conflicts are deliberately independent of SSH usernames:
                # different login users still mutate the same machine/devices.
                keys = {"host:" + str(host).rsplit("@", 1)[-1].lower() for host in hosts} if hosts else {"*"}
                executor = getattr(cluster, "executor_config", None)
                if executor:
                    import json

                    keys.add("destination:" + hashlib.sha256(json.dumps(executor, sort_keys=True).encode()).hexdigest())
                with reserve_resources(config.config_path.parent, keys):
                    return function(*bound.args, **bound.kwargs)

        return invoke

    return decorate


def reservations(root):
    """Inspect claims, including dead owners whose remote effects are unknown."""
    import json
    from sparkrun.core.process_identity import alive

    path = Path(root) / "operations.sqlite3"
    if not path.exists():
        return []
    with sqlite3.connect(path, timeout=10) as db:
        rows = db.execute("SELECT owner,resource,process FROM reservations ORDER BY owner,resource").fetchall()
    return [
        {"owner": owner, "resource": resource, "process": json.loads(process), "alive": alive(json.loads(process))}
        for owner, resource, process in rows
    ]


def reconcile_reservation(root, owner, *, acknowledge_effects=False):
    """Release only a verified dead owner's claims after explicit reconciliation."""
    import json
    from sparkrun.core.process_identity import alive

    if not acknowledge_effects:
        raise ValueError("Inspect the remote workloads, then acknowledge that effects have been reconciled")
    path = Path(root) / "operations.sqlite3"
    if not path.exists():
        raise LookupError("No reservations for this controller")
    with sqlite3.connect(path, timeout=10) as db:
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute("SELECT process FROM reservations WHERE owner=?", (owner,)).fetchall()
        if not rows:
            raise LookupError("Reservation not found")
        if any(alive(json.loads(row[0])) for row in rows):
            raise ResourceBusy("The reservation owner is still alive")
        db.execute("DELETE FROM reservations WHERE owner=?", (owner,))
