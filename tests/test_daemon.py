"""Contract, isolation, and real-process tests for the durable API service."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
import subprocess
import sys
import time

import pytest

from sparkrun.core.daemon_registry import DaemonOp
from sparkrun.daemon.jobs import Conflict, JobStore, ReplayGap
from sparkrun.daemon.state import Binding


@pytest.fixture
def binding(tmp_path, monkeypatch):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core import daemon_registry

    monkeypatch.setattr(daemon_registry, "_OPERATIONS", {})
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    config = tmp_path / "config.yaml"
    config.write_text("features:\n  daemon: true\n")
    return Binding.from_config(SparkrunConfig(config)).prepare()


def operation():
    return DaemonOp("test.operation", lambda: None)


def test_idempotency_atomic_claims_and_terminal_events(binding):
    store = JobStore(binding.directory)
    op = operation()
    identifier = store.submit(op, {}, "alice", "display", {}, key="one")
    assert store.submit(op, {}, "alice", "different-label", {}, key="one") == identifier
    with pytest.raises(Conflict, match="different arguments"):
        store.submit(op, {"different": True}, "alice", "", {}, key="one")
    with ThreadPoolExecutor(8) as pool:
        claims = list(pool.map(lambda _: store.claim(), range(8)))
    assert len([c for c in claims if c]) == 1
    assert store.get(identifier)["state"] == "starting"
    assert store.finish(identifier, "succeeded", result={"answer": 42})
    assert not store.finish(identifier, "failed", error={"detail": "late error"})
    assert store.events(identifier)[-1]["state"] == "succeeded"
    assert store.get(identifier)["result"] == {"answer": 42}


def test_replay_retention_and_idempotency_tombstones(binding):
    store = JobStore(binding.directory, max_events=3)
    identifier = store.submit(operation(), {}, "alice", "", {}, key="once")
    for i in range(7):
        store.emit(identifier, {"type": "progress", "index": i})
    with pytest.raises(ReplayGap):
        store.events(identifier, 1)
    store.prune(age=0, count=0)
    assert store.get(identifier)["state"] == "queued"
    store.finish(identifier, "succeeded")
    store.prune(age=0, count=0)
    with pytest.raises(KeyError):
        store.get(identifier)
    with pytest.raises(Conflict, match="expired"):
        store.submit(operation(), {}, "alice", "", {}, key="once")


def test_prompt_transitions_and_cancellation_race(binding):
    store = JobStore(binding.directory)
    identifier = store.submit(operation(), {}, "alice", "", {})
    claim = store.claim()
    store.register_worker(identifier, claim["nonce"], {"process": {}})
    store.authorize(identifier, claim["nonce"])
    store.prompt(identifier, {"prompt_id": "approval", "kind": "approval"})
    store.answer(identifier, "approval")
    with pytest.raises(Conflict):
        store.answer(identifier, "approval")
    store.finish(identifier, "succeeded", result="done")
    with pytest.raises(Conflict):
        store.cancel(identifier)
    assert store.get(identifier)["result"] == "done"


def test_operation_state_follows_fanout_without_cross_request_budget():
    from sparkrun.core.operation import OperationState, operation_scope, submit
    from sparkrun.models.hub import _STATE

    seen = []
    first = OperationState(events=seen.append)
    second = OperationState()
    with operation_scope(state=first), ThreadPoolExecutor(2) as pool:
        _STATE.spent_s = 12
        assert submit(pool, lambda: _STATE.spent_s).result() == 12
        submit(pool, lambda: first.emit("progress", step=1)).result()
    with operation_scope(state=second):
        assert _STATE.spent_s == 0
    assert seen[0]["operation_id"] == first.id
    assert first.id != second.id


def test_closed_schema_and_explicit_redaction():
    from dataclasses import dataclass
    from sparkrun.api._wire import object_schema, validate, reject_secrets, to_wire

    with pytest.raises(ValueError, match="unknown fields"):
        validate({"hidden": True}, object_schema())
    with pytest.raises(ValueError, match="Secrets"):
        reject_secrets({"overrides": {"api_key": "never-persist"}})
    assert to_wire({"api_key": "never-return", "value": 2}) == {"api_key": "[redacted]", "value": 2}

    @dataclass
    class Private:
        password: str

    with pytest.raises(TypeError, match="No public wire codec"):
        to_wire(Private("not-exported"))


def test_auth_scopes_body_limits_and_ownership(binding):
    from fastapi.testclient import TestClient
    from sparkrun.daemon.auth import TokenStore
    from sparkrun.daemon.server import create_app

    tokens = TokenStore(binding)
    _, reader = tokens.create("reader", scope="read")
    _, operator = tokens.create("operator", scope="operate")
    _, other = tokens.create("other", scope="operate")
    app = create_app(binding, start_workers=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/info", headers={"X-Sparkrun-Client": "uid:0"}).status_code == 401
        headers = {"Authorization": "Bearer " + reader}
        assert client.get("/v1/ops", headers=headers).status_code == 200
        assert client.post("/v1/ops/stop", headers=headers, json={}).status_code == 403
        headers = {"Authorization": "Bearer " + operator}
        response = client.post("/v1/ops/stop", headers=headers, json={"cluster_id": "example"})
        assert response.status_code == 202, response.text
        job = response.json()["job_id"]
        assert "nonce" not in client.get("/v1/jobs/" + job, headers=headers).json()
        assert (
            client.get("/v1/jobs/" + job, headers={"Authorization": "Bearer " + other, "X-Sparkrun-Client": "operator"}).status_code == 403
        )
        assert client.post("/v1/ops/run", headers=headers, json={"recipe": "x", "trust": True}).status_code == 403
        assert client.post("/v1/ops/stop", headers=headers, content=b"x" * (1024 * 1024 + 1)).status_code == 413
        assert client.post("/v1/ops/stop", headers=headers, json={"unknown": 1}).status_code == 422
        assert client.get("/openapi.json").status_code == 401
        schema = client.get("/openapi.json", headers=headers).json()
        assert "runRequest" in schema["components"]["schemas"]


def test_binding_uses_config_file_and_short_worker_paths(tmp_path, monkeypatch):
    from sparkrun.core.config import SparkrunConfig

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / ("long" * 30)))
    one = Binding.from_config(SparkrunConfig(tmp_path / "one.yaml"))
    two = Binding.from_config(SparkrunConfig(tmp_path / "two.yaml"))
    assert one.key != two.key
    assert len(str(one.runtime / ("f" * 32 + ".sock")).encode()) < 104


def test_runtime_lease_blocks_removal_and_core_update(binding):
    from sparkrun.core.runtime_lease import assert_generations_available, live_leases
    from sparkrun.core.process_identity import process_identity
    from sparkrun.core.plugin_manifest import PluginError
    from sparkrun.utils.fs import atomic_private_write

    root = binding.config_path.parent
    directory = root / "runtime-leases"
    directory.mkdir()
    atomic_private_write(
        directory / "fixture.json", json.dumps({"process": process_identity(), "runtime": {"plugins": {"example": "generation-a"}}})
    )
    with pytest.raises(PluginError, match="in use"):
        assert_generations_available(root, {})
    assert_generations_available(root, {"example": {"active_generation": "generation-b", "previous_generations": ["generation-a"]}})
    assert len(live_leases(root)) == 1


def test_fresh_imports_leave_server_dependencies_unloaded(binding):
    code = """
import sys
import sparkrun.cli
import sparkrun.api
import sparkrun.daemon.worker
from sparkrun.application import initialize
initialize(config_path=sys.argv[1])
assert not {'fastapi','starlette','uvicorn','pydantic','anyio'} & sys.modules.keys(), sorted({'fastapi','starlette','uvicorn','pydantic','anyio'} & sys.modules.keys())
"""
    result = subprocess.run([sys.executable, "-c", code, str(binding.config_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@contextmanager
def real_server(binding, *, extra=()):
    from sparkrun.daemon.client import Client
    from sparkrun.core.application_profile import child_environment

    log = (binding.directory / "test-server.log").open("a")
    child = subprocess.Popen(
        [sys.executable, "-m", "sparkrun.daemon", "--config", str(binding.config_path), "serve", *extra],
        env={**os.environ, **child_environment(config_path=binding.config_path)},
        stdout=log,
        stderr=log,
    )
    client = Client(binding=binding)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if child.poll() is not None:
                pytest.fail((binding.directory / "test-server.log").read_text())
            try:
                client.info()
                break
            except (OSError, ValueError):
                time.sleep(0.05)
        else:
            pytest.fail("Daemon did not start")
        yield child, client
    finally:
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        log.close()


def test_real_worker_and_restart_preserve_result_and_idempotency(binding):
    with real_server(binding) as (process, client):
        job = client.call("cluster.create", {"name": "example", "hosts": ["host-a"]}, key="create-one")["job_id"]
        assert client.wait(job, timeout=15)["name"] == "example"
        process.kill()
        process.wait()
    with real_server(binding) as (_, client):
        assert client.request("GET", "/v1/jobs/" + job)["state"] == "succeeded"
        assert client.call("cluster.create", {"name": "example", "hosts": ["host-a"]}, key="create-one")["job_id"] == job
        events = list(client.events(job))
        assert events[-1]["state"] == "succeeded"


def test_real_queued_worker_reconnects_to_new_coordinator(binding):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.runtime_lease import runtime_identity
    from sparkrun.daemon.ops import register_core_ops
    from sparkrun.core.daemon_registry import get_daemon_op
    from sparkrun.daemon.supervisor import launch, Coordinator

    store = JobStore(binding.directory)
    register_core_ops()
    runtime = runtime_identity(SparkrunConfig(binding.config_path))
    identifier = store.submit(get_daemon_op("cluster.create"), {"name": "adopted", "hosts": ["host-a"]}, "local", "", runtime)
    store.claim()
    worker = launch(binding, identifier)
    try:
        deadline = time.monotonic() + 15
        while not store.get(identifier)["worker"] and time.monotonic() < deadline:
            assert worker.poll() is None
            time.sleep(0.05)
        # No coordinator was present to authorize execution. A replacement adopts it.
        assert store.get(identifier)["state"] == "starting"
        coordinator = Coordinator(binding, store)
        while store.get(identifier)["state"] not in {"succeeded", "failed", "lost"} and time.monotonic() < deadline:
            coordinator.tick(runtime)
            time.sleep(0.1)
        assert store.get(identifier)["state"] == "succeeded", store.get(identifier)
        worker.wait(timeout=5)
    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=10)


def test_mcp_negotiates_and_uses_same_registry():
    from sparkrun.daemon.mcp import Session

    class Fake:
        def info(self):
            return {"protocol": 1}

        def request(self, *args):
            return [{"name": "status", "scope": "read", "kind": "query", "description": "Status", "request_schema": {"type": "object"}}]

        def call(self, name, arguments, key):
            return {"job_id": "durable"}

    session = Session(Fake())
    with pytest.raises(ValueError, match="Initialize"):
        session.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert session.dispatch({"jsonrpc": "2.0", "id": 2, "method": "initialize"})["protocolVersion"] == "2025-11-25"
    session.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"})
    tools = session.dispatch({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})["tools"]
    assert tools[0]["annotations"]["readOnlyHint"]
    result = session.dispatch({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "status"}})
    assert not result["isError"]


@pytest.fixture
def controlled_plugin(binding, monkeypatch):
    """Real external operation with no hardware/network side effects."""
    import textwrap

    plugin = binding.directory / "plugins"
    plugin.mkdir()
    (plugin / "daemon_fixture.py").write_text(
        textwrap.dedent(r"""        SPARKRUN_PLUGIN_API_VERSION = 1
        def execute(args, sctx, prompt):
            import time
            from pathlib import Path
            marker = Path(sctx.config.config_path).parent / "executions.txt"
            with marker.open("a") as file:
                file.write(sctx.operation.id + "\n")
            if args.get("mode") == "tree":
                import subprocess, sys, signal
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(90)"], start_new_session=True)
                sctx.operation.emit("child", pid=child.pid)
                while True: time.sleep(.1)
            secret = prompt("secret", "Fixture credential", default=None)
            sctx.operation.emit("progress", credential_echo=secret)
            approved = prompt("approval", "Continue?", default=False)
            if not approved:
                raise InterruptedError("Declined")
            return {"accepted": True, "echo": secret}
        def loader(): return execute
        def register(v):
            from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op
            register_daemon_op(DaemonOp("fixture.wait", loader,
                {"type":"object", "properties":{"mode":{"type":"string"}}, "additionalProperties":False},
                owner="daemon_fixture"))
    """)
    )
    binding.config_path.write_text(
        "features:\n  daemon: true\n  core.external_plugins: true\nplugins:\n  paths:\n    - " + str(plugin) + "\n"
    )
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS", raising=False)
    return binding


def wait_state(client, job, states, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = client.request("GET", "/v1/jobs/" + job)
        if row["state"] in states:
            return row
        if row["state"] in {"failed", "cancelled", "lost"}:
            pytest.fail(str(row))
        time.sleep(0.05)
    pytest.fail("Timed out waiting for " + str(states) + ": " + str(row))


def test_worker_survives_api_crash_with_secret_prompt_and_single_execution(controlled_plugin):
    binding = controlled_plugin
    with real_server(binding) as (process, client):
        job = client.call("fixture.wait", {}, key="once")["job_id"]
        row = wait_state(client, job, {"waiting_input"})
        original_prompt = row["prompt"]
        process.kill()
        process.wait()
    with real_server(binding) as (_, client):
        assert wait_state(client, job, {"waiting_input"})["prompt"] == original_prompt
        answer = {"prompt_id": original_prompt["prompt_id"], "value": "credential-must-not-persist"}
        client.request("POST", f"/v1/jobs/{job}/input", answer)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            row = client.request("GET", "/v1/jobs/" + job)
            if row["prompt"] and row["prompt"]["kind"] == "approval":
                break
            time.sleep(0.05)
        # Retry an answered secret while a different prompt is pending.
        client.request("POST", f"/v1/jobs/{job}/input", answer)
        from sparkrun.daemon.client import DaemonError

        with pytest.raises(DaemonError, match="differently"):
            client.request("POST", f"/v1/jobs/{job}/input", {**answer, "value": "different"})
        client.request("POST", f"/v1/jobs/{job}/input", {"prompt_id": row["prompt"]["prompt_id"], "value": True})
        assert client.wait(job, timeout=10) == {"accepted": True, "echo": "[redacted]"}
        assert client.call("fixture.wait", {}, key="once")["job_id"] == job
        assert (binding.config_path.parent / "executions.txt").read_text().splitlines() == [job]
    for path in binding.directory.iterdir():
        if path.is_file():
            assert b"credential-must-not-persist" not in path.read_bytes(), path


def test_cancel_prompt_and_reconcile_dead_resource_claim(controlled_plugin):
    from sparkrun.core.coordination import reservations, reconcile_reservation, ResourceBusy

    binding = controlled_plugin
    with real_server(binding) as (_, client):
        job = client.call("fixture.wait")["job_id"]
        wait_state(client, job, {"waiting_input"})
        claims = reservations(binding.config_path.parent)
        with pytest.raises(ResourceBusy):
            reconcile_reservation(binding.config_path.parent, claims[0]["owner"], acknowledge_effects=True)
        client.request("DELETE", "/v1/jobs/" + job)
        wait_state(client, job, {"cancelled"})
        assert not reservations(binding.config_path.parent)


def test_forced_cancel_reaps_child_in_separate_session(controlled_plugin):
    from sparkrun.core.process_identity import process_identity, alive
    from sparkrun.core.coordination import reservations, reconcile_reservation

    binding = controlled_plugin
    with real_server(binding) as (_, client):
        job = client.call("fixture.wait", {"mode": "tree"})["job_id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            children = [e for e in JobStore(binding.directory).events(job) if e["type"] == "child"]
            if children:
                break
            time.sleep(0.1)
        child = process_identity(children[0]["pid"])
        assert alive(child)
        client.request("DELETE", "/v1/jobs/" + job)
        wait_state(client, job, {"lost"}, timeout=25)
        assert not alive(child)
        claims = reservations(binding.config_path.parent)
        assert claims and not claims[0]["alive"]
        with pytest.raises(ValueError, match="acknowledge"):
            reconcile_reservation(binding.config_path.parent, claims[0]["owner"])
        reconcile_reservation(binding.config_path.parent, claims[0]["owner"], acknowledge_effects=True)
        assert not reservations(binding.config_path.parent)


def test_actual_systemd_api_restart_preserves_independent_worker(controlled_plugin):
    import shutil
    from sparkrun.core.application_profile import child_environment
    from sparkrun.daemon.client import Client
    from sparkrun.core.process_identity import alive

    binding = controlled_plugin
    if not shutil.which("systemd-run"):
        pytest.skip("systemd unavailable")
    bus = subprocess.run(["systemctl", "--user", "show-environment"], capture_output=True)
    if bus.returncode:
        pytest.skip("No user service manager")
    unit = "sparkrun-daemon-test-" + binding.key
    env = {
        **child_environment(config_path=binding.config_path),
        "XDG_RUNTIME_DIR": os.environ["XDG_RUNTIME_DIR"],
        "SPARKRUN_NO_EXTERNAL_PLUGINS": "0",
        "SPARKRUN_NO_TELEMETRY": "1",
    }
    command = [
        "systemd-run",
        "--user",
        "--quiet",
        "--unit=" + unit,
        "--property=Type=exec",
        "--property=KillMode=control-group",
        *[f"--setenv={k}={v}" for k, v in env.items()],
        "--",
        sys.executable,
        "-m",
        "sparkrun.daemon",
        "--config",
        str(binding.config_path),
        "serve",
        "--worker-supervisor",
        "systemd",
    ]
    client = Client(binding=binding)
    job = None
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=20)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                client.info()
                break
            except (OSError, ValueError):
                time.sleep(0.1)
        job = client.call("fixture.wait")["job_id"]
        row = wait_state(client, job, {"waiting_input"})
        worker = JobStore(binding.directory).get(job)["worker"]["process"]
        subprocess.run(["systemctl", "--user", "restart", unit], check=True, capture_output=True, timeout=20)
        assert alive(worker)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                client.info()
                break
            except (OSError, ValueError):
                time.sleep(0.1)
        assert client.request("GET", "/v1/jobs/" + job)["prompt"] == row["prompt"]
        client.request("DELETE", "/v1/jobs/" + job)
        wait_state(client, job, {"cancelled"})
        assert (binding.config_path.parent / "executions.txt").read_text().splitlines() == [job]
    finally:
        subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, timeout=20)
        if job:
            subprocess.run(["systemctl", "--user", "stop", f"sparkrun-job-{binding.key}-{job}"], capture_output=True, timeout=20)
        subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True, timeout=20)
