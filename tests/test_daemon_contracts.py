"""Wire, permissions, stream cleanup, and update-coordination acceptance."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from test_daemon import binding as binding, controlled_plugin as controlled_plugin, real_server, wait_state


def test_public_api_coverage_and_closed_schemas(binding):
    from sparkrun import api
    from sparkrun.daemon.coverage import API_OPERATIONS, EXCLUSIONS, K8S_OPERATIONS, K8S_EXCLUSIONS
    from sparkrun.daemon.ops import register_core_ops, check_request
    from sparkrun.core.daemon_registry import registered_daemon_ops, get_daemon_op
    from sparkrun.plugins.k8s import api as k8s
    from sparkrun.plugins.k8s.daemon import register

    functions = set()
    for prefix, module in [("", api), *((n + ".", getattr(api, n)) for n in ("proxy", "setup", "tailscale"))]:
        functions.update(prefix + n for n in module.__all__ if inspect.isfunction(getattr(module, n, None)))
    functions.update(
        "model_cache." + n
        for n, obj in vars(api.model_cache).items()
        if inspect.isfunction(obj) and obj.__module__ == api.model_cache.__name__ and not n.startswith("_")
    )
    assert functions == API_OPERATIONS.keys() | EXCLUSIONS.keys()
    assert {n for n in k8s.__all__ if inspect.isfunction(getattr(k8s, n, None))} == K8S_OPERATIONS.keys() | K8S_EXCLUSIONS.keys()
    from sparkrun.core.plugin_identity import plugin_registration, discovered_identity

    register_core_ops()
    with plugin_registration(discovered_identity("k8s", "sparkrun.plugins.k8s", ("in-tree", "sparkrun.plugins.k8s", None))):
        register()
    names = {op.name for op in registered_daemon_ops()}
    for endpoint in [*API_OPERATIONS.values(), *K8S_OPERATIONS.values()]:
        assert all(target.startswith("/v1/streams/") or target in names for target in endpoint.split(" / "))
    for op in registered_daemon_ops():
        assert op.request_schema["additionalProperties"] is False
        assert op.result_schema, op.name
        with pytest.raises(ValueError):
            check_request(op, {"unadvertised_argument": "ignored inputs are forbidden"})
    with pytest.raises(ValueError):
        check_request(get_daemon_op("workloads.run"), {"recipe": None})
    assert check_request(get_daemon_op("proxy.start"), {"auto_discover": False, "restart": True})
    assert check_request(get_daemon_op("model_cache.inspect"), {"model": "org/model", "cache_dir": "/tmp/cache", "file_selection": "all"})
    assert check_request(get_daemon_op("plugins.k8s.configure_service_account"), {"token_duration": "1h"})


def test_plugin_route_uses_selected_operation_authorization(binding, monkeypatch):
    from fastapi.testclient import TestClient
    from sparkrun.core import daemon_registry as registry
    from sparkrun.core.daemon_registry import DaemonOp, DaemonRoute, register_daemon_op, register_daemon_route
    from sparkrun.daemon.auth import TokenStore
    from sparkrun.daemon.server import create_app

    monkeypatch.setattr(registry, "_ROUTES", {})
    from sparkrun.plugins import application_daemon_registration

    with application_daemon_registration():
        register_daemon_op(DaemonOp("mutate", lambda: None))
        declaration = DaemonRoute(
            "fixture",
            lambda: lambda args: ("mutate", {}),
            ("mutate",),
            {"type": "object", "properties": {}, "additionalProperties": False},
        )
        route = register_daemon_route(declaration)
        register_daemon_route(declaration)
        with pytest.raises(ValueError):
            register_daemon_route(replace(declaration, owner="other"))
        with pytest.raises(ValueError):
            replace(declaration, path="/v1/jobs")
    tokens = TokenStore(binding)
    _, read = tokens.create("read", scope="read")
    _, operate = tokens.create("operate", scope="operate")
    with TestClient(create_app(binding, start_workers=False), raise_server_exceptions=False) as client:
        assert client.post(route.path, json={}, headers={"Authorization": "Bearer " + read}).status_code == 403
        response = client.post(route.path, json={}, headers={"Authorization": "Bearer " + operate})
        assert response.status_code == 202, response.text
        assert response.json()["job_id"]


def test_quiet_log_stream_cancels_and_closes_iterator(monkeypatch):
    from sparkrun.daemon.streams import LogStream
    from sparkrun import api
    from sparkrun.core.context import SparkrunContext

    closed = threading.Event()
    entered = threading.Event()

    def quiet(*args, cancel, **kwargs):
        entered.set()
        try:
            cancel.wait(3)
            yield from ()
        finally:
            closed.set()

    monkeypatch.setattr(api, "logs", quiet)
    stream = LogStream({}, SparkrunContext(None, None))
    assert entered.wait(2)
    stream.close()
    assert closed.is_set() and not stream.thread.is_alive()


def test_monitor_subscribers_share_one_session_and_release(monkeypatch):
    from sparkrun.daemon.streams import MonitorHub
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.config import SparkrunConfig
    from sparkrun import api
    from sparkrun.core.monitoring import HistoryPoint, HostHistory, MonitorFrame, MonitorSample
    import sparkrun.api._resolve

    cluster = ClusterDefinition(name="fixture", hosts=["host-a"])
    monkeypatch.setattr(sparkrun.api._resolve, "resolve_cluster", lambda *args, **kw: cluster)
    sessions = []

    class Session:
        closed = False

        def frame(self):
            return MonitorFrame(hosts=(), queried_at=time.time())

        def history_snapshot(self):
            return [
                HostHistory(
                    host="host-a", points=(HistoryPoint(recorded_at=1.0, sample=MonitorSample(gpu_util_pct="7", net_recv_bytes_s="9")),)
                )
            ]

        def close(self):
            self.closed = True

    def opened(*args, **kwargs):
        session = Session()
        sessions.append(session)
        return session

    monkeypatch.setattr(api, "open_live_monitor", opened)
    hub = MonitorHub(linger=0)
    sctx = SparkrunContext(None, SparkrunConfig())
    with hub.subscribe({}, sctx) as first:
        with hub.subscribe({}, sctx) as second:
            assert len(sessions) == 1
            assert first.frame()["hosts"] == second.frame()["hosts"] == []
            backfill = first.history()
            assert backfill["type"] == "history" and "hosts" not in backfill  # frame-row readers ignore it
            (series,) = backfill["series"]
            assert series["host"] == "host-a"
            assert series["points"][0]["recorded_at"] == 1.0
            assert series["points"][0]["sample"]["gpu_util_pct"] == "7"
            assert series["points"][0]["sample"]["net_recv_bytes_s"] == "9"  # new sample fields reach the wire
        hub.prune()
        assert not sessions[0].closed
    hub.prune()
    assert sessions[0].closed and not hub.entries


@pytest.mark.parametrize("allow_insecure", [False, True])
def test_real_mixed_tls_unix_and_ephemeral_token_lifecycle(binding, monkeypatch, allow_insecure):
    import shutil
    from sparkrun.daemon.client import Client, DaemonError
    from sparkrun.daemon.auth import TokenStore

    if not shutil.which("openssl"):
        pytest.skip("openssl unavailable")
    cert = binding.directory / "cert.pem"
    key = binding.directory / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    connection = binding.directory / "connection.json"
    with real_server(
        binding,
        extra=("--tcp", "127.0.0.1:0", "--tls-cert", str(cert), "--tls-key", str(key), "--conn-file", str(connection))
        + (("--allow-insecure",) if allow_insecure else ()),
    ) as (_, local):
        data = json.loads(connection.read_text())
        assert data["url"].startswith("https://")
        remote = Client(url=data["url"], token=data["token"])
        assert remote.info()["binding"] == local.info()["binding"] == binding.key
        with pytest.raises(DaemonError):
            Client(url=data["url"]).info()
        assert connection.stat().st_mode & 0o777 == 0o600
    assert not connection.exists()
    assert not TokenStore(binding).list()


def test_remote_plaintext_refused(binding):
    from sparkrun.core.application_profile import child_environment

    result = subprocess.run(
        [sys.executable, "-m", "sparkrun.daemon", "--config", str(binding.config_path), "serve", "--tcp", "0.0.0.0:0"],
        env={**os.environ, **child_environment(config_path=binding.config_path)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0 and "Remote TCP requires TLS" in result.stderr
    assert not binding.socket.exists()


def test_secret_permission_and_revoke_affect_existing_principals(controlled_plugin):
    from sparkrun.daemon.auth import TokenStore
    from sparkrun.daemon.client import Client, DaemonError

    binding = controlled_plugin
    tokens = TokenStore(binding)
    token_id, token = tokens.create("admin without secret", scope="admin")
    principal = tokens.authenticate(token)
    connection = binding.directory / "tcp.json"
    with real_server(binding, extra=("--tcp", "127.0.0.1:0", "--conn-file", str(connection))) as (_, local):
        remote = Client(url=json.loads(connection.read_text())["url"], token=token)
        job = local.call("plugins.daemon-fixture.wait")["job_id"]
        row = wait_state(local, job, {"waiting_input"})
        with pytest.raises(DaemonError) as caught:
            remote.request("POST", f"/v1/jobs/{job}/input", {"prompt_id": row["prompt"]["prompt_id"], "value": "not-authorized"})
        assert caught.value.status == 403
        tokens.revoke(token_id)
        with pytest.raises(PermissionError):
            tokens.validate_principal(principal)
        with pytest.raises(DaemonError) as caught:
            remote.info()
        assert caught.value.status == 401
        local.request("DELETE", "/v1/jobs/" + job)
        wait_state(local, job, {"cancelled"})


def test_core_upgrade_guard_includes_another_config_binding(binding):
    from sparkrun.core.runtime_lease import assert_runtime_idle
    from sparkrun.core.config import SparkrunConfig

    code = """
import sys,time
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.runtime_lease import runtime_lease
with runtime_lease(SparkrunConfig(sys.argv[1]),kind='test-worker'):
 print('leased',flush=True)
 time.sleep(30)
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(binding.config_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "leased"
        with pytest.raises(RuntimeError, match="draining workers"):
            assert_runtime_idle(SparkrunConfig(binding.config_path.parent / "other.yaml"))
    finally:
        child.terminate()
        child.wait(timeout=10)
    # Strong process identity removes its stale lease; PID presence alone is insufficient.
    assert_runtime_idle(SparkrunConfig(binding.config_path))


def test_disjoint_destinations_and_interrupted_owner_claim(binding):
    from sparkrun.core.coordination import reserve_resources, ResourceBusy
    from concurrent.futures import ThreadPoolExecutor

    root = binding.config_path.parent

    def attempt(key):
        with reserve_resources(root, {key}):
            return key

    with reserve_resources(root, {"host:a"}), ThreadPoolExecutor() as pool:
        assert pool.submit(attempt, "host:b").result() == "host:b"
        with pytest.raises(ResourceBusy):
            pool.submit(attempt, "host:a").result()
        with pytest.raises(ResourceBusy):
            pool.submit(attempt, "*").result()


def test_recovery_help_does_not_import_broken_external_plugin(controlled_plugin):
    plugin = next((controlled_plugin.directory / "plugins").glob("*.py"))
    plugin.write_text("raise SystemExit('BROKEN PLUGIN IMPORTED')\n")
    from sparkrun.core.application_profile import child_environment

    env = {**os.environ, **child_environment(config_path=controlled_plugin.config_path)}
    for command in (("--help",), ("status",), ("jobs", "--help"), ("token", "--help")):
        result = subprocess.run(
            [sys.executable, "-m", "sparkrun.daemon", "--config", str(controlled_plugin.config_path), *command],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr
        assert "BROKEN PLUGIN" not in result.stderr


def test_active_idempotency_survives_retry_window(binding):
    from sparkrun.daemon.jobs import JobStore
    from sparkrun.core.daemon_registry import DaemonOp
    import time

    store = JobStore(binding.directory)
    op = DaemonOp("test.retention", lambda: None)
    job = store.submit(op, {}, "alice", "", {}, key="long-operation")
    with store.transaction() as db:
        db.execute("UPDATE submissions SET created=?", (time.time() - 31 * 86400,))
    store.prune(age=0, count=0)
    assert store.submit(op, {}, "alice", "", {}, key="long-operation") == job


def test_run_approval_executes_same_live_plan_and_context(monkeypatch):
    from types import SimpleNamespace
    from sparkrun import api
    from sparkrun.core.operation import OperationState
    from sparkrun.daemon import ops

    plan = SimpleNamespace(recipe=SimpleNamespace(post_exec=[], post_commands=[]))
    context = SimpleNamespace(operation=OperationState())
    result = SimpleNamespace(rc=0, workload_id="same-plan", already_running=False, launch_result=None)
    observed = []
    monkeypatch.setattr(ops, "to_wire", lambda obj: {"display": obj.workload_id} if obj is result else {"display": "plan"})

    def resolve(options, *, sctx):
        assert sctx is context
        observed.append(("planned", options))
        return plan

    def approve(kind, message, **kwargs):
        assert kind == "approval"
        observed.append(("approved", plan))
        return True

    def run(options, *, sctx, plan):
        assert plan is observed[1][1]
        assert options is observed[0][1]
        assert sctx is context
        observed.append(("run", plan))
        return result

    monkeypatch.setattr(api, "plan", resolve)
    monkeypatch.setattr(api, "run", run)
    assert ops.invoke_options("run", "RunOptions", {"recipe": "fixture", "confirm": True, "dry_run": True}, context, approve) == {
        "display": "same-plan"
    }
    assert [entry[0] for entry in observed] == ["planned", "approved", "run"]


def test_private_state_rejects_symlink_ancestor(tmp_path):
    from sparkrun.daemon.state import private_directory

    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="ancestor"):
        private_directory(tmp_path / "link" / "child")
    assert not (real / "child").exists()


def test_reconciliation_keeps_old_active_jobs_beyond_history_page(binding, monkeypatch):
    from sparkrun.core.daemon_registry import DaemonOp
    from sparkrun.daemon.jobs import JobStore
    from sparkrun.daemon.supervisor import Coordinator

    store = JobStore(binding.directory)
    job = store.submit(DaemonOp("test.old", lambda: None), {}, "alice", "", {})
    store.claim()
    with store.transaction() as db:
        db.execute("UPDATE jobs SET created=0,updated=0 WHERE id=?", (job,))
        db.executemany(
            "INSERT INTO jobs(id,operation,version,arguments,principal,client,scope,runtime,state,created,updated) "
            "VALUES(?, 'test.old', 1, '{}', 'alice', '', 'operate', '{}', 'succeeded', ?, ?)",
            [(f"finished-{n}", n + 1, n + 1) for n in range(1001)],
        )
    assert store.list(limit=1)[0]["id"] == job
    monkeypatch.setattr("sparkrun.core.runtime_lease.runtime_identity", lambda config: {})
    coordinator = Coordinator(binding, store)
    coordinator.tick({})
    assert store.get(job)["state"] == "lost"
    assert "not authorized" in store.get(job)["error"]["detail"]


def test_cli_restart_preserves_tcp_and_rotates_connection_credential(binding):
    from sparkrun.core.application_profile import child_environment
    from sparkrun.daemon.auth import TokenStore
    from sparkrun.daemon.client import Client

    connection = binding.directory / "restart.json"
    command = [sys.executable, "-m", "sparkrun.daemon", "--config", str(binding.config_path)]
    env = {**os.environ, **child_environment(config_path=binding.config_path)}
    with real_server(binding, extra=("--tcp", "127.0.0.1:0", "--conn-file", str(connection))) as (_, local):
        before = json.loads(connection.read_text())
        try:
            result = subprocess.run([*command, "restart"], env=env, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stderr
            after = json.loads(connection.read_text())
            assert after["binding"] == before["binding"]
            assert after["process"] != before["process"]
            assert after["token"] != before["token"]
            assert after["launch"]["tcp"] == before["launch"]["tcp"]
            assert Client(url=after["url"], token=after["token"]).info()["binding"] == binding.key
            assert local.call("catalog.clusters") == []
            with pytest.raises(PermissionError):
                TokenStore(binding).authenticate(before["token"])
        finally:
            subprocess.run([*command, "stop"], env=env, capture_output=True, timeout=30)


def test_update_state_does_not_apply_posix_ownership_rules_without_uid(tmp_path, monkeypatch):
    """Windows CLI updates use the lease directory even though serving is Linux-only."""
    from sparkrun.daemon.state import private_directory

    parent = tmp_path / "windows-mode-bits"
    parent.mkdir(mode=0o777)
    parent.chmod(0o777)
    monkeypatch.delattr(os, "getuid")
    assert private_directory(parent / "state").is_dir()


def test_service_worker_guard_and_launch_diagnostics(binding, monkeypatch, caplog):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.runtime_lease import runtime_identity
    from sparkrun.daemon import supervisor
    from sparkrun.daemon.jobs import JobStore
    from sparkrun.core.daemon_registry import DaemonOp

    monkeypatch.setenv("INVOCATION_ID", "test-service")
    runtime = runtime_identity(SparkrunConfig(binding.config_path))
    store = JobStore(binding.directory)
    job = store.submit(DaemonOp("test.launch", lambda: None), {}, "test", "test", runtime)
    supervisor.Coordinator(binding, store).tick(runtime)
    assert store.get(job)["state"] == "lost"
    assert "Service mode requires --worker-supervisor systemd" in caplog.text
    assert job in caplog.text


@pytest.mark.parametrize(
    "error,reason",
    [
        (subprocess.CalledProcessError(23, ["systemd-run", "--setenv=KEY=private-key"], stderr="private-key"), "status 23"),
        (subprocess.TimeoutExpired(["systemd-run", "--setenv=KEY=private-key"], 20, stderr="private-key"), "timed out after 20"),
    ],
)
def test_supervisor_launch_diagnostics_do_not_expose_command_secrets(binding, monkeypatch, caplog, error, reason):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.runtime_lease import runtime_identity
    from sparkrun.daemon import supervisor
    from sparkrun.daemon.jobs import JobStore
    from sparkrun.core.daemon_registry import DaemonOp

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(supervisor, "launch", fail)
    runtime = runtime_identity(SparkrunConfig(binding.config_path))
    store = JobStore(binding.directory)
    job = store.submit(DaemonOp("test.launch", lambda: None), {}, "test", "test", runtime)
    supervisor.Coordinator(binding, store).tick(runtime)
    assert store.get(job)["state"] == "lost"
    assert reason in caplog.text and job in caplog.text
    assert "private-key" not in caplog.text


def test_public_workload_api_coverage():
    from sparkrun.api import workloads
    from sparkrun.daemon.coverage import WORKLOAD_OPERATIONS, WORKLOAD_EXCLUSIONS
    from sparkrun.daemon.ops import register_core_ops
    from sparkrun.core.daemon_registry import registered_daemon_ops

    public = {
        name
        for name, obj in vars(workloads).items()
        if inspect.isfunction(obj) and obj.__module__ == workloads.__name__ and not name.startswith("_")
    }
    assert public == WORKLOAD_OPERATIONS.keys() | WORKLOAD_EXCLUSIONS.keys()
    register_core_ops()
    names = {op.name for op in registered_daemon_ops()}
    for targets in WORKLOAD_OPERATIONS.values():
        assert set(targets.split(" / ")) <= names
