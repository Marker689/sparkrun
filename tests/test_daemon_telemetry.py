"""Usage telemetry attribution and serving lifecycle, with local-only collectors."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
from types import SimpleNamespace

import pytest

from sparkrun.core.operation import OperationState, operation_scope, submit
from sparkrun.daemon.client import Client
from sparkrun.telemetry.client import prepare_event
from test_daemon import binding as binding, controlled_plugin as controlled_plugin, real_server, wait_state


@pytest.fixture
def collector(monkeypatch):
    """Subprocess telemetry is explicitly routed to this loopback-only sink."""
    received = Queue()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.put(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "0")
    monkeypatch.setenv("SPARKRUN_TELEMETRY_ENDPOINT", f"http://127.0.0.1:{server.server_port}/")
    monkeypatch.setenv("SPARKRUN_TELEMETRY_KEY", "local-test-only")
    try:
        yield received
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_event_origin_is_operation_local_and_propagates_to_fanout(binding, monkeypatch):
    from sparkrun.core.config import SparkrunConfig

    monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "0")
    config = SparkrunConfig(binding.config_path)
    config.set("telemetry.installation_id", "unit-test")

    def event():
        return prepare_event(config, {"event_type": "run", "via_daemon": "untrusted-override"})

    assert event()["via_daemon"] is False
    with ThreadPoolExecutor(2) as pool:
        with operation_scope(state=OperationState(via_daemon=True)):
            assert submit(pool, event).result()["via_daemon"] is True
            with operation_scope():
                assert event()["via_daemon"] is True
            with operation_scope(state=OperationState()):
                assert event()["via_daemon"] is False
            assert event()["via_daemon"] is True
        assert submit(pool, event).result()["via_daemon"] is False
    assert event()["via_daemon"] is False


@pytest.mark.parametrize("via_daemon", [False, True])
def test_existing_run_and_benchmark_emitters_include_origin(binding, monkeypatch, via_daemon):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.telemetry.emit import emit_run_telemetry, emit_benchmark_telemetry

    config = SparkrunConfig(binding.config_path)
    monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "0")
    monkeypatch.setenv("SPARKRUN_TELEMETRY_ENDPOINT", "https://never-contact.invalid/")
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            return b""

    def capture(request, **kwargs):
        captured.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("sparkrun.telemetry.client.urlopen", capture)
    recipe = SimpleNamespace(model="/private/model", metadata={})
    options = SimpleNamespace(recipe=recipe, overrides={}, dry_run=False, skip_run=False, bench_args={})
    result = SimpleNamespace(
        success=True,
        category="inference",
        framework="test",
        profile="default",
        integration_results={},
        resumed=False,
        host_list=["private-host"],
        results={"throughput": 1},
        outputs={},
        metadata={},
        container_image_sha_pinned=False,
        container_image_longterm_pinned=False,
    )
    with operation_scope(state=OperationState(via_daemon=via_daemon)):
        emit_run_telemetry(config, result=result, recipe=recipe, cluster=None, options=options)
        emit_benchmark_telemetry(config, result=result, options=options, recipe=recipe)
    assert [event["event_type"] for event in captured] == ["run", "benchmark"]
    assert all(event["via_daemon"] is via_daemon for event in captured)
    assert "/private/model" not in json.dumps(captured) and "private-host" not in json.dumps(captured)


def test_cli_start_and_restart_report_one_ready_event_each(binding, collector):
    def cli(command):
        result = subprocess.run(
            [sys.executable, "-m", "sparkrun.daemon", "--config", str(binding.config_path), command],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    try:
        cli("start")
        first = collector.get(timeout=10)
        assert first["event_type"] == "daemon_serve"
        assert first["via_daemon"] is False
        assert first["worker_supervisor"] == "process"
        assert first["startup_ms"] >= 0
        cli("start")  # Attaching to the existing server is not another serve.
        with pytest.raises(Empty):
            collector.get(timeout=0.2)
        cli("restart")
        second = collector.get(timeout=10)
        assert second["event_type"] == "daemon_serve"
        assert first["installation_id"] == second["installation_id"]
        assert first["event_id"] != second["event_id"]
        assert str(binding.config_path) not in json.dumps([first, second])
        assert binding.key not in json.dumps([first, second])
        cli("stop")
        with pytest.raises(Empty):
            collector.get(timeout=0.2)
    finally:
        try:
            Client(binding=binding).request("POST", "/v1/shutdown", {})
        except (OSError, ValueError):
            pass


def test_multiple_listeners_report_one_event_and_rejected_start_reports_none(binding, collector):
    extra = ("--tcp", "0.0.0.0:0", "--allow-insecure")
    with real_server(binding, extra=extra):
        event = collector.get(timeout=10)
        assert event["tcp"] and event["remote_tcp"] and not event["tls"]
        assert event["unix_socket"] is sys.platform.startswith("linux")
        assert set(event) == {
            "schema_version",
            "event_id",
            "sent_at",
            "installation_id",
            "sparkrun_version",
            "via_daemon",
            "event_type",
            "unix_socket",
            "tcp",
            "tls",
            "remote_tcp",
            "worker_supervisor",
            "parent_managed",
            "startup_ms",
            "system",
        }
        with pytest.raises(Empty):
            collector.get(timeout=0.2)
    result = subprocess.run(
        [sys.executable, "-m", "sparkrun.daemon", "--config", str(binding.config_path), "serve", "--tcp", "0.0.0.0:0"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    with pytest.raises(Empty):
        collector.get(timeout=0.2)


@pytest.mark.parametrize("disabled_by", ["environment", "config"])
def test_serving_respects_opt_out(binding, collector, monkeypatch, disabled_by):
    if disabled_by == "environment":
        monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "1")
    else:
        monkeypatch.delenv("SPARKRUN_NO_TELEMETRY")
        with binding.config_path.open("a") as file:
            file.write("telemetry:\n  enabled: false\n")
    with real_server(binding, extra=("--no-unix", "--tcp", "127.0.0.1:0")):
        with pytest.raises(Empty):
            collector.get(timeout=0.3)
    assert "installation_id" not in binding.config_path.read_text()


def test_real_worker_and_query_are_marked_as_daemon_requests(controlled_plugin, collector):
    binding = controlled_plugin
    plugin = binding.directory / "plugins/daemon_fixture.py"
    content = plugin.read_text()
    content = content.replace(
        "    secret = prompt(",
        """    from sparkrun.telemetry.client import send_event
    send_event(sctx.config, {"event_type": "fixture_worker"})
    secret = prompt(""",
    )
    content += """
def query(args, sctx, prompt):
    from sparkrun.telemetry.client import send_event
    send_event(sctx.config, {"event_type": "fixture_query"})
    return {"ok": True}
def query_loader(): return query
original_register = register
def register(v):
    original_register(v)
    from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op
    register_daemon_op(DaemonOp("query", query_loader,
        {"type": "object", "properties": {}, "additionalProperties": False}, kind="query", scope="read"))
"""
    plugin.write_text(content)
    with real_server(binding, extra=("--no-unix", "--tcp", "127.0.0.1:0")) as (_, client):
        assert collector.get(timeout=10)["event_type"] == "daemon_serve"
        job = client.call("plugins.daemon-fixture.wait", key="once")["job_id"]
        try:
            event = collector.get(timeout=10)
            assert event["event_type"] == "fixture_worker" and event["via_daemon"] is True
            wait_state(client, job, {"waiting_input"})
            assert client.call("plugins.daemon-fixture.wait", key="once")["job_id"] == job
            assert client.call("plugins.daemon-fixture.query") == {"ok": True}
            event = collector.get(timeout=10)
            assert event["event_type"] == "fixture_query" and event["via_daemon"] is True
            with pytest.raises(Empty):
                collector.get(timeout=0.2)
        finally:
            client.request("DELETE", "/v1/jobs/" + job)
            wait_state(client, job, {"cancelled"})


def test_systemd_workers_forward_telemetry_overrides(binding, monkeypatch):
    from sparkrun.daemon.supervisor import launch
    from sparkrun.core.application_profile import get_application_profile

    monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "1")
    monkeypatch.setenv("SPARKRUN_TELEMETRY_ENDPOINT", "http://127.0.0.1:1/")
    monkeypatch.setenv("SPARKRUN_TELEMETRY_KEY", "local-key")
    monkeypatch.setenv("SPARKRUN_TELEMETRY_TIMEOUT", "0.1")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-forward")
    profile = replace(get_application_profile(), env_prefix="TESTAPP", env_aliases={"NO_TELEMETRY": ("LEGACY_OPT_OUT",)})
    monkeypatch.setattr("sparkrun.core.application_profile.get_application_profile", lambda: profile)
    monkeypatch.setattr("sparkrun.telemetry.config.get_application_profile", lambda: profile)
    monkeypatch.setenv("TESTAPP_NO_TELEMETRY", "1")
    monkeypatch.setenv("TESTAPP_TELEMETRY_ENDPOINT", "http://127.0.0.1:2/")
    monkeypatch.setenv("LEGACY_OPT_OUT", "1")
    calls = []
    monkeypatch.setattr("sparkrun.daemon.supervisor.subprocess.run", lambda argv, **kwargs: calls.append(argv))
    launch(binding, "test-job", systemd=True)
    flags = calls[0]
    assert "--setenv=SPARKRUN_NO_TELEMETRY=1" in flags
    assert "--setenv=TESTAPP_NO_TELEMETRY=1" in flags
    assert "--setenv=TESTAPP_TELEMETRY_ENDPOINT=http://127.0.0.1:2/" in flags
    assert "--setenv=LEGACY_OPT_OUT=1" in flags
    assert "--setenv=SPARKRUN_TELEMETRY_TIMEOUT=0.1" in flags
    assert not any("UNRELATED_SECRET" in value for value in flags)


def test_daemon_emitter_is_best_effort(binding, monkeypatch):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.telemetry.emit import emit_daemon_serve_telemetry

    monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "0")

    def failure(*args, **kwargs):
        raise RuntimeError("delivery failed")

    monkeypatch.setattr("sparkrun.telemetry.emit.send_event", failure)
    emit_daemon_serve_telemetry(
        SparkrunConfig(binding.config_path),
        unix_socket=False,
        tcp=True,
        tls=False,
        remote_tcp=False,
        worker_supervisor="process",
        parent_managed=True,
        startup_ms=1,
    )
