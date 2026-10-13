"""Daemon capacity validation and exception-safe subscription admission."""

import asyncio
import json
from unittest.mock import Mock

import pytest
import yaml
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from sparkrun.daemon.server import create_app, Subscriptions
from sparkrun.daemon.auth import TokenStore
from sparkrun.daemon.jobs import Conflict
from test_daemon import binding as binding


def test_log_stream_forwards_source_filters_without_rewriting_hosts(binding, monkeypatch):
    from sparkrun.api import LogLine

    _, token = TokenStore(binding).create("reader", scope="read")
    headers = {"Authorization": "Bearer " + token}
    reader = Mock(side_effect=lambda **kw: iter([LogLine("user@worker", "workload_node_1", "ready", role="node_1", rank=1)]))
    monkeypatch.setattr("sparkrun.api.logs", reader)
    params = {
        "cluster_id": "workload",
        "hosts": "head,user@worker",
        "host": "worker",
        "container": "workload_node_1",
        "follow": "false",
        "tail": "10",
    }
    with TestClient(create_app(binding, start_workers=False)) as client:
        response = client.get("/v1/streams/logs", params=params, headers=headers)
    assert response.status_code == 200
    records = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert records[0]["type"] == "log" and records[0]["rank"] == 1
    args = reader.call_args.kwargs
    assert args["hosts"] == ["head", "user@worker"]
    assert (args["host"], args["container"], args["follow"], args["tail"]) == ("worker", "workload_node_1", False, 10)


@pytest.mark.parametrize(
    "query", ["host=", "host=%20", "container=", "container=%20", "scope=workers", "tail=-1", "tail=abc", "follow=yes", "unknown=value"]
)
def test_log_stream_rejects_bad_queries_before_opening_reader(binding, monkeypatch, query):
    _, token = TokenStore(binding).create("reader", scope="read")
    reader = Mock(side_effect=lambda **kw: iter(()))
    monkeypatch.setattr("sparkrun.api.logs", reader)
    with TestClient(create_app(binding, start_workers=False)) as client:
        response = client.get("/v1/streams/logs?" + query, headers={"Authorization": "Bearer " + token})
    assert response.status_code == 422, response.text
    reader.assert_not_called()


def test_log_stream_reports_failed_finite_reader(binding, monkeypatch):
    import sys
    from sparkrun import api
    from sparkrun.orchestration import logs

    def target(*args, **kwargs):
        return api.LogTarget("cid", "sglang", (api.LogSource("h1", "c"),), Mock(), {}, kwargs["cancel"])

    monkeypatch.setattr("sparkrun.api._logs.open_log_target", target)
    monkeypatch.setattr(logs, "build_read_command", lambda *a, **kw: [sys.executable, "-c", "import sys; sys.exit(7)"])
    _, token = TokenStore(binding).create("reader", scope="read")
    with TestClient(create_app(binding, start_workers=False)) as client:
        response = client.get("/v1/streams/logs?follow=false", headers={"Authorization": "Bearer " + token})
    assert response.status_code == 200
    records = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert records[-1]["type"] == "error"


def test_closing_daemon_stream_cancels_silent_finite_reader(monkeypatch):
    import sys
    import threading
    from sparkrun import api
    from sparkrun.daemon.streams import LogStream
    from sparkrun.orchestration import logs

    def target(*args, **kwargs):
        return api.LogTarget("cid", "sglang", (api.LogSource("h1", "c"),), Mock(), {}, kwargs["cancel"])

    monkeypatch.setattr("sparkrun.api._logs.open_log_target", target)
    monkeypatch.setattr(logs, "build_read_command", lambda *a, **kw: [sys.executable, "-c", "import time; time.sleep(60)"])
    original = logs._spawn
    started = threading.Event()
    processes = []

    def spawn(cmd):
        process = original(cmd)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(logs, "_spawn", spawn)
    stream = LogStream({"follow": False}, api.default_sctx())
    try:
        assert started.wait(5)
    finally:
        stream.close()
    assert stream.done.is_set() and not stream.thread.is_alive()
    assert stream.error is None
    assert all(p.poll() is not None for p in processes)


@pytest.mark.parametrize("kind", ["logs", "monitor"])
def test_stream_setup_failure_recovers_after_config_is_restored(binding, monkeypatch, kind):
    binding.config_path.write_text("features:\n  daemon: true\ndaemon:\n  max_streams: 1\n")
    original = binding.config_path.read_bytes()
    _, token = TokenStore(binding).create("fixture", scope="admin")
    headers = {"Authorization": "Bearer " + token}
    logs = Mock(side_effect=lambda **kwargs: iter(()))
    monkeypatch.setattr("sparkrun.api.logs", logs)
    with TestClient(create_app(binding, start_workers=False), raise_server_exceptions=False) as client:
        try:
            binding.config_path.write_text("- temporarily incomplete config\n")
            response = client.get("/v1/streams/" + kind, headers=headers)
            assert response.status_code == 422, response.text
        finally:
            binding.config_path.write_bytes(original)
        assert client.post("/v1/catalog/clusters", json={}, headers=headers).status_code == 200
        for _ in range(2):
            response = client.get("/v1/streams/logs?follow=false", headers=headers)
            assert response.status_code == 200, response.text
        assert logs.call_count == 2


@pytest.mark.parametrize("fail_at", ["start", "body", "cleanup"])
def test_subscription_released_on_send_and_cleanup_failures(fail_at):
    subscriptions = Subscriptions(1)
    closed = []

    async def producer():
        try:
            yield "event\n"
        finally:
            closed.append(True)
            if fail_at == "cleanup":
                raise RuntimeError("cleanup failed")

    def admitted_factory():
        with pytest.raises(Conflict, match="Subscriber limit"):
            subscriptions.response(producer)
        return producer()

    response = subscriptions.response(admitted_factory)

    async def send(message):
        if (fail_at == "start" and message["type"] == "http.response.start") or (
            fail_at == "body" and message["type"] == "http.response.body"
        ):
            raise OSError("client disconnected")

    async def receive():
        return {"type": "http.disconnect"}

    with pytest.raises(RuntimeError if fail_at == "cleanup" else ClientDisconnect):
        asyncio.run(response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send))
    assert not subscriptions.active
    assert closed == ([] if fail_at == "start" else [True])
    retry = subscriptions.response(lambda: empty())
    asyncio.run(retry({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, discard))
    assert not subscriptions.active


async def empty():
    if False:
        yield ""


async def discard(message):
    pass


COUNTS = [
    "max_jobs",
    "max_services",
    "max_streams",
    "max_query_threads",
    "max_queued_queries",
    "max_body_bytes",
    "retained_jobs",
    "max_queued_jobs",
    "max_events",
    "max_event_bytes",
]


@pytest.mark.parametrize("key", COUNTS)
@pytest.mark.parametrize("value", [0, -1, 0.5, 1024.5, True, None, float("inf"), "nan", "no"])
def test_daemon_rejects_invalid_integer_limits(binding, key, value):
    binding.config_path.write_text(yaml.safe_dump({"daemon": {key: value}}))
    with pytest.raises(ValueError, match="daemon\\." + key):
        create_app(binding, start_workers=False)


@pytest.mark.parametrize("value", [2, 2.0, "2", "2.0"])
def test_integral_limits_are_normalized_and_can_claim_jobs(binding, value):
    from sparkrun.core.daemon_registry import DaemonOp

    binding.config_path.write_text(yaml.safe_dump({"daemon": {"max_jobs": value}}))
    app = create_app(binding, start_workers=False)
    coordinator, store = app.state.coordinator, app.state.store
    assert type(coordinator.max_active) is int
    assert coordinator.max_active == 2
    identifier = store.submit(DaemonOp("test.job", lambda: None), {}, "fixture", "", {})
    assert store.claim(max_active=coordinator.max_active)["id"] == identifier


@pytest.mark.parametrize("key", ["cancel_grace", "worker_start_timeout", "query_timeout", "job_retention_days"])
@pytest.mark.parametrize("value", [0, -1, True, None, float("inf"), "nan"])
def test_daemon_rejects_invalid_durations(binding, key, value):
    binding.config_path.write_text(yaml.safe_dump({"daemon": {key: value}}))
    with pytest.raises(ValueError, match="daemon\\." + key):
        create_app(binding, start_workers=False)
