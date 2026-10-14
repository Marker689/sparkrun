"""Bundles are shared, isolated, bounded, and independent of workload outcome."""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from threading import Barrier
from unittest.mock import Mock

import pytest
from scitrera_app_framework import Variables

from sparkrun import api
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.context import SparkrunContext
from sparkrun.core.log_source import LogLine, LogSource
from sparkrun.core.operation import CURRENT, OperationState, submit
from sparkrun.orchestration.ssh import RemoteResult


@pytest.fixture
def sctx(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("hosts: [host-a]\ncache_dir: " + str(tmp_path / "cache") + "\n")
    return SparkrunContext(variables=Variables(), config=SparkrunConfig(path))


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_standalone_reports_partial_hosts_and_redacts_every_section(sctx, tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "key-from-environment")
    remote = Mock(
        return_value=[
            RemoteResult("host-a", 0, "DIAG_COMPLETE=1\nDIAG_HOSTNAME=key-from-environment\n", ""),
            RemoteResult("host-b", 1, "", "Bearer leaked-secret"),
        ]
    )
    monkeypatch.setattr("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel", remote)
    path = tmp_path / "bundle.ndjson"
    result = api.collect_diagnostics(path, hosts=["host-a", "host-b"], sctx=sctx)
    assert result.partial
    assert result.hosts == {"host-a": True, "host-b": False}
    assert result.sections[-1]["status"] == "error"
    assert records(path)[0]["schema_version"] == 1
    assert records(path)[-1]["_type"] == "bundle_complete"
    assert "key-from-environment" not in path.read_text()
    assert "leaked-secret" not in path.read_text()
    if os.name != "nt":
        assert path.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        api.collect_diagnostics(path, hosts=["host-a"], sctx=sctx)
    assert sctx.timing is None
    assert CURRENT.get() is None


def test_captures_isolate_concurrent_logs_feedback_and_propagated_fanout(sctx, tmp_path):
    from copy import copy

    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    barrier = Barrier(2)

    def run(label):
        context = copy(sctx)
        context.operation = None
        path = tmp_path / (label + ".ndjson")
        with api.capture_diagnostics(path, sctx=context) as capture:
            barrier.wait(timeout=10)
            logging.getLogger("test.capture").warning("only-" + label)
            CURRENT.get().feedback({"type": "feedback", "message": "structured-" + label})
            with ThreadPoolExecutor(1) as pool:
                submit(pool, logging.getLogger("test.capture").warning, "fanout-" + label).result()
            with api.capture_diagnostics(tmp_path / "ignored-nested.ndjson", sctx=context) as nested:
                assert capture is nested
            assert capture.result is None
        assert capture.result is not None
        return path.read_text()

    with ThreadPoolExecutor(2) as pool:
        left, right = list(pool.map(run, ["left", "right"]))
    for label, other, text in (("left", "right", left), ("right", "left", right)):
        for prefix in ("only-", "structured-", "fanout-"):
            assert prefix + label in text
            assert prefix + other not in text
    assert root.level == level and root.handlers == handlers
    assert CURRENT.get() is None
    assert not (tmp_path / "ignored-nested.ndjson").exists()


def test_capture_limits_keep_valid_completion_and_do_not_change_original_exception(sctx, tmp_path):
    path = tmp_path / "bounded.ndjson"
    original = RuntimeError("original-operation-error")
    with pytest.raises(RuntimeError) as caught:
        with api.capture_diagnostics(path, sctx=sctx, max_bytes=16384) as capture:
            for i in range(100):
                CURRENT.get().feedback({"type": "feedback", "message": "x" * 1000, "index": i})
            raise original
    assert caught.value is original
    assert path.stat().st_size <= 16384
    assert capture.result.partial and capture.result.truncated
    assert records(path)[-1] == {**records(path)[-1], "_type": "bundle_complete", "operation_success": False, "truncated": True}


def test_live_operation_secrets_and_text_forms_redacted(sctx, tmp_path):
    from pathlib import Path

    path = tmp_path / "secrets.ndjson"
    with api.capture_diagnostics(path, sctx=sctx) as capture:
        CURRENT.get().secrets.add("arrived-from-prompt")
        capture.writer.emit("object_value", {"path": Path("arrived-from-prompt")})
        CURRENT.get().feedback(
            {
                "type": "feedback",
                "password": "dict-secret",
                "message": "arrived-from-prompt --api-key 'quoted secret' https://alice:password@host/path?token=query-secret Bearer auth-secret",
            }
        )
    content = path.read_text()
    for secret in ("arrived-from-prompt", "dict-secret", "quoted secret", "alice:password", "query-secret", "auth-secret"):
        assert secret not in content


def test_shared_log_pages_continue_after_a_source_fails(sctx, tmp_path, monkeypatch):
    a, b = LogSource("a", "container-a"), LogSource("b", "container-b")
    target = Mock(sources=(a, b))
    target.page.side_effect = [OSError("source unavailable"), [LogLine(host="b", container="container-b", text="read through shared API")]]
    monkeypatch.setattr("sparkrun.api._logs.open_log_target", Mock(return_value=target))
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.load_job_metadata",
        lambda *a, **kw: {"hosts": ["control"], "executor": "k8s", "executor_user_scoped": False},
    )
    monkeypatch.setattr("sparkrun.orchestration.executor.cluster_status_scope", lambda *a, **kw: "k8s")
    probe = Mock(side_effect=AssertionError("native sources must not trigger SSH probes"))
    monkeypatch.setattr("sparkrun.diagnostics.spark_collector.collect_spark_diagnostics", probe)
    path = tmp_path / "native.ndjson"
    result = api.collect_diagnostics(path, workload_id="saved-workload", sctx=sctx, log_tail=75)
    assert result.partial
    assert any(s["section"] == "hosts" and s["status"] == "skipped" for s in result.sections)
    assert any(s["section"].startswith("logs:") and s["status"] == "error" for s in result.sections)
    assert "read through shared API" in path.read_text()
    assert target.page.call_count == 2
    assert target.page.call_args.kwargs == {"skip": 0, "count": 75, "timeout": 30}
    probe.assert_not_called()


def test_unknown_workload_never_falls_back_to_default_hosts(sctx, tmp_path, monkeypatch):
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **kw: None)
    probe = Mock(side_effect=AssertionError("must not probe unrelated hosts"))
    monkeypatch.setattr("sparkrun.diagnostics.spark_collector.collect_spark_diagnostics", probe)
    result = api.collect_diagnostics(tmp_path / "unknown.ndjson", workload_id="unknown", log_tail=0, sctx=sctx)
    assert result.partial
    assert any(s["section"] == "target" and s["status"] == "error" for s in result.sections)
    probe.assert_not_called()


def test_cancelled_capture_finishes_without_collecting_more_logs(sctx, tmp_path, monkeypatch):
    sctx.operation = OperationState()
    reader = Mock(side_effect=AssertionError("must not start cancelled read"))
    monkeypatch.setattr("sparkrun.api._logs.open_log_target", reader)
    path = tmp_path / "cancelled.ndjson"
    with pytest.raises(InterruptedError):
        with api.capture_diagnostics(path, sctx=sctx) as capture:
            capture._workloads.add("known-workload")
            sctx.operation.cancel.set()
            sctx.operation.check()
    assert records(path)[-1]["operation_success"] is False
    assert any(s["status"] == "cancelled" for s in capture.result.sections)
    reader.assert_not_called()


def test_python_benchmark_capture_records_failure_without_frontend(sctx, tmp_path, monkeypatch):
    def measure(*args, **kwargs):
        CURRENT.get().feedback({"type": "feedback", "message": "test case completed"})
        raise api.BenchmarkFailed("measurement failed", exit_code=1)

    monkeypatch.setattr("sparkrun.api._benchmark._execute_benchmark", measure)
    path = tmp_path / "benchmark.ndjson"
    with pytest.raises(api.BenchmarkFailed, match="measurement failed"):
        with api.capture_diagnostics(path, sctx=sctx):
            api.benchmark(api.BenchmarkOptions(recipe="test/model", dry_run=True), sctx=sctx)
    data = records(path)
    assert any(r.get("phase") == "benchmark" and r.get("status") == "error" for r in data)
    assert "test case completed" in path.read_text()
    assert data[-1]["operation_success"] is False


def test_write_failure_does_not_replace_workload_exception(sctx, tmp_path):
    path = tmp_path / "disk-full.ndjson"
    original = RuntimeError("operation failed")
    with pytest.raises(RuntimeError) as caught:
        with api.capture_diagnostics(path, sctx=sctx) as capture:
            file = capture.writer._fh
            capture.writer._fh = Mock(write=Mock(side_effect=OSError("disk full")), close=file.close)
            raise original
    assert caught.value is original
    assert capture.result.partial
    assert any(s["section"] == "storage" and s["status"] == "error" for s in capture.result.sections)


def test_config_source_failure_does_not_hide_other_sections(sctx, tmp_path, monkeypatch):
    monkeypatch.setattr(sctx.cluster_manager, "list_clusters", Mock(side_effect=OSError("inventory unavailable")))
    path = tmp_path / "config-partial.ndjson"
    with api.capture_diagnostics(path, sctx=sctx) as capture:
        pass
    assert capture.result.partial
    assert any(s["section"] == "clusters" and s["status"] == "error" for s in capture.result.sections)
    assert any(r["_type"] == "config_registries" for r in records(path))


def test_keyboard_interrupt_during_probe_finalizes_and_propagates(sctx, tmp_path, monkeypatch):
    monkeypatch.setattr("sparkrun.diagnostics.spark_collector.collect_spark_diagnostics", Mock(side_effect=KeyboardInterrupt))
    path = tmp_path / "interrupted.ndjson"
    with pytest.raises(KeyboardInterrupt):
        api.collect_diagnostics(path, hosts=["host-a"], sctx=sctx)
    data = records(path)
    assert data[-1]["_type"] == "bundle_complete"
    assert data[-1]["partial"] is True
    assert data[-1]["operation_success"] is False


def test_nested_standalone_dry_run_never_probes(sctx, tmp_path, monkeypatch):
    hosts = Mock(side_effect=AssertionError("dry run must not probe"))
    logs = Mock(side_effect=AssertionError("dry run must not read logs"))
    monkeypatch.setattr("sparkrun.diagnostics.spark_collector.collect_spark_diagnostics", hosts)
    monkeypatch.setattr("sparkrun.api._logs.open_log_target", logs)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **kw: {"hosts": ["host-a"]})
    path = tmp_path / "outer.ndjson"
    with api.capture_diagnostics(path, sctx=sctx):
        result = api.collect_diagnostics(tmp_path / "inner.ndjson", workload_id="saved", dry_run=True, sctx=sctx)
        assert result.output_path == str(path)
    hosts.assert_not_called()
    logs.assert_not_called()


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit(130)])
def test_interrupt_does_not_start_failure_log_reads(sctx, tmp_path, monkeypatch, interrupt):
    logs = Mock(side_effect=AssertionError("must not read after interrupt"))
    monkeypatch.setattr("sparkrun.api._logs.open_log_target", logs)
    with pytest.raises((KeyboardInterrupt, SystemExit)):
        with api.capture_diagnostics(tmp_path / "interrupt.ndjson", sctx=sctx) as capture:
            capture._workloads.add("saved-workload")
            raise interrupt
    assert capture.result.partial
    logs.assert_not_called()


def test_benchmark_unsuccessful_result_is_captured_without_changing_return(sctx, tmp_path, monkeypatch):
    outcome = api.BenchmarkResult(success=False, benchmark_id="bench-test", category="tools", framework="tool-eval-bench", profile=None)
    monkeypatch.setattr("sparkrun.api._benchmark._execute_benchmark", lambda *a, **kw: object())
    monkeypatch.setattr("sparkrun.api._benchmark._build_result", lambda *a: outcome)
    monkeypatch.setattr("sparkrun.telemetry.emit_benchmark_telemetry", lambda *a, **kw: None)
    path = tmp_path / "benchmark-outcome.ndjson"
    with api.capture_diagnostics(path, sctx=sctx):
        returned = api.benchmark(api.BenchmarkOptions(recipe="test", dry_run=True), sctx=sctx)
    assert returned is outcome
    data = records(path)
    benchmark = next(r for r in data if r["_type"] == "benchmark_result")
    assert benchmark["success"] is False and "workload_id" in benchmark
    assert data[-1]["operation_success"] is False
