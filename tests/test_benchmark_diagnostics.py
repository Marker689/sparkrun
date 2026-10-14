"""Capture inference logs while benchmark-owned sources are still readable."""

from contextlib import nullcontext
from dataclasses import replace
import json
import sys
from unittest.mock import Mock

import pytest

from sparkrun import api
from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration
from sparkrun.core.log_source import LogLine, LogSource
from sparkrun.core.operation import CURRENT
from sparkrun.diagnostics.session import capture_before_cleanup
from test_benchmark_startup_collection import bench_env as bench_env


@pytest.fixture
def logs(bench_env, monkeypatch):
    def page(*args, **kwargs):
        # Stopping inference removes the sources: capture-exit reads are too late.
        assert not bench_env.stop.called
        return [LogLine(host="localhost", container="inference", text="last inference output")]

    target = Mock(sources=(LogSource("localhost", "inference"),), page=Mock(side_effect=page))
    reader = Mock(return_value=target)
    monkeypatch.setattr("sparkrun.api._logs.open_log_target", reader)
    return reader, target


@pytest.mark.parametrize("failure", [None, "startup", "schedule", "parse", "state", "export", "cleanup", "publication", "notification"])
def test_snapshot_is_flushed_before_cleanup_and_not_repeated(bench_env, logs, tmp_path, monkeypatch, failure):
    env = bench_env
    path = tmp_path / "benchmark.ndjson"
    options = env.options
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})
    if failure == "startup":
        env.probe.side_effect = RuntimeError("inference failed")
    elif failure == "schedule":
        env.fw.build_benchmark_command.side_effect = lambda *a, **kw: [sys.executable, "-c", "raise SystemExit(1)"]
    elif failure == "parse":
        env.fw.parse_results.side_effect = ValueError("invalid result")
    elif failure == "state":
        monkeypatch.setattr("sparkrun.api._benchmark._save_completed_results", Mock(side_effect=OSError("state full")))
    elif failure == "export":
        options = replace(options, output_file=str(tmp_path))
    elif failure == "publication":
        register_benchmark_integration(BenchmarkIntegration("test", on_complete=Mock(side_effect=RuntimeError("publish failed"))))
        options = replace(options, integrations={"test": {}})
    elif failure == "notification":

        def notify(event):
            if event.kind == "run_complete":
                raise RuntimeError("notification failed")

        options = replace(options, progress_callback=notify)

    def stop(*args, **kwargs):
        assert '"_type": "workload_log"' in path.read_text()
        if failure == "cleanup":
            raise OSError("stop failed")

    env.stop.side_effect = stop
    with pytest.raises(api.SparkrunError) if failure else nullcontext():
        with api.capture_diagnostics(path, sctx=env.sctx, log_tail=23):
            api.benchmark(options, sctx=env.sctx)
    reader, target = logs
    reader.assert_called_once_with(env.launch.cluster_id, scope="all", sctx=env.sctx)
    target.page.assert_called_once_with(target.sources[0], skip=0, count=23, timeout=30)
    env.stop.assert_called_once()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert sum(record["_type"] == "workload_log" for record in records) == 1
    assert records[-1]["operation_success"] is (failure is None)


@pytest.mark.parametrize("unavailable", ["target", "source"])
def test_diagnostic_failure_does_not_replace_benchmark_error_or_prevent_cleanup(bench_env, logs, tmp_path, unavailable):
    env = bench_env
    reader, target = logs
    (reader if unavailable == "target" else target.page).side_effect = OSError("logs unavailable")
    env.fw.parse_results.side_effect = ValueError("primary failure")
    env.stop.side_effect = OSError("secondary cleanup failure")
    with pytest.raises(api.SparkrunError, match="primary failure") as caught:
        with api.capture_diagnostics(tmp_path / "partial.ndjson", sctx=env.sctx) as capture:
            api.benchmark(env.options, sctx=env.sctx)
    assert capture.result.partial
    assert any(section.get("detail") == "logs unavailable" for section in capture.result.sections)
    assert "secondary cleanup failure" in " ".join(caught.value.__cause__.__notes__)
    reader.assert_called_once()
    env.stop.assert_called_once()


@pytest.mark.parametrize("mode", ["disabled", "dry_run", "no_stop", "skip_run", "reused", "zero_tail", "other_operation"])
def test_snapshot_honors_capture_opt_in_and_workload_ownership(bench_env, logs, tmp_path, mode):
    from copy import copy

    env = bench_env
    options = replace(env.options, **{mode: True}) if mode in {"dry_run", "no_stop", "skip_run"} else env.options
    if mode == "reused":
        env.run.return_value.already_running = True
    capture_context = env.sctx
    if mode == "other_operation":
        capture_context = copy(env.sctx)
    context = (
        nullcontext()
        if mode == "disabled"
        else api.capture_diagnostics(tmp_path / "optional.ndjson", sctx=capture_context, log_tail=0 if mode == "zero_tail" else 200)
    )
    with context:
        if mode == "other_operation":
            from sparkrun.core.operation import OperationState

            env.sctx.operation = OperationState()
        assert api.benchmark(options, sctx=env.sctx).success
    logs[0].assert_not_called()
    assert env.stop.call_count == int(mode in {"disabled", "zero_tail", "other_operation"})


@pytest.mark.parametrize("stage", ["measurement", "export"])
@pytest.mark.parametrize("cancel", [False, True])
def test_interrupted_benchmark_skips_new_reads_but_still_cleans_up(bench_env, logs, tmp_path, monkeypatch, stage, cancel):
    primary = KeyboardInterrupt()

    def interrupt(*args, **kwargs):
        if cancel:
            CURRENT.get().cancel.set()
        raise primary

    if stage == "measurement":
        bench_env.fw.parse_results.side_effect = interrupt
    else:
        monkeypatch.setattr("sparkrun.benchmarking.base._write_measurement", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        with api.capture_diagnostics(tmp_path / "interrupted.ndjson", sctx=bench_env.sctx) as capture:
            api.benchmark(bench_env.options, sctx=bench_env.sctx)
    assert caught.value is primary
    assert any(section["status"] == "cancelled" for section in capture.result.sections)
    logs[0].assert_not_called()
    bench_env.stop.assert_called_once()


@pytest.mark.parametrize("already_failed", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_interrupt_during_snapshot_preserves_primary_and_attempts_cleanup(bench_env, logs, tmp_path, already_failed, cleanup_fails):
    env = bench_env
    interruption = KeyboardInterrupt()
    logs[1].page.side_effect = interruption
    if already_failed:
        env.fw.parse_results.side_effect = ValueError("primary failure")
    if cleanup_fails:
        env.stop.side_effect = OSError("cleanup failed")
    with pytest.raises(api.SparkrunError if already_failed else KeyboardInterrupt) as caught:
        with api.capture_diagnostics(tmp_path / "read-interrupted.ndjson", sctx=env.sctx):
            api.benchmark(env.options, sctx=env.sctx)
    if already_failed:
        assert "primary failure" in str(caught.value)
    else:
        assert caught.value is interruption
    if cleanup_fails:
        primary = caught.value.__cause__ if already_failed else interruption
        assert "cleanup failed" in " ".join(primary.__notes__)
    logs[0].assert_called_once()
    env.stop.assert_called_once()


def test_expired_deadline_skips_snapshot_but_allows_cleanup(bench_env, logs, tmp_path):
    cleanup = Mock()
    with api.capture_diagnostics(tmp_path / "deadline.ndjson", sctx=bench_env.sctx) as capture:
        CURRENT.get().deadline = 0
        with capture_before_cleanup("test-job", sctx=bench_env.sctx):
            cleanup()
    logs[0].assert_not_called()
    cleanup.assert_called_once()
    assert capture.result.partial


def test_cleanup_refreshes_logs_collected_earlier_in_the_same_capture(bench_env, logs, tmp_path):
    env = bench_env
    path = tmp_path / "fresh.ndjson"
    env.fw.parse_results.side_effect = ValueError("measurement failed")

    def stop(*args, **kwargs):
        assert "new output before cleanup" in path.read_text()

    env.stop.side_effect = stop
    with pytest.raises(api.SparkrunError, match="measurement failed"):
        with api.capture_diagnostics(path, sctx=env.sctx) as capture:
            capture.collect_logs(env.launch.cluster_id)
            logs[1].page.return_value = [LogLine(host="localhost", container="inference", text="new output before cleanup")]
            logs[1].page.side_effect = None
            api.benchmark(env.options, sctx=env.sctx)
    assert logs[0].call_count == logs[1].page.call_count == 2
    assert "new output before cleanup" in path.read_text()
    env.stop.assert_called_once()
