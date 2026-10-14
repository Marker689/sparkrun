"""Post-measurement failures preserve results, cleanup and measurement identity."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
import json

import pytest

from sparkrun.api import benchmark, resume_benchmark, BenchmarkFinalizationFailed, BenchmarkIntegrationFailed
from sparkrun.benchmarking.run_state import BenchmarkRunState
from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration, STATE_KEY
from test_benchmark_startup_collection import bench_env as bench_env
from test_benchmark_api_contract import scheduled_env as scheduled_env


@pytest.fixture(autouse=True)
def integration_registry(monkeypatch):
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("lifecycle", ["normal", "no_stop", "skip_run"])
def test_export_failure_preserves_results_and_honors_cleanup(request, tmp_path, scheduled, lifecycle):
    env = request.getfixturevalue("scheduled_env" if scheduled else "bench_env")
    publish = Mock()
    register_benchmark_integration(BenchmarkIntegration("review", on_complete=publish))
    with pytest.raises(BenchmarkFinalizationFailed) as caught:
        benchmark(
            replace(
                env.options,
                output_file=str(tmp_path),
                integrations={"review": {}},
                no_stop=lifecycle == "no_stop",
                skip_run=lifecycle == "skip_run",
            ),
            sctx=env.sctx,
        )
    error = caught.value
    assert error.stage == "export" and isinstance(error.__cause__, IsADirectoryError)
    assert error.result.success and error.result.results == ({"rows": [env.rows]} if scheduled else env.rows)
    assert env.stop.call_count == int(lifecycle == "normal")
    publish.assert_not_called()
    if scheduled:
        state = BenchmarkRunState.load(error.result.benchmark_id, str(env.sctx.config.cache_dir))
        assert state.extras["measurement_complete"]
        assert (state.state_dir(str(env.sctx.config.cache_dir)) / "result.yaml").is_file()
        retried = resume_benchmark(error.result.benchmark_id, sctx=env.sctx)
        assert retried.results == error.result.results
        publish.assert_called_once()
        env.fw.parse_results.assert_called_once()  # no remeasurement on publication retry


def test_secondary_format_failure_retains_already_written_paths(bench_env, monkeypatch):
    env = bench_env
    values = {**env.rows, "json": {"rate": 12}, "csv": "rate\n12\n"}
    env.fw.parse_results.side_effect = lambda *a, **kw: values
    original = Path.write_text

    def write(path, *a, **kw):
        if path.suffix == ".csv":
            raise OSError("CSV disk full")
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, "write_text", write)
    with pytest.raises(BenchmarkFinalizationFailed, match="CSV disk full") as caught:
        benchmark(env.options, sctx=env.sctx)
    assert caught.value.stage == "export" and caught.value.result.results == values
    assert set(caught.value.result.outputs) == {"yaml", "json"}
    assert all(Path(path).is_file() for path in caught.value.result.outputs.values())
    env.stop.assert_called_once()


@pytest.mark.parametrize("export_fails", [False, True])
def test_cleanup_failure_is_retained_without_masking_primary_export_error(bench_env, tmp_path, export_fails):
    env = bench_env
    env.stop.side_effect = OSError("stop unavailable")
    options = replace(env.options, output_file=str(tmp_path)) if export_fails else env.options
    with pytest.raises(BenchmarkFinalizationFailed) as caught:
        benchmark(options, sctx=env.sctx)
    assert caught.value.stage == ("export" if export_fails else "cleanup")
    assert caught.value.errors["cleanup"] == "stop unavailable"
    if export_fails:
        assert isinstance(caught.value.__cause__, IsADirectoryError)
    assert caught.value.result.success and caught.value.result.results == env.rows
    env.stop.assert_called_once()


def test_state_commit_failure_still_stops_owned_inference(scheduled_env, monkeypatch):
    env = scheduled_env
    monkeypatch.setattr("sparkrun.api._benchmark._save_completed_results", Mock(side_effect=OSError("state full")))
    with pytest.raises(BenchmarkIntegrationFailed, match="state full") as caught:
        benchmark(env.options, sctx=env.sctx)
    assert isinstance(caught.value, BenchmarkFinalizationFailed) and caught.value.stage == "state"
    assert caught.value.integration == "<state>" and caught.value.result.success
    env.stop.assert_called_once()


def test_export_interrupt_cleans_up_once_and_propagates(bench_env, monkeypatch):
    monkeypatch.setattr("sparkrun.benchmarking.base._write_measurement", Mock(side_effect=KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        benchmark(bench_env.options, sctx=bench_env.sctx)
    bench_env.stop.assert_called_once()


def test_dry_run_never_exports_or_stops(bench_env, monkeypatch):
    export = Mock(side_effect=AssertionError("no export during preview"))
    monkeypatch.setattr("sparkrun.benchmarking.base.export_results", export)
    assert benchmark(replace(bench_env.options, dry_run=True), sctx=bench_env.sctx).success
    export.assert_not_called()
    bench_env.stop.assert_not_called()


def test_completion_notification_has_same_result_bearing_error_for_both_entrypoints(scheduled_env):
    env = scheduled_env
    register_benchmark_integration(BenchmarkIntegration("review", on_complete=lambda _: None))

    def notify(event):
        if event.kind == "run_complete":
            raise RuntimeError("frontend disconnected")

    with pytest.raises(BenchmarkFinalizationFailed) as initial:
        benchmark(replace(env.options, integrations={"review": {}}, progress_callback=notify), sctx=env.sctx)
    with pytest.raises(BenchmarkFinalizationFailed) as resumed:
        resume_benchmark(initial.value.result.benchmark_id, sctx=env.sctx, progress_callback=notify)
    assert initial.value.stage == resumed.value.stage == "notification"
    assert initial.value.result.results == resumed.value.result.results == {"rows": [env.rows]}
    for error in (initial.value, resumed.value):
        assert isinstance(error.__cause__, RuntimeError) and error.result.success
    env.stop.assert_called_once()


@pytest.mark.parametrize("failing_retry", [False, True])
def test_publication_retries_keep_legacy_measurement_interval(monkeypatch, failing_retry):
    from sparkrun.api._context import default_sctx
    from test_distribution_api_contracts import _completed_state

    original, retry_one, retry_two = ("2026-01-0%dT12:00:00+00:00" % day for day in (1, 2, 3))
    monkeypatch.setattr("sparkrun.benchmarking.run_state._now_iso", lambda: original)
    sctx = default_sctx()
    state = _completed_state(sctx, {})
    seen = []

    def complete(context):
        seen.append((context.result.measured_at, context.result.completed_at))
        if failing_retry:
            raise RuntimeError("publication unavailable")

    register_benchmark_integration(BenchmarkIntegration("review", on_complete=complete))
    for clock in (retry_one, retry_two):
        monkeypatch.setattr("sparkrun.benchmarking.run_state._now_iso", lambda clock=clock: clock)
        if failing_retry:
            with pytest.raises(BenchmarkIntegrationFailed):
                resume_benchmark(state.benchmark_id, sctx=sctx)
        else:
            assert resume_benchmark(state.benchmark_id, sctx=sctx).success
    assert seen == [(original, original), (original, original)]
    saved = BenchmarkRunState.load(state.benchmark_id, str(sctx.config.cache_dir))
    assert saved.updated_at == retry_two
    assert saved.extras["measurement_started_at"] == saved.extras["measurement_completed_at"] == original


def test_new_measurement_interval_is_unchanged_by_publication(scheduled_env):
    env = scheduled_env
    seen = []
    register_benchmark_integration(
        BenchmarkIntegration("review", on_complete=lambda ctx: seen.append((ctx.result.measured_at, ctx.result.completed_at)))
    )
    result = benchmark(replace(env.options, integrations={"review": {}}), sctx=env.sctx)
    resume_benchmark(result.benchmark_id, sctx=env.sctx)
    assert seen[0] == seen[1] and all(seen[0])
    assert seen[0][0] <= seen[0][1]


def test_arena_retry_metadata_uses_measurement_interval(monkeypatch):
    from sparkrun.api._context import default_sctx
    from sparkrun.plugins.sparkarena.integration import complete
    from test_distribution_api_contracts import _completed_state

    original = "2026-01-01T12:00:00+00:00"
    monkeypatch.setattr("sparkrun.benchmarking.run_state._now_iso", lambda: original)
    sctx = default_sctx()
    state = _completed_state(sctx, {"local_test": True})
    state.extras[STATE_KEY]["review"]["data"] = {
        "submission_id": "test-local-submission",
        "effective_recipe_text": "model: test/model\n",
        "metadata_json": {"timing": {"start": original, "end": original}},
    }
    state.save(str(sctx.config.cache_dir))
    (state.state_dir(str(sctx.config.cache_dir)) / "result.yaml").write_text("csv: 'rate,12'\n")
    register_benchmark_integration(BenchmarkIntegration("review", on_complete=complete))
    metadata = []
    for day in (2, 3):
        monkeypatch.setattr("sparkrun.benchmarking.run_state._now_iso", lambda day=day: "2026-01-0%dT12:00:00+00:00" % day)
        resume_benchmark(state.benchmark_id, sctx=sctx)
        metadata.append(json.loads((sctx.config.cache_dir / "benchmarks/test-local-submission/metadata.json").read_text()))
    assert metadata[0]["benchmark"]["measured_at"] == metadata[1]["benchmark"]["measured_at"] == original
    assert metadata[0]["timing"] == metadata[1]["timing"] == {"start": original, "end": original, "duration": 0}


def test_failed_stop_result_is_a_cleanup_failure(bench_env):
    from sparkrun.api import StopResult

    bench_env.stop.return_value = StopResult("test-job", ("localhost",), 0, hosts_failed=("localhost",))
    with pytest.raises(BenchmarkFinalizationFailed, match="cleanup incomplete") as caught:
        benchmark(bench_env.options, sctx=bench_env.sctx)
    assert caught.value.stage == "cleanup" and caught.value.result.results == bench_env.rows
    bench_env.stop.assert_called_once()


def test_incomplete_resume_export_failure_preserves_results_without_stopping_unowned_inference(scheduled_env, tmp_path):
    import sys
    from sparkrun.api import BenchmarkFailed

    env = scheduled_env
    register_benchmark_integration(BenchmarkIntegration("review", on_complete=lambda _: None))
    original = env.fw.build_benchmark_command.side_effect
    env.fw.build_benchmark_command.side_effect = lambda *a, **kw: [sys.executable, "-c", "raise SystemExit(1)"]
    with pytest.raises(BenchmarkFailed, match="incomplete"):
        benchmark(replace(env.options, integrations={"review": {}}), sctx=env.sctx)
    state_path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    env.fw.build_benchmark_command.side_effect = original
    env.stop.reset_mock()
    output_dir = tmp_path / "invalid-output-directory"
    output_dir.write_text("a file, not a directory")
    env.sctx.config.set("defaults.benchmark_output_dir", str(output_dir))
    with pytest.raises(BenchmarkFinalizationFailed) as caught:
        resume_benchmark(state_path.parent.name, sctx=env.sctx)
    assert caught.value.stage == "export"
    assert caught.value.result.results == {"rows": [env.rows]}
    env.stop.assert_not_called()
    assert resume_benchmark(state_path.parent.name, sctx=env.sctx).results == caught.value.result.results


@pytest.mark.parametrize("failure", ["checkpoint", "parse", "progress", "readiness"])
@pytest.mark.parametrize("lifecycle", ["owned", "no_stop", "skip_run"])
def test_early_benchmark_failure_honors_launch_ownership(bench_env, monkeypatch, failure, lifecycle):
    from sparkrun.api import SparkrunError

    env = bench_env
    options = replace(env.options, no_stop=lifecycle == "no_stop", skip_run=lifecycle == "skip_run")
    if failure == "checkpoint":
        register_benchmark_integration(BenchmarkIntegration("early", on_checkpoint=Mock(side_effect=RuntimeError("early failure"))))
        options = replace(options, integrations={"early": {}})
    elif failure == "parse":
        env.fw.parse_results.side_effect = ValueError("early failure")
    elif failure == "progress":

        def progress(event):
            if event.kind == "schedule_started":
                raise RuntimeError("early failure")

        options = replace(options, progress_callback=progress)
    else:
        if lifecycle == "skip_run":
            # There is no readiness wait for inference the benchmark does not own.
            assert benchmark(options, sctx=env.sctx).success
            env.stop.assert_not_called()
            return
        monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", Mock(side_effect=RuntimeError("early failure")))
    with pytest.raises(SparkrunError, match="early failure"):
        benchmark(options, sctx=env.sctx)
    assert env.run.call_count == int(lifecycle != "skip_run")
    assert env.stop.call_count == int(lifecycle == "owned")


def test_ownership_is_established_before_serve_command_notification(bench_env):
    from sparkrun.api import SparkrunError

    bench_env.run.return_value.serve_command = "serve"

    def progress(event):
        if bench_env.run.called:
            raise RuntimeError("frontend failed immediately after launch")

    with pytest.raises(SparkrunError, match="frontend failed"):
        benchmark(replace(bench_env.options, progress_callback=progress), sctx=bench_env.sctx)
    bench_env.stop.assert_called_once()


@pytest.mark.parametrize("launched", [True, False])
def test_interrupt_survives_broken_notifications_and_secondary_cleanup(bench_env, launched):
    primary = KeyboardInterrupt()
    if launched:
        bench_env.fw.build_benchmark_command.side_effect = primary
    else:
        bench_env.run.side_effect = primary
    bench_env.stop.side_effect = OSError("cleanup failed")

    def progress(event):
        if event.kind == "info" and event.data.get("msg", "").startswith("Interrupted"):
            raise RuntimeError("notification failed")

    with pytest.raises(KeyboardInterrupt) as caught:
        benchmark(replace(bench_env.options, progress_callback=progress), sctx=bench_env.sctx)
    assert caught.value is primary
    assert bench_env.stop.call_count == int(launched)
    if launched:
        assert "cleanup failed" in " ".join(primary.__notes__)


def test_early_failure_preserves_cause_when_cleanup_fails(bench_env):
    from sparkrun.api import SparkrunError

    primary = ValueError("parse failed")
    bench_env.fw.parse_results.side_effect = primary
    bench_env.stop.side_effect = OSError("cleanup failed")
    with pytest.raises(SparkrunError, match="parse failed") as caught:
        benchmark(bench_env.options, sctx=bench_env.sctx)
    assert caught.value.__cause__ is primary
    assert "cleanup failed" in " ".join(primary.__notes__)
    bench_env.stop.assert_called_once()


def test_existing_deployment_returned_by_run_is_not_owned(bench_env):
    bench_env.run.return_value.already_running = True
    assert benchmark(bench_env.options, sctx=bench_env.sctx).success
    bench_env.stop.assert_not_called()


@pytest.mark.parametrize("missing_plugin", [False, True])
def test_completed_resume_without_available_integrations_returns_saved_result(scheduled_env, monkeypatch, capsys, missing_plugin):
    from sparkrun.api import BenchmarkResult

    env = scheduled_env
    initial = benchmark(replace(env.options, export_files=False), sctx=env.sctx)
    state = BenchmarkRunState.load(initial.benchmark_id, str(env.sctx.config.cache_dir))
    if missing_plugin:
        state.extras[STATE_KEY] = {"missing": {"settings": {}, "data": {"retained": True}}}
        state.save(str(env.sctx.config.cache_dir))
    state_path = state.state_dir(str(env.sctx.config.cache_dir)) / "state.yaml"
    before = state_path.read_bytes()
    monkeypatch.setattr("sparkrun.core.resolve.load_recipe", Mock(side_effect=AssertionError("must not reload")))
    export = Mock(side_effect=AssertionError("must not export"))
    monkeypatch.setattr("sparkrun.api._benchmark._export_measurement", export)
    events = []
    capsys.readouterr()
    result = resume_benchmark(initial.benchmark_id, sctx=env.sctx, progress_callback=events.append)
    assert capsys.readouterr().out == ""
    assert isinstance(result, BenchmarkResult) and result.success and result.already_complete and result.resumed
    assert result.results == initial.results and result.run_result is None
    assert state_path.read_bytes() == before
    assert [event.kind for event in events] == ["schedule_started", "results_update", "schedule_finished", "run_complete"]
    assert events[1].data["results"] == initial.results
    env.run.assert_called_once()
    env.fw.parse_results.assert_called_once()
    export.assert_not_called()


@pytest.mark.parametrize("export_files", [True, False])
def test_resumed_measurement_respects_output_controls(scheduled_env, monkeypatch, tmp_path, export_files):
    from sparkrun.api import BenchmarkFailed

    env = scheduled_env
    original = env.fw.build_benchmark_command.side_effect
    import sys

    env.fw.build_benchmark_command.side_effect = lambda *a, **kw: [sys.executable, "-c", "raise SystemExit(1)"]
    with pytest.raises(BenchmarkFailed):
        benchmark(replace(env.options, export_files=False), sctx=env.sctx)
    path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    env.fw.build_benchmark_command.side_effect = original
    output = tmp_path / "resumed.yaml"
    result = resume_benchmark(path.parent.name, sctx=env.sctx, export_files=export_files, output_file=str(output))
    assert result.success and not result.already_complete
    assert output.exists() is export_files
    assert result.outputs == ({"yaml": str(output)} if export_files else {})
    state = BenchmarkRunState.load(result.benchmark_id, str(env.sctx.config.cache_dir))
    assert state.extras["measurement_complete"]
    env.stop.assert_called_once()  # failed initial measurement only; resume owns no launch


def test_cli_reports_completed_resume_without_failure_exception(scheduled_env, monkeypatch):
    from click.testing import CliRunner
    from sparkrun.cli._benchmark import benchmark_resume

    initial = benchmark(replace(scheduled_env.options, export_files=False), sctx=scheduled_env.sctx)
    monkeypatch.setattr("sparkrun.cli._benchmark._get_context", lambda _: scheduled_env.sctx)
    result = CliRunner().invoke(benchmark_resume, [initial.benchmark_id])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "already complete. Nothing to resume." in result.output
    assert result.stderr == ""


@pytest.mark.parametrize("private_rc", [None, 0, 23])
@pytest.mark.parametrize("dry_run", [False, True])
def test_public_launch_failure_precedes_benchmark_work(bench_env, private_rc, dry_run):
    from sparkrun.api import RunResult, BenchmarkFailed

    env = bench_env
    env.launch.rc = private_rc or 0
    env.run.return_value = RunResult(
        workload_id="test-job",
        host_list=("localhost",),
        placement=None,
        scheduler="test",
        runtime=env.recipe.runtime,
        executor="test-plugin",
        started_at=0,
        dry_run=dry_run,
        is_solo=True,
        rc=23,
        serve_port=8000,
        container_image=env.recipe.container,
        launch_result=env.launch if private_rc is not None else None,
    )
    checkpoint, complete = Mock(), Mock()
    register_benchmark_integration(BenchmarkIntegration("status", on_checkpoint=checkpoint, on_complete=complete))
    options = replace(env.options, dry_run=dry_run, export_files=False, integrations={"status": {}})
    if dry_run:
        assert benchmark(options, sctx=env.sctx).success
    else:
        with pytest.raises(BenchmarkFailed, match="exit code 23") as caught:
            benchmark(options, sctx=env.sctx)
        assert caught.value.exit_code == 23
        for callback in (checkpoint, complete, env.fw.build_benchmark_command, env.fw.parse_results, env.endpoint, env.probe):
            callback.assert_not_called()
    env.stop.assert_not_called()


def test_successful_handle_free_launch_still_benchmarks(bench_env):
    from sparkrun.api import RunResult

    env = bench_env
    env.run.return_value = RunResult(
        workload_id="test-job",
        host_list=("localhost",),
        placement=None,
        scheduler="test",
        runtime=env.recipe.runtime,
        executor="test-plugin",
        started_at=0,
        dry_run=False,
        is_solo=True,
        serve_port=8000,
        container_image=env.recipe.container,
    )
    assert benchmark(replace(env.options, export_files=False), sctx=env.sctx).success
    env.endpoint.assert_called_once()
    env.fw.parse_results.assert_called_once()
    env.stop.assert_called_once()


@pytest.mark.parametrize("resume", [False, True])
def test_cancellation_survives_notification_failure_and_releases_state(scheduled_env, monkeypatch, resume):
    import sys
    from sparkrun.api import BenchmarkFailed
    from sparkrun.benchmarking.run_state import hold_state_dir

    env = scheduled_env
    if resume:
        env.fw.build_benchmark_command.side_effect = lambda *a, **kw: [sys.executable, "-c", "raise SystemExit(1)"]
        with pytest.raises(BenchmarkFailed):
            benchmark(replace(env.options, export_files=False), sctx=env.sctx)
        env.stop.reset_mock()
    primary = KeyboardInterrupt("cancel measurement")
    monkeypatch.setattr("sparkrun.benchmarking.scheduler.run_schedule", Mock(side_effect=primary))
    notices = []

    def progress(event):
        if event.kind == "info" and event.data.get("msg", "").startswith("Interrupted"):
            notices.append(event)
            raise RuntimeError("frontend disconnected")

    with pytest.raises(KeyboardInterrupt) as caught:
        if resume:
            state_path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
            resume_benchmark(state_path.parent.name, sctx=env.sctx, progress_callback=progress)
        else:
            benchmark(replace(env.options, progress_callback=progress, export_files=False), sctx=env.sctx)
    assert caught.value is primary and len(notices) == 1
    assert env.stop.call_count == int(not resume)
    state_path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    assert BenchmarkRunState.load(state_path.parent.name, str(env.sctx.config.cache_dir)) is not None
    with hold_state_dir(state_path.parent.name, str(env.sctx.config.cache_dir)):
        pass


@pytest.mark.parametrize("failure", ["decode", "consolidate", "commit"])
@pytest.mark.parametrize("via_benchmark", [False, True])
def test_resume_finishes_interrupted_processing_without_remeasurement(scheduled_env, monkeypatch, failure, via_benchmark):
    from sparkrun.api import _benchmark, SparkrunError, ResumeMode

    env = scheduled_env
    publish = Mock()
    register_benchmark_integration(BenchmarkIntegration("recovery", on_complete=publish))
    options = replace(env.options, export_files=False, integrations={"recovery": {}})
    with monkeypatch.context() as failing:
        if failure == "decode":
            failing.setattr(env.fw, "parse_results", Mock(side_effect=RuntimeError("temporary decoder failure")))
        else:
            failing.setattr(
                _benchmark,
                "_write_consolidated" if failure == "consolidate" else "_save_completed_results",
                Mock(side_effect=KeyboardInterrupt),
            )
        with pytest.raises(SparkrunError if failure == "decode" else KeyboardInterrupt):
            benchmark(options, sctx=env.sctx)
    publish.assert_not_called()
    cache_dir = str(env.sctx.config.cache_dir)
    path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    saved = BenchmarkRunState.load(path.parent.name, cache_dir)
    assert saved.completed_indices == [0] and not saved.extras.get("measurement_complete")
    original_artifact = (path.parent / "runs/000.json").read_bytes()
    commands = env.fw.build_benchmark_command.call_count
    launches = env.run.call_count
    # Result recovery works even after cleanup removed inference and its record.
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **kw: None)
    live_check = Mock(side_effect=AssertionError("recovery does not need inference"))
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.check_job_running", live_check)
    if via_benchmark:
        result = benchmark(replace(options, resume=ResumeMode.REQUIRED), sctx=env.sctx)
    else:
        env.fw.check_prerequisites.side_effect = AssertionError("processing does not need command prerequisites")
        result = resume_benchmark(path.parent.name, sctx=env.sctx, export_files=False)
    assert result.success and result.resumed and result.results == {"rows": [env.rows]}
    assert env.fw.build_benchmark_command.call_count == commands
    assert env.run.call_count == launches
    assert (path.parent / "runs/000.json").read_bytes() == original_artifact
    assert BenchmarkRunState.load(path.parent.name, cache_dir).extras["measurement_complete"]
    publish.assert_called_once()
    live_check.assert_not_called()


def test_processing_recovery_retries_only_missing_artifacts(scheduled_env, monkeypatch):
    from sparkrun.api import SparkrunError
    from sparkrun.benchmarking.scheduler import BenchTask

    env = scheduled_env
    env.fw.build_task_list.return_value = [BenchTask(0, "first"), BenchTask(1, "second")]
    env.fw.consolidated_coverage_keys.side_effect = None
    env.fw.consolidated_coverage_keys.return_value = None
    with monkeypatch.context() as failing:
        failing.setattr(env.fw, "parse_results", Mock(side_effect=RuntimeError("decoder unavailable")))
        with pytest.raises(SparkrunError, match="decoder unavailable"):
            benchmark(replace(env.options, export_files=False), sctx=env.sctx)
    path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    survivor = path.parent / "runs/001.json"
    original = survivor.read_bytes()
    (path.parent / "runs/000.json").unlink()
    result = resume_benchmark(path.parent.name, sctx=env.sctx, export_files=False)
    assert result.success and result.results == {"rows": [env.rows, env.rows]}
    assert env.fw.build_benchmark_command.call_count == 3
    assert survivor.read_bytes() == original
    saved = BenchmarkRunState.load(result.benchmark_id, str(env.sctx.config.cache_dir))
    assert sorted(saved.completed_indices) == [0, 1] and saved.failed_indices == []
