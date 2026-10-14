"""Public benchmark contracts exercised through orchestration and the scheduler."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.api import BenchmarkResult, benchmark, resume_benchmark, BenchmarkFailed, ResumeMode
from sparkrun.benchmarking.run_state import BenchmarkRunState
from sparkrun.benchmarking.scheduler import BenchTask
from test_benchmark_startup_collection import bench_env as bench_env


@pytest.fixture
def scheduled_env(bench_env, monkeypatch):
    env = bench_env
    from sparkrun.benchmarking.base import ProgressColumn, ProgressTableSpec

    fw = env.fw
    fw.progress_table_spec.return_value = ProgressTableSpec(
        [ProgressColumn("speed")], lambda result: [(row["speed"],) for row in result.get("rows", [])]
    )
    fw.build_task_list.return_value = [BenchTask(0, "first")]
    fw.detect_version.return_value = None
    fw.apply_session_warmup_state.side_effect = lambda args, **kw: dict(args)
    fw.result_filename_suffix.return_value = ""
    fw.consolidate_per_task_results.side_effect = lambda rows: {"rows": rows}
    fw.task_coverage_key.return_value = "first"
    fw.consolidated_coverage_keys.side_effect = lambda result: {"first"} if result.get("rows") else set()
    env.rows = {"speed": 42}

    def command(*args, result_file, **kwargs):
        return [sys.executable, "-c", "from pathlib import Path; Path(%r).write_text(%r)" % (result_file, json.dumps(env.rows))]

    fw.build_benchmark_command.side_effect = command
    monkeypatch.setattr("sparkrun.orchestration.primitives.resolve_image_sha", lambda *a, **kw: None)
    from copy import deepcopy

    saved_job = {}

    def record_launch(*args, **kwargs):
        result = env.run.return_value
        launch = result.launch_result
        if launch is not None:
            saved_job.update(
                deepcopy(
                    {
                        "hosts": list(launch.host_list),
                        "port": launch.serve_port,
                        "recipe_state": launch.recipe.__getstate__(),
                        "overrides": launch.overrides,
                        "effective_container_image": launch.container_image,
                    }
                )
            )
        return result

    env.run.side_effect = record_launch
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.load_job_metadata",
        lambda *a, **kw: deepcopy(saved_job) or {"hosts": ["localhost"], "port": 8000},
    )
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.check_job_running", lambda **kw: SimpleNamespace(running=True))
    env.sctx.config.set("defaults.benchmark_output_dir", str(Path(env.options.output_file).parent))
    return env


@pytest.mark.parametrize("callback", [False, True])
def test_scheduled_api_is_silent_and_reports_tasks(scheduled_env, capsys, callback):
    env = scheduled_env
    events = []
    capsys.readouterr()
    result = benchmark(replace(env.options, progress_callback=events.append if callback else None), sctx=env.sctx)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert result.success and result.results == {"rows": [env.rows]}
    assert result.run_result is env.run.return_value
    if callback:
        kinds = [event.kind for event in events]
        assert kinds.index("task_start") < kinds.index("task_end") < kinds.index("run_complete")
        task = next(e for e in events if e.kind == "task_start")
        assert task.data == {"benchmark_id": result.benchmark_id, "total_tasks": 1, "index": 0, "label": "first"}
        assert next(e for e in events if e.kind == "results_update").data["results"] == result.results


def test_resume_api_is_silent_with_same_result_and_task_events(scheduled_env, capsys):
    env = scheduled_env
    original = env.fw.build_benchmark_command.side_effect
    env.fw.build_benchmark_command.side_effect = lambda *a, **kw: [sys.executable, "-c", "raise SystemExit(1)"]
    with pytest.raises(BenchmarkFailed, match="incomplete"):
        benchmark(env.options, sctx=env.sctx)
    state_path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    env.fw.build_benchmark_command.side_effect = original
    events = []
    capsys.readouterr()
    result = resume_benchmark(state_path.parent.name, sctx=env.sctx, progress_callback=events.append)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert isinstance(result, BenchmarkResult) and result.success and result.resumed
    assert result.framework == "test-bench" and result.category == "performance"
    assert result.results == {"rows": [env.rows]}
    assert result.run_result is None
    assert result.outputs["yaml"] and Path(result.outputs["yaml"]).is_file()
    assert {"task_start", "task_end", "run_complete"} <= {e.kind for e in events}


@pytest.mark.parametrize("edited", [False, True])
def test_preloaded_recipe_is_used_without_registry_lookup(bench_env, monkeypatch, edited):
    env = bench_env
    if edited:
        env.recipe.model = "locally/edited-model"
    lookup = Mock(side_effect=AssertionError("A supplied Recipe must not be reloaded"))
    monkeypatch.setattr("sparkrun.core.resolve.load_recipe", lookup)
    result = benchmark(replace(env.options, recipe=env.recipe), sctx=env.sctx)
    assert result.success
    assert env.run.call_args.args[0].recipe is env.recipe
    assert env.fw.build_benchmark_command.call_args.args[1] == env.recipe.model
    lookup.assert_not_called()


@pytest.mark.parametrize("complete", [False, True])
def test_typed_decision_controls_auto_resume(scheduled_env, complete):
    env = scheduled_env
    first = benchmark(env.options, sctx=env.sctx)
    state = BenchmarkRunState.load(first.benchmark_id, str(env.sctx.config.cache_dir))
    if not complete:
        state.completed_indices = []
        state.save(str(env.sctx.config.cache_dir))
    decisions = []

    def decide(request):
        decisions.append(request)
        return complete  # remeasure complete; discard incomplete

    result = benchmark(
        replace(env.options, resume=ResumeMode.AUTO, decision_callback=decide),
        sctx=env.sctx,
    )
    assert result.success and not result.resumed
    assert len(decisions) == 1
    assert decisions[0].kind == ("remeasure_complete" if complete else "resume_incomplete")
    assert decisions[0].benchmark_id == first.benchmark_id
    assert decisions[0].default is (not complete)


def test_cli_keeps_terminal_task_rendering(scheduled_env, monkeypatch, tmp_path):
    from click.testing import CliRunner
    from sparkrun.cli import main

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        main, ["benchmark", "run", "startup-test", "--hosts", "localhost", "--solo", "--fresh", "--skip-run", "--framework", "test-bench"]
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "[1/1]" in result.output and "first" in result.output


def test_public_result_keeps_integration_selected_category(bench_env, monkeypatch):
    from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration

    env = bench_env
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})
    monkeypatch.setattr("sparkrun.core.bootstrap.get_benchmarking_frameworks_for_category", lambda *_: [env.fw])
    register_benchmark_integration(BenchmarkIntegration("category", prepare=lambda options, ctx: replace(options, category="custom")))
    result = benchmark(replace(env.options, integrations={"category": {}}), sctx=env.sctx)
    assert result.category == "custom"


@pytest.mark.parametrize("fail_fast", [False, True])
def test_single_task_nonzero_exit_never_publishes_partial_measurements(bench_env, monkeypatch, fail_fast):
    from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration

    env = bench_env
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})
    publish = Mock()
    register_benchmark_integration(BenchmarkIntegration("failure-contract", on_complete=publish))
    env.fw.build_benchmark_command.side_effect = None
    env.fw.build_benchmark_command.return_value = [
        sys.executable,
        "-c",
        "import sys; print(%r); sys.exit(7)" % json.dumps(env.rows),
    ]
    with pytest.raises(BenchmarkFailed) as failure:
        benchmark(
            replace(env.options, exit_on_first_fail=fail_fast, integrations={"failure-contract": {}}, export_files=False), sctx=env.sctx
        )
    assert failure.value.exit_code == 1
    state_path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    state = BenchmarkRunState.load(state_path.parent.name, str(env.sctx.config.cache_dir), strict=True)
    assert state.failed_indices == [0] and state.completed_indices == []
    publish.assert_not_called()
    env.fw.parse_results.assert_not_called()
    env.stop.assert_called_once()


@pytest.mark.parametrize("fail_fast", [False, True])
@pytest.mark.parametrize("failure", ["nonzero", "timeout"])
def test_failed_schedule_advances_once_and_retries_on_resume(scheduled_env, monkeypatch, fail_fast, failure):
    import subprocess

    env = scheduled_env
    env.fw.build_task_list.return_value = [BenchTask(0, "first"), BenchTask(1, "second")]
    calls = []
    failing = True

    def process(cmd, **kwargs):
        # The fixture command contains the actual result-file path.
        index = int("001.json" in cmd[-1])
        calls.append(index)
        assert len(calls) <= 2, "a failed task was retried within the same invocation"
        if index == 0 and failing:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(cmd, 1)
            return 7
        subprocess.run(cmd, check=True, capture_output=True)
        return 0

    monkeypatch.setattr("sparkrun.benchmarking.scheduler.run_benchmark_process", process)
    with pytest.raises(BenchmarkFailed, match="incomplete"):
        benchmark(replace(env.options, exit_on_first_fail=fail_fast, timeout=1, export_files=False), sctx=env.sctx)
    assert calls == ([0] if fail_fast else [0, 1])
    env.stop.assert_called_once()
    state_path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    saved = BenchmarkRunState.load(state_path.parent.name, str(env.sctx.config.cache_dir))
    assert saved.failed_indices == [0]
    assert saved.completed_indices == ([] if fail_fast else [1])
    assert saved.sessions[-1]["ended_at"] is not None
    failing = False
    calls.clear()
    resumed = resume_benchmark(state_path.parent.name, sctx=env.sctx, export_files=False)
    assert resumed.success and resumed.resumed
    assert calls == ([0, 1] if fail_fast else [0])
    saved = BenchmarkRunState.load(state_path.parent.name, str(env.sctx.config.cache_dir))
    assert sorted(saved.completed_indices) == [0, 1] and saved.failed_indices == []


@pytest.mark.parametrize("tasks", [None, [], [BenchTask(1, "gap")], [BenchTask(True, "boolean")], [object()]])
def test_invalid_framework_task_contract_fails_before_launch(bench_env, tasks):
    from sparkrun.api import SparkrunError

    env = bench_env
    env.fw.build_task_list.return_value = tasks
    with pytest.raises(SparkrunError, match="BenchTask"):
        benchmark(env.options, sctx=env.sctx)
    env.run.assert_not_called()
    env.fw.build_benchmark_command.assert_not_called()


@pytest.mark.parametrize("interrupted", [False, True])
def test_cli_and_callback_consume_same_scheduler_notifications(monkeypatch, capsys, interrupted):
    from sparkrun.api._benchmark import _CallbackProgressEmitter
    from sparkrun.benchmarking.tool_eval_bench import ToolEvalBenchFramework
    from sparkrun.cli._benchmark import _CliEmitter

    monkeypatch.setattr("sparkrun.benchmarking.progress_ui._HAS_RICH", False)
    events = []
    rows = {"runs": [{"score": 42}]}
    for emitter in (_CliEmitter(), _CallbackProgressEmitter(events.append)):
        try:
            with emitter.schedule_progress(
                total_tasks=1, benchmark_id="bench_contract", fw=ToolEvalBenchFramework(), title="suite"
            ) as progress:
                progress.start_task(0, "suite task")
                progress.log("measurement log")
                progress.end_task(0, not interrupted, 1.25)
                progress.update_results_table(rows)
                if interrupted:
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
    output = capsys.readouterr().out
    assert "[1/1] running suite task" in output
    assert "measurement log" in output and "1.2s" in output
    assert ("FAILED" if interrupted else "ok") in output
    assert "Benchmark schedule finished" in output
    assert [event.kind for event in events] == [
        "schedule_started",
        "task_start",
        "schedule_log",
        "task_end",
        "results_update",
        "schedule_finished",
    ]
    assert all(event.data["benchmark_id"] == "bench_contract" and event.data["total_tasks"] == 1 for event in events)
    assert events[3].data["success"] is not interrupted
    rows["runs"][0]["score"] = -1
    assert events[4].data["results"]["runs"][0]["score"] == 42


@pytest.mark.parametrize("skip_run", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_native_benchmark_never_resolves_unused_images(scheduled_env, monkeypatch, skip_run, dry_run):
    from sparkrun.orchestration.executors.local import LocalExecutor

    env = scheduled_env
    env.recipe.container = ""
    env.recipe.executor = "local"
    env.launch.container_image = ""
    env.launch.runtime.executor = LocalExecutor()
    unused_image = Mock(side_effect=AssertionError("native benchmark must not resolve image defaults"))
    env.launch.builder = Mock(resolve_long_term_image=unused_image)
    monkeypatch.setattr("sparkrun.orchestration.primitives.resolve_image_sha", unused_image)
    monkeypatch.setattr(env.launch.runtime, "default_image_for", unused_image)
    monkeypatch.setattr("sparkrun.core.bootstrap.get_runtime", lambda *a, **kw: env.launch.runtime)
    result = benchmark(replace(env.options, recipe=env.recipe, skip_run=skip_run, dry_run=dry_run), sctx=env.sctx)
    assert result.success and not result.container_image and result.container_image_sha is None
    if skip_run:
        env.run.assert_not_called()
    else:
        env.run.assert_called_once()
    if not dry_run:
        reused = resume_benchmark(result.benchmark_id, sctx=env.sctx)
        assert reused.already_complete and not reused.container_image
    unused_image.assert_not_called()


@pytest.mark.parametrize("value", [{"invalid": "object"}, [1], "not-a-number"])
def test_invalid_benchmark_numeric_config_fails_before_launch(scheduled_env, value):
    env = scheduled_env
    with pytest.raises(BenchmarkFailed, match="tensor_parallel.*numeric"):
        benchmark(replace(env.options, overrides={"tensor_parallel": value}), sctx=env.sctx)
    env.run.assert_not_called()


@pytest.mark.parametrize(
    "raised,expected",
    [(RuntimeError("disk full"), "benchmark resume failed: disk full"), (BenchmarkFailed("typed", exit_code=2), "typed")],
)
def test_resume_benchmark_wraps_only_untyped_failures(v, monkeypatch, raised, expected):
    from sparkrun.api import SparkrunError, default_sctx

    monkeypatch.setattr("sparkrun.api._benchmark._resume_benchmark", Mock(side_effect=raised))
    with pytest.raises(SparkrunError, match=expected) as error:
        resume_benchmark("bench-id", sctx=default_sctx())
    assert type(error.value) is (SparkrunError if isinstance(raised, RuntimeError) else BenchmarkFailed)
    if isinstance(raised, RuntimeError):
        assert error.value.__cause__ is raised


def test_benchmark_module_exports_only_public_names():
    from sparkrun.api import _benchmark

    assert _benchmark.__all__ == ["benchmark", "resume_benchmark"]
