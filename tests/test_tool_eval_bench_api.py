"""Real tool-eval framework scheduling, recovery, and version identity contracts."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from sparkrun.api import BenchmarkFailed, ResumeMode, benchmark, resume_benchmark
from sparkrun.benchmarking.run_state import BenchmarkRunState
from sparkrun.benchmarking.tool_eval_bench import ToolEvalBenchFramework
from test_benchmark_api_contract import scheduled_env as scheduled_env
from test_benchmark_startup_collection import bench_env as bench_env


def test_tool_suite_retries_failed_attempt_and_reuses_completed_output(scheduled_env, monkeypatch):
    env = scheduled_env
    fw = ToolEvalBenchFramework()
    monkeypatch.setattr("sparkrun.core.bootstrap.get_benchmarking_framework", lambda *a, **kw: fw)
    monkeypatch.setattr(fw, "check_prerequisites", lambda: [])
    payload = {
        "schema_version": "1",
        "tool_eval_bench_version": "2.6.0",
        "final_score": 50,
        "scores": {"max_points": 2, "scenario_results": [{"scenario_id": "TC-01", "status": "fail", "failure_kind": "missing_step"}]},
    }
    commands = []
    failed = True
    actual_command = fw.build_benchmark_command

    def command(target_url, model, args, result_file):
        commands.append(actual_command(target_url, model, args, result_file))
        source = "from pathlib import Path; Path(%r).write_text(%r); raise SystemExit(%d)" % (
            result_file,
            json.dumps(payload),
            1 if failed else 0,
        )
        return [sys.executable, "-c", source]

    monkeypatch.setattr(fw, "build_benchmark_command", command)
    options = replace(env.options, framework=fw.framework_name, bench_args={"label": "first, second", "ref": "v2.6.0"})
    with pytest.raises(BenchmarkFailed, match="incomplete"):
        benchmark(options, sctx=env.sctx)
    state_path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    state = BenchmarkRunState.load(state_path.parent.name, str(env.sctx.config.cache_dir))
    assert state.failed_indices == [0] and state.completed_indices == []
    assert state.base_args["ref"] == "v2.6.0"
    assert state.base_args["label"] == "first, second"
    failed = False
    result = resume_benchmark(state.benchmark_id, sctx=env.sctx)
    assert result.success and result.resumed and result.results["json"] == payload
    assert result.category == "tools"
    assert len(commands) == 2
    assert "--json-file" in commands[1] and "first, second" in commands[1]
    assert commands[1][commands[1].index("--from") + 1].endswith("@v2.6.0")
    assert json.loads(Path(result.outputs["json"]).read_text()) == payload
    assert "missing_step" in Path(result.outputs["csv"]).read_text()

    forbidden = Mock(side_effect=AssertionError("completed output must not require execution"))
    monkeypatch.setattr(fw, "build_benchmark_command", forbidden)
    monkeypatch.setattr(fw, "check_prerequisites", forbidden)
    saved_events = []
    reused = resume_benchmark(state.benchmark_id, sctx=env.sctx, progress_callback=saved_events.append)
    assert next(e for e in saved_events if e.kind == "results_update").data["results"] == payload
    assert not any(e.kind == "task_start" for e in saved_events)
    assert reused.already_complete and reused.results["json"] == payload
    forbidden.assert_not_called()


def test_tool_version_participates_in_measurement_identity(scheduled_env, monkeypatch):
    env = scheduled_env
    fw = ToolEvalBenchFramework()
    monkeypatch.setattr("sparkrun.core.bootstrap.get_benchmarking_framework", lambda *a, **kw: fw)
    options = replace(env.options, framework=fw.framework_name, dry_run=True, resume=ResumeMode.FRESH)
    old = benchmark(replace(options, bench_args={"ref": "v2.5.0"}), sctx=env.sctx)
    new = benchmark(options, sctx=env.sctx)
    assert old.benchmark_id != new.benchmark_id
    assert all(call.args[0].dry_run for call in env.run.call_args_list)


@pytest.mark.parametrize("failure_kind", ["timeout", "connection_error", "server_error", "missing_measurements"])
def test_unmeasured_suite_cannot_publish_and_resume_retries(scheduled_env, monkeypatch, failure_kind):
    from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration

    env = scheduled_env
    fw = ToolEvalBenchFramework()
    monkeypatch.setattr("sparkrun.core.bootstrap.get_benchmarking_framework", lambda *a, **kw: fw)
    monkeypatch.setattr(fw, "check_prerequisites", lambda: [])
    payload = (
        {
            "schema_version": "1",
            "scores": {
                "final_score": 0,
                "max_points": 0,
                "completion_rate": 0.0,
                "excluded_scenarios": ["TC-01"],
                "scenario_results": [{"scenario_id": "TC-01", "status": "fail", "failure_kind": failure_kind}],
            },
        }
        if failure_kind != "missing_measurements"
        else {"schema_version": "1"}
    )

    def command(target_url, model, args, result_file):
        return [sys.executable, "-c", "from pathlib import Path; Path(%r).write_text(%r)" % (result_file, json.dumps(payload))]

    command = Mock(side_effect=command)
    monkeypatch.setattr(fw, "build_benchmark_command", command)
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})
    publish = Mock(return_value={"published": True})
    register_benchmark_integration(BenchmarkIntegration("test-publication", on_complete=publish))
    options = replace(env.options, framework=fw.framework_name, integrations={"test-publication": {}})
    with pytest.raises(BenchmarkFailed, match="incomplete"):
        benchmark(options, sctx=env.sctx)
    publish.assert_not_called()
    state_path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    state = BenchmarkRunState.load(state_path.parent.name, str(env.sctx.config.cache_dir))
    assert state.failed_indices == [0] and state.completed_indices == []
    before_resume = command.call_count
    # A genuine model-quality zero is still a measurement, unlike an unavailable endpoint.
    payload = {
        "schema_version": "1",
        "scores": {
            "final_score": 0,
            "max_points": 2,
            "completion_rate": 1.0,
            "excluded_scenarios": [],
            "scenario_results": [{"scenario_id": "TC-01", "status": "fail", "failure_kind": "missing_step"}],
        },
    }
    result = resume_benchmark(state.benchmark_id, sctx=env.sctx)
    assert result.success and result.resumed and result.results["json"] == payload
    assert command.call_count == before_resume + 1
    publish.assert_called_once()
    command.reset_mock()
    reused = resume_benchmark(state.benchmark_id, sctx=env.sctx)
    assert reused.success and reused.results["json"] == payload
    command.assert_not_called()


@pytest.mark.parametrize(
    "mode", ["perf_only", "mmlu_only", "skip_tool_eval", "context_pressure_sweep", "needle_only", "decision_bench_only", "decision_live"]
)
def test_non_artifact_modes_rejected_before_launch(scheduled_env, monkeypatch, mode):
    from sparkrun.api import SparkrunError

    env = scheduled_env
    fw = ToolEvalBenchFramework()
    monkeypatch.setattr("sparkrun.core.bootstrap.get_benchmarking_framework", lambda *a, **kw: fw)
    with pytest.raises(SparkrunError, match="JSON"):
        benchmark(
            replace(env.options, framework=fw.framework_name, bench_args={mode: "0,0.5" if mode == "context_pressure_sweep" else True}),
            sctx=env.sctx,
        )
    env.run.assert_not_called()


@pytest.mark.parametrize("label", ["true", "false", "0012", "one, two", "quoted 'value' \"here\""])
def test_api_preserves_text_labels(scheduled_env, monkeypatch, label):
    env = scheduled_env
    fw = ToolEvalBenchFramework()
    monkeypatch.setattr("sparkrun.core.bootstrap.get_benchmarking_framework", lambda *a, **kw: fw)
    task_builder = Mock(wraps=fw.build_task_list)
    monkeypatch.setattr(fw, "build_task_list", task_builder)
    result = benchmark(
        replace(env.options, framework=fw.framework_name, dry_run=True, bench_args={"label": label, "no_think": "true"}), sctx=env.sctx
    )
    assert result.success
    args = fw.build_benchmark_command("http://localhost/v1", "test/model", task_builder.call_args.args[0])
    assert args[args.index("--label") + 1] == label
    assert "--no-think" in args
