"""Run and benchmark render the launch they actually hand to the API."""

from dataclasses import replace
from unittest import mock

import pytest
from click.testing import CliRunner

import sparkrun.api as api
from sparkrun.api._benchmark import _execute_benchmark
from sparkrun.cli import main
from sparkrun.cli._benchmark import _CliEmitter
from sparkrun.core.cluster_manager import ClusterDefinition, ClusterManager
from sparkrun.core.hardware_assessment import assess_launch_hardware
from sparkrun.models.vram import estimate_vram
from sparkrun.runtimes.sglang import SglangRuntime

from test_benchmark_startup_collection import bench_env as bench_env
from test_cli_run_single_placement import (
    _FREE,
    _HOSTS,
    _RECIPE_NAME,
    busy_cluster_status as busy_cluster_status,
    cluster_env as cluster_env,
)
from test_launch_summary import _hw, _placement


@pytest.mark.parametrize("category", ["performance", "tools"])
def test_benchmark_matches_run_without_replanning(cluster_env, busy_cluster_status, monkeypatch, category):
    # Inventory with different capacities proves the memory block uses the
    # actual winning hosts, not an estimate with unknown target capacity.
    ClusterManager(cluster_env).update("wopr", hosts_hardware={h: _hw(memory=96.0 if h == _FREE[-1] else 121.7) for h in _HOSTS})
    monkeypatch.setattr("sparkrun.core.version.display_version", lambda config: "0.4.0-alpha+test")
    outputs = []
    for command in (["run"], ["benchmark", category]):
        with (
            mock.patch.object(api, "plan", wraps=api.plan) as plan,
            mock.patch.object(api, "schedule", wraps=api.schedule) as schedule,
            mock.patch.object(api, "run", wraps=api.run) as run,
            mock.patch.object(SglangRuntime, "run", return_value=0),
        ):
            argv = command + [_RECIPE_NAME, "--cluster", "wopr", "--dry-run", "--gpu-mem", "0.8"]
            if command[0] == "benchmark":
                argv += ["--fresh"]
            result = CliRunner().invoke(main, argv)
        assert result.exit_code == 0, result.output
        assert plan.call_count == schedule.call_count == run.call_count == 1
        assert plan.call_args.args[0].hosts == tuple(_HOSTS)
        launch_plan = run.call_args.kwargs["plan"]
        assert launch_plan.host_list == tuple(_FREE)
        assert launch_plan.hardware_reported is True
        assert launch_plan.candidate_hosts == tuple(_HOSTS)
        out = result.output
        assert out.count("sparkrun v0.4.0-alpha+test") == 1
        assert "VRAM Estimation:" not in out
        assert "96.0 GiB x 80% gpu_memory_utilization" in out
        assert out.index("Planning:") < out.index("Placement:") < out.index("Memory per GPU")
        assert "Placement: 2 of 4 hosts from cluster 'wopr' (occupancy-sparse scheduler)" in out
        # Exclude benchmark-only details and the executor's launch progress.
        start = out.index("sparkrun v0.4.0-alpha+test")
        end = out.index("Recipe:    ") if command[0] == "benchmark" else out.index("[1/6]")
        outputs.append(out[start:end].rstrip())
    # Framework-specific advisories may follow the common launch block.
    assert outputs[1].startswith(outputs[0])


@pytest.fixture
def observed_plan(bench_env, monkeypatch):
    env = bench_env
    hosts = ("localhost", "worker")
    hardware = {hosts[0]: _hw(), hosts[1]: _hw(driver="580.173.02", memory=96.0)}
    cluster = ClusterDefinition(name="test", hosts=list(hosts), hosts_hardware=hardware)
    placement = _placement(hosts)
    runtime = env.launch.runtime
    plan = api.RunPlan(
        recipe=env.recipe,
        runtime=runtime,
        cluster=cluster,
        candidate_hosts=hosts,
        host_list=hosts,
        is_solo=False,
        placement=placement,
        scheduler="occupancy-sparse",
        hardware_assessment=assess_launch_hardware(runtime, hosts, cluster, placement, hardware),
    )
    monkeypatch.setattr(
        env.recipe,
        "estimate_vram",
        lambda **kw: estimate_vram(model_vram=4, tensor_parallel=2, total_gpu_memory_gb=96, gpu_memory_utilization=0.8),
    )
    monkeypatch.setattr(api, "plan", mock.Mock(return_value=plan))
    return plan


@pytest.mark.parametrize("render", ["cli", "api", "failed"])
def test_hardware_is_acknowledged_only_after_cli_render(bench_env, observed_plan, monkeypatch, capsys, render):
    env = bench_env
    events = []
    options = replace(env.options, dry_run=True, progress_callback=events.append)
    if render == "api":
        api.benchmark(options, sctx=env.sctx)
    else:
        if render == "failed":
            monkeypatch.setattr("sparkrun.cli._benchmark.render_run_plan", mock.Mock(side_effect=ValueError("display failed")))
        _execute_benchmark(options, sctx=env.sctx, emitter=_CliEmitter(show_recipe=True, host_source="cluster 'test'"))
    out = capsys.readouterr()
    launched_plan = env.run.call_args.kwargs["plan"]
    assert launched_plan.hardware_reported is (render == "cli")
    # Reporting may replace the frozen plan; all execution decisions survive.
    assert replace(launched_plan, hardware_reported=False) == observed_plan
    assert observed_plan.hardware_reported is False
    if render == "cli":
        assert out.err.count("nvidia driver branches differ") == 1
        assert "worker" in out.out and "580.173.02 !" in out.out
        assert "Memory per GPU" in out.out
        assert "96.0 GiB x 80% gpu_memory_utilization" in out.out
        assert "VRAM Estimation:" not in out.out
        assert "Host localhost hardware:" not in out.out
    if render == "api":
        assert out.out == out.err == ""
        assert any(e.kind == "banner" and "Benchmark Framework:" in e.data["line"] for e in events)


def test_skip_run_does_not_plan_or_claim_hardware_fit(bench_env, monkeypatch, capsys):
    env = bench_env
    plan = mock.Mock(side_effect=AssertionError("skip-run must not plan a new deployment"))
    monkeypatch.setattr(api, "plan", plan)
    _execute_benchmark(replace(env.options, skip_run=True, dry_run=True), sctx=env.sctx, emitter=_CliEmitter(show_recipe=True))
    out = capsys.readouterr().out
    plan.assert_not_called()
    env.run.assert_not_called()
    assert "Target:    existing deployment (--skip-run)" in out
    assert "Hosts:     localhost" in out
    assert "Mode:      solo" in out
    assert "Benchmark: test-bench" in out
    for absent in ("Planning:", "Placement:", "Memory per GPU", "VRAM Estimation:"):
        assert absent not in out


def test_planning_failure_ends_line_without_claiming_success(bench_env, monkeypatch, capsys):
    env = bench_env
    monkeypatch.setattr(api, "plan", mock.Mock(side_effect=api.SparkrunError("cannot place workload")))
    with pytest.raises(api.BenchmarkFailed, match="cannot place workload"):
        _execute_benchmark(env.options, sctx=env.sctx, emitter=_CliEmitter(show_recipe=True))
    out = capsys.readouterr().out
    assert out.endswith("model metadata...\n")
    assert " done" not in out
    assert "Placement:" not in out
    env.run.assert_not_called()
