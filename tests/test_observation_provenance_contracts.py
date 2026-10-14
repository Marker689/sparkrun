"""Cross-surface regressions for observation coverage and benchmark provenance."""

from dataclasses import replace
import time
from unittest.mock import Mock

import pytest
import yaml

from _status_fixtures import host_coverage

from test_benchmark_startup_collection import bench_env as bench_env
from test_benchmark_api_contract import scheduled_env as scheduled_env
from test_run_option_contracts import run_env as run_env
from test_native_lifecycle_contracts import lifecycle_env as lifecycle_env


def test_host_launch_preserves_unobserved_live_native_metadata(lifecycle_env, monkeypatch):
    from sparkrun import api
    from sparkrun.api._run import run
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy
    from sparkrun.orchestration.job_metadata import load_job_metadata

    env = lifecycle_env
    native = run(env.options, sctx=env.sctx)
    for path in env.sctx.config.cache_dir.glob("jobs/*.yaml"):
        data = yaml.safe_load(path.read_text())
        if data["cluster_id"] == native.workload_id:
            data["started_at"] = time.time() - 31 * 86400
            path.write_text(yaml.safe_dump(data))
    assert load_job_metadata(native.workload_id, cache_dir=str(env.sctx.config.cache_dir))
    assert env.objects[env.target]
    env.sctx.config.set("jobs.autoprune", True)

    def host_launch(**kwargs):
        kwargs["before_start"]()
        env.launch.cluster_id = kwargs["cluster_id_override"]
        return env.launch

    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", host_launch)
    # This query covers the host substrate only; native resources stay live.
    monkeypatch.setattr(
        "sparkrun.orchestration.executor.query_status_for_cluster",
        lambda *a, **kw: ClusterStatus(hosts=(HostOccupancy(host="localhost"),), coverage=host_coverage(["localhost"])),
    )
    docker = run(replace(env.options, executor="docker", executor_config={}), sctx=env.sctx)
    assert docker.rc == 0
    assert env.objects[env.target], "the native workload is still live"
    assert load_job_metadata(native.workload_id, cache_dir=str(env.sctx.config.cache_dir)) is not None
    assert api.stop(workload_id=native.workload_id, sctx=env.sctx).success


def test_completion_rejects_other_native_destination_snapshot(lifecycle_env):
    from sparkrun import api
    from sparkrun.api._run import plan, run
    from sparkrun.cli._common import _completion_running, _job_is_live

    env = lifecycle_env
    first = run(env.options, sctx=env.sctx)
    other_options = replace(env.options, executor_config={"kubectl_path": "/fake/kubectl", "k8s_namespace": "second-ns"})
    other_plan = plan(other_options, sctx=env.sctx)
    second = run(other_options, sctx=env.sctx, plan=other_plan)
    assert first.workload_id != second.workload_id
    first_plan = plan(env.options, sctx=env.sctx)
    api.status(["localhost"], cluster=first_plan.cluster, sctx=env.sctx)
    calls = len(env.calls)
    snapshot = _completion_running(other_plan.cluster)
    assert len(env.calls) > calls, "completion must query the second namespace"
    job = next(job for job in api.list_jobs(sctx=env.sctx) if job.workload_id == second.workload_id)
    assert snapshot.cluster_ids == frozenset({second.workload_id})
    assert _job_is_live(job, snapshot, {"localhost"})
    assert env.objects[(env.kubeconfig, "launch-context", "second-ns")]


@pytest.mark.parametrize("image_source", ["runtime_default", "builder"])
def test_processing_recovery_preserves_effective_image(scheduled_env, monkeypatch, image_source):
    from sparkrun import api
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration

    env = scheduled_env
    if image_source == "runtime_default":
        env.recipe.container = ""
        actual_image = env.launch.runtime.resolve_container(env.recipe)
    else:
        actual_image = "builder/image:actual"
    assert actual_image and actual_image != env.recipe.container
    env.launch.container_image = actual_image
    published = []
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})
    register_benchmark_integration(BenchmarkIntegration("review", on_complete=lambda ctx: published.append(ctx.result)))
    options = replace(env.options, export_files=False, integrations={"review": {}})
    with monkeypatch.context() as failing:
        failing.setattr(env.fw, "parse_results", Mock(side_effect=RuntimeError("temporary decoder failure")))
        with pytest.raises(api.SparkrunError, match="temporary decoder failure"):
            api.benchmark(options, sctx=env.sctx)
    path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    state = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir))
    assert state.completed_indices == [0]
    assert not state.extras.get("container_image_sha")
    assert state.extras["container_image"] == actual_image
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **kw: None)
    result = api.resume_benchmark(path.parent.name, sctx=env.sctx, export_files=False)
    assert result.success
    assert env.fw.build_benchmark_command.call_count == 1
    assert published[0].container_image == actual_image
    assert published[0].provenance["recipe"]["container"] == actual_image


@pytest.mark.parametrize("image_source", ["runtime_default", "builder"])
def test_export_image_agrees_with_integration_provenance(scheduled_env, monkeypatch, image_source):
    from sparkrun import api
    from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration

    env = scheduled_env
    if image_source == "runtime_default":
        env.recipe.container = ""
        actual_image = env.launch.runtime.resolve_container(env.recipe)
    else:
        actual_image = "builder/image:actual"
    assert actual_image and actual_image != env.recipe.container
    env.launch.container_image = actual_image
    published = []
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})
    register_benchmark_integration(BenchmarkIntegration("review", on_complete=lambda ctx: published.append(ctx.result)))
    result = api.benchmark(replace(env.options, integrations={"review": {}}), sctx=env.sctx)
    from pathlib import Path

    exported = yaml.safe_load(Path(result.outputs["yaml"]).read_text())["sparkrun_benchmark"]
    assert published[0].provenance["recipe"]["container"] == actual_image
    assert exported["recipe"]["container"] == actual_image
    assert exported["recipe"]["text"] == published[0].recipe_yaml
    assert yaml.safe_load(exported["recipe"]["text"])["container"] == actual_image
    assert env.recipe.container != actual_image


def test_native_prune_removes_only_confirmed_absent_destination(lifecycle_env):
    from sparkrun import api
    from sparkrun.api._run import run, plan, _prune_stale_job_metadata
    from sparkrun.orchestration.job_metadata import load_job_metadata

    env = lifecycle_env
    first = run(env.options, sctx=env.sctx)
    second = run(replace(env.options, executor_config={"k8s_namespace": "second-ns"}), sctx=env.sctx)
    for path in env.sctx.config.cache_dir.glob("jobs/*.yaml"):
        data = yaml.safe_load(path.read_text())
        data["started_at"] = time.time() - 31 * 86400
        path.write_text(yaml.safe_dump(data))
    env.objects[env.target].clear()  # resource disappeared independently
    scoped = plan(env.options, sctx=env.sctx).cluster
    status = api.status(["localhost"], cluster=scoped, sctx=env.sctx)
    assert not status.running_cluster_ids()
    _prune_stale_job_metadata(env.sctx.config, observed_running=status.observation, keep=(), sctx=env.sctx)
    assert load_job_metadata(first.workload_id, sctx=env.sctx) is None
    assert load_job_metadata(second.workload_id, sctx=env.sctx) is not None


def test_discovered_native_target_survives_defaults_and_missing_metadata(lifecycle_env):
    from sparkrun import api
    from sparkrun.api._run import run, plan
    from sparkrun.orchestration.job_metadata import remove_job_metadata

    env = lifecycle_env
    launched = run(env.options, sctx=env.sctx)
    discovered = api.status_report(["localhost"], cluster=plan(env.options, sctx=env.sctx).cluster, sctx=env.sctx)
    remove_job_metadata(launched.workload_id, sctx=env.sctx)
    env.sctx.config.set("k8s", {"kubeconfig": "/other", "context": "other", "namespace": "other"})
    result = api.stop_all(["localhost"], discovered=discovered, sctx=env.sctx)
    assert result.success and result.jobs_stopped == 1
    assert not env.objects[env.target]


def test_host_coverage_does_not_hide_failed_executor_or_unqueried_hosts(monkeypatch):
    from sparkrun.api import JobInfo
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy
    from sparkrun.orchestration.executor import query_status_for_cluster
    from sparkrun.orchestration.executors.docker import DockerExecutor
    from sparkrun.orchestration.executors.local import LocalExecutor

    monkeypatch.setattr(
        DockerExecutor, "query_status", lambda *a, **kw: ClusterStatus(hosts=(HostOccupancy("h1"),), errors={"h2": "unreachable"})
    )
    monkeypatch.setattr(LocalExecutor, "query_status", Mock(side_effect=RuntimeError("failed local query")))
    observation = query_status_for_cluster(ClusterDefinition("c", ["h1", "h2"]), ["h1", "h2"]).observation
    for executor, host, absent in [("docker", "h1", True), ("docker", "h2", False), ("docker", "h3", False), ("local", "h1", False)]:
        job = JobInfo("sparkrun_old", hosts=(host,), metadata={"executor": executor, "executor_destination_key": ""})
        assert observation.confirms_absent(job) is absent


def test_legacy_snapshot_cannot_prove_native_absence(tmp_path):
    import json
    from sparkrun.orchestration.job_metadata import load_running_snapshot, RUNNING_SNAPSHOT_FILE

    (tmp_path / RUNNING_SNAPSHOT_FILE).write_text(json.dumps({"cluster_ids": [], "hosts": ["localhost"], "at": time.time()}))
    assert load_running_snapshot(cache_dir=str(tmp_path)) is None


@pytest.mark.parametrize("entrypoint", ["resume", "initial"])
@pytest.mark.parametrize("failure", ["partial", "decoder", "consolidation", "commit"])
@pytest.mark.parametrize("digest", [False, True])
def test_effective_context_survives_every_processing_boundary(scheduled_env, monkeypatch, entrypoint, failure, digest):
    import sys
    from sparkrun import api
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration

    env = scheduled_env
    actual = "builder/actual:version"
    sha = "sha256:" + "a" * 64 if digest else None
    env.launch.container_image = actual
    env.launch.overrides = {"max_model_len": 1024}
    env.launch.runtime_info = {"vllm": "test-version"}
    monkeypatch.setattr("sparkrun.orchestration.primitives.resolve_image_sha", lambda *a, **kw: sha)
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})
    published = []
    register_benchmark_integration(BenchmarkIntegration("context", on_complete=lambda ctx: published.append(ctx.result)))
    options = replace(env.options, integrations={"context": {}})
    recipe_before = env.recipe.__getstate__()
    with monkeypatch.context() as failing:
        if failure == "partial":
            failing.setattr(env.fw.build_benchmark_command, "side_effect", lambda *a, **kw: [sys.executable, "-c", "raise SystemExit(7)"])
        elif failure == "decoder":
            failing.setattr(env.fw.parse_results, "side_effect", RuntimeError("interrupted decoder"))
        elif failure == "consolidation":
            original = env.fw.consolidate_per_task_results.side_effect

            def consolidate(rows):
                if rows:
                    raise RuntimeError("interrupted consolidation")
                return original(rows)

            failing.setattr(env.fw.consolidate_per_task_results, "side_effect", consolidate)
        else:
            failing.setattr("sparkrun.api._benchmark._save_completed_results", Mock(side_effect=RuntimeError("interrupted commit")))
        with pytest.raises(api.SparkrunError):
            api.benchmark(options, sctx=env.sctx)
    path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    saved = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir))
    assert saved.extras["container_image"] == actual
    result = (
        api.resume_benchmark(path.parent.name, sctx=env.sctx)
        if entrypoint == "resume"
        else api.benchmark(replace(options, resume=api.ResumeMode.REQUIRED), sctx=env.sctx)
    )
    assert result.success and result.container_image == actual
    assert env.fw.build_benchmark_command.call_count == (2 if failure == "partial" else 1)
    snapshot = published[0]
    assert result.measured_at == snapshot.measured_at == saved.extras["measurement_started_at"]
    assert result.completed_at == snapshot.completed_at
    assert snapshot.container_image == actual
    assert snapshot.provenance["recipe"]["container"] == (sha or actual)
    from pathlib import Path

    exported = yaml.safe_load(Path(result.outputs["yaml"]).read_text())["sparkrun_benchmark"]
    assert exported["recipe"]["text"] == snapshot.recipe_yaml
    assert exported["benchmark"]["measured_at"] == result.measured_at
    assert exported["benchmark"]["completed_at"] == result.completed_at
    assert exported["cluster"]["runtime_info"] == {"vllm": "test-version"}
    assert yaml.safe_load(snapshot.recipe_yaml)["defaults"]["max_model_len"] == 1024
    assert env.recipe.__getstate__() == recipe_before


def test_skip_run_records_effective_image_and_unknown_legacy_stays_unknown(scheduled_env, monkeypatch):
    from sparkrun import api
    from sparkrun.benchmarking.run_state import BenchmarkRunState

    env = scheduled_env
    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.load_job_metadata",
        lambda *a, **kw: {
            "hosts": ["localhost"],
            "port": 8000,
            "effective_container_image": "running/image:actual",
        },
    )
    with monkeypatch.context() as failing:
        failing.setattr(env.fw.parse_results, "side_effect", RuntimeError("decoder failed"))
        with pytest.raises(api.SparkrunError):
            api.benchmark(replace(env.options, skip_run=True), sctx=env.sctx)
    env.run.assert_not_called()
    path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    saved = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir))
    assert saved.extras["container_image"] == "running/image:actual"
    for key in ("container_image", "container_image_sha", "measurement_context_version"):
        saved.extras.pop(key, None)
    saved.save(str(env.sctx.config.cache_dir))
    # A legacy processing-only record with no image evidence must not guess the recipe image.
    result = api.resume_benchmark(path.parent.name, sctx=env.sctx)
    assert result.success and result.container_image == ""
    from pathlib import Path

    exported = yaml.safe_load(Path(result.outputs["yaml"]).read_text())["sparkrun_benchmark"]
    assert exported["recipe"]["container"] is None
    assert yaml.safe_load(exported["recipe"]["text"])["container"] is None


def test_stale_completion_fallback_never_uses_other_destination(lifecycle_env, monkeypatch):
    from sparkrun import api
    from sparkrun.api._run import run, plan
    from sparkrun.cli._common import _completion_running

    env = lifecycle_env
    run(env.options, sctx=env.sctx)
    api.status(["localhost"], cluster=plan(env.options, sctx=env.sctx).cluster, sctx=env.sctx)
    other = plan(replace(env.options, executor_config={"k8s_namespace": "other"}), sctx=env.sctx).cluster
    monkeypatch.setattr(api, "status", Mock(side_effect=RuntimeError("offline")))
    assert _completion_running(other) is None


def test_local_state_directory_is_part_of_observation_target():
    from sparkrun.api import JobInfo
    from sparkrun.core.status_observation import ExecutorCoverage, RunningSnapshot
    from sparkrun.orchestration.executor import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    first = LocalExecutor(ExecutorConfig(pid_dir="/state/a", log_dir="/logs/a")).resolve_target()
    second = LocalExecutor(ExecutorConfig(pid_dir="/state/b", log_dir="/logs/b")).resolve_target()
    assert first.destination_key != second.destination_key
    observation = RunningSnapshot(frozenset(), (ExecutorCoverage(first, "host", {"h1"}, {"h1"}),))
    job = JobInfo("sparkrun_other", hosts=("h1",), metadata={"executor": "local", "executor_destination_key": second.destination_key})
    assert not observation.confirms_absent(job)
    assert not observation.covers(second, {"h1"})
    assert first.overrides["log_dir"] == "/logs/a"
    assert LocalExecutor().resolve_target().destination_key == ""  # default path needs no extra path key


def test_missing_executor_metadata_is_unknown_even_on_observed_hosts():
    from sparkrun.api import JobInfo
    from _status_fixtures import host_snapshot

    assert not host_snapshot(set(), ["h1"]).confirms_absent(JobInfo("legacy", hosts=("h1",)))


def test_legacy_local_destination_is_unknown_on_current_default():
    from sparkrun.api import JobInfo
    from sparkrun.core.status_observation import ExecutorCoverage, RunningSnapshot
    from sparkrun.orchestration.executors.local import LocalExecutor

    observation = RunningSnapshot(set(), (ExecutorCoverage(LocalExecutor().resolve_target(), "host", {"h1"}, {"h1"}),))
    job = JobInfo("legacy-local", hosts=("h1",), metadata={"executor": "local", "executor_config": {"pid_dir": "/old/state"}})
    assert not observation.confirms_absent(job)


def test_bulk_stop_reports_incomplete_backend_discovery(monkeypatch):
    from sparkrun import api
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.orchestration.executors.docker import DockerExecutor
    from sparkrun.orchestration.executors.local import LocalExecutor

    monkeypatch.setattr(DockerExecutor, "query_status", lambda *a, **kw: ClusterStatus(hosts=(HostOccupancy("h1"),)))
    monkeypatch.setattr(LocalExecutor, "query_status", Mock(side_effect=RuntimeError("offline")))
    result = api.stop_all(["h1"], cluster=ClusterDefinition("c", ["h1"]))
    assert not result.success
    assert result.discovery_errors == {"h1": "local status was not observed"}


def test_unscoped_native_discovery_requires_target_or_saved_metadata(lifecycle_env):
    from sparkrun import api
    from sparkrun.api._run import run, plan
    from sparkrun.orchestration.job_metadata import remove_job_metadata

    env = lifecycle_env
    launched = run(env.options, sctx=env.sctx)
    discovered = api.status_report(["localhost"], cluster=plan(env.options, sctx=env.sctx).cluster, sctx=env.sctx)
    discovered.coverage = ()  # legacy/manual snapshot with no destination evidence
    remove_job_metadata(launched.workload_id, sctx=env.sctx)
    env.sctx.config.set("k8s", {"context": "other", "namespace": "other"})
    result = api.stop_all(["localhost"], discovered=discovered, sctx=env.sctx)
    assert not result.success and result.jobs_stopped == 0
    assert env.objects[env.target]
    assert not any(args[0] == "delete" for _, args in env.calls)
