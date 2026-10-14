"""Public contracts for recorded measurement identity and observed destinations."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import yaml

from test_benchmark_startup_collection import bench_env as bench_env
from test_benchmark_api_contract import scheduled_env as scheduled_env


@pytest.mark.parametrize("resume_path", ["direct", "automatic", "gap"])
@pytest.mark.parametrize("image_case", ["changed", "unpinned_changed", "same", "digest_alias", "unknown", "legacy_unknown"])
def test_resume_verifies_image_before_more_measurements(scheduled_env, monkeypatch, resume_path, image_case):
    from sparkrun import api
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from sparkrun.benchmarking.scheduler import BenchTask
    from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration

    env = scheduled_env
    original_image = None if image_case in {"unknown", "legacy_unknown"} else "builder/image:A"
    env.launch.container_image = original_image
    digest = "sha256:" + "a" * 64 if image_case not in {"unknown", "legacy_unknown", "unpinned_changed"} else None
    meta = {
        "hosts": ["localhost"],
        "port": 8000,
        "recipe_state": env.recipe.__getstate__(),
        "overrides": {},
        "effective_container_image": original_image,
    }
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **kw: dict(meta))
    monkeypatch.setattr("sparkrun.orchestration.primitives.resolve_image_sha", lambda *a, **kw: digest)
    env.fw.build_task_list.return_value = [BenchTask(0, "first"), BenchTask(1, "second")]
    env.fw.task_coverage_key.side_effect = lambda task: task.label
    env.fw.consolidated_coverage_keys.side_effect = lambda result: {r["label"] for r in result.get("rows", [])}
    resuming = False
    commands = []

    def command(*args, result_file, **kwargs):
        idx = int(Path(result_file).stem)
        commands.append((idx, resuming))
        if idx == 1 and not resuming and resume_path != "gap":
            return [sys.executable, "-c", "raise SystemExit(7)"]
        row = {"speed": 42, "label": "first" if idx == 0 else "second"}
        return [sys.executable, "-c", "from pathlib import Path; Path(%r).write_text(%r)" % (result_file, json.dumps(row))]

    env.fw.build_benchmark_command.side_effect = command
    published = []
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})
    register_benchmark_integration(BenchmarkIntegration("review", on_complete=lambda ctx: published.append(ctx.result)))
    options = replace(env.options, integrations={"review": {}})
    parser = env.fw.parse_results.side_effect
    if resume_path == "gap":
        env.fw.parse_results.side_effect = RuntimeError("decoder interrupted")
    with pytest.raises(api.SparkrunError, match="incomplete|decoder interrupted"):
        api.benchmark(options, sctx=env.sctx)
    env.fw.parse_results.side_effect = parser
    path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    state = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir))
    assert state.completed_indices == ([0, 1] if resume_path == "gap" else [0])
    assert state.measurement_spec["job_fingerprint"]
    if image_case == "legacy_unknown":
        state.extras.pop("measurement_context_version")
        state.save(str(env.sctx.config.cache_dir))
    first_artifact = (path.parent / "runs/000.json").read_bytes()
    if resume_path == "gap":
        (path.parent / "runs/001.json").unlink()
    saved_state = path.read_bytes()
    resuming = True
    image = "builder/image:B" if image_case in {"changed", "unpinned_changed", "unknown", "legacy_unknown"} else original_image
    if image_case == "digest_alias":
        image = "alias/image@" + digest
    meta["effective_container_image"] = env.launch.container_image = image

    def resume():
        if resume_path == "automatic":
            return api.benchmark(replace(options, resume=api.ResumeMode.IF_EXISTS), sctx=env.sctx)
        return api.resume_benchmark(path.parent.name, sctx=env.sctx)

    if image_case in {"changed", "unpinned_changed"}:
        with pytest.raises(api.SparkrunError, match="image differs"):
            resume()
        assert commands == [(0, False), (1, False)]
        assert not published
        assert (path.parent / "runs/000.json").read_bytes() == first_artifact
        assert BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir)).extras["container_image"] == original_image
        if resume_path != "automatic":
            assert path.read_bytes() == saved_state
    else:
        result = resume()
        assert result.success and commands == [(0, False), (1, False), (1, True)]
        assert result.container_image == (original_image or "")
        assert result.container_image_sha == digest
        expected = digest or original_image
        exported = yaml.safe_load(Path(result.outputs["yaml"]).read_text())["sparkrun_benchmark"]
        assert exported["recipe"]["container"] == expected
        assert published[0].container_image == original_image
        assert published[0].provenance["recipe"]["container"] == expected


@pytest.fixture(params=[None, "$HOME/custom/pids"])
def local_job(bench_env, monkeypatch, request):
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.application_profile import get_application_profile
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.ssh import RemoteResult
    from sparkrun.orchestration.job_metadata import generate_cluster_id, save_job_metadata

    env = bench_env
    cid = generate_cluster_id("a" * 16, "b" * 12)
    from sparkrun.orchestration.executors._base import ExecutorConfig

    cluster = ClusterDefinition(name="review", hosts=["worker"], user="alice", executor="local", executor_config={"pid_dir": request.param})
    save_job_metadata(
        cid, env.recipe, cluster.hosts, ssh_user="alice", executor=LocalExecutor(ExecutorConfig(pid_dir=request.param)), sctx=env.sctx
    )
    path = next(env.sctx.config.cache_dir.glob("jobs/*.yaml"))
    data = yaml.safe_load(path.read_text())
    data["started_at"] = time.time() - 31 * 86400
    path.write_text(yaml.safe_dump(data))
    result = SimpleNamespace(env=env, cid=cid, cluster=cluster, calls=[], live=True)

    def remote(hosts, script, **kwargs):
        user = kwargs.get("ssh_user")
        result.calls.append(dict(kwargs))
        stdout = cid + "_solo\t123\t" + get_application_profile().id + "\n" if "*.pid" in script and user == "alice" and result.live else ""
        return [RemoteResult(host=h, returncode=0, stdout=stdout, stderr="") for h in hosts]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote)
    return result


@pytest.mark.parametrize("user", ["bob", None])
def test_local_observation_preserves_other_or_unknown_user_metadata(local_job, user):
    from sparkrun import api
    from sparkrun.api._run import _prune_stale_job_metadata
    from sparkrun.orchestration.job_metadata import load_job_metadata

    f = local_job
    alice = api.status(["worker"], cluster=f.cluster, sctx=f.env.sctx)
    assert f.cid in alice.running_cluster_ids()
    snapshot = api.status(["worker"], cluster=f.cluster, ssh_kwargs={"ssh_user": user}, sctx=f.env.sctx)
    job = next(j for j in api.list_jobs(sctx=f.env.sctx) if j.workload_id == f.cid)
    assert not snapshot.observation.confirms_absent(job)
    f.env.sctx.config.set("jobs.autoprune", True)
    _prune_stale_job_metadata(f.env.sctx.config, observed_running=snapshot.observation, keep=(), sctx=f.env.sctx)
    assert load_job_metadata(f.cid, sctx=f.env.sctx)
    f.live = False
    alice = api.status(["worker"], cluster=f.cluster, sctx=f.env.sctx)
    assert alice.observation.confirms_absent(job)
    _prune_stale_job_metadata(f.env.sctx.config, observed_running=alice.observation, keep=(), sctx=f.env.sctx)
    assert load_job_metadata(f.cid, sctx=f.env.sctx) is None


@pytest.mark.parametrize("primary_executor", ["local", "docker"])
def test_completion_queries_the_requested_principal(local_job, primary_executor):
    from sparkrun import api
    from sparkrun.cli._common import _completion_running
    from sparkrun.orchestration.job_metadata import load_running_snapshot

    f = local_job
    f.cluster = replace(f.cluster, executor=primary_executor)
    api.status(["worker"], cluster=replace(f.cluster, user="bob"), sctx=f.env.sctx)
    cached = load_running_snapshot(sctx=f.env.sctx)
    assert cached and all(c.ssh_kwargs is None for c in cached.coverage)
    count = len(f.calls)
    observation = _completion_running(f.cluster)
    assert len(f.calls) > count
    assert f.cid in observation.cluster_ids
    count = len(f.calls)
    assert _completion_running(f.cluster).cluster_ids == observation.cluster_ids
    assert len(f.calls) == count


@pytest.mark.parametrize("entrypoint", ["status", "status_report", "stop_all"])
@pytest.mark.parametrize("cluster_user", [None, "bob"])
def test_status_entrypoints_resolve_transport_without_mutating_context(bench_env, monkeypatch, entrypoint, cluster_user):
    from sparkrun import api
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.orchestration.ssh import RemoteResult

    env = bench_env
    env.sctx.config.ssh_user = "alice"
    env.sctx.config.set("ssh.key", "/config/key")
    env.sctx.config.set("ssh.options", ["-o", "Port=2222"])
    cluster = ClusterDefinition(name="review", hosts=["worker"], user=cluster_user, executor="local")
    calls = []

    def remote(hosts, script, **kwargs):
        calls.append(kwargs)
        return [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote)
    operation = getattr(api, entrypoint)
    operation(["worker"], cluster=cluster, sctx=env.sctx)
    assert calls and all(c["ssh_user"] == (cluster_user or "alice") and c["ssh_key"] == "/config/key" for c in calls)
    calls.clear()
    operation(["worker"], cluster=cluster, sctx=env.sctx, ssh_kwargs={"ssh_key": None, "ssh_options": []})
    assert calls and all(c["ssh_user"] == (cluster_user or "alice") and c["ssh_key"] is None and not c["ssh_options"] for c in calls)
    calls.clear()
    operation(["worker"], cluster=replace(cluster, user=None), sctx=env.sctx)
    assert calls and all(c["ssh_user"] == "alice" for c in calls)
    assert env.sctx.config.ssh_user == "alice"


@pytest.mark.parametrize("override", [False, True])
def test_bulk_stop_uses_observed_connection_after_defaults_change(local_job, monkeypatch, override):
    from sparkrun import api
    from sparkrun.orchestration.ssh import RemoteResult

    f = local_job
    f.env.sctx.config.set("ssh.key", "/old/key")
    report = api.status_report(["worker"], cluster=f.cluster, sctx=f.env.sctx)
    assert report.total_containers == 1
    f.env.sctx.config.ssh_user = "bob"
    f.env.sctx.config.set("ssh.key", "/new/key")
    calls = []

    def cleanup(grouped, **kwargs):
        calls.append(kwargs["ssh_kwargs"])
        return {host: RemoteResult(host=host, returncode=0, stdout="sparkrun_removed=1\n", stderr="") for host in grouped}

    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", cleanup)
    result = api.stop_all(["worker"], discovered=report, sctx=f.env.sctx, ssh_kwargs={"ssh_key": "/rotated/key"} if override else None)
    assert result.success and not result.dry_run
    assert calls == [{"ssh_user": "alice", "ssh_key": "/rotated/key" if override else "/old/key", "ssh_options": []}]


def test_bulk_stop_rejects_a_different_explicit_principal(local_job, monkeypatch):
    from sparkrun import api

    f = local_job
    report = api.status_report(["worker"], cluster=f.cluster, sctx=f.env.sctx)
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.cleanup_containers_by_host", lambda *a, **kw: pytest.fail("must reject before teardown")
    )
    with pytest.raises(api.SparkrunError, match="SSH user differs"):
        api.stop_all(["worker"], discovered=report, sctx=f.env.sctx, ssh_kwargs={"ssh_user": "bob"})


@pytest.mark.parametrize("has_error", [True, False])
def test_host_stop_all_preview_preserves_discovery_errors(bench_env, monkeypatch, has_error):
    from sparkrun import api
    from sparkrun.orchestration.ssh import RemoteResult
    from test_stop_all_dispatch import _solo_report

    report = _solo_report()
    if has_error:
        report.errors["unreachable-worker"] = "SSH failed"
    report.host_count = 2
    calls = []

    def cleanup(grouped, **kwargs):
        calls.append(kwargs["dry_run"])
        return {host: RemoteResult(host=host, returncode=0, stdout="", stderr="") for host in grouped}

    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", cleanup)
    result = api.stop_all(["localhost", "unreachable-worker"], discovered=report, dry_run=True, sctx=bench_env.sctx)
    assert calls == [True]
    assert result.discovery_errors == report.errors
    assert result.success is (not has_error)
    assert result.dry_run and result.containers_removed == 1


@pytest.mark.parametrize("scheduler_name", ["occupancy-sparse", "occupancy-dense"])
@pytest.mark.parametrize("failure", ["partial_backend", "all_queries", "target_resolution"])
def test_occupancy_schedulers_do_not_allocate_unobserved_hosts(bench_env, monkeypatch, scheduler_name, failure):
    from sparkrun import api
    from sparkrun.core.cluster_manager import ClusterDefinition, classify_cluster_status
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy
    from sparkrun.core.parallelism import ParallelismConfig
    from sparkrun.core.scheduler import SchedulingRequest, InfeasibleScheduleError, get_scheduler
    from sparkrun.orchestration.executors.docker import DockerExecutor
    from sparkrun.orchestration.executors.local import LocalExecutor

    cluster = ClusterDefinition(name="review", hosts=["bad", "good"])
    monkeypatch.setattr("sparkrun.orchestration.executor.list_executors", lambda *a: ["docker", "local"])

    def docker_status(self, hosts, **kwargs):
        if failure == "all_queries":
            return ClusterStatus(errors={h: "unreachable" for h in hosts})
        return ClusterStatus(hosts=tuple(HostOccupancy(host=h) for h in hosts))

    monkeypatch.setattr(DockerExecutor, "query_status", docker_status)
    if failure == "target_resolution":
        monkeypatch.setattr(LocalExecutor, "resolve_target", lambda *a, **kw: (_ for _ in ()).throw(ValueError("target unavailable")))
    else:

        def local_status(self, hosts, **kwargs):
            covered = [] if failure == "all_queries" else [h for h in hosts if h == "good"]
            return ClusterStatus(
                hosts=tuple(HostOccupancy(host=h) for h in covered), errors={h: "not observed" for h in hosts if h not in covered}
            )

        monkeypatch.setattr(LocalExecutor, "query_status", local_status)
    status = api.status(cluster.hosts, cluster=cluster, sctx=bench_env.sctx)
    report = classify_cluster_status(status, host_list=cluster.hosts, cache_dir=str(bench_env.sctx.config.cache_dir))
    assert report.errors == status.observation_errors
    assert "bad" in status.observation_errors and status.free_slots("bad") == 0
    request = SchedulingRequest(hosts=tuple(cluster.hosts), parallelism=ParallelismConfig(tensor_parallel=1), status=status)
    scheduler = get_scheduler(scheduler_name)
    if failure == "partial_backend":
        assert scheduler.schedule(request).assignment.hosts_used == ("good",)
    else:
        with pytest.raises(InfeasibleScheduleError):
            scheduler.schedule(request)


def test_old_user_unscoped_cache_is_ignored(local_job):
    from sparkrun import api
    from sparkrun.orchestration.job_metadata import RUNNING_SNAPSHOT_FILE, load_running_snapshot

    f = local_job
    api.status(["worker"], cluster=f.cluster, sctx=f.env.sctx)
    path = f.env.sctx.config.cache_dir / RUNNING_SNAPSHOT_FILE
    payload = json.loads(path.read_text())
    payload["version"] = 1
    path.write_text(json.dumps(payload))
    assert load_running_snapshot(sctx=f.env.sctx) is None


@pytest.mark.parametrize("entrypoint", ["status", "status_report", "stop_all"])
def test_provider_preparation_precedes_ssh_resolution_once(bench_env, monkeypatch, entrypoint):
    from sparkrun import api
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.orchestration.ssh import RemoteResult

    cluster = ClusterDefinition(name="provider", hosts=["worker"])
    calls = []

    def prepare(definition, **kwargs):
        calls.append("prepare")
        definition.user = "provider-user"

    def remote(hosts, script, **kwargs):
        calls.append(kwargs["ssh_user"])
        return [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts]

    monkeypatch.setattr("sparkrun.api._resolve.prepare_transport", prepare)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote)
    getattr(api, entrypoint)(["worker"], cluster=cluster, sctx=bench_env.sctx)
    assert calls.count("prepare") == 1
    assert calls[0] == "prepare" and all(c == "provider-user" for c in calls[1:])


@pytest.mark.parametrize("cleared_user", [None, ""])
def test_explicit_empty_ssh_user_is_an_unknown_principal(bench_env, monkeypatch, cleared_user):
    from sparkrun import api
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.orchestration.ssh import RemoteResult

    cluster = ClusterDefinition(name="review", hosts=["worker"], user="alice", executor="local")
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        lambda hosts, script, **kw: [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts],
    )
    snapshot = api.status(["worker"], cluster=cluster, sctx=bench_env.sctx, ssh_kwargs={"ssh_user": cleared_user})
    assert snapshot.coverage and not snapshot.observation_errors
    assert all(c.ssh_user is None for c in snapshot.coverage)
