"""Behavioral acceptance across namespace and measurement recovery boundaries."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import yaml

from test_benchmark_startup_collection import bench_env as bench_env
from test_benchmark_api_contract import scheduled_env as scheduled_env
from test_run_option_contracts import run_env as run_env
from test_transport_measurement_contracts import local_job as local_job


@pytest.mark.parametrize("pid_dir", [None, "$HOME/custom/pids", "/shared/pids"])
def test_local_destinations_keep_independent_ids_and_lifecycle(run_env, monkeypatch, pid_dir):
    from sparkrun import api
    from sparkrun.api._run import plan
    from sparkrun.api._stop import stop
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata
    from sparkrun.orchestration.ssh import RemoteResult

    env = run_env
    monkeypatch.setattr("sparkrun.api._hosts.resolve_effective_hosts", lambda hosts, *a, **kw: (hosts, True, [], None))
    plans = []
    for user in ("alice", "bob"):
        cluster = ClusterDefinition(name=user, hosts=["worker"], user=user, executor="local", executor_config={"pid_dir": pid_dir})
        options = api.RunOptions(recipe=env.recipe, cluster=cluster, solo=True, scheduler="greedy", dry_run=True)
        planned = plan(options, sctx=env.sctx)
        plans.append(planned)
        env.sctx.config.set("ssh.key", "/rotated/key")
        assert plan(options, sctx=env.sctx).cluster_id == planned.cluster_id
        save_job_metadata(
            planned.cluster_id,
            env.recipe,
            ["worker"],
            ssh_user=user,
            executor=LocalExecutor(ExecutorConfig(pid_dir=pid_dir)),
            sctx=env.sctx,
        )
    assert plans[0].cluster_id != plans[1].cluster_id
    assert len(list(env.sctx.config.cache_dir.glob("jobs/*.yaml"))) == 2
    calls = []

    def cleanup(grouped, **kwargs):
        calls.append(kwargs["ssh_kwargs"])
        return {host: RemoteResult(host=host, returncode=0, stdout="sparkrun_removed=1\n", stderr="") for host in grouped}

    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", cleanup)
    for user, planned in zip(("alice", "bob"), plans, strict=True):
        assert load_job_metadata(planned.cluster_id, sctx=env.sctx)["ssh_user"] == user
        result = stop(cluster_id=planned.cluster_id, sctx=env.sctx)
        assert result.success and calls[-1]["ssh_user"] == user and calls[-1]["ssh_key"] == "/rotated/key"
        assert load_job_metadata(planned.cluster_id, sctx=env.sctx) is None


def test_local_default_principal_has_stable_identity(run_env, monkeypatch):
    from sparkrun import api
    from sparkrun.api._run import plan

    env = run_env
    env.sctx.config.ssh_user = None
    options = api.RunOptions(recipe=env.recipe, hosts=("localhost",), solo=True, scheduler="greedy", executor="local", dry_run=True)
    assert plan(options, sctx=env.sctx).cluster_id == plan(options, sctx=env.sctx).cluster_id


@pytest.mark.parametrize("operation", ["stop", "logs", "liveness"])
@pytest.mark.parametrize("legacy", [False, True])
def test_saved_local_principal_survives_cluster_edits(local_job, monkeypatch, operation, legacy):
    from sparkrun.api._stop import stop
    from sparkrun.api._logs import logs
    from sparkrun.orchestration.job_metadata import check_job_running, load_job_metadata
    from sparkrun.orchestration.ssh import RemoteResult

    f = local_job
    env = f.env
    env.sctx.cluster_manager.create("review", ["worker"], user="alice", executor="local")
    path = next(env.sctx.config.cache_dir.glob("jobs/*.yaml"))
    meta = yaml.safe_load(path.read_text())
    meta["cluster"] = "review"
    if legacy:
        meta.pop("executor_user_scoped", None)
    path.write_text(yaml.safe_dump(meta))
    env.sctx.cluster_manager.update("review", user="bob")
    env.sctx.config.set("ssh.key", "/rotated/key")
    calls = []
    if operation == "stop":

        def cleanup(grouped, **kwargs):
            calls.append(kwargs["ssh_kwargs"])
            assert kwargs["ssh_kwargs"]["ssh_user"] == "alice"
            return {host: RemoteResult(host=host, returncode=0, stdout="sparkrun_removed=1\n", stderr="") for host in grouped}

        monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", cleanup)
        assert stop(cluster_id=f.cid, sctx=env.sctx).success
        assert load_job_metadata(f.cid, sctx=env.sctx) is None
    elif operation == "logs":

        def read(*args, **kwargs):
            calls.append(kwargs["ssh_kwargs"])
            return iter(())

        monkeypatch.setattr("sparkrun.orchestration.logs.read_log_sources", read)
        assert list(logs(workload_id=f.cid, sctx=env.sctx)) == []
        assert load_job_metadata(f.cid, sctx=env.sctx)
    else:
        assert check_job_running(cluster_id=f.cid, cache_dir=str(env.sctx.config.cache_dir)).running
        calls = f.calls
    assert calls and all(c["ssh_user"] == "alice" for c in calls)
    if operation != "liveness":
        assert calls[-1]["ssh_key"] == "/rotated/key"
    assert env.sctx.cluster_manager.get("review").user == "bob"


@pytest.mark.parametrize("unknown", [False, True])
def test_local_stop_rejects_unknown_or_explicitly_changed_principal(local_job, monkeypatch, unknown):
    from sparkrun import api
    from sparkrun.api._stop import stop
    from sparkrun.orchestration.job_metadata import load_job_metadata

    f = local_job
    if unknown:
        path = next(f.env.sctx.config.cache_dir.glob("jobs/*.yaml"))
        meta = yaml.safe_load(path.read_text())
        meta.pop("ssh_user")
        path.write_text(yaml.safe_dump(meta))
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.cleanup_containers_by_host", lambda *a, **kw: pytest.fail("unverified destination")
    )
    with pytest.raises(api.SparkrunError, match="SSH user.*unknown|SSH user differs"):
        stop(cluster_id=f.cid, cluster=replace(f.cluster, user="bob"), sctx=f.env.sctx)
    assert load_job_metadata(f.cid, sctx=f.env.sctx)


@pytest.mark.parametrize("failure", ["daemon", "permission", "missing", None])
def test_docker_failure_is_unknown_through_public_status_and_pruning(bench_env, monkeypatch, failure):
    from sparkrun import api
    from sparkrun.api._run import _prune_stale_job_metadata
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.orchestration.executors.docker import DockerExecutor
    from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata, generate_cluster_id
    from sparkrun.orchestration.ssh import RemoteResult

    env = bench_env
    cid = generate_cluster_id("a" * 16, "b" * 12)
    save_job_metadata(cid, env.recipe, ["worker"], executor=DockerExecutor(), ssh_user="alice", sctx=env.sctx)
    path = next(env.sctx.config.cache_dir.glob("jobs/*.yaml"))
    meta = yaml.safe_load(path.read_text())
    meta["started_at"] = time.time() - 31 * 86400
    path.write_text(yaml.safe_dump(meta))
    message, code = {
        "daemon": ("Cannot connect to Docker daemon", 1),
        "permission": ("permission denied", 1),
        "missing": ("docker: command not found", 127),
        None: ("", 0),
    }[failure]

    def remote(hosts, script, **kwargs):
        if script.startswith("docker ps"):
            prefix = "docker() { printf '%s' %r >&2; return %d; }\n" % ("%s", message, code)
            result = subprocess.run(["bash", "-c", prefix + script], capture_output=True, text=True)
            assert result.returncode == code
            return [RemoteResult(host=h, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr) for h in hosts]
        return [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote)
    status = api.status(
        ["worker"], cluster=ClusterDefinition(name="review", hosts=["worker"], executor="docker", user="alice"), sctx=env.sctx
    )
    job = next(j for j in api.list_jobs(sctx=env.sctx) if j.cluster_id == cid)
    assert bool(status.observation_errors) is bool(failure)
    assert status.observation.confirms_absent(job) is (failure is None)
    if failure:
        assert message in status.observation_errors["worker"] or "docker" in status.observation_errors["worker"]
        assert status.free_slots("worker") == 0
        assert status.for_host("worker") is not None  # successful peer observation survives
    else:
        assert status.free_slots("worker") > 0
    env.sctx.config.set("jobs.autoprune", True)
    _prune_stale_job_metadata(env.sctx.config, observed_running=status.observation, keep=(), sctx=env.sctx)
    assert bool(load_job_metadata(cid, sctx=env.sctx)) is bool(failure)


@pytest.mark.parametrize("resume_path", ["direct", "automatic", "automatic_without_launch", "skip_run", "gap"])
@pytest.mark.parametrize("change", ["override", "recipe_default", "image", "same", "pinned", "legacy_pinned"])
def test_every_resume_path_verifies_effective_measurement_inputs(scheduled_env, monkeypatch, resume_path, change):
    from sparkrun import api
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from sparkrun.benchmarking.scheduler import BenchTask
    from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration
    from sparkrun.core.recipe import Recipe

    env = scheduled_env
    from sparkrun.orchestration.job_metadata import derive_cluster_id

    cid = derive_cluster_id(env.recipe, ["localhost"])
    env.launch.cluster_id = env.run.return_value.cluster_id = cid
    monkeypatch.setattr("sparkrun.api._benchmark._resolve_running_deployment", lambda *a, **kw: (["localhost"], True, cid))
    digest = "sha256:" + "a" * 64
    env.launch.recipe = Recipe._deserialize(env.recipe.__getstate__())
    env.launch.overrides = {"max_model_len": 1024}
    if change == "recipe_default":
        env.launch.recipe.defaults["max_model_len"] = env.launch.overrides.pop("max_model_len")
    env.launch.runtime_info = {"observed": "original"}
    meta = {
        "hosts": ["localhost"],
        "port": 8000,
        "recipe_state": env.launch.recipe.__getstate__(),
        "overrides": dict(env.launch.overrides),
        "effective_container_image": env.launch.container_image,
        "runtime_info": dict(env.launch.runtime_info),
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
        row = {
            "label": "first" if idx == 0 else "second",
            "max_model_len": env.launch.overrides.get("max_model_len", env.launch.recipe.defaults.get("max_model_len")),
        }
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
    original = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir))
    if change == "legacy_pinned":
        original.measurement_spec.pop("job_configuration_fingerprint")
        original.save(str(env.sctx.config.cache_dir))
    baseline = dict(original.measurement_spec)
    if resume_path == "gap":
        (path.parent / "runs/001.json").unlink()
    artifact = (path.parent / "runs/000.json").read_bytes()
    before = path.read_bytes()
    resuming = True
    if change == "recipe_default":
        env.launch.recipe.defaults["max_model_len"] = 2048
    elif change == "override":
        env.launch.overrides["max_model_len"] = 2048
    elif change == "image":
        env.launch.container_image = "different/image:B"
    elif change in {"pinned", "legacy_pinned"}:
        env.launch.container_image = "alias/image@" + digest
        env.launch.overrides["image"] = digest
    env.launch.runtime_info = {"observed": "later"}
    meta.update(
        recipe_state=env.launch.recipe.__getstate__(),
        overrides=dict(env.launch.overrides),
        effective_container_image=env.launch.container_image,
        runtime_info=dict(env.launch.runtime_info),
    )

    if resume_path == "automatic_without_launch":
        env.run.return_value.launch_result = None
        env.run.return_value.container_image = env.launch.container_image
        if change == "image":
            meta.pop("effective_container_image")  # API result still supplies known image evidence

    def resume():
        if resume_path in {"automatic", "automatic_without_launch", "skip_run"}:
            return api.benchmark(replace(options, resume=api.ResumeMode.IF_EXISTS, skip_run=resume_path == "skip_run"), sctx=env.sctx)
        return api.resume_benchmark(path.parent.name, sctx=env.sctx)

    if change in {"override", "recipe_default", "image"}:
        with pytest.raises(api.SparkrunError, match="configuration differs|image differs"):
            resume()
        assert commands == [(0, False), (1, False)] and not published
        assert (path.parent / "runs/000.json").read_bytes() == artifact
        if resume_path in {"direct", "gap"}:
            assert path.read_bytes() == before
    else:
        result = resume()
        assert result.success and commands == [(0, False), (1, False), (1, True)]
        assert [r["max_model_len"] for r in result.results["rows"]] == [1024, 1024]
        assert yaml.safe_load(published[0].recipe_yaml)["defaults"]["max_model_len"] == 1024
        assert published[0].provenance["runtime_info"] == {"observed": "original"}
        assert published[0].provenance["recipe"]["container"] == digest
    saved = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir))
    assert saved.measurement_spec == baseline


def test_transport_refresh_cannot_redirect_a_saved_local_job(local_job, monkeypatch):
    from sparkrun.api._stop import stop
    from sparkrun.orchestration.ssh import RemoteResult

    f = local_job

    prepared = False
    from sparkrun.orchestration.executor import resolve_executor

    def resolve(*args, **kwargs):
        assert prepared, "saved namespace policy must not require executor resolution before transport refresh"
        return resolve_executor(*args, **kwargs)

    def refresh(cluster, **kwargs):
        nonlocal prepared
        prepared = True
        cluster.user = "bob"

    monkeypatch.setattr("sparkrun.api._resolve.prepare_transport", refresh)
    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", resolve)
    calls = []

    def cleanup(grouped, **kwargs):
        calls.append(kwargs["ssh_kwargs"]["ssh_user"])
        return {host: RemoteResult(host=host, returncode=0, stdout="sparkrun_removed=1\n", stderr="") for host in grouped}

    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", cleanup)
    assert stop(cluster_id=f.cid, sctx=f.env.sctx).success
    assert calls == ["alice"]
