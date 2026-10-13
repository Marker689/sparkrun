"""Public lifecycle, resume, and monitoring boundaries share resolved evidence."""

from dataclasses import replace
from unittest.mock import Mock
import json
from pathlib import Path
import sys

import pytest

from test_benchmark_startup_collection import bench_env as bench_env
from test_run_option_contracts import run_env as run_env
from test_benchmark_api_contract import scheduled_env as scheduled_env


@pytest.mark.parametrize("entrypoint", ["telemetry", "monitor"])
@pytest.mark.parametrize("supplied_context", [True, False])
@pytest.mark.parametrize("overrides", [None, {"ssh_key": None}, {"ssh_user": None}, {"ssh_user": "charlie"}, {"ssh_options": []}])
def test_monitor_uses_same_cluster_user_as_status(bench_env, monkeypatch, entrypoint, supplied_context, overrides):
    from sparkrun import api
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.api._telemetry import LiveMonitorSession
    from sparkrun.orchestration.ssh import RemoteResult

    env = bench_env
    env.sctx.config.ssh_user = "alice"
    env.sctx.config.set("ssh.key", "/global/key")
    cluster = ClusterDefinition(name="bob", hosts=["worker"], user="stale-user", executor="local")
    monkeypatch.setattr("sparkrun.api._resolve.prepare_transport", lambda cluster, **kw: setattr(cluster, "user", "bob"))
    default_context = Mock(return_value=env.sctx)
    monkeypatch.setattr("sparkrun.api._context.default_sctx", default_context)
    expected = {"ssh_user": "bob", "ssh_key": "/global/key", "ssh_options": [], **(overrides or {})}
    provider = Mock()
    provider.open.return_value = None
    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda *a: provider)
    monkeypatch.setattr(LiveMonitorSession, "_poll_loop", lambda self: None)
    calls = []

    def remote(hosts, script, **kwargs):
        calls.append(kwargs)
        return [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote)
    # Control: ordinary status obeys the cluster's effective transport.
    api.status(["worker"], cluster=cluster, sctx=env.sctx, ssh_kwargs=overrides)
    assert calls and all(all(c[k] == value for k, value in expected.items()) for c in calls)
    calls.clear()
    if entrypoint == "telemetry":
        api.open_telemetry(["worker"], cluster=cluster, sctx=env.sctx if supplied_context else None, ssh_kwargs=overrides)
    else:
        with api.open_live_monitor(
            ["worker"], cluster=cluster, sctx=env.sctx if supplied_context else None, ssh_kwargs=overrides
        ) as session:
            session._poll_once()
        assert calls and all(all(c[k] == value for k, value in expected.items()) for c in calls)
    seen = provider.open.call_args.kwargs["ssh_kwargs"]
    assert seen == expected
    assert default_context.call_count == int(not supplied_context)
    assert env.sctx.config.ssh_user == "alice"


@pytest.mark.parametrize("failure,active", [("docker", False), ("local", False), ("both", False), (None, False), ("docker", True)])
def test_monitor_preserves_partial_observation_failure(bench_env, monkeypatch, failure, active):
    from sparkrun import api
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.api._telemetry import LiveMonitorSession
    from sparkrun.orchestration.ssh import RemoteResult

    env = bench_env
    cluster = ClusterDefinition(name="review", hosts=["worker"], user="alice", executor="docker")
    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda *a: None)
    monkeypatch.setattr(LiveMonitorSession, "_poll_loop", lambda self: None)

    def remote(hosts, script, **kwargs):
        docker = script.startswith("docker ps")
        failed = failure == "both" or failure == ("docker" if docker else "local")
        cid = "sparkrun_" + "a" * 16 + "_" + "b" * 12
        stdout = cid + "_solo\t123\t" + env.sctx.application_profile.id + "\n" if active and not docker and not failed else ""
        return [
            RemoteResult(host=h, returncode=1 if failed else 0, stdout=stdout, stderr="permission denied" if failed else "") for h in hosts
        ]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote)
    status = api.status(["worker"], cluster=cluster, sctx=env.sctx)
    assert bool(status.observation_errors) is bool(failure)
    with api.open_live_monitor(["worker"], cluster=cluster, sctx=env.sctx) as session:
        session._poll_once()
        activity = session.frame().hosts[0]
    assert activity.status_error == status.observation_errors.get("worker")
    assert activity.free_slots == status.free_slots("worker")
    assert (activity.free_slots == 0) is bool(failure)
    assert bool(activity.workloads) is active


def test_plan_preserves_principal_when_launch_default_changes(run_env, monkeypatch):
    from sparkrun import api
    from sparkrun.api._run import plan, run

    env = run_env
    env.sctx.config.ssh_user = "alice"
    options = api.RunOptions(recipe=env.recipe, hosts=("localhost",), solo=True, scheduler="greedy", executor="local", dry_run=True)
    alice = plan(options, sctx=env.sctx)
    env.sctx.config.ssh_user = "bob"
    bob = plan(options, sctx=env.sctx)
    calls = []

    def launch(**kwargs):
        calls.append((kwargs["config"].ssh_user, kwargs["cluster_id_override"]))
        env.launch.cluster_id = kwargs["cluster_id_override"]
        return env.launch

    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", launch)
    result = run(options, sctx=env.sctx, plan=alice)
    assert alice.cluster_id != bob.cluster_id
    assert calls == [("alice", alice.cluster_id)]
    assert result.cluster_id == alice.cluster_id


def test_equivalent_api_image_evidence_survives_baseline_recording(scheduled_env, monkeypatch):
    from sparkrun import api
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from sparkrun.benchmarking.scheduler import BenchTask
    from sparkrun.orchestration.job_metadata import derive_cluster_id

    env = scheduled_env
    cid = derive_cluster_id(env.recipe, ["localhost"])
    env.launch.cluster_id = env.run.return_value.cluster_id = cid
    monkeypatch.setattr("sparkrun.api._benchmark._resolve_running_deployment", lambda *a, **kw: (["localhost"], True, cid))
    digest = "sha256:" + "a" * 64
    env.launch.overrides = {"max_model_len": 1024}
    meta = {
        "hosts": ["localhost"],
        "port": 8000,
        "recipe_state": env.recipe.__getstate__(),
        "overrides": dict(env.launch.overrides),
        "effective_container_image": env.launch.container_image,
    }
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **kw: dict(meta))
    monkeypatch.setattr("sparkrun.orchestration.primitives.resolve_image_sha", lambda *a, **kw: digest)
    env.fw.build_task_list.return_value = [BenchTask(0, "first"), BenchTask(1, "second")]
    env.fw.task_coverage_key.side_effect = lambda task: task.label
    env.fw.consolidated_coverage_keys.side_effect = lambda result: {r["label"] for r in result.get("rows", [])}
    commands = []
    resuming = False

    def command(*args, result_file, **kwargs):
        idx = int(Path(result_file).stem)
        commands.append((idx, resuming))
        if idx == 1 and not resuming:
            return [sys.executable, "-c", "raise SystemExit(7)"]
        row = {"label": "first" if idx == 0 else "second", "value": 1024}
        return [sys.executable, "-c", "from pathlib import Path; Path(%r).write_text(%r)" % (result_file, json.dumps(row))]

    env.fw.build_benchmark_command.side_effect = command
    with pytest.raises(api.SparkrunError, match="incomplete"):
        api.benchmark(env.options, sctx=env.sctx)
    path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    original = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir))
    assert original.extras["container_image_sha"] == digest
    artifact = (path.parent / "runs/000.json").read_bytes()
    meta["overrides"]["image"] = digest
    meta.pop("effective_container_image")
    env.run.return_value.launch_result = None
    env.run.return_value.container_image = "alias/image@" + digest
    resuming = True
    result = api.benchmark(replace(env.options, resume=api.ResumeMode.IF_EXISTS), sctx=env.sctx)
    assert result.success and commands == [(0, False), (1, False), (1, True)]
    assert (path.parent / "runs/000.json").read_bytes() == artifact
    saved = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir))
    assert saved.measurement_spec == original.measurement_spec
    assert saved.extras["container_image"] == original.extras["container_image"]


@pytest.mark.parametrize("transport", ["localhost", "ssh_alias"])
def test_new_local_launch_without_explicit_user_is_recoverable(run_env, monkeypatch, tmp_path, transport):
    from sparkrun import api
    from sparkrun.api._run import plan
    from sparkrun.api._stop import stop
    from sparkrun.core.launcher import launch_inference
    from sparkrun.orchestration.job_metadata import load_job_metadata
    from test_launcher import _StubRuntime

    env = run_env
    env.sctx.config.ssh_user = None
    from sparkrun.orchestration.ssh import _local_user

    host, expected_user = "localhost", _local_user()
    if transport == "ssh_alias":
        import shutil

        if not shutil.which("ssh"):
            pytest.skip("OpenSSH is required for offline alias resolution")
        host, expected_user = "worker-a", "alice"
        ssh_config = tmp_path / "ssh.conf"
        ssh_config.write_text("Host worker-a\n  User alice\n")
        env.sctx.config.set("ssh.options", ["-F", str(ssh_config)])
        monkeypatch.setattr("sparkrun.orchestration._ssh_identity.should_run_locally", lambda *a: False)
    monkeypatch.setattr("sparkrun.api._hosts.resolve_effective_hosts", lambda hosts, *a, **kw: (hosts, True, [], None))
    env.sctx.config.set("runtime_cache.enabled", False)
    options = api.RunOptions(recipe=env.recipe, hosts=(host,), solo=True, executor="local", dry_run=True)
    planned = plan(options, sctx=env.sctx)
    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", lambda *a, **kw: (None, {}, {}, {}))
    monkeypatch.setattr("sparkrun.core.launcher.resolve_effective_cache_dir", lambda *a, **kw: str(tmp_path))
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **kw: None)
    monkeypatch.setattr("sparkrun.tuning.distribute.distribute_tuning_to_hosts", lambda *a, **kw: [])
    runtime = _StubRuntime()
    # Actual launcher, executor resolution, and job writer; only preparation and workload runtime are stubs.
    launched = launch_inference(
        recipe=env.recipe,
        runtime=runtime,
        host_list=[host],
        overrides={},
        config=env.sctx.config,
        sctx=env.sctx,
        v=env.sctx.variables,
        cluster=planned.cluster,
        executor_config=options.executor_overrides(),
        cluster_id_override=planned.cluster_id,
        is_solo=True,
        dry_run=False,
        sync_tuning=False,
        transfer_mode="local",
        runtime_cache_override={"enabled": False},
    )
    assert launched.rc == 0 and runtime.last_kwargs["executor"].executor_name == "local"
    meta = load_job_metadata(launched.cluster_id, sctx=env.sctx)
    from sparkrun.orchestration.ssh import _local_user, RemoteResult
    from sparkrun.orchestration.job_metadata import check_job_running
    from sparkrun.api._logs import logs

    assert meta["executor_user_scoped"] and meta["ssh_user"] == expected_user
    if transport == "ssh_alias":
        ssh_config.write_text("Host worker-a\n  User bob\n")
    calls = []

    def remote(hosts, script, **kwargs):
        calls.append(kwargs)
        return [
            RemoteResult(
                host=h, returncode=0, stdout=launched.cluster_id + "_solo\t123\t" + env.sctx.application_profile.id + "\n", stderr=""
            )
            for h in hosts
        ]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote)
    assert check_job_running(cluster_id=launched.cluster_id, cache_dir=str(env.sctx.config.cache_dir)).running
    monkeypatch.setattr("sparkrun.orchestration.logs.read_log_sources", lambda *a, **kw: iter(()))
    assert list(logs(workload_id=launched.cluster_id, sctx=env.sctx)) == []

    def cleanup(grouped, **kwargs):
        calls.append(kwargs["ssh_kwargs"])
        return {h: RemoteResult(host=h, returncode=0, stdout="sparkrun_removed=1\n", stderr="") for h in grouped}

    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", cleanup)
    assert stop(cluster_id=launched.cluster_id, sctx=env.sctx).success
    assert load_job_metadata(launched.cluster_id, sctx=env.sctx) is None
    assert calls and all(c["ssh_user"] == expected_user for c in calls)
    assert env.sctx.config.ssh_user is None


def test_failed_monitor_poll_preserves_workloads_and_marks_capacity_unknown(bench_env, monkeypatch):
    from sparkrun import api
    from sparkrun.api._telemetry import LiveMonitorSession
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload

    healthy = ClusterStatus(hosts=(HostOccupancy(host="worker", workloads=(RunningWorkload("known-job"),), used_slots=1, free_slots=1),))
    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda *a: None)
    monkeypatch.setattr(LiveMonitorSession, "_poll_loop", lambda self: None)
    monkeypatch.setattr(api, "status", Mock(side_effect=[healthy, api.SparkrunError("transport refresh failed"), healthy]))
    with api.open_live_monitor(["worker"], sctx=bench_env.sctx) as session:
        session._poll_once()
        initial = session.frame().hosts[0]
        assert initial.free_slots == 1 and initial.status_error is None
        session._poll_once()
        failed = session.frame().hosts[0]
        assert failed.workloads == initial.workloads and failed.used_slots == initial.used_slots
        assert failed.free_slots == 0 and "transport refresh failed" in failed.status_error
        session._poll_once()
        assert session.frame().hosts[0] == initial
