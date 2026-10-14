"""Native managed destinations and launch claims across actual API/shell boundaries."""

import os
from pathlib import Path
import re
import resource
import shlex
import subprocess
import time

import pytest
import yaml

from sparkrun.orchestration.executors._base import ExecutorConfig, ExecutorTarget
from sparkrun.orchestration.executors.local import LocalExecutor
from test_local_workload_boundaries import alive, await_file, bash, cleanup_group
from test_run_option_contracts import run_env as run_env
from test_benchmark_startup_collection import bench_env as bench_env

NAME = "sparkrun_aaaabbbbccccdddd_eeeeffff0000_solo"


@pytest.mark.parametrize("field", ["pid_dir", "log_dir", "log_file"])
@pytest.mark.parametrize("path", ["pids", "./$HOME/pids", "../state"])
def test_managed_relative_paths_rejected_before_scripts_or_submission(tmp_path, field, path):
    executor = LocalExecutor(ExecutorConfig(**{field: path}))
    operations = [
        lambda: executor.resolve_target(),
        lambda: executor.query_status(["localhost"]),
        lambda: executor.run_cmd("", "true", NAME),
        lambda: executor.generate_launch_script("", NAME, "true"),
        lambda: executor.generate_exec_serve_script(NAME, "true"),
        lambda: executor.generate_node_script("", NAME, "true"),
    ]
    for operation in operations:
        with pytest.raises(ValueError, match=field + " requires an absolute or home-relative"):
            operation()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["pid_dir", "log_dir", "log_file"])
def test_api_rejects_relative_managed_paths_before_submission(run_env, monkeypatch, field):
    from sparkrun import api
    from sparkrun.api._run import run

    options = api.RunOptions(
        recipe=run_env.recipe,
        hosts=("localhost",),
        solo=True,
        executor="local",
        executor_config={field: "relative/state"},
    )
    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", lambda **kw: pytest.fail("must reject before submission"))
    with pytest.raises(api.SparkrunError, match=field + " requires an absolute or home-relative"):
        run(options, sctx=run_env.sctx)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("home_relative", [False, True])
def test_api_recovery_across_entry_directories_preserves_destination(tmp_path, monkeypatch, legacy, home_relative):
    from sparkrun import api
    from sparkrun.api._run import _prune_stale_job_metadata
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.status_observation import ExecutorCoverage, RunningSnapshot
    from sparkrun.orchestration.job_metadata import generate_cluster_id, save_job_metadata, load_job_metadata
    from sparkrun.orchestration.ssh import RemoteResult, _local_user

    sctx = api.default_sctx()
    recipe = Recipe({"name": "native-claim", "runtime": "vllm-distributed", "model": "test/model", "container": "test/image"})
    cid = generate_cluster_id("a" * 16, "b" * 12)
    name = cid + "_solo"
    entry_a, entry_b = tmp_path / "a", tmp_path / "b"
    entry_a.mkdir()
    entry_b.mkdir()
    monkeypatch.setenv("HOME", str(entry_a))
    monkeypatch.chdir(entry_a)
    config = {"pid_dir": "${HOME}/pids" if home_relative else str(entry_a / "pids"), "log_dir": str(entry_a / "logs")}
    executor = LocalExecutor(ExecutorConfig(**config))
    record = entry_a / "pids" / (name + ".pid")
    pid = None

    def remote(hosts, script, **kwargs):
        if script.startswith("docker ps"):
            return [RemoteResult(host, 0, "", "") for host in hosts]
        result = bash(script)
        return [RemoteResult(host, result.returncode, result.stdout, result.stderr) for host in hosts]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote)
    cluster = ClusterDefinition(name="native", hosts=["localhost"], user=_local_user(), executor="local", executor_config=config)
    try:
        result = bash(executor.run_cmd("", "exec sleep 300", name))
        assert result.returncode == 0, result.stderr
        pid = int(record.read_text())
        save_job_metadata(cid, recipe, ["localhost"], ssh_user=_local_user(), executor=executor, sctx=sctx)
        assert cid in api.status(["localhost"], cluster=cluster, sctx=sctx).running_cluster_ids()
        path = next(sctx.config.cache_dir.glob("jobs/*.yaml"))
        metadata = yaml.safe_load(path.read_text())
        metadata["started_at"] = time.time() - 31 * 86400
        if legacy:
            # Model a pre-migration relative-path job without asking the new
            # launcher to accept an unsupported destination.
            metadata["executor_config"]["pid_dir"] = "pids"
            metadata["executor_destination_key"] = "pids"
            cluster.executor_config = {**config, "pid_dir": "pids"}
        path.write_text(yaml.safe_dump(metadata))
        monkeypatch.chdir(entry_b)
        status = api.status(["localhost"], cluster=cluster, sctx=sctx)
        job = next(job for job in api.list_jobs(sctx=sctx) if job.workload_id == cid)
        assert bool(status.observation_errors) is legacy
        assert not status.observation.confirms_absent(job)
        sctx.config.set("jobs.autoprune", True)
        _prune_stale_job_metadata(sctx.config, observed_running=status.observation, keep=(), sctx=sctx)
        assert load_job_metadata(cid, sctx=sctx) is not None
        assert alive(pid)
        if legacy:
            with pytest.raises(api.SparkrunError, match="Recorded native destination is unsupported or unanchored"):
                api.stop(workload_id=cid, sctx=sctx)
            assert load_job_metadata(cid, sctx=sctx) is not None and alive(pid)
            # Even an old complete cached observation with exactly matching
            # strings must not authorize absence or reuse after migration.
            target = ExecutorTarget("local", {"pid_dir": "pids"}, destination_key="pids", user_scoped=True)
            coverage = ExecutorCoverage(target, "host", frozenset({"localhost"}), frozenset({"localhost"}), ssh_user=_local_user())
            old_snapshot = RunningSnapshot.from_dict(RunningSnapshot(frozenset(), (coverage,)).to_dict())
            assert not old_snapshot.confirms_absent(job)
            assert not old_snapshot.covers(target, ["localhost"], ssh_user=_local_user())
            _prune_stale_job_metadata(sctx.config, observed_running=old_snapshot, keep=(), sctx=sctx)
            # A healthy sweep of a different, absolute directory also cannot
            # make the unresolved historical destination become known.
            cluster.executor_config = {"pid_dir": str(entry_b / "pids")}
            healthy = api.status(["localhost"], cluster=cluster, sctx=sctx)
            assert not healthy.observation_errors and not healthy.observation.confirms_absent(job)
            assert load_job_metadata(cid, sctx=sctx) is not None
            # Explicit low-level recovery in the original directory still works.
            recovery = LocalExecutor(ExecutorConfig(pid_dir="pids"))
            monkeypatch.chdir(entry_a)
            assert bash(recovery.status_cmd(name)).returncode == 0
            assert bash(recovery.stop_cmd(name)).returncode == 0
            assert not alive(pid) and not record.exists()
        else:
            result = api.stop(workload_id=cid, sctx=sctx)
            assert result.success and result.containers_removed == 1
            assert load_job_metadata(cid, sctx=sctx) is None
            assert not alive(pid) and not record.exists()
    finally:
        cleanup_group(pid)


@pytest.mark.parametrize("key", ["pids", "./$HOME/pids"])
def test_saved_relative_destination_key_cannot_fall_back_to_default_stop(bench_env, monkeypatch, key):
    from sparkrun.api._errors import SparkrunError
    from sparkrun.api._stop import stop
    from sparkrun.orchestration.job_metadata import generate_cluster_id, save_job_metadata, load_job_metadata
    from sparkrun.orchestration.ssh import _local_user

    env = bench_env
    cid = generate_cluster_id("a" * 16, "b" * 12)
    save_job_metadata(cid, env.recipe, ["localhost"], ssh_user=_local_user(), executor=LocalExecutor(), sctx=env.sctx)
    path = next(env.sctx.config.cache_dir.glob("jobs/*.yaml"))
    metadata = yaml.safe_load(path.read_text())
    metadata["executor_destination_key"] = key
    metadata["executor_config"].pop("pid_dir", None)
    path.write_text(yaml.safe_dump(metadata))
    monkeypatch.setattr("sparkrun.api._stop._stop_containers", lambda *a, **kw: pytest.fail("unanchored stop must not dispatch"))
    with pytest.raises(SparkrunError, match="Recorded native destination is unsupported or unanchored"):
        stop(workload_id=cid, sctx=env.sctx)
    assert load_job_metadata(cid, sctx=env.sctx) is not None


@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("failure,ignore_xfsz", [("write", False), ("write", True), ("rename", False), ("temporary", False)])
def test_pid_commit_failure_rolls_back_child_and_allows_retry(tmp_path, failure, ignore_xfsz, stale):
    from sparkrun.core.application_profile import ApplicationProfile, select_application_profile

    select_application_profile(ApplicationProfile(id="s", display_name="Test", command="s", package="s"))
    name = "s_aaaabbbbccccdddd_eeeeffff0000_solo"
    directory = tmp_path / "pids"
    directory.mkdir()
    record = directory / (name + ".pid")
    if stale:
        record.write_text("999999999")
        Path(str(record) + ".owner").write_text("s")
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(directory), log_dir=str(tmp_path / "logs")))
    environment = dict(os.environ)
    if failure != "write":
        # Fault injection at the final rename or temporary-file creation. The
        # owner write succeeds; only the post-spawn PID operation fails.
        binary = "mv" if failure == "rename" else "mktemp"
        shim_dir = tmp_path / "bin"
        shim_dir.mkdir()
        shim = shim_dir / binary
        pattern = "*.pid" if binary == "mv" else "*.pid.pending.XXXXXX"
        shim.write_text(f'#!/bin/bash\ncase "${{!#}}" in {pattern}) exit 73 ;; esac\nexec /usr/bin/{binary} "$@"\n')
        shim.chmod(0o700)
        environment["PATH"] = str(shim_dir) + os.pathsep + environment["PATH"]

    def limit():
        resource.setrlimit(resource.RLIMIT_FSIZE, (1, 1))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    pid = None
    try:
        script = ("trap '' XFSZ\n" if ignore_xfsz else "") + executor.run_cmd("", "exec sleep 300", name)
        result = subprocess.run(
            ["bash", "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            preexec_fn=limit if failure == "write" else None,
        )
        assert result.returncode != 0 and "Launched" not in result.stdout
        pid = int(re.search(r"Cannot persist native PID (\d+)", result.stderr)[1])
        assert re.search(r"^Rolled back native launch: PID/group %d$" % pid, result.stderr, re.MULTILINE)
        assert not alive(pid)
        if failure == "write":
            assert ("File too large" if ignore_xfsz else "File size limit exceeded") in result.stderr
        assert record.read_text() == "999999999" if stale else not record.exists()
        assert not list(directory.glob("*.pending.*"))
        ready = tmp_path / "ready"
        retry = bash(executor.run_cmd("", "printf ready > %s; exec sleep 300" % shlex.quote(str(ready)), name))
        assert retry.returncode == 0, retry.stderr
        pid = int(record.read_text())
        await_file(ready)
        assert alive(pid)
        assert bash(executor.stop_cmd(name)).returncode == 0
        assert not record.exists() and not alive(pid)
    finally:
        cleanup_group(pid)


@pytest.mark.parametrize("ignore_xfsz", [False, True])
def test_owner_write_failure_prevents_spawn_and_preserves_stale_claim(tmp_path, ignore_xfsz):
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path), log_dir=str(tmp_path)))
    record = tmp_path / (NAME + ".pid")
    record.write_text("999999999")
    Path(str(record) + ".owner").write_text("sparkrun")
    ready = tmp_path / "ready"

    def limit():
        resource.setrlimit(resource.RLIMIT_FSIZE, (1, 1))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    script = ("trap '' XFSZ\n" if ignore_xfsz else "") + executor.run_cmd("", "touch %s" % shlex.quote(str(ready)), NAME)
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, preexec_fn=limit, timeout=5)
    assert result.returncode != 0 and "Launched" not in result.stdout
    assert "Cannot persist native PID" not in result.stderr
    assert not ready.exists()
    assert record.read_text() == "999999999"
    assert Path(str(record) + ".owner").read_text() == "sparkrun"
    assert not list(tmp_path.glob("*.pending.*"))


@pytest.mark.parametrize("generator", ["run_cmd", "generate_exec_serve_script", "generate_launch_script"])
def test_foreground_launch_is_rejected_and_detached_mode_works(tmp_path, generator):
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path / "pids"), log_dir=str(tmp_path / "logs")))
    ready = tmp_path / "ready"
    command = "printf ready > %s; exec sleep 300" % shlex.quote(str(ready))

    def generate(detached):
        if generator == "generate_exec_serve_script":
            return executor.generate_exec_serve_script(NAME, command, detached=detached)
        return (
            getattr(executor, generator)("", NAME, command, detach=detached)
            if generator == "generate_launch_script"
            else executor.run_cmd("", command, NAME, detach=detached)
        )

    with pytest.raises(ValueError, match="detached workload launch only"):
        generate(False)
    assert not ready.exists() and not (tmp_path / "pids").exists()
    script = generate(True)
    pid = None
    try:
        result = bash(script)
        assert result.returncode == 0, result.stderr
        if generator == "generate_launch_script":
            assert not ready.exists()  # preflight has no serving process
        else:
            pid = int((tmp_path / "pids" / (NAME + ".pid")).read_text())
            await_file(ready)
            assert alive(pid)
            assert bash(executor.stop_cmd(NAME)).returncode == 0
            assert not alive(pid)
    finally:
        cleanup_group(pid)


def test_unconfirmed_pid_commit_rollback_reports_child_for_manual_recovery(tmp_path):
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path / "pids"), log_dir=str(tmp_path / "logs")))
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shims = {
        "mv": '#!/bin/bash\ncase "${!#}" in *.pid) exit 73 ;; esac\nexec /usr/bin/mv "$@"\n',
        "ps": "#!/bin/bash\nexit 74\n",
    }
    for name, text in shims.items():
        path = shim_dir / name
        path.write_text(text)
        path.chmod(0o700)
    ready = tmp_path / "ready"
    script = executor.run_cmd("", "printf ready > %s; exec sleep 300" % shlex.quote(str(ready)), NAME)
    pid = None
    try:
        result = bash(script, env={**os.environ, "PATH": str(shim_dir) + os.pathsep + os.environ["PATH"]})
        assert result.returncode != 0 and "Launched" not in result.stdout
        pid = int(re.search(r"Cannot persist native PID (\d+)", result.stderr)[1])
        assert f"rollback could not be confirmed: PID/group {pid}; manual recovery required" in result.stderr
        await_file(ready)
        assert alive(pid)
        assert not (tmp_path / "pids" / (NAME + ".pid")).exists()
        assert not list((tmp_path / "pids").glob("*.pending.*"))
    finally:
        cleanup_group(pid)
