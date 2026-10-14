"""Real filesystem acceptance for local path and state acquisition contracts."""

import os
from pathlib import Path
import subprocess
import time

import pytest
import yaml

from test_benchmark_startup_collection import bench_env as bench_env


@pytest.mark.parametrize("problem", ["directory", "pid_file"])
def test_unreadable_pid_state_preserves_live_job_metadata(bench_env, monkeypatch, tmp_path, problem):
    from sparkrun import api
    from sparkrun.api._run import _prune_stale_job_metadata
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.ssh import RemoteResult, _local_user
    from sparkrun.orchestration.job_metadata import generate_cluster_id, save_job_metadata, load_job_metadata

    if os.geteuid() == 0:
        pytest.skip("Permission-denied reproduction requires an unprivileged account")
    env = bench_env
    cid = generate_cluster_id("a" * 16, "b" * 12)
    directory = tmp_path / "pids"
    directory.mkdir()
    pid = directory / (cid + "_solo.pid")
    pid.write_text(str(os.getpid()))
    Path(str(pid) + ".owner").write_text(env.sctx.application_profile.id)
    config = {"pid_dir": str(directory)}
    executor = LocalExecutor(ExecutorConfig(**config))
    save_job_metadata(cid, env.recipe, ["localhost"], ssh_user=_local_user(), executor=executor, sctx=env.sctx)
    path = next(env.sctx.config.cache_dir.glob("jobs/*.yaml"))
    metadata = yaml.safe_load(path.read_text())
    metadata["started_at"] = time.time() - 31 * 86400
    path.write_text(yaml.safe_dump(metadata))

    def remote(hosts, script, **kwargs):
        if script.startswith("docker ps"):
            return [RemoteResult(host, 0, "", "") for host in hosts]
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        return [RemoteResult(host, result.returncode, result.stdout, result.stderr) for host in hosts]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote)
    cluster = ClusterDefinition(name="review", hosts=["localhost"], user=_local_user(), executor="local", executor_config=config)
    control = api.status(["localhost"], cluster=cluster, sctx=env.sctx)
    assert cid in control.running_cluster_ids()
    blocked = directory if problem == "directory" else pid
    blocked.chmod(0o111 if problem == "directory" else 0)
    try:
        if problem == "directory":
            with pytest.raises(PermissionError):
                os.listdir(directory)
        else:
            with pytest.raises(PermissionError):
                pid.read_text()
        status = api.status(["localhost"], cluster=cluster, sctx=env.sctx)
        job = next(job for job in api.list_jobs(sctx=env.sctx) if job.workload_id == cid)
        assert status.observation_errors
        assert not status.observation.confirms_absent(job)
        assert status.free_slots("localhost") == 0
        env.sctx.config.set("jobs.autoprune", True)
        _prune_stale_job_metadata(env.sctx.config, observed_running=status.observation, keep=(), sctx=env.sctx)
        assert load_job_metadata(cid, sctx=env.sctx) is not None
        os.kill(os.getpid(), 0)
    finally:
        blocked.chmod(0o700 if problem == "directory" else 0o600)


@pytest.mark.parametrize("home_relative", [False, True])
def test_remote_path_preserves_symlink_parent_meaning(tmp_path, home_relative):
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    base = tmp_path / "base"
    other = tmp_path / "other"
    (other / "child").mkdir(parents=True)
    base.mkdir()
    (base / "alias").symlink_to(other / "child", target_is_directory=True)
    actual = other / "serve.log"
    wrong = base / "serve.log"
    actual.write_text("actual saved workload log\n")
    wrong.write_text("another workload log\n")
    configured = str(base / "alias" / ".." / "serve.log")
    if home_relative:
        configured = "$HOME/base/alias/../serve.log"
    env = {**os.environ, "HOME": str(tmp_path)}
    # The filesystem's ordinary path traversal reaches the intended file.
    control = subprocess.run(["bash", "-c", 'cat "$HOME/base/alias/../serve.log"'], capture_output=True, text=True, check=True, env=env)
    assert control.stdout == "actual saved workload log\n"
    executor = LocalExecutor(ExecutorConfig(log_file=configured))
    result = subprocess.run(["bash", "-c", executor.logs_cmd("job")], capture_output=True, text=True, check=True, env=env)
    assert result.stdout == "actual saved workload log\n"


def test_legacy_recovery_preserves_configured_traversal(tmp_path):
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.core.application_profile import get_application_profile

    base, other = tmp_path / "base", tmp_path / "other"
    base.mkdir()
    (other / "child").mkdir(parents=True)
    (base / "alias").symlink_to(other / "child", target_is_directory=True)
    for directory in (base, other):
        (directory / "legacy.pid").write_text("999999999")
        (directory / "legacy.pid.owner").write_text(get_application_profile().id)
    executor = LocalExecutor(ExecutorConfig(pid_file=str(base / "alias" / ".." / "legacy.pid")))
    name = "sparkrun_" + "a" * 16 + "_" + "b" * 12 + "_solo"
    subprocess.run(["bash", "-c", executor.stop_cmd(name)], check=True, capture_output=True, timeout=5)
    assert not (other / "legacy.pid").exists()
    assert (base / "legacy.pid").exists()


def test_unreadable_foreign_owner_blocks_teardown(tmp_path):
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    if os.geteuid() == 0:
        pytest.skip("Permission-denied reproduction requires an unprivileged account")
    name = "sparkrun_" + "a" * 16 + "_" + "b" * 12 + "_solo"
    pid = tmp_path / (name + ".pid")
    owner = tmp_path / (name + ".pid.owner")
    pid.write_text("999999999")
    owner.write_text("other-application")
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path)))
    control = subprocess.run(["bash", "-c", executor.stop_cmd(name)], capture_output=True, text=True, timeout=5)
    assert control.returncode and pid.exists() and owner.exists()
    owner.chmod(0)
    try:
        with pytest.raises(PermissionError):
            owner.read_text()
        result = subprocess.run(["bash", "-c", executor.stop_cmd(name)], capture_output=True, text=True, timeout=5)
        assert result.returncode != 0 and pid.exists() and owner.exists()
    finally:
        if owner.exists():
            owner.chmod(0o600)


@pytest.mark.parametrize("unreadable", [False, True])
def test_stop_preserves_records_until_the_process_is_confirmed_stopped(bench_env, tmp_path, unreadable):
    from sparkrun.api._stop import stop
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.ssh import _local_user
    from sparkrun.orchestration.job_metadata import generate_cluster_id, save_job_metadata, load_job_metadata

    if os.geteuid() == 0:
        pytest.skip("Permission-denied reproduction requires an unprivileged account")
    env = bench_env
    cid = generate_cluster_id("c" * 16, "d" * 12)
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path)))
    pid = tmp_path / (cid + "_solo.pid")
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        pid.write_text(str(child.pid))
        Path(str(pid) + ".owner").write_text(env.sctx.application_profile.id)
        save_job_metadata(cid, env.recipe, ["localhost"], ssh_user=_local_user(), executor=executor, sctx=env.sctx)
        if unreadable:
            pid.chmod(0)
            with pytest.raises(PermissionError):
                pid.read_text()
        result = stop(workload_id=cid, sctx=env.sctx)
        if unreadable:
            assert not result.success and result.containers_removed == 0
            assert result.errors
            assert child.poll() is None and pid.exists()
            assert Path(str(pid) + ".owner").exists()
            assert load_job_metadata(cid, sctx=env.sctx) is not None
        else:
            assert result.success and result.containers_removed == 1
            assert child.poll() is not None and not pid.exists()
            assert not Path(str(pid) + ".owner").exists()
            assert load_job_metadata(cid, sctx=env.sctx) is None
    finally:
        child.kill()
        child.wait(timeout=5)
        if pid.exists():
            pid.chmod(0o600)


@pytest.mark.parametrize("owner", ["current", "missing", "foreign", "empty", "invalid", "nul", "directory", "dangling", "unreadable"])
@pytest.mark.parametrize("operation", ["stop", "launch"])
def test_owner_acquisition_controls_mutation(tmp_path, owner, operation):
    from sparkrun.core.application_profile import get_application_profile
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    name = "sparkrun_" + "a" * 16 + "_" + "b" * 12 + "_solo"
    pid = tmp_path / (name + ".pid")
    marker = Path(str(pid) + ".owner")
    pid.write_text("999999999")
    profile = get_application_profile().id
    values = {
        "current": profile,
        "foreign": "other-app",
        "empty": "",
        "invalid": "not/an/id",
        "nul": profile + "\x00",
        "unreadable": "other-app",
    }
    if owner in values:
        marker.write_text(values[owner])
    elif owner == "directory":
        marker.mkdir()
    elif owner == "dangling":
        marker.symlink_to(tmp_path / "missing-owner")
    if owner == "unreadable":
        if os.geteuid() == 0:
            pytest.skip("Permission tests require an unprivileged account")
        marker.chmod(0)
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path), log_dir=str(tmp_path)))
    script = executor.stop_cmd(name) if operation == "stop" else executor.run_cmd("", "true", name)
    try:
        status = executor.query_status(["localhost"])
        assert bool(status.errors) is (owner not in {"current", "missing", "foreign"})
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=5)
        if owner in {"current", "missing"}:
            assert result.returncode == 0, result.stderr
            if operation == "stop":
                assert not pid.exists() and not marker.exists()
            else:
                assert marker.read_text() == profile
        else:
            assert result.returncode != 0
            assert pid.read_text() == "999999999"
            assert marker.exists() or marker.is_symlink()
            if owner in values and owner != "unreadable":
                assert marker.read_text() == values[owner]
    finally:
        if owner == "unreadable" and marker.exists():
            marker.chmod(0o600)


@pytest.mark.parametrize("content", ["", "0", "1", "-1", "999999999999999999999999", "42\n43", "123\x00", "not-a-pid"])
def test_invalid_pid_cannot_authorize_stop_or_replacement(tmp_path, content):
    from sparkrun.core.application_profile import get_application_profile
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    name = "sparkrun_" + "a" * 16 + "_" + "b" * 12 + "_solo"
    pid = tmp_path / (name + ".pid")
    marker = Path(str(pid) + ".owner")
    pid.write_text(content)
    marker.write_text(get_application_profile().id)
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path)))
    for script in (
        executor.status_cmd(name),
        executor.stop_cmd(name),
        executor.run_cmd("", "true", name),
        executor.teardown_script([name]),
    ):
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=5)
        assert result.returncode != 0
        assert "invalid" in result.stderr
        assert pid.read_text() == content and marker.exists()
    assert executor.query_status(["localhost"]).errors


@pytest.mark.parametrize("record", ["directory", "fifo", "dangling"])
def test_nonregular_pid_fails_without_hanging_or_deleting(tmp_path, record):
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    name = "sparkrun_" + "a" * 16 + "_" + "b" * 12 + "_solo"
    pid = tmp_path / (name + ".pid")
    if record == "directory":
        pid.mkdir()
    elif record == "fifo":
        os.mkfifo(pid)
    else:
        pid.symlink_to(tmp_path / "missing-pid")
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path)))
    result = subprocess.run(["bash", "-c", executor.teardown_script([name])], capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert pid.exists() or pid.is_symlink()
    assert executor.query_status(["localhost"]).errors


@pytest.mark.parametrize("condition", ["missing", "empty", "file", "inaccessible_parent"])
def test_directory_acquisition_distinguishes_absence_from_failure(tmp_path, condition):
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    parent = tmp_path / "parent"
    parent.mkdir()
    directory = parent / "pids"
    if condition == "empty":
        directory.mkdir()
    elif condition == "file":
        directory.write_text("not a directory")
    elif condition == "inaccessible_parent":
        if os.geteuid() == 0:
            pytest.skip("Permission tests require an unprivileged account")
        directory.mkdir()
        parent.chmod(0)
    try:
        status = LocalExecutor(ExecutorConfig(pid_dir=str(directory))).query_status(["localhost"])
        assert bool(status.errors) is (condition in {"file", "inaccessible_parent"})
        if not status.errors:
            assert status.for_host("localhost") is not None
            assert not status.running_cluster_ids()
    finally:
        parent.chmod(0o700)


def test_parent_traversal_is_part_of_destination_identity(tmp_path):
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    configured = str(tmp_path) + "/link/../pids"
    target = LocalExecutor(ExecutorConfig(pid_dir=configured)).resolve_target()
    direct = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path / "pids"))).resolve_target()
    assert target.config["pid_dir"] == configured
    assert target.destination_key != direct.destination_key


@pytest.mark.parametrize("failure", ["cat", "ls", "liveness", "ps"])
def test_acquisition_command_failures_remain_errors(tmp_path, monkeypatch, failure):
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.ssh import RemoteResult

    name = "sparkrun_" + "a" * 16 + "_" + "b" * 12 + "_solo"
    pid = tmp_path / (name + ".pid")
    pid.write_text(str(os.getpid()))
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path)))
    command = "kill" if failure == "liveness" else failure
    injected = '%s() { echo "simulated acquisition failure" >&2; return 2; };\n' % command

    def remote(hosts, script, **kwargs):
        result = subprocess.run(["bash", "-c", injected + script], capture_output=True, text=True, timeout=5)
        return [RemoteResult(host, result.returncode, result.stdout, result.stderr) for host in hosts]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote)
    assert executor.query_status(["localhost"]).errors
    # Use a dead PID when checking mutation, so the test can never signal itself.
    pid.write_text("999999999")
    if failure != "ps":  # ps is consulted only for a live PID
        result = subprocess.run(["bash", "-c", injected + executor.teardown_script([name])], capture_output=True, text=True, timeout=5)
        assert result.returncode != 0
        assert pid.exists()
