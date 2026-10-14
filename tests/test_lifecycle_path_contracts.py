"""Lifecycle acceptance checks use real Bash and offline OpenSSH evaluation."""

import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest
import yaml

from test_benchmark_startup_collection import bench_env as bench_env
from test_run_option_contracts import run_env as run_env


@pytest.mark.parametrize("user_option", [["-o", "User=bob"], ["-l", "bob"], ["-oUser=bob"], ["-lbob"]])
def test_saved_job_lifecycle_keeps_effective_principal(bench_env, monkeypatch, user_option):
    from sparkrun.api._stop import stop
    from sparkrun.orchestration.ssh import RemoteResult
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.job_metadata import generate_cluster_id, save_job_metadata, load_job_metadata, check_job_running

    if not shutil.which("ssh"):
        pytest.skip("OpenSSH required for local configuration evaluation")
    env = bench_env
    cid = generate_cluster_id("a" * 16, "b" * 12)
    save_job_metadata(cid, env.recipe, ["review-worker"], ssh_user="alice", executor=LocalExecutor(), sctx=env.sctx)
    options = ["-F", "/dev/null", *user_option]
    env.sctx.config.set("ssh.options", options)
    key = Path(env.sctx.config.cache_dir) / "rotated-key"
    key.write_text("test key")
    env.sctx.config.set("ssh.key", str(key))
    calls = []
    alive = True

    def remote(cmd, host, label, **kwargs):
        nonlocal alive
        # Evaluate the actual generated SSH command with OpenSSH, without connecting.
        target_idx = next(i for i, part in enumerate(cmd) if part == host or part.endswith("@" + host))
        effective = subprocess.run([cmd[0], "-G", *cmd[1 : target_idx + 1]], capture_output=True, text=True, check=True)
        user = next(line.split()[1] for line in effective.stdout.splitlines() if line.startswith("user "))
        calls.append((label, cmd[target_idx], user))
        if label == "SSH cmd":
            assert cmd[cmd.index("-i") + 1] == str(key)
            removed = alive and user == "alice"
            if removed:
                alive = False
            return RemoteResult(host=host, returncode=0, stdout=f"sparkrun_removed={int(removed)}\n", stderr="")
        stdout = cid + "_solo\t123\t" + env.sctx.application_profile.id + "\n" if alive and user == "alice" else ""
        return RemoteResult(host=host, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("sparkrun.orchestration.ssh._run_subprocess", remote)
    # Control: the saved principal is live using normal explicit-user SSH settings.
    assert check_job_running(
        cluster_id=cid, cache_dir=str(env.sctx.config.cache_dir), ssh_kwargs={"ssh_options": ["-F", "/dev/null"]}
    ).running
    assert check_job_running(cluster_id=cid, cache_dir=str(env.sctx.config.cache_dir), ssh_kwargs={"ssh_options": options}).running
    result = stop(workload_id=cid, sctx=env.sctx)
    assert result.success and result.containers_removed == 1 and not alive
    assert load_job_metadata(cid, sctx=env.sctx) is None
    assert calls and all(call[1:] == ("alice@review-worker", "alice") for call in calls)


@pytest.mark.parametrize("mode", ["normal", "spaces", "metacharacters", "tilde", "home", "braced_home"])
def test_local_status_path_contract(bench_env, monkeypatch, tmp_path, mode):
    from sparkrun import api
    from sparkrun.api._run import _prune_stale_job_metadata
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.orchestration.ssh import RemoteResult, _local_user
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.job_metadata import generate_cluster_id, save_job_metadata, load_job_metadata

    env = bench_env
    cid = generate_cluster_id("c" * 16, "d" * 12)
    name = {"normal": "pids", "metacharacters": "pids ' ; $(touch SHOULD_NOT_EXIST) $var [ab]"}.get(mode, "pid files")
    directory = tmp_path / name
    directory.mkdir()
    prefix = {"tilde": "~/", "home": "$HOME/", "braced_home": "${HOME}/"}.get(mode)
    config = {"pid_dir": prefix + name if prefix else str(directory)}
    executor = LocalExecutor(ExecutorConfig(**config))
    pid_path = directory / (cid + "_solo.pid")
    pid_path.write_text(str(os.getpid()))
    Path(str(pid_path) + ".owner").write_text(env.sctx.application_profile.id)
    save_job_metadata(cid, env.recipe, ["localhost"], ssh_user=_local_user(), executor=executor, sctx=env.sctx)
    meta_path = next(env.sctx.config.cache_dir.glob("jobs/*.yaml"))
    meta = yaml.safe_load(meta_path.read_text())
    meta["started_at"] = time.time() - 31 * 86400
    meta_path.write_text(yaml.safe_dump(meta))
    scripts = []

    def remote(hosts, script, **kwargs):
        scripts.append(script)
        if script.startswith("docker ps"):
            return [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts]
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True, cwd=tmp_path, env={**os.environ, "HOME": str(tmp_path)}
        )
        return [RemoteResult(host=h, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr) for h in hosts]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote)
    cluster = ClusterDefinition(name="review", hosts=["localhost"], user=_local_user(), executor="local", executor_config=config)
    status = api.status(["localhost"], cluster=cluster, sctx=env.sctx)
    job = next(job for job in api.list_jobs(sctx=env.sctx) if job.workload_id == cid)
    absent = status.observation.confirms_absent(job)
    os.kill(int(pid_path.read_text()), 0)  # real live process, real PID-file format
    assert not absent and not status.observation_errors
    assert cid in status.running_cluster_ids()
    assert status.free_slots("localhost") == 0
    env.sctx.config.set("jobs.autoprune", True)
    _prune_stale_job_metadata(env.sctx.config, observed_running=status.observation, keep=(), sctx=env.sctx)
    assert load_job_metadata(cid, sctx=env.sctx) is not None
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()
    assert pid_path.exists()


def test_pid_file_rejected_before_planning_or_launch(run_env, monkeypatch, tmp_path):
    from sparkrun import api
    from sparkrun.api._run import run
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    pid = tmp_path / "old.pid"
    pid.write_text(str(os.getpid()))
    options = api.RunOptions(
        recipe=run_env.recipe, hosts=("localhost",), solo=True, executor="local", executor_config={"pid_file": str(pid)}
    )
    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", lambda **kw: pytest.fail("must reject before submission"))
    with pytest.raises(api.SparkrunError, match="pid_file.*use pid_dir"):
        run(options, sctx=run_env.sctx)
    executor = LocalExecutor(ExecutorConfig(pid_file=str(pid)))
    operations = [
        lambda: executor.resolve_target(),
        lambda: executor.query_status(["localhost"]),
        lambda: executor.generate_launch_script(image="", container_name="demo", command="true"),
        lambda: executor.generate_exec_serve_script(container_name="demo", serve_command="true"),
    ]
    for operation in operations:
        with pytest.raises(ValueError, match="pid_file.*use pid_dir"):
            operation()
    assert pid.read_text() == str(os.getpid())


@pytest.mark.parametrize("legacy_cluster_config", [True, False])
def test_fixed_file_history_cannot_be_pruned_by_directory_discovery(bench_env, monkeypatch, tmp_path, legacy_cluster_config):
    from sparkrun import api
    from sparkrun.api._run import _prune_stale_job_metadata
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.ssh import RemoteResult, _local_user
    from sparkrun.orchestration.job_metadata import generate_cluster_id, save_job_metadata, load_job_metadata

    env = bench_env
    cid = generate_cluster_id("a" * 16, "b" * 12)
    save_job_metadata(cid, env.recipe, ["localhost"], ssh_user=_local_user(), executor=LocalExecutor(), sctx=env.sctx)
    path = next(env.sctx.config.cache_dir.glob("jobs/*.yaml"))
    metadata = yaml.safe_load(path.read_text())
    metadata["started_at"] = time.time() - 31 * 86400
    metadata["executor_config"]["pid_file"] = str(tmp_path / "legacy.pid")
    path.write_text(yaml.safe_dump(metadata))
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel", lambda hosts, *a, **kw: [RemoteResult(h, 0, "", "") for h in hosts]
    )
    cluster = ClusterDefinition(
        name="legacy",
        hosts=["localhost"],
        user=_local_user(),
        executor="local",
        executor_config={"pid_file": str(tmp_path / "legacy.pid")} if legacy_cluster_config else {},
    )
    snapshot = api.status(["localhost"], cluster=cluster, sctx=env.sctx)
    assert bool(snapshot.observation_errors) is legacy_cluster_config
    job = next(job for job in api.list_jobs(sctx=env.sctx) if job.workload_id == cid)
    assert not snapshot.observation.confirms_absent(job)
    env.sctx.config.set("jobs.autoprune", True)
    _prune_stale_job_metadata(env.sctx.config, observed_running=snapshot.observation, keep=(), sctx=env.sctx)
    assert load_job_metadata(cid, sctx=env.sctx) is not None


def test_legacy_fixed_file_command_recovery_uses_literal_paths(tmp_path):
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.core.application_profile import get_application_profile

    pid = tmp_path / "legacy ' file.pid"
    log = tmp_path / "legacy ' file.log"
    pid.write_text(str(os.getpid()))
    Path(str(pid) + ".owner").write_text(get_application_profile().id)
    log.write_text("saved output\n")
    executor = LocalExecutor(ExecutorConfig(pid_file=str(pid), log_file=str(log)))
    name = "sparkrun_" + "a" * 16 + "_" + "b" * 12 + "_solo"
    subprocess.run(["bash", "-c", executor.status_cmd(name)], check=True)
    result = subprocess.run(["bash", "-c", executor.logs_cmd(name)], check=True, capture_output=True, text=True)
    assert result.stdout == "saved output\n"
    pid.write_text("999999999")  # never signal the test process
    subprocess.run(["bash", "-c", executor.stop_cmd(name)], check=True, timeout=5)
    assert not pid.exists() and not Path(str(pid) + ".owner").exists()
    assert log.read_text() == "saved output\n"


@pytest.mark.parametrize("prefix", ["~/", "$HOME/", "${HOME}/"])
def test_equivalent_remote_directories_have_one_destination(prefix):
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    executor = LocalExecutor(ExecutorConfig(pid_dir=prefix + "pids/./", log_dir=prefix + "logs/./"))
    target = executor.resolve_target()
    assert target.destination_key == "$HOME/pids"
    assert target.config == {"pid_dir": "$HOME/pids", "log_dir": "$HOME/logs"}
    assert executor._resolve_pid_file("job") == "$HOME/pids/job.pid"
    assert executor._resolve_log_file("job") == "$HOME/logs/job.log"


def test_remote_home_parent_remains_relative_to_remote_home():
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    executor = LocalExecutor(ExecutorConfig(pid_dir="~/../pids"))
    assert executor.resolve_target().destination_key == "$HOME/../pids"
    assert executor._resolve_pid_file("job") == "$HOME/../pids/job.pid"


def test_terminated_log_discovery_expands_only_remote_home(tmp_path):
    from sparkrun.core.log_source import LogSource
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    # Actual local dispatch still exercises the emitted remote-home script.
    # Use an absolute path reachable through HOME/.. without changing HOME.
    import posixpath

    relative = posixpath.relpath(str(tmp_path), os.path.expanduser("~"))
    directory = tmp_path / "log ' files"
    directory.mkdir()
    name = "demo"
    log = directory / (name + ".log")
    log.write_text("saved output\n")
    executor = LocalExecutor(ExecutorConfig(log_dir="$HOME/" + relative + "/log ' files"))
    info = executor.describe_terminated([LogSource("localhost", name)])[("localhost", name)]
    assert info.exists
    output = subprocess.run(["bash", "-c", info.investigate_hints[0]], capture_output=True, text=True, check=True)
    assert output.stdout == "saved output\n"
