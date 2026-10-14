"""Implicit SSH configuration becomes explicit, recoverable launch identity."""

from dataclasses import replace
import shutil

import pytest

from sparkrun.orchestration._ssh_identity import resolve_ssh_user
from sparkrun.orchestration.ssh import _local_user, should_run_locally
from test_benchmark_startup_collection import bench_env as bench_env
from test_run_option_contracts import run_env as run_env


def test_local_identity_uses_os_principal_even_with_stale_environment(monkeypatch):
    monkeypatch.setenv("USER", "not-the-process-owner")
    assert resolve_ssh_user(["localhost", "127.0.0.1"]) == _local_user()
    assert should_run_locally("localhost", _local_user())


@pytest.fixture
def ssh_config(tmp_path, monkeypatch):
    if not shutil.which("ssh"):
        pytest.skip("OpenSSH is required for offline effective-configuration tests")
    monkeypatch.setattr("sparkrun.orchestration._ssh_identity.should_run_locally", lambda *a: False)
    path = tmp_path / "ssh.conf"
    path.write_text("Host worker-a worker-b\n  User alice\nHost worker-c\n  User bob\n")
    return path


def test_ssh_alias_and_custom_config_resolve_without_a_connection(ssh_config):
    assert resolve_ssh_user(["worker-a", "worker-b"], ssh_options=["-F", str(ssh_config)]) == "alice"
    assert resolve_ssh_user(["worker-a"], ssh_user="charlie", ssh_options=["-F", str(ssh_config)]) == "charlie"
    assert resolve_ssh_user(["worker-a"], ssh_options=["-F", str(ssh_config), "-o", "User=delta"]) == "delta"
    assert resolve_ssh_user(["worker-a"], ssh_user="alice", ssh_options=["-F", str(ssh_config), "-o", "User=bob"]) == "alice"


@pytest.mark.parametrize("mode", ["mixed", "invalid"])
def test_unrepresentable_destination_fails_during_planning(run_env, monkeypatch, ssh_config, mode):
    from sparkrun import api
    from sparkrun.api._run import run

    env = run_env
    env.sctx.config.ssh_user = None
    env.sctx.config.set("ssh.options", ["-F", str(ssh_config)])
    if mode == "invalid":
        ssh_config.write_text("InvalidSSHOption yes\n")
    hosts = ("worker-a", "worker-c") if mode == "mixed" else ("worker-a",)
    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", lambda **kw: pytest.fail("must reject before submitting"))
    with pytest.raises(api.SparkrunError, match="effective SSH user|Cannot resolve SSH user"):
        run(api.RunOptions(recipe=env.recipe, hosts=hosts, solo=True, executor="local"), sctx=env.sctx)


def test_plan_pins_implicit_alias_user_and_preserves_key_rotation(run_env, monkeypatch, ssh_config):
    from sparkrun import api
    from sparkrun.api._run import plan, run
    from sparkrun.core.cluster_manager import ClusterDefinition

    env = run_env
    env.sctx.config.ssh_user = None
    env.sctx.config.set("ssh.options", ["-F", str(ssh_config)])
    monkeypatch.setattr("sparkrun.api._hosts.resolve_effective_hosts", lambda hosts, *a, **kw: (hosts, True, [], None))
    cluster = ClusterDefinition(name="review", hosts=["worker-a"], executor="local")
    options = api.RunOptions(recipe=env.recipe, cluster=cluster, solo=True, dry_run=True)
    planned = plan(options, sctx=env.sctx)
    assert planned.cluster.user == "alice" and cluster.user is None
    # Even mutable nested plan inputs cannot replace the resolved principal.
    planned.cluster.user = "bob"
    ssh_config.write_text("Host worker-a\n  User bob\n")
    env.sctx.config.set("ssh.key", "/rotated/key")
    seen = []

    def launch(**kwargs):
        seen.append((kwargs["config"].ssh_user, kwargs["config"].ssh_key, kwargs["cluster_id_override"]))
        env.launch.cluster_id = kwargs["cluster_id_override"]
        return env.launch

    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", launch)
    assert run(options, sctx=env.sctx, plan=planned).rc == 0
    assert seen == [("alice", "/rotated/key", planned.workload_id)]
    assert plan(replace(options, cluster=cluster), sctx=env.sctx).workload_id != planned.workload_id
    assert env.sctx.config.ssh_user is None


def test_plan_keeps_explicit_user_when_options_change(run_env, monkeypatch, ssh_config):
    from sparkrun import api
    from sparkrun.api._run import plan, run

    env = run_env
    env.sctx.config.ssh_user = "alice"
    options = api.RunOptions(recipe=env.recipe, hosts=("worker-a",), solo=True, executor="local", ensure=True, dry_run=True)
    monkeypatch.setattr("sparkrun.api._hosts.resolve_effective_hosts", lambda hosts, *a, **kw: (hosts, True, [], None))
    planned = plan(options, sctx=env.sctx)
    env.sctx.config.set("ssh.options", ["-F", str(ssh_config), "-o", "User=bob"])

    def find(*args, **kwargs):
        assert kwargs["sctx"].config.ssh_user == "alice"
        return None

    monkeypatch.setattr("sparkrun.api._intent.find_running_intent", find)
    seen = []

    def launch(**kwargs):
        from sparkrun.orchestration.ssh import build_ssh_cmd
        import subprocess

        config = kwargs["config"]
        cmd = build_ssh_cmd("worker-a", config.ssh_user, config.ssh_key, config.ssh_options)
        effective = subprocess.run([cmd[0], "-G", *cmd[1:]], capture_output=True, text=True, check=True)
        seen.append(next(line for line in effective.stdout.splitlines() if line.startswith("user ")))
        return env.launch

    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", launch)
    assert run(options, sctx=env.sctx, plan=planned).rc == 0
    assert seen == ["user alice"]


@pytest.mark.parametrize("format", ["command", "embedded"])
@pytest.mark.parametrize("user", [None, "alice"])
@pytest.mark.parametrize("option", [["-o", "User=bob"], ["-l", "bob"], ["-oUser=bob"], ["-lbob"]])
def test_all_ssh_builders_share_effective_user_precedence(ssh_config, format, user, option):
    import shlex
    import subprocess
    from sparkrun.orchestration.ssh import build_ssh_cmd, build_ssh_opts_string

    kwargs = {"ssh_user": user, "ssh_options": ["-F", str(ssh_config), *option]}
    if format == "command":
        cmd = build_ssh_cmd("worker-a", **kwargs)
    else:
        cmd = ["ssh", *shlex.split(build_ssh_opts_string(**kwargs)), "worker-a"]
    result = subprocess.run([cmd[0], "-G", *cmd[1:]], capture_output=True, text=True, check=True)
    assert next(line for line in result.stdout.splitlines() if line.startswith("user ")) == "user " + (user or "bob")
