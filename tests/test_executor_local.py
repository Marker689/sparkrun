"""Unit tests for the experimental LocalExecutor.

Verifies that:
- ``ExecutorConfig`` honors the new ``executor`` selector and the
  local-only path fields (working_dir, log_dir, pid_dir, ...).
- ``LocalExecutor`` generates well-formed shell scripts whose intent
  matches the design (setsid launcher, kill-by-pidfile teardown, tail
  -F for follow logs, no-op pull/inspect, raises on Ray).
- The launcher-side factory + ``build_executor`` helper dispatch
  correctly between Docker and Local.
- Multi-host clusters get per-rank pidfile/logfile paths automatically.
"""

from __future__ import annotations

import subprocess

import pytest

from sparkrun.orchestration.executor import (
    EXECUTOR_DEFAULTS,
    ExecutorConfig,
    resolve_executor,
)
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.executors.local import LocalExecutor, _hostify_env


# ---------------------------------------------------------------------------
# ExecutorConfig (new fields)
# ---------------------------------------------------------------------------


class TestExecutorConfigLocalFields:
    """The experimental local-only ExecutorConfig fields."""

    def test_default_executor_type_is_docker(self):
        cfg = ExecutorConfig()
        assert cfg.executor_type == "docker"

    def test_local_executor_selector_via_executor_key(self):
        cfg = ExecutorConfig.from_chain({"executor": "local"})
        assert cfg.executor_type == "local"

    def test_local_executor_selector_via_executor_type_key(self):
        cfg = ExecutorConfig.from_chain({"executor_type": "local"})
        assert cfg.executor_type == "local"

    def test_executor_key_takes_priority_over_executor_type(self):
        cfg = ExecutorConfig.from_chain({"executor": "local", "executor_type": "docker"})
        assert cfg.executor_type == "local"

    def test_unknown_executor_type_warns_and_falls_back(self, caplog):
        cfg = ExecutorConfig.from_chain({"executor": "wasm"})
        assert cfg.executor_type == "docker"
        assert any("wasm" in rec.message.lower() or "wasm" in str(rec.args).lower() for rec in caplog.records)

    def test_local_fields_round_trip_from_chain(self):
        chain = {
            "executor": "local",
            "working_dir": "/srv/inference",
            "log_dir": "/var/log/sparkrun",
            "log_file": "/var/log/sparkrun/custom.log",
            "pid_dir": "/run/sparkrun",
            "pid_file": "/run/sparkrun/custom.pid",
            "env_file": "/etc/sparkrun/env",
            "command_prefix": "uv run",
        }
        cfg = ExecutorConfig.from_chain(chain)
        assert cfg.executor_type == "local"
        assert cfg.working_dir == "/srv/inference"
        assert cfg.log_dir == "/var/log/sparkrun"
        assert cfg.log_file == "/var/log/sparkrun/custom.log"
        assert cfg.pid_dir == "/run/sparkrun"
        assert cfg.pid_file == "/run/sparkrun/custom.pid"
        assert cfg.env_file == "/etc/sparkrun/env"
        assert cfg.command_prefix == "uv run"

    def test_local_fields_default_to_none(self):
        cfg = ExecutorConfig()
        assert cfg.working_dir is None
        assert cfg.log_dir is None
        assert cfg.log_file is None
        assert cfg.pid_dir is None
        assert cfg.pid_file is None
        assert cfg.env_file is None
        assert cfg.command_prefix is None


# ---------------------------------------------------------------------------
# resolve_executor dispatch (Local-flavoured cases)
# ---------------------------------------------------------------------------


class TestResolveExecutorLocalDispatch:
    """Lightweight cases that exercise the Local path through ``resolve_executor``.

    More exhaustive coverage of the resolution chain lives in
    ``tests/test_executor_resolution.py`` — these are the
    Local-executor-flavoured slices.
    """

    def test_default_returns_docker(self):
        ex = resolve_executor()
        assert isinstance(ex, DockerExecutor)
        assert ex.config.executor_type == "docker"

    def test_explicit_docker_via_cli_overrides(self):
        ex = resolve_executor(cli_overrides={"executor": "docker"})
        assert isinstance(ex, DockerExecutor)

    def test_local_selector_via_cli_overrides(self):
        ex = resolve_executor(
            cli_overrides={"executor": "local", "log_dir": "/tmp/x", "pid_dir": "/tmp/y"},
            rootless=False,
            auto_user=False,
        )
        assert isinstance(ex, LocalExecutor)
        assert ex.config.log_dir == "/tmp/x"
        assert ex.config.pid_dir == "/tmp/y"


# ---------------------------------------------------------------------------
# LocalExecutor: command generators
# ---------------------------------------------------------------------------


def _local(**cfg_kwargs) -> LocalExecutor:
    """Helper: build a LocalExecutor with explicit pid_dir/log_dir for
    deterministic substring assertions."""
    return LocalExecutor(
        ExecutorConfig(
            executor_type="local",
            pid_dir=cfg_kwargs.pop("pid_dir", "/tmp/sparkrun-test/pids"),
            log_dir=cfg_kwargs.pop("log_dir", "/tmp/sparkrun-test/logs"),
            **cfg_kwargs,
        )
    )


class TestLocalExecutorBasics:
    def test_inspect_exists_is_noop(self):
        assert _local().inspect_exists_cmd("any/image:tag") == "true"

    def test_pull_is_noop(self):
        assert _local().pull_cmd("any/image:tag") == "true"

    def test_run_cmd_requires_container_name(self):
        with pytest.raises(ValueError, match="container_name"):
            _local().run_cmd(image="", command="echo hi", container_name=None)

    def test_run_cmd_requires_command(self):
        with pytest.raises(ValueError, match="command"):
            _local().run_cmd(image="", command="", container_name="foo")

    def test_run_cmd_uses_setsid_and_pidfile(self):
        script = _local().run_cmd(image="", command="echo hi", container_name="foo_solo")
        assert "setsid" in script
        assert "/tmp/sparkrun-test/pids/foo_solo.pid" in script
        assert "/tmp/sparkrun-test/logs/foo_solo.log" in script
        # Background detach: the bash `&` token must be present so the
        # script returns immediately to the SSH transport.
        assert " &\n" in script

    def test_run_cmd_writes_pid_after_setsid(self):
        script = _local().run_cmd(image="", command="echo hi", container_name="foo_solo")
        setsid_idx = script.index("\nsetsid bash ")
        commit_pid_idx = script.index('_sr_local_commit_pid "$_pid"')
        assert setsid_idx < commit_pid_idx, "PID must be captured AFTER setsid backgrounds the child"

    def test_run_cmd_rejects_pid_file_override(self):
        ex = LocalExecutor(ExecutorConfig(pid_file="/tmp/explicit.pid", log_file="/tmp/explicit.log"))
        with pytest.raises(ValueError, match="pid_file.*use pid_dir"):
            ex.run_cmd(image="", command="echo hi", container_name="foo_solo")

    def test_run_cmd_includes_working_dir(self):
        script = _local(working_dir="/srv/inference").run_cmd(image="", command="echo hi", container_name="foo_solo")
        # A plain path is shlex-quoted (safe chars stay bare) — NOT bare double-quoted.
        assert "cd -- /srv/inference || exit $?" in script

    def test_run_cmd_sources_env_file(self):
        script = _local(env_file="/etc/sparkrun.env").run_cmd(image="", command="echo hi", container_name="foo_solo")
        # ``set -a`` makes sourced KEY=VAL lines export automatically.
        assert "set -a" in script
        assert ". /etc/sparkrun.env" in script
        assert "set +a" in script

    def test_run_cmd_env_file_home_relative_expands(self):
        # Only a leading $HOME/ ~/ prefix expands; the remainder is shlex-quoted literal.
        script = _local(env_file="$HOME/.cache/sparkrun/uv-venv/x/env.sh").run_cmd(image="", command="echo hi", container_name="foo_solo")
        assert '. "$HOME"/.cache/sparkrun/uv-venv/x/env.sh' in script
        # A leading ~/ is rewritten to $HOME/ (tilde doesn't expand inside quotes).
        script2 = _local(env_file="~/venv/env.sh").run_cmd(image="", command="echo hi", container_name="foo_solo")
        assert '. "$HOME"/venv/env.sh' in script2

    def test_run_cmd_working_dir_does_not_expand_or_inject(self):
        # A path with shell metacharacters must be a shlex-quoted literal — the
        # single quotes neutralize command substitution / variable expansion
        # (unlike a bare double-quote wrapper, which the shell would interpret).
        script = _local(working_dir="/srv/$(touch pwned)/$data").run_cmd(image="", command="echo hi", container_name="foo_solo")
        assert "cd -- '/srv/$(touch pwned)/$data' || exit $?" in script  # single-quoted → inert
        assert 'cd "/srv' not in script  # NOT double-quoted (which would expand $())

    def test_run_cmd_prepends_command_prefix(self):
        script = _local(command_prefix="taskset -c 0-7").run_cmd(image="", command="vllm serve foo", container_name="foo_solo")
        # Command body is b64-encoded; decode and verify.
        # The decoded body should start with the prefix.
        import base64
        import re

        m = re.search(r"printf %s ([A-Za-z0-9+/=]+) \|", script)
        assert m is not None
        decoded = base64.b64decode(m.group(1)).decode()
        assert decoded.startswith("taskset -c 0-7 vllm serve foo")

    def test_run_cmd_exports_explicit_env(self):
        script = _local().run_cmd(
            image="",
            command="echo hi",
            container_name="foo_solo",
            env={"HF_TOKEN": "secret", "ANOTHER": "value"},
        )
        assert "export ANOTHER=value" in script
        assert "export HF_TOKEN=secret" in script

    def test_run_cmd_gpus_device_to_cuda_visible_devices(self):
        ex = LocalExecutor(ExecutorConfig(executor_type="local", gpus="device=0,2", pid_dir="/tmp", log_dir="/tmp"))
        script = ex.run_cmd(image="", command="echo hi", container_name="foo_solo")
        assert "export CUDA_VISIBLE_DEVICES=0,2" in script

    def test_run_cmd_gpus_all_omits_cuda_export(self):
        ex = LocalExecutor(ExecutorConfig(executor_type="local", gpus="all", pid_dir="/tmp", log_dir="/tmp"))
        script = ex.run_cmd(image="", command="echo hi", container_name="foo_solo")
        assert "CUDA_VISIBLE_DEVICES" not in script

    def test_stop_cmd_targets_process_group(self):
        cmd = _local().stop_cmd("foo_solo")
        # -- - sends SIGTERM to the negative PID == process group.
        assert "kill -TERM -- -" in cmd
        assert "kill -KILL" in cmd  # fallback after grace
        assert "/tmp/sparkrun-test/pids/foo_solo.pid" in cmd
        assert "rm -f" in cmd

    def test_status_cmd_uses_kill_zero(self):
        cmd = _local().status_cmd("foo_solo")
        assert "kill -0" in cmd
        assert "/tmp/sparkrun-test/pids/foo_solo.pid" in cmd

    def test_logs_cmd_follow_uses_tail_F(self):
        cmd = _local().logs_cmd("foo_solo", follow=True, tail=100)
        # ``-F`` (capital) survives file rotation/recreation; -f does not.
        assert cmd.startswith("tail -F")
        assert "-n 100" in cmd
        assert cmd.endswith("/foo_solo.log")

    def test_logs_cmd_no_follow_no_dash_F(self):
        cmd = _local().logs_cmd("foo_solo", follow=False, tail=None)
        assert "-F" not in cmd
        assert "-n +1" in cmd

    def test_exec_cmd_runs_in_subshell_with_prelude(self):
        ex = _local(working_dir="/srv")
        cmd = ex.exec_cmd(container_name="foo_solo", command="echo hi")
        # exec_cmd uses ( ... ) so prelude (cd) doesn't pollute caller.
        assert cmd.startswith("(")
        assert cmd.endswith(")")
        assert "cd -- /srv || exit $?" in cmd

    def test_exec_cmd_without_prelude_is_bare_bash_c(self):
        cmd = LocalExecutor(ExecutorConfig()).exec_cmd(container_name="x", command="echo hi")
        assert cmd.startswith("bash -c")


# ---------------------------------------------------------------------------
# LocalExecutor: high-level script generators
# ---------------------------------------------------------------------------


class TestLocalExecutorScripts:
    def test_generate_launch_script_is_preflight_only(self):
        script = _local().generate_launch_script(
            image="ignored",
            container_name="foo_solo",
            command="should not run yet",
        )
        # Preflight should *only* clean up; the actual setsid launch
        # must wait for generate_exec_serve_script.
        assert "\nsetsid bash " not in script
        assert "kill" in script  # stop_cmd is reused as cleanup
        assert "preflight complete" in script

    def test_generate_exec_serve_script_actually_launches(self):
        script = _local().generate_exec_serve_script(
            container_name="foo_solo",
            serve_command="vllm serve foo",
        )
        assert "setsid" in script
        assert '_sr_local_commit_pid "$_pid"' in script

    def test_generate_node_script_per_rank(self):
        ex = _local()
        s0 = ex.generate_node_script(image="", container_name="sparkrun_abc_node_0", serve_command="cmd rank 0")
        s1 = ex.generate_node_script(image="", container_name="sparkrun_abc_node_1", serve_command="cmd rank 1")
        # Each rank gets its own pidfile/logfile by default — no clashes.
        assert "sparkrun_abc_node_0.pid" in s0 and "sparkrun_abc_node_0.log" in s0
        assert "sparkrun_abc_node_1.pid" in s1 and "sparkrun_abc_node_1.log" in s1
        assert "sparkrun_abc_node_0" not in s1
        assert "sparkrun_abc_node_1" not in s0

    def test_ray_head_script_raises(self):
        with pytest.raises(NotImplementedError, match="Ray"):
            _local().generate_ray_head_script(image="x", container_name="y")

    def test_ray_worker_script_raises(self):
        with pytest.raises(NotImplementedError, match="Ray"):
            _local().generate_ray_worker_script(image="x", container_name="y", head_ip="1.2.3.4")


# ---------------------------------------------------------------------------
# LocalExecutor: bash-level syntax check
# ---------------------------------------------------------------------------


class TestLocalExecutorBashSyntax:
    """Generated scripts must be valid bash.  Run them through ``bash -n``."""

    def _check(self, script: str) -> None:
        result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert result.returncode == 0, "bash -n failed:\n%s\n---\n%s" % (script, result.stderr)

    def test_launch_script_syntax(self):
        self._check(_local().generate_launch_script(image="", container_name="foo_solo", command=""))

    def test_exec_serve_script_syntax(self):
        self._check(_local().generate_exec_serve_script(container_name="foo_solo", serve_command="echo hi"))

    def test_node_script_syntax(self):
        self._check(
            _local().generate_node_script(
                image="",
                container_name="sparkrun_abc_node_0",
                serve_command="echo rank 0",
            )
        )

    def test_stop_status_logs_syntax(self):
        ex = _local()
        for snippet in (ex.stop_cmd("foo_solo"), ex.status_cmd("foo_solo"), ex.logs_cmd("foo_solo", follow=True, tail=10)):
            self._check("#!/bin/bash\n" + snippet + "\n")


# ---------------------------------------------------------------------------
# End-to-end: launch + status + stop against a real local process
# ---------------------------------------------------------------------------


class TestLocalExecutorRoundTrip:
    """Exercise the full lifecycle against a ``sleep`` subprocess.

    Tests are filesystem-touching but stay inside ``tmp_path``, so they
    are safe to run in CI on Linux runners.
    """

    @pytest.mark.parametrize("directory", ["plain", "spaces ' [literal] $(touch SHOULD_NOT_EXIST)"])
    def test_launch_status_stop(self, tmp_path, directory):
        pid_dir = tmp_path / directory / "pids"
        log_dir = tmp_path / directory / "logs"
        ex = LocalExecutor(
            ExecutorConfig(
                executor_type="local",
                pid_dir=str(pid_dir),
                log_dir=str(log_dir),
            )
        )
        name = "rt_test_solo"
        # Preflight + launch:
        launch = ex.generate_launch_script(image="", container_name=name, command="")
        exec_serve = ex.generate_exec_serve_script(container_name=name, serve_command="sleep 30")
        script = "%s\n%s" % (launch, exec_serve)
        # Run the launcher and let it return:
        subprocess.run(["bash", "-c", script], check=True, timeout=10)

        # PID file must exist.
        pid_file = pid_dir / ("%s.pid" % name)
        assert pid_file.exists()
        pid = int(pid_file.read_text().strip())
        assert pid > 0

        # status_cmd should report alive.
        rc = subprocess.run(["bash", "-c", ex.status_cmd(name)]).returncode
        assert rc == 0

        # stop_cmd should kill it and remove the pidfile.
        subprocess.run(["bash", "-c", ex.stop_cmd(name)], check=True, timeout=15)
        assert not pid_file.exists()

        # The process must really be gone.
        import os

        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


# ---------------------------------------------------------------------------
# _hostify_env: reverse-map container-path env values to host source paths
# ---------------------------------------------------------------------------


class TestHostifyEnv:
    """The container-path → host-path env reverse-mapper."""

    _VOLUMES = {"/home/ubuntu/.cache/huggingface": "/cache/huggingface"}

    def test_exact_target_maps_to_host_source(self):
        out = _hostify_env({"HF_HOME": "/cache/huggingface"}, self._VOLUMES)
        assert out == {"HF_HOME": "/home/ubuntu/.cache/huggingface"}

    def test_prefix_under_target_is_rewritten(self):
        out = _hostify_env({"HF_HUB_CACHE": "/cache/huggingface/hub"}, self._VOLUMES)
        assert out == {"HF_HUB_CACHE": "/home/ubuntu/.cache/huggingface/hub"}

    def test_non_path_value_passes_through(self):
        out = _hostify_env({"FOO": "bar", "HF_HUB_OFFLINE": "1"}, self._VOLUMES)
        assert out == {"FOO": "bar", "HF_HUB_OFFLINE": "1"}

    def test_mixed_env_maps_only_matching_values(self):
        env = {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "FOO": "bar",
        }
        out = _hostify_env(env, self._VOLUMES)
        assert out == {
            "HF_HOME": "/home/ubuntu/.cache/huggingface",
            "HF_HUB_CACHE": "/home/ubuntu/.cache/huggingface/hub",
            "FOO": "bar",
        }

    def test_longest_target_wins(self):
        # Two overlapping mounts: the more specific (longer) target must win.
        volumes = {
            "/host/hf": "/cache/huggingface",
            "/host/hf/hub": "/cache/huggingface/hub",
        }
        out = _hostify_env({"X": "/cache/huggingface/hub/models"}, volumes)
        assert out == {"X": "/host/hf/hub/models"}

    def test_no_volumes_passes_through_unchanged(self):
        env = {"HF_HOME": "/cache/huggingface"}
        assert _hostify_env(env, None) == env
        assert _hostify_env(env, {}) == env

    def test_none_env_passes_through(self):
        assert _hostify_env(None, self._VOLUMES) is None

    def test_target_prefix_without_slash_is_not_matched(self):
        # ``/cache/huggingface-old`` must NOT be treated as under ``/cache/huggingface``.
        out = _hostify_env({"X": "/cache/huggingface-old"}, self._VOLUMES)
        assert out == {"X": "/cache/huggingface-old"}


class TestLocalExecutorHostifiesEnv:
    """LocalExecutor emits host-path exports for container-path env values."""

    _VOLUMES = {"/home/ubuntu/.cache/huggingface": "/cache/huggingface"}
    _ENV = {
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_CACHE": "/cache/huggingface/hub",
        "FOO": "bar",
    }

    def test_exec_serve_script_exports_host_paths(self):
        script = _local().generate_exec_serve_script(
            container_name="foo_solo",
            serve_command="vllm serve x",
            env=self._ENV,
            volumes=self._VOLUMES,
        )
        # Exports are shlex-quoted; these plain paths need no quoting.
        assert "export HF_HOME=/home/ubuntu/.cache/huggingface" in script
        assert "export HF_HUB_CACHE=/home/ubuntu/.cache/huggingface/hub" in script
        assert "export FOO=bar" in script
        # The raw container path must NOT leak into the exports.
        assert "export HF_HOME=/cache/huggingface\n" not in script

    def test_exec_serve_script_without_volumes_keeps_container_paths(self):
        script = _local().generate_exec_serve_script(
            container_name="foo_solo",
            serve_command="vllm serve x",
            env=self._ENV,
        )
        assert "export HF_HOME=/cache/huggingface" in script

    def test_node_script_hostifies_env(self):
        script = _local().generate_node_script(
            image="",
            container_name="sparkrun_abc_node_0",
            serve_command="cmd rank 0",
            env=self._ENV,
            volumes=self._VOLUMES,
        )
        assert "export HF_HOME=/home/ubuntu/.cache/huggingface" in script
        assert "export HF_HUB_CACHE=/home/ubuntu/.cache/huggingface/hub" in script


class TestDockerExecServeUnchangedByVolumes:
    """DockerExecutor's exec-serve output ignores volumes (no regression)."""

    def test_volumes_do_not_change_docker_exec_serve(self):
        ex = DockerExecutor(ExecutorConfig())
        env = {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "FOO": "bar",
        }
        without = ex.generate_exec_serve_script(
            container_name="foo_solo",
            serve_command="vllm serve x",
            env=env,
        )
        with_vols = ex.generate_exec_serve_script(
            container_name="foo_solo",
            serve_command="vllm serve x",
            env=env,
            volumes={"/home/ubuntu/.cache/huggingface": "/cache/huggingface"},
        )
        assert without == with_vols


# ---------------------------------------------------------------------------
# Launcher-side: recipe ``executor`` field flows through
# ---------------------------------------------------------------------------


class TestLauncherFactoryDispatch:
    """The launcher honours the recipe-level ``executor`` selector.

    Tests the chain build in isolation (without spinning up a full
    runtime) because the launcher's full path requires SSH + real
    hosts.  The chain build is the only piece that picks the executor.
    """

    def test_recipe_executor_local_selects_local_executor(self):
        from sparkrun.core.recipe import Recipe
        from scitrera_app_framework.api import Variables, EnvPlacement

        r = Recipe.from_dict(
            {
                "model": "test/model",
                "runtime": "vllm",
                "container": "img:tag",
                "executor": "local",
                "executor_config": {"log_dir": "/tmp/x", "pid_dir": "/tmp/y"},
            }
        )
        # Reproduce the launcher's chain-build (without the rootless /
        # auto_user adjustment layers, which don't affect the selector).
        recipe_cfg = dict(r.executor_config)
        if r.executor:
            recipe_cfg["executor"] = r.executor
        chain = Variables(sources=({}, recipe_cfg, {}, EXECUTOR_DEFAULTS), env_placement=EnvPlacement.IGNORED)
        cfg = ExecutorConfig.from_chain(chain)
        assert cfg.executor_type == "local"
        assert cfg.log_dir == "/tmp/x"
        assert cfg.pid_dir == "/tmp/y"

    def test_recipe_without_executor_defaults_to_docker(self):
        from sparkrun.core.recipe import Recipe

        r = Recipe.from_dict({"model": "m", "runtime": "vllm", "container": "img"})
        assert r.executor == ""
        ex = resolve_executor(recipe=r)
        assert isinstance(ex, DockerExecutor)
