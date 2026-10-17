"""Regression tests: ``bench_args`` must be final before anything consumes it.

``_execute_benchmark`` merged the framework's recipe-derived args
(``prepare_benchmark_args``) and the runtime-resolved ``api_key`` *after* the
scheduled task list had already been built.  Three consumers snapshot
``bench_args`` before that point::

    tasks        = fw.build_task_list(bench_args, …)   # copies into task.run_args
    benchmark_id = derive_benchmark_id(…, bench_args, …)
    state        = BenchmarkRunState(base_args=bench_args, …)

The scheduler renders its command from ``task.run_args``, so on the scheduled
path — the default — none of those contributions ever reached a command.

Two things were silently dropped:

* ``served_model_name`` — issue #257's fix.  A recipe whose served name differs
  from its model id still asked the server for the model id (HTTP 404).
* the runtime-resolved ``api_key`` — an authenticated server would reject every
  request.

Unit tests of ``prepare_benchmark_args`` cannot see this: they exercise the
plugin in isolation and pass whether or not the flow ever uses the result.
These assert on what the flow actually hands downstream. Authentication is now
injected only into command construction; test_benchmark_auth_and_state covers it.
"""

from __future__ import annotations

from unittest import mock

import pytest
import yaml
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def _no_ssh(monkeypatch):
    """Stub the occupancy sweep — the fixture host is unroutable, and every
    ``api.status`` call site would otherwise pay a full connect timeout."""
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy

    def _stub(hosts, **kwargs):
        return ClusterStatus(
            hosts=tuple(HostOccupancy(host=h, workloads=(), used_slots=0, free_slots=1) for h in hosts),
            executor="docker",
        )

    monkeypatch.setattr("sparkrun.api._status.status", _stub)
    monkeypatch.setattr("sparkrun.api.status", _stub)


@pytest.fixture
def bench_env(tmp_path, monkeypatch, v, _no_ssh):
    """A single-host cluster plus a recipe with a served name and an api key.

    ``served_model_name`` differs from ``model`` deliberately — that difference
    is the whole point of issue #257, and a recipe where they match cannot
    detect the regression.
    """
    import sparkrun.core.config
    from sparkrun.core.cluster_manager import ClusterManager

    config_root = tmp_path / "config"
    config_root.mkdir()
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)
    ClusterManager(config_root).create("solo1", ["10.0.9.9"])

    recipe_file = tmp_path / "args-ordering.yaml"
    recipe_file.write_text(
        yaml.safe_dump(
            {
                "sparkrun_version": "2",
                "name": "args-ordering",
                "model": "Qwen/Qwen3-1.7B",
                "runtime": "sglang",
                "container": "scitrera/dgx-spark-sglang:latest",
                "defaults": {
                    "port": 30000,
                    "tensor_parallel": 1,
                    "served_model_name": "served-alias",
                    "api_key": "sk-from-recipe",
                },
                "metadata": {"model_params": 1700000000, "model_dtype": "float16"},
            }
        )
    )
    return str(recipe_file)


def _spy_task_args(runner, recipe_path, extra_argv=()):
    """Return the ``base_args`` the flow hands ``build_task_list``."""
    from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework as _FW
    from sparkrun.cli import main

    seen: dict = {}
    real = _FW.build_task_list

    def _spy(self, base_args, schedule):
        seen["base_args"] = dict(base_args)
        return real(self, base_args, schedule)

    argv = ["benchmark", recipe_path, "--cluster", "solo1", "--dry-run", "--fresh", "--skip-run"]
    with mock.patch.object(_FW, "build_task_list", _spy):
        result = runner.invoke(main, argv + list(extra_argv))
    return result, seen.get("base_args")


@pytest.mark.usefixtures("benchmark_version", "external_control")
def test_served_model_name_reaches_the_scheduled_tasks(runner, bench_env):
    """Issue #257's fix, which was inert on this path."""
    result, base_args = _spy_task_args(runner, bench_env)

    assert result.exit_code == 0, result.output
    assert base_args is not None, "build_task_list was never called"
    assert base_args.get("served_model_name") == "served-alias"


@pytest.mark.usefixtures("benchmark_version", "external_control")
def test_api_key_stays_out_of_scheduled_task_definitions(runner, bench_env):
    """Authentication is injected into execution, not persisted task inputs."""
    result, base_args = _spy_task_args(runner, bench_env)

    assert result.exit_code == 0, result.output
    assert "api_key" not in base_args


@pytest.mark.usefixtures("benchmark_version", "external_control")
def test_the_rendered_command_carries_them(runner, bench_env):
    """End of the chain: what the scheduler would actually exec."""
    from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework

    result, base_args = _spy_task_args(runner, bench_env)
    assert result.exit_code == 0, result.output

    fw = LlamaBenchyFramework()
    task = fw.build_task_list(base_args, None)[0]
    cmd = fw.build_benchmark_command("http://h:30000/v1", "Qwen/Qwen3-1.7B", task.run_args)

    assert "--served-model-name" in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == "served-alias"


@pytest.mark.usefixtures("benchmark_version", "external_control")
def test_user_bench_args_still_win(runner, bench_env):
    """The merge is ``setdefault``; moving it earlier must not change that."""
    result, base_args = _spy_task_args(runner, bench_env, ["-b", "served_model_name=explicit"])

    assert result.exit_code == 0, result.output
    assert base_args.get("served_model_name") == "explicit"


@pytest.mark.usefixtures("benchmark_version", "external_control")
def test_return_token_ids_suppression_reaches_the_scheduled_tasks(runner, bench_env):
    """The sglang workaround travels the same road, and hit the same pothole.

    Its unit tests passed while it was completely inert in practice — the
    framework produced the arg and the flow discarded it.  The fixture recipe is
    an sglang one, so this asserts the whole chain rather than the plugin.
    """
    result, base_args = _spy_task_args(runner, bench_env)

    assert result.exit_code == 0, result.output
    assert base_args.get("extra_body") == "return_token_ids=false"


@pytest.mark.usefixtures("benchmark_version", "external_control")
def test_user_can_re_enable_token_ids_end_to_end(runner, bench_env):
    result, base_args = _spy_task_args(runner, bench_env, ["-b", "extra_body=return_token_ids=true"])

    assert result.exit_code == 0, result.output
    assert base_args.get("extra_body") == "return_token_ids=true"
