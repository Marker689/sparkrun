"""Regression tests: ``sparkrun benchmark`` honours the recipe-override flags.

``sparkrun run <recipe> --tp 4`` placed the launch on four nodes; ``sparkrun
benchmark <recipe> --tp 4`` resolved to **solo**.  The CLI flattened every
recipe-override flag into ``BenchmarkOptions.overrides`` correctly, but
``_execute_benchmark`` then rebuilt the overrides dict from scratch, forwarding
only ``image`` and ``port`` and passing ``tensor_parallel=None`` (and friends)
explicitly.  Placement reads ``tensor_parallel`` off the recipe's config chain,
so with the override dropped the recipe's own value stood — one rank, one host.

The whole dict is now forwarded, so ``--tp/--pp/--dp``, ``--gpu-mem``,
``--max-model-len`` and ``-o key=value`` reach the benchmark's launch on the
same terms they reach ``run``'s.
"""

from __future__ import annotations

from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.cli import main

_RECIPE_NAME = "test-benchmark-overrides"

#: Deliberately ``tensor_parallel: 1`` — the flag has to be what widens it.
_RECIPE_DATA = {
    "sparkrun_version": "2",
    "name": "Benchmark override regression recipe",
    "description": "single-rank recipe widened via --tp",
    "model": "Qwen/Qwen3-1.7B",
    "runtime": "sglang",
    "mode": "cluster",
    "min_nodes": 1,
    "max_nodes": 8,
    "container": "scitrera/dgx-spark-sglang:latest",
    "defaults": {
        "port": 30000,
        "host": "0.0.0.0",
        "tensor_parallel": 1,
    },
    "metadata": {
        "model_params": 1700000000,
        "model_dtype": "float16",
    },
}

_HOSTS = ["10.0.4.30", "10.0.4.31", "10.0.4.32", "10.0.4.33"]


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def idle_cluster_status(monkeypatch):
    """Point every ``api.status`` call site at an all-idle 4-host snapshot.

    Without this the occupancy sweep SSHes to the (unroutable) fixture hosts
    and each test pays the full connect timeout.
    """
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy

    def _stub(hosts, **kwargs):
        return ClusterStatus(
            hosts=tuple(HostOccupancy(host=h, workloads=(), used_slots=0, free_slots=1) for h in hosts),
            executor="docker",
        )

    monkeypatch.setattr("sparkrun.api._status.status", _stub)
    monkeypatch.setattr("sparkrun.api.status", _stub)


@pytest.fixture
def bench_env(tmp_path, monkeypatch, v, idle_cluster_status):
    """A 4-host cluster plus a discoverable single-rank recipe."""
    import sparkrun.core.config
    from sparkrun.core.cluster_manager import ClusterManager

    config_root = tmp_path / "config"
    config_root.mkdir()
    monkeypatch.setattr(sparkrun.core.config, "DEFAULT_CONFIG_DIR", config_root)

    ClusterManager(config_root).create("wopr", list(_HOSTS))

    recipe_dir = tmp_path / "recipes"
    recipe_dir.mkdir()
    recipe_file = recipe_dir / ("%s.yaml" % _RECIPE_NAME)
    recipe_file.write_text(yaml.safe_dump(_RECIPE_DATA))

    import sparkrun.core.recipe

    original_discover = sparkrun.core.recipe.discover_cwd_recipes
    monkeypatch.setattr(
        sparkrun.core.recipe,
        "discover_cwd_recipes",
        lambda directory=None: [recipe_file] + original_discover(directory),
    )
    return config_root


def _invoke(runner, extra_argv):
    """Run ``sparkrun benchmark`` with a spied ``api.plan``.

    Returns ``(click_result, captured)`` where *captured* holds the
    ``RunOptions`` and ``RunPlan`` the benchmark flow produced.
    """
    import sparkrun.api as api

    captured: dict = {}
    real_plan = api.plan

    def _spy(options, *, sctx=None):
        plan = real_plan(options, sctx=sctx)
        captured["options"] = options
        captured["plan"] = plan
        return plan

    with mock.patch.object(api, "plan", _spy):
        result = runner.invoke(main, ["benchmark", _RECIPE_NAME, "--cluster", "wopr", "--dry-run"] + extra_argv)
    return result, captured


def test_tp_flag_reaches_the_launch_overrides(runner, bench_env):
    """``--tp 4`` lands in ``RunOptions.overrides`` — the reported bug."""
    result, captured = _invoke(runner, ["--tp", "4"])

    assert result.exit_code == 0, result.output
    assert captured, "the benchmark never planned a launch"
    assert captured["options"].overrides.get("tensor_parallel") == 4


def test_tp_flag_places_on_four_nodes(runner, bench_env):
    """The end-user symptom: ``--tp 4`` must not resolve to solo."""
    result, captured = _invoke(runner, ["--tp", "4"])

    assert result.exit_code == 0, result.output
    plan = captured["plan"]
    assert plan.is_solo is False
    assert len(plan.host_list) == 4
    assert "Placement: 4 hosts from" in result.output


def test_without_the_flag_the_recipe_value_still_wins(runner, bench_env):
    """No ``--tp`` → the recipe's own ``tensor_parallel: 1`` → solo."""
    result, captured = _invoke(runner, [])

    assert result.exit_code == 0, result.output
    assert "tensor_parallel" not in captured["options"].overrides
    assert captured["plan"].is_solo is True


def test_every_recipe_override_flag_is_forwarded(runner, bench_env):
    """``--pp``/``--dp``/``--gpu-mem``/``--max-model-len``/``-o`` share the defect.

    ``--gpu-mem`` is the one flag whose CLI spelling differs from its override
    key: ``apply_recipe_overrides`` maps ``gpu_mem`` onto
    ``gpu_memory_utilization``, which only happens because the dict is fed
    through it rather than merged blindly.
    """
    result, captured = _invoke(
        runner,
        ["--tp", "2", "--pp", "2", "--gpu-mem", "0.85", "--max-model-len", "4096", "-o", "attention_backend=triton"],
    )

    assert result.exit_code == 0, result.output
    overrides = captured["options"].overrides
    assert overrides.get("tensor_parallel") == 2
    assert overrides.get("pipeline_parallel") == 2
    assert overrides.get("gpu_memory_utilization") == 0.85
    assert overrides.get("max_model_len") == 4096
    assert overrides.get("attention_backend") == "triton"
    # ``gpu_mem`` is a CLI spelling, not an override key — it must not survive
    # alongside the canonical one.
    assert "gpu_mem" not in overrides


def test_dash_o_values_are_coerced(runner, bench_env):
    """``-o max_model_len=4096`` yields an int, as it does on the ``run`` path."""
    result, captured = _invoke(runner, ["-o", "max_model_len=4096", "-o", "enforce_eager=true"])

    assert result.exit_code == 0, result.output
    overrides = captured["options"].overrides
    assert overrides.get("max_model_len") == 4096
    assert overrides.get("enforce_eager") is True


def test_malformed_dash_o_is_rejected(runner, bench_env):
    """A bare ``-o foo`` is an error, not a silently ignored no-op."""
    result = runner.invoke(main, ["benchmark", _RECIPE_NAME, "--cluster", "wopr", "--dry-run", "-o", "foo"])

    assert result.exit_code != 0
    assert "--option must be key=value" in result.output


def test_dash_o_executor_keys_configure_the_executor(runner, bench_env):
    """``-o entrypoint=''`` is what the ENTRYPOINT preflight tells users to
    pass; on the benchmark path it used to land in the serve overrides, be
    reported unmapped and dropped."""
    result, captured = _invoke(runner, ["-o", "entrypoint=", "-o", "shm_size=8g", "-o", "max_model_len=4096"])

    assert result.exit_code == 0, result.output
    options = captured["options"]
    assert options.overrides.get("max_model_len") == 4096
    assert "entrypoint" not in options.overrides and "shm_size" not in options.overrides
    assert options.executor_overrides()["entrypoint"] == ""
    assert options.executor_overrides()["shm_size"] == "8g"


def test_untrusted_hooks_are_decided_before_the_benchmark_launches(runner, bench_env):
    """Non-interactive: the ``recipe_trust`` decision declines, and the
    benchmark fails naming ``--trust`` before ``api.run`` is reached."""
    import sparkrun.api as api

    run = mock.Mock(side_effect=AssertionError("launched"))
    with (
        mock.patch.object(api, "untrusted_hooks", return_value={"mods": ("patch-a",)}),
        mock.patch.object(api, "run", run),
        # A real (non-dry) plan probes its hosts; keep it off the network.
        mock.patch("sparkrun.core.hardware_probe.probe_hosts", return_value={}),
    ):
        result = runner.invoke(main, ["benchmark", _RECIPE_NAME, "--cluster", "wopr"])
    assert result.exit_code == 1
    assert "--trust" in result.output and "mods" in result.output
    run.assert_not_called()
