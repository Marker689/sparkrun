"""One hook-trust verdict per launch, decided before anything runs.

Covers the contract between ``api.run`` (refuses untrusted hooks, dry runs
report instead), ``launch_inference`` (decides before ``before_start`` can
evict anything), the post-launch hooks (reuse the launch's verdict) and the
CLI frontends (``run`` / ``proxy load`` / ``benchmark`` ask on a TTY, refuse
otherwise, naming ``--trust``).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml
from click.testing import CliRunner

from sparkrun import api
from sparkrun.core.recipe import Recipe


def _untrusted_recipe(**extra) -> Recipe:
    """A recipe from a registry the local registries.yaml does not trust."""
    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m", **extra})
    recipe.source_registry = "third-party"
    recipe.source_registry_url = "https://example.invalid/third-party.git"
    return recipe


class _FakeRuntime:
    runtime_name = "vllm"
    executor = None

    def world_size(self, parallelism, recipe=None, cluster=None):
        return 1


def _run(options):
    """``api.run`` with a hermetic launcher; returns (launch kwargs, evict mock)."""
    calls = []

    def _fake_launch(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            rc=0,
            cluster_id=kwargs["cluster_id_override"],
            host_list=kwargs["host_list"],
            is_solo=kwargs["is_solo"],
            runtime=_FakeRuntime(),
            recipe=kwargs["recipe"],
            overrides={},
            container_image="test:latest",
            effective_cache_dir="/tmp",
            serve_port=8000,
            config=None,
            recipe_ref=None,
            serve_command="",
            runtime_info={},
            timeline=None,
            recipe_trusted=kwargs.get("recipe_trusted"),
        )

    from sparkrun.core.cluster_status import empty_status

    with (
        patch("sparkrun.api.status", side_effect=lambda hosts, **kw: empty_status(hosts)),
        patch("sparkrun.core.launcher.launch_inference", side_effect=_fake_launch),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=_FakeRuntime()),
        patch("sparkrun.core.hardware_probe.probe_hosts", return_value={}),
        patch("sparkrun.api._run._evict_superseded_deployments", return_value=([], set())) as evict,
    ):
        api.run(options)
    return calls, evict


# --------------------------------------------------------------------------
# api.run: the gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hooks",
    [
        {"mods": ["@third-party/patch-kernels"]},
        {"pre_exec": ["echo pre"]},
        {"post_exec": ["echo post"]},
        {"post_commands": ["echo local"]},
    ],
)
def test_untrusted_hooks_are_refused_before_preparation(hooks):
    """``mods`` become pre_exec entries mid-launch, so they are gated with the
    hooks — and the refusal comes before preparation, launch or eviction."""
    recipe = _untrusted_recipe(**hooks)
    forbidden = Mock(side_effect=AssertionError("untrusted recipe reached preparation"))
    with patch("sparkrun.core.execution.resolve_recipe_execution", forbidden):
        with pytest.raises(api.TrustRejected, match=r"Recipe hooks \(%s\)" % next(iter(hooks))):
            _run(api.RunOptions(recipe=recipe, hosts=("h1",), follow=False))
    forbidden.assert_not_called()


def test_untrusted_recipe_without_hooks_is_not_refused():
    calls, _ = _run(api.RunOptions(recipe=_untrusted_recipe(), hosts=("h1",), follow=False))
    assert calls[0]["recipe_trusted"] is False


def test_explicit_trust_is_the_verdict_launch_inference_receives():
    """api.run decides once (with its sctx) and hands the verdict down."""
    calls, evict = _run(api.RunOptions(recipe=_untrusted_recipe(mods=["m"]), hosts=("h1",), follow=False, trust=True))
    assert calls[0]["recipe_trusted"] is True


def test_trusted_verdict_resolved_once_per_run():
    import sparkrun.core.launcher as launcher

    recipe = _untrusted_recipe(pre_exec=["echo pre"])
    with patch.object(launcher, "resolve_recipe_trust", wraps=launcher.resolve_recipe_trust) as resolve:
        calls, _ = _run(api.RunOptions(recipe=recipe, hosts=("h1",), follow=False, trust=True))
    assert resolve.call_count == 1
    assert calls[0]["recipe_trusted"] is True


def test_dry_run_of_untrusted_hooks_proceeds_and_reports(caplog):
    """A dry run executes nothing, so it is not refused; it says what a real
    run would need authorized and launches with an untrusted verdict."""
    recipe = _untrusted_recipe(pre_exec=["echo pre"], mods=["m"])
    with caplog.at_level(logging.WARNING, logger="sparkrun.api._run"):
        calls, evict = _run(api.RunOptions(recipe=recipe, hosts=("h1",), dry_run=True))
    assert calls[0]["dry_run"] is True
    assert calls[0]["recipe_trusted"] is False
    assert calls[0]["before_start"] is None
    assert "pre_exec, mods would require authorization (--trust)" in caplog.text
    evict.assert_not_called()


def test_untrusted_hooks_lists_only_what_needs_authorization():
    recipe = _untrusted_recipe(pre_exec=["echo a", {"copy": "/src", "dest": "/dst"}], mods=["fix"], post_commands="echo b")
    assert api.untrusted_hooks(recipe) == {
        "pre_exec": ("echo a", "copy /src -> /dst"),
        "mods": ("fix",),
        "post_commands": ("echo b",),
    }
    assert api.untrusted_hooks(recipe, trust=True) == {}
    local = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m", "pre_exec": ["echo a"]})
    assert api.untrusted_hooks(local) == {}


# --------------------------------------------------------------------------
# launch_inference: direct callers decide before before_start
# --------------------------------------------------------------------------


def test_launch_inference_decides_untrusted_hooks_before_eviction(monkeypatch, tmp_path, v):
    """Without a verdict, launch_inference confirms up front: on a refusal the
    ``before_start`` replacement hook never fires, so the serving deployment
    survives a declined launch."""
    from sparkrun.core.bootstrap import get_runtime
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.launcher import launch_inference

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    before_start = Mock()
    with pytest.raises(RuntimeError, match="--trust"):
        launch_inference(
            recipe=_untrusted_recipe(mods=["m"]),
            runtime=get_runtime("vllm-distributed", v),
            host_list=["h1"],
            overrides={},
            config=SparkrunConfig(),
            v=v,
            is_solo=True,
            dry_run=False,
            sync_tuning=False,
            before_start=before_start,
        )
    before_start.assert_not_called()


# --------------------------------------------------------------------------
# post-launch hooks reuse the launch's verdict
# --------------------------------------------------------------------------


def test_post_launch_hooks_reuse_the_launch_verdict(monkeypatch):
    from sparkrun.core import launcher

    recipe = _untrusted_recipe(post_commands=["echo post"])
    result = SimpleNamespace(
        recipe=recipe,
        runtime=SimpleNamespace(runtime_name="vllm"),
        host_list=["h1"],
        overrides={},
        config=SimpleNamespace(ssh_user=None),
        cluster_id="sparkrun_x",
        hook_launch_context=None,
        runtime_cache=None,
        recipe_trusted=True,
    )
    readiness = SimpleNamespace(ready=True, head_host="h1", head_ip="10.0.0.1", port=8000, container="c")
    seen = {}
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda config: {})
    monkeypatch.setattr("sparkrun.orchestration.hooks.run_post_commands", lambda *a, **kw: seen.setdefault("trust", kw["trust"]))
    monkeypatch.setattr(launcher, "resolve_recipe_trust", Mock(side_effect=AssertionError("verdict re-resolved")))
    launcher.complete_post_launch(result, "/tmp/cache", trust=False, readiness=readiness)
    assert seen["trust"] is True


def test_dry_run_hook_confirmation_reports_instead_of_refusing(monkeypatch, caplog):
    from sparkrun.orchestration.hooks import _confirm_hook_execution, run_post_commands

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with caplog.at_level(logging.WARNING, logger="sparkrun.orchestration.hooks"):
        _confirm_hook_execution("pre_exec", ["echo a"], False, dry_run=True)
        run_post_commands(["echo b"], {}, dry_run=True, trust=False)
    assert "pre_exec would require authorization (--trust)" in caplog.text
    assert "post_commands would require authorization (--trust)" in caplog.text
    with pytest.raises(RuntimeError, match="--trust"):
        _confirm_hook_execution("pre_exec", ["echo a"], False)


# --------------------------------------------------------------------------
# CLI prompt helper
# --------------------------------------------------------------------------


def test_cli_trust_passes_through_and_trusted_sources_are_not_asked(monkeypatch):
    from sparkrun.cli._common import authorize_recipe_hooks

    confirm = Mock(side_effect=AssertionError("asked"))
    monkeypatch.setattr("click.confirm", confirm)
    assert authorize_recipe_hooks(_untrusted_recipe(pre_exec=["x"]), trust=True, dry_run=False) is True
    local = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/m", "pre_exec": ["x"]})
    assert authorize_recipe_hooks(local, trust=False, dry_run=False) is False


def test_cli_non_interactive_refusal_names_trust(monkeypatch):
    import click

    from sparkrun.cli._common import authorize_recipe_hooks

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(click.ClickException, match="--trust") as error:
        authorize_recipe_hooks(_untrusted_recipe(mods=["m"]), trust=False, dry_run=False)
    assert "mods" in error.value.message


@pytest.mark.parametrize(("answer", "expected"), [(True, True), (False, None)])
def test_cli_interactive_prompt_defaults_to_no(monkeypatch, capsys, answer, expected):
    import click

    from sparkrun.cli._common import authorize_recipe_hooks

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    confirm = Mock(return_value=answer)
    monkeypatch.setattr("click.confirm", confirm)
    recipe = _untrusted_recipe(pre_exec=["pip install something\nsecond line"], mods=["patch-a"])
    if expected is None:
        with pytest.raises(click.ClickException, match="not authorized"):
            authorize_recipe_hooks(recipe, trust=False, dry_run=False)
    else:
        assert authorize_recipe_hooks(recipe, trust=False, dry_run=False) is expected
    assert confirm.call_args.kwargs["default"] is False
    shown = capsys.readouterr().err
    assert "- pip install something ..." in shown
    assert "mods: patch-a" in shown


def test_cli_dry_run_reports_without_asking(monkeypatch, capsys):
    from sparkrun.cli._common import authorize_recipe_hooks

    monkeypatch.setattr("click.confirm", Mock(side_effect=AssertionError("asked")))
    assert authorize_recipe_hooks(_untrusted_recipe(post_exec=["x"]), trust=False, dry_run=True) is False
    assert "post_exec" in capsys.readouterr().err


def test_run_trust_flag_is_visible_and_covers_mods():
    from sparkrun.cli import main

    result = CliRunner().invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--trust" in result.output
    assert "mods" in result.output


def test_benchmark_recipe_trust_decision_uses_the_shared_prompt(monkeypatch):
    from sparkrun.api._benchmark_models import BenchmarkDecision
    from sparkrun.cli._benchmark import _cli_decision

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert _cli_decision(BenchmarkDecision("recipe_trust", "summary", False)) is False
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    confirm = Mock(return_value=True)
    monkeypatch.setattr("click.confirm", confirm)
    assert _cli_decision(BenchmarkDecision("recipe_trust", "summary", False)) is True
    assert confirm.call_args.kwargs["default"] is False


# --------------------------------------------------------------------------
# proxy load
# --------------------------------------------------------------------------


def _public_result(**fields):
    """A RunResult as a native run handler returns it: no launcher handle."""
    values = dict(
        workload_id="sparkrun_x",
        host_list=("10.0.4.30",),
        placement=None,
        scheduler="greedy",
        runtime="sglang",
        executor="docker",
        started_at=0.0,
        dry_run=False,
        is_solo=True,
        serve_port=30000,
        launch_result=None,
    )
    values.update(fields)
    return api.RunResult(**values)


@pytest.fixture
def proxy_load_env(tmp_path, monkeypatch, idle_occupancy):
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import DEFAULT_CONFIG_DIR
    from sparkrun.orchestration.ssh import RemoteResult

    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        lambda hosts, *args, **kwargs: [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts],
    )
    ClusterManager(DEFAULT_CONFIG_DIR).create("lab", ["10.0.4.30"], scheduler="greedy")
    path = tmp_path / "model.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "sparkrun_version": "2",
                "model": "test/model",
                "runtime": "sglang",
                "container": "test/image:latest",
                "defaults": {"port": 30000, "tensor_parallel": 1},
                "metadata": {"model_params": 1000000, "model_dtype": "float16"},
            }
        )
    )
    monkeypatch.setattr(api.proxy, "status", Mock(return_value=SimpleNamespace(running=False, autodiscover_running=False)))
    return path


def test_proxy_load_handles_a_result_without_launch_handle(proxy_load_env, monkeypatch):
    """A run handler may return only the public result; proxy load reported it
    as an AttributeError instead of an unconfirmed readiness."""
    from sparkrun.cli import main

    monkeypatch.setattr(api, "run", Mock(return_value=_public_result()))
    out = CliRunner().invoke(main, ["proxy", "load", str(proxy_load_env), "--cluster", "lab"])
    assert out.exit_code == 1
    assert "no supported readiness handle" in out.output
    assert not isinstance(out.exception, AttributeError)


def test_proxy_load_reports_failure_from_the_public_result(proxy_load_env, monkeypatch):
    from sparkrun.cli import main

    monkeypatch.setattr(api, "run", Mock(return_value=_public_result(rc=3)))
    out = CliRunner().invoke(main, ["proxy", "load", str(proxy_load_env), "--cluster", "lab"])
    assert out.exit_code == 1
    assert "exit code 3" in out.output


def test_proxy_load_forwards_trust(proxy_load_env, monkeypatch):
    from sparkrun.cli import main

    run = Mock(return_value=_public_result(dry_run=True))
    monkeypatch.setattr(api, "run", run)
    out = CliRunner().invoke(main, ["proxy", "load", str(proxy_load_env), "--cluster", "lab", "--trust", "--dry-run"])
    assert out.exit_code == 0, out.output
    assert run.call_args.args[0].trust is True
