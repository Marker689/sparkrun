"""The advanced command is a thin, preparation-only API frontend."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

import sparkrun.api as api
from sparkrun.cli import main
from sparkrun.cli._common import HIDE_ADVANCED_OPTIONS


@pytest.fixture
def build_recipe(tmp_path):
    path = tmp_path / "build-test.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "recipe_version": "2",
                "model": "org/model",
                "runtime": "vllm",
                "container": "org/image:tag",
                "builder": "docker-pull",
            }
        )
    )
    return str(path)


def test_build_visibility_uses_existing_advanced_gate():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert ("\n  build " in result.output) is not HIDE_ADVANCED_OPTIONS
    help_result = CliRunner().invoke(main, ["build", "--help"])
    assert help_result.exit_code == 0
    assert "without starting inference" in help_result.output
    assert "--hosts" in help_result.output and "--executor" in help_result.output


@pytest.mark.parametrize("advanced,hidden", [("0", True), ("1", False)])
def test_build_visibility_reads_advanced_environment(advanced, hidden):
    script = "from sparkrun.cli._build import build; print(build.hidden)"
    result = subprocess.run(
        [sys.executable, "-c", script], env={**os.environ, "SPARKRUN_ADVANCED": advanced}, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == str(hidden)


def test_build_cli_reuses_plan_and_maps_options(build_recipe, monkeypatch):
    seen = []

    def plan(request, *, sctx):
        seen.append(("plan", request, sctx))
        return SimpleNamespace(recipe=request.recipe, host_list=request.hosts, issues=())

    def build(request, *, plan, sctx):
        assert request is seen[0][1] and sctx is seen[0][2]
        assert plan.host_list == request.hosts
        seen.append(("build", request, sctx))
        return api.BuildResult(
            recipe="build-test",
            host_list=request.hosts,
            executor="local",
            dry_run=True,
            environment_file="/venv/env.sh",
        )

    monkeypatch.setattr(api, "plan_build", plan)
    monkeypatch.setattr(api, "build", build)
    result = CliRunner().invoke(
        main,
        [
            "build",
            build_recipe,
            "--hosts",
            "localhost",
            "--executor",
            "local",
            "--rebuild",
            "--dry-run",
            "--no-sync-tuning",
            "--cache-dir",
            "/cache/hf",
            "--builder-option",
            "snapshot_driver=n580",
            "-o",
            "max_num_seqs=6",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["environment_file"] == "/venv/env.sh"
    assert "cluster_id" not in payload
    request = seen[0][1]
    assert request.rebuild is True and request.executor == "local"
    assert request.overrides["max_num_seqs"] == 6
    assert request.builder_context == {"snapshot_driver": "n580"}
    assert request.sync_tuning is False and request.cache_dir == "/cache/hf"
    assert [c[0] for c in seen] == ["plan", "build"]


def test_build_cli_reports_preparation_failure(build_recipe, monkeypatch):
    def fail(*a, **k):
        raise api.SparkrunError("model transfer failed on worker-2")

    monkeypatch.setattr(api, "plan_build", fail)
    result = CliRunner().invoke(main, ["build", build_recipe, "--hosts", "localhost"])
    assert result.exit_code != 0
    assert "model transfer failed on worker-2" in result.output
    assert "Build complete" not in result.output


def test_build_cli_preserves_default_cluster(build_recipe, monkeypatch):
    sctx = api.default_sctx()
    sctx.cluster_manager.create(name="native", hosts=["localhost"], executor="local", user="native-user")
    sctx.cluster_manager.set_default("native")
    seen = []

    def plan(request, *, sctx):
        seen.append(request)
        return SimpleNamespace(recipe=request.recipe, host_list=request.hosts, issues=())

    monkeypatch.setattr(api, "plan_build", plan)
    monkeypatch.setattr(api, "build", lambda request, **k: api.BuildResult("build-test", request.hosts, "local", True))
    result = CliRunner().invoke(main, ["build", build_recipe, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert seen[0].cluster == "native"
    assert seen[0].hosts == ("localhost",)


def test_build_cli_offline_keeps_recipe_load_local_and_lifts_executor_overrides(build_recipe, monkeypatch):
    from sparkrun.core import resolve
    from sparkrun.core.application_profile import env_name

    monkeypatch.setattr("sparkrun.models.hub.disable_hub_metadata", lambda: None)
    monkeypatch.setenv(env_name("NO_TELEMETRY"), "0")  # restored after _enter_offline_mode sets it
    loads, seen = [], []
    real = resolve.load_recipe

    def load(*args, **kwargs):
        loads.append(kwargs.get("offline"))
        return real(*args, **kwargs)

    def plan(request, **kwargs):
        seen.append(request)
        return SimpleNamespace(host_list=request.hosts, recipe=request.recipe, issues=())

    monkeypatch.setattr(resolve, "load_recipe", load)
    monkeypatch.setattr(api, "plan_build", plan)
    monkeypatch.setattr(api, "build", lambda request, **k: api.BuildResult("build-test", request.hosts, "docker", True))
    result = CliRunner().invoke(main, ["build", build_recipe, "--hosts", "localhost", "--offline", "--dry-run", "-o", "entrypoint="])
    assert result.exit_code == 0, result.output
    assert loads == [True]
    assert seen[0].offline is True
    assert seen[0].executor_config == {"entrypoint": ""} and "entrypoint" not in seen[0].overrides
