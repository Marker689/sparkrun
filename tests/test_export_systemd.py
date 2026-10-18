"""Test systemd export correctly outputs YAML."""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from click.testing import CliRunner
from sparkrun.cli import main


def test_export_systemd_preserves_single_quotes(monkeypatch, tmp_path):
    """Test systemd export does not escape single quotes in YAML."""
    runner = CliRunner()

    # Mock detect_remote_sparkrun to avoid SSH
    monkeypatch.setattr(
        "sparkrun.orchestration.systemd_service.detect_remote_command",
        lambda host, ssh_kwargs, dry_run=False: ("/usr/local/bin/sparkrun", "/home/user"),
    )

    # A recipe whose env values are quoted strings. This used to name
    # `@official/qwen3-coder-next-int4-autoround-vllm`, which made the test
    # depend on git-cloning the official registry *and* on that remote recipe
    # keeping this exact env var. The subject here is the export's YAML
    # quoting, so a local file (find_recipe resolves a direct path) tests it
    # without either dependency.
    recipe = tmp_path / "quoted-env-vllm.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "name": "quoted-env-vllm",
                "model": "Qwen/Qwen3-1.7B",
                "runtime": "vllm",
                "container": "vllm/vllm-openai:latest",
                "env": {"VLLM_MARLIN_USE_ATOMIC_ADD": "1"},
            }
        )
    )

    result = runner.invoke(main, ["export", "systemd", str(recipe), "--hosts", "127.0.0.1"])

    assert result.exit_code == 0, result.output
    # Look for the env var VLLM_MARLIN_USE_ATOMIC_ADD which should have single quotes
    # Before the fix, it would be VLLM_MARLIN_USE_ATOMIC_ADD: '\''1'\''
    # After the fix, it should be VLLM_MARLIN_USE_ATOMIC_ADD: '1'
    assert "VLLM_MARLIN_USE_ATOMIC_ADD: '1'" in result.output
    assert "VLLM_MARLIN_USE_ATOMIC_ADD: '\\''1'\\''" not in result.output

    # Another test: verify the bash script structure uses << 'SPARKRUN_RECIPE_EOF'
    assert "<< 'SPARKRUN_RECIPE_EOF'" in result.output


def test_render_only_missing_remote_command_cannot_install(monkeypatch, tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"model": "test/model", "runtime": "vllm", "container": "test/image"}), encoding="utf-8")
    install = Mock(side_effect=AssertionError("render-only export cannot install"))
    monkeypatch.setattr("sparkrun.orchestration.systemd_service.detect_remote_command", lambda *a, **kw: (None, None))
    monkeypatch.setattr("sparkrun.orchestration.systemd_service.bootstrap_remote_command", install)
    result = CliRunner().invoke(main, ["export", "systemd", str(recipe), "--hosts", "example.invalid"])
    assert result.exit_code != 0
    assert "no installation was attempted" in result.output and "--install" in result.output
    install.assert_not_called()


@pytest.mark.skipif(os.name == "nt", reason="systemd bootstrap targets Linux bash hosts")
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("alternate_profile", [False, True])
def test_remote_bootstrap_is_fresh_only_and_preserves_application_source(tmp_path, monkeypatch, existing, alternate_profile):
    from sparkrun.orchestration.systemd_service import bootstrap_remote_command
    from sparkrun.core.application_profile import UpdateSource
    from test_application_profiles import alternate

    package, requirement = "sparkrun", "sparkrun"
    if alternate_profile:
        package = "acme-tool"
        requirement = "Acme_Tool @ git+https://example.invalid/repo.git@release"
        alternate(package="Acme_Tool", update_sources={"stable": UpdateSource(requirement, strategy="git")})
    tools_root = tmp_path / "tools with spaces"
    tools_root.mkdir()
    if existing:
        (tools_root / package).mkdir()
    binary = tmp_path / ".local/bin/uv"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        '#!/bin/sh\ncase "$1 $2" in\n'
        '"tool dir") printf "%s\\n" "$TEST_TOOLS_ROOT";;\n'
        '"tool install") printf "%s\\n" "$@" > "$TEST_INSTALL_ARGS";;\n'
        "*) exit 42;;\nesac\n",
        encoding="utf-8",
    )
    binary.chmod(0o700)
    args = tmp_path / "install-args"

    def transport(host, script, **kwargs):
        result = subprocess.run(
            ["bash", "-c", script],
            env={**os.environ, "HOME": str(tmp_path), "TEST_TOOLS_ROOT": str(tools_root), "TEST_INSTALL_ARGS": str(args)},
            capture_output=True,
            text=True,
            timeout=10,
        )
        return SimpleNamespace(success=result.returncode == 0, stderr=result.stderr)

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", transport)
    result = bootstrap_remote_command("unused.invalid", {}, channel="stable")
    assert result.success is not existing and result.requirement == requirement
    if existing:
        assert not args.exists()
    else:
        assert args.read_text(encoding="utf-8").splitlines() == ["tool", "install", requirement]


def test_remote_bootstrap_dry_run_is_read_only(monkeypatch, tmp_path):
    from sparkrun.orchestration.systemd_service import bootstrap_remote_command

    run = Mock(side_effect=AssertionError("dry run cannot execute SSH"))
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", run)
    result = bootstrap_remote_command("unused.invalid", {}, channel="stable", dry_run=True)
    assert result.success and result.dry_run and result.requirement == "sparkrun"
    run.assert_not_called()

    # The CLI renders the would-install line from that result.
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"model": "test/model", "runtime": "vllm", "container": "test/image"}), encoding="utf-8")
    found = iter([(None, None), ("/home/user/.local/bin/sparkrun", "/home/user")])
    monkeypatch.setattr("sparkrun.orchestration.systemd_service.detect_remote_command", lambda *a, **kw: next(found))
    monkeypatch.setattr("sparkrun.cli._export._do_install", lambda *args: None)
    out = CliRunner().invoke(main, ["export", "systemd", str(recipe), "--hosts", "example.invalid", "--install", "--dry-run"])
    assert out.exit_code == 0, out.output
    assert "uv tool install sparkrun" in out.output and "only if its uv environment is absent" in out.output
    run.assert_not_called()


def _plain_recipe(tmp_path, **extra):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"model": "test/model", "runtime": "vllm", "container": "test/image", **extra}), encoding="utf-8")
    return recipe


def test_cli_bakes_executor_overrides_into_executor_config(monkeypatch, tmp_path):
    """``export systemd -o entrypoint=''`` reaches the service recipe's executor_config (#323)."""
    recipe = _plain_recipe(tmp_path, executor_config={"shm_size": "8g"})
    result = CliRunner().invoke(
        main, ["export", "systemd", str(recipe), "--hosts", "h1", "--dry-run", "-o", "entrypoint=", "-o", "port=9000"]
    )
    assert result.exit_code == 0, result.output
    baked = yaml.safe_load(result.output.split("--- Baked recipe ---\n", 1)[1].split("--- Cluster definition", 1)[0])
    assert baked["executor_config"] == {"shm_size": "8g", "entrypoint": ""}
    assert baked["defaults"]["port"] == 9000 and "entrypoint" not in baked["defaults"]


def test_cli_offline_needs_pin(tmp_path):
    result = CliRunner().invoke(main, ["export", "systemd", str(_plain_recipe(tmp_path)), "--hosts", "h1", "--dry-run", "--offline"])
    assert result.exit_code == 2 and "--pin" in result.output


def test_cli_pin_reports_what_it_pinned(monkeypatch, tmp_path):
    from sparkrun.containers import digest as digest_mod
    from sparkrun.models import revision as revision_mod

    monkeypatch.setattr(digest_mod, "resolve_registry_digest", lambda ref: "sha256:" + "a" * 64)
    monkeypatch.setattr(revision_mod, "resolve_hub_commit", lambda *a, **kw: "c" * 40)
    result = CliRunner().invoke(main, ["export", "systemd", str(_plain_recipe(tmp_path)), "--hosts", "h1", "--dry-run", "--pin"])
    assert result.exit_code == 0, result.output
    assert "Pinned container: test/image -> test/image@sha256:%s (registry)" % ("a" * 64) in result.output
    assert "model_revision: %s" % ("c" * 40) in result.output


def test_cli_pin_and_trust_do_not_apply_to_uninstall(tmp_path):
    result = CliRunner().invoke(main, ["export", "systemd", str(_plain_recipe(tmp_path)), "--hosts", "h1", "--uninstall", "--pin"])
    assert result.exit_code == 2 and "--uninstall" in result.output
