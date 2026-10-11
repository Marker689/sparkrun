"""Offline lifecycle acceptance for immutable, externally trusted Git plugins."""

from __future__ import annotations

from dataclasses import replace
import os
import subprocess

import pytest
from click.testing import CliRunner

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.plugin_compatibility import HostSnapshot, evaluate_plugin_compatibility
from sparkrun.core.plugin_manager import PluginManager
from sparkrun.core.plugin_manifest import PluginError, repository_identity
from sparkrun.core.plugin_sources import acquire_git


_REAL_RUN = subprocess.run


@pytest.fixture
def repository(tmp_path, monkeypatch):
    offline_run = subprocess.run

    def local_run(args, *a, **kwargs):
        if args[0] == "git" and "clone" in args:
            assert any(arg.startswith(tmp_path.as_uri() + "/") for arg in args)
            return _REAL_RUN(args, *a, **kwargs)
        if args[0] == "git" and "fetch" in args:
            origin = (
                _REAL_RUN(["git", "config", "remote.origin.url"], cwd=kwargs["cwd"], capture_output=True, check=True)
                .stdout.decode()
                .strip()
            )
            assert origin.startswith(tmp_path.as_uri() + "/")
            return _REAL_RUN(args, *a, **kwargs)
        return offline_run(args, *a, **kwargs)

    monkeypatch.setattr(subprocess, "run", local_run)
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        return (
            subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                env={
                    **os.environ,
                    "GIT_AUTHOR_NAME": "Tests",
                    "GIT_AUTHOR_EMAIL": "tests@example.invalid",
                    "GIT_COMMITTER_NAME": "Tests",
                    "GIT_COMMITTER_EMAIL": "tests@example.invalid",
                },
            )
            .stdout.decode()
            .strip()
        )

    git("init", "-b", "main")

    def commit(version="1.0", *, name="managedfixture", module=None, requires=">=0.4,<0.5", body="", feature=None):
        module = module or "sparkrun.plugins." + name
        source = "src/" + module.replace(".", "/")
        package = root / source
        package.mkdir(parents=True, exist_ok=True)
        (root / "plugin.toml").write_text(
            f'schema = 1\nname = "{name}"\nmodule = "{module}"\nsource = "{source}"\n'
            f'repository = "{root.as_uri()}"\nsparkrun = "{requires}"\n' + (f'feature = "{feature}"\n' if feature else "")
        )
        (root / "pyproject.toml").write_text(
            f'[project]\nname = "{name}-plugin"\nversion = "{version}"\nrequires-python = ">=3.12"\ndependencies = ["sparkrun{requires}"]\n'
        )
        (package / "__init__.py").write_text("SPARKRUN_PLUGIN_API_VERSION = 1\n" + f"from {module}.helper import value\n" + body)
        (package / "helper.py").write_text(f'value = "{version}"\n')
        (package / "data.txt").write_text(version)
        (root / "LICENSE").write_text("Test fixture license\n")
        git("add", ".")
        git("commit", "-m", version, "--allow-empty")
        return git("rev-parse", "HEAD")

    commit()
    return root, commit, git


@pytest.fixture
def manager(tmp_path):
    # Tests of host dependency compatibility use synthetic complete snapshots;
    # lifecycle tests should not depend on unrelated dev-environment drift.
    return PluginManager(SparkrunConfig(tmp_path / "config" / "config.yaml"), host=HostSnapshot("0.4.0", python_version="3.12.10"))


def state(manager):
    return manager.store.snapshot()[1]["managedfixture"]


def generations(manager):
    root = manager.store.directory / "managedfixture"
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


def test_git_lifecycle_and_resources(repository, manager):
    root, commit, _git = repository
    first = manager.add(str(root), trust=True)
    a = first["generation"]
    saved = SparkrunConfig(manager.config.config_path)
    assert saved.feature_override("core.external_plugins") is True
    payload = manager.store.generation_path("managedfixture", a)
    assert (payload / "payload/sparkrun/plugins/managedfixture/data.txt").read_text() == "1.0"
    assert (payload / "metadata/LICENSE").is_file()
    commit("1.1")
    b = manager.update("managedfixture")["generation"]
    assert state(manager)["previous_generations"] == [a]
    commit("1.2")
    c = manager.update(root.as_uri())["generation"]
    assert generations(manager) == sorted([b, c])
    assert state(manager)["previous_generations"] == [b]
    manager.rollback("managedfixture")
    assert state(manager)["active_generation"] == b
    assert state(manager)["metadata"]["version"] == "1.1"
    assert state(manager)["previous_generations"] == [c]
    # Advancing to the already-retained branch tip reuses it.
    assert manager.update("managedfixture")["generation"] == c
    assert generations(manager) == sorted([b, c])
    assert manager.update("managedfixture")["status"] == "unchanged"
    manager.remove("managedfixture")
    assert not generations(manager)
    assert SparkrunConfig(manager.config.config_path).feature_override("core.external_plugins") is True


@pytest.mark.parametrize("retain", [1, 3])
def test_retention_and_lowering_on_noop(repository, manager, retain):
    root, commit, _git = repository
    config = manager.config
    config.set("plugins.retain_generations", retain)
    config.save()
    manager.add(str(root), trust=True)
    for n in range(1, 4):
        commit(f"1.{n}")
        manager.update("managedfixture")
    assert len(generations(manager)) == retain
    if retain == 1:
        with pytest.raises(PluginError, match="No retained generation"):
            manager.rollback("managedfixture")
    else:
        cfg = SparkrunConfig(config.config_path)
        cfg.set("plugins.retain_generations", 1)
        cfg.save()
        manager.update("managedfixture", dry_run=True)
        assert len(generations(manager)) == 3
        manager.update("managedfixture")
        assert len(generations(manager)) == 1


@pytest.mark.parametrize("retain", [0, -1, True, "2", 2.0])
def test_invalid_retention_does_not_install(repository, manager, retain):
    manager.config.set("plugins.retain_generations", retain)
    manager.config.save()
    with pytest.raises(PluginError, match="positive integer"):
        manager.add(str(repository[0]), trust=True)
    assert not manager.store.ledger.exists()


def test_untrusted_and_dry_run_do_not_execute(repository, manager, tmp_path):
    root, commit, _git = repository
    marker = tmp_path / "executed"
    commit(body=f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    with pytest.raises(PluginError, match="Source trust is required"):
        manager.add(str(root))
    assert not marker.exists()
    result = manager.add(str(root), dry_run=True)
    assert result["status"] == "planned"
    assert not marker.exists()
    assert not manager.store.ledger.exists()


def test_legacy_paths_need_separate_consent(repository, manager, tmp_path):
    manager.config.set("plugins.paths", [str(tmp_path / "legacy")])
    manager.config.save()
    with pytest.raises(PluginError, match="existing plugins.paths"):
        manager.add(str(repository[0]), trust=True)
    assert not manager.store.ledger.exists()
    manager.add(str(repository[0]), trust=True, legacy_consent=lambda _: True)


def test_incompatible_and_failed_import_preserve_activation(repository, manager):
    root, commit, _git = repository
    manager.add(str(root), trust=True)
    before = manager.store.ledger.read_bytes()
    commit("2.0", requires=">=9")
    with pytest.raises(PluginError, match="incompatible"):
        manager.update("managedfixture")
    assert manager.store.ledger.read_bytes() == before
    commit("1.1", body='raise SystemExit("broken candidate")\n')
    with pytest.raises(PluginError, match="load validation failed"):
        manager.update("managedfixture")
    assert manager.store.ledger.read_bytes() == before
    assert len(generations(manager)) == 1


def test_pin_same_version_commit_and_moved_tag(repository, manager):
    root, commit, git = repository
    a = git("rev-parse", "HEAD")
    manager.add(str(root), ref=a, trust=True)
    commit()
    assert manager.update("managedfixture")["status"] == "pinned"
    manager.update("managedfixture", ref="main")
    assert len(generations(manager)) == 2
    git("tag", "v1")
    manager.update("managedfixture", ref="v1")
    commit()
    git("tag", "-f", "v1")
    with pytest.raises(PluginError, match="Pinned tag moved"):
        manager.update("managedfixture")


def test_cleanup_failure_is_committed_bounded_and_recoverable(repository, manager, monkeypatch):
    root, commit, _git = repository
    manager.config.set("plugins.retain_generations", 1)
    manager.config.save()
    manager.add(str(root), trust=True)
    original = manager.store._prune

    def fail_after_commit(records, **kwargs):
        if manager.store.journal.exists():
            raise PermissionError("busy")
        return original(records, **kwargs)

    monkeypatch.setattr(manager.store, "_prune", fail_after_commit)
    commit("1.1")
    result = manager.update("managedfixture")
    assert "cleanup pending" in result["warning"]
    assert state(manager)["metadata"]["version"] == "1.1"
    assert len(generations(manager)) == 2
    commit("1.2")
    with pytest.raises(PermissionError):
        manager.update("managedfixture")
    assert len(generations(manager)) == 2
    monkeypatch.setattr(manager.store, "_prune", original)
    manager.update("managedfixture")
    assert len(generations(manager)) == 1
    assert not manager.store.journal.exists()


def test_remove_missing_receipt_and_payload(repository, manager):
    manager.add(str(repository[0]), trust=True)
    import shutil

    path = manager.store.generation_path("managedfixture", state(manager)["active_generation"])
    shutil.rmtree(path / "payload")
    (path / "receipt.json").unlink()
    manager.remove("managedfixture")
    assert not generations(manager)


def test_rejected_commit_cleans_candidate_preserving_concurrent_config(repository, manager, monkeypatch):
    root, commit, _git = repository
    manager.add(str(root), trust=True)
    old = state(manager)
    original = manager.store.publish

    def edit_after_publish(candidate):
        result = original(candidate)
        config = SparkrunConfig(manager.config.config_path)
        config.set("plugins.retain_generations", 3)
        config.save()
        return result

    monkeypatch.setattr(manager.store, "publish", edit_after_publish)
    commit("1.1")
    with pytest.raises(PluginError, match="changed during validation"):
        manager.update("managedfixture")
    assert state(manager) == old
    assert len(generations(manager)) == 1
    assert SparkrunConfig(manager.config.config_path).get("plugins.retain_generations") == 3


def test_symlink_export_rejected(repository, manager):
    root, _commit, git = repository
    (root / "src/sparkrun/plugins/managedfixture/escape").symlink_to("/tmp")
    git("add", ".")
    git("commit", "-m", "symlink")
    with pytest.raises(PluginError, match="symlink"):
        manager.add(str(root), trust=True)


def test_management_routes_bypass_plugin_initialization(repository, manager, monkeypatch):
    from sparkrun.cli import main
    from sparkrun.cli._plugins import management_config

    monkeypatch.setattr("sparkrun.cli._common._initialize_application", lambda *_: pytest.fail("management imported plugins"))
    # Both routes share the bound default config location from conftest.
    runner = CliRunner()
    for args in (["plugins", "--help"], ["setup", "plugins", "--help"], ["plugins", "list", "--json"]):
        result = runner.invoke(main, args)
        assert result.exit_code == 0, result.output
    assert management_config  # Recovery does not use _get_context.
    result = runner.invoke(main, ["plugins", "add", str(repository[0])])
    assert result.exit_code == 1
    assert "--trust" in result.output


def test_compatibility_is_shared_and_reports_unknown_and_dependency_conflicts(repository):
    with acquire_git(str(repository[0])) as candidate:
        data = candidate.metadata.to_dict()
    host = HostSnapshot("0.4.0", python_version="3.12.1")
    assert evaluate_plugin_compatibility(data, host).status == "compatible"
    assert evaluate_plugin_compatibility(data, replace(host, core_version=None)).status == "unknown"
    assert evaluate_plugin_compatibility(data, replace(host, core_version="1.0")).status == "incompatible"
    data["dependencies"] += ["other[feature]>=1"]
    graph = replace(host, packages={"other": "1.0"}, requirements={"other": ('child>=2; extra == "feature"',)})
    assert "Missing dependency" in "; ".join(evaluate_plugin_compatibility(data, graph).reasons)
    graph = replace(graph, packages={"other": "1.0", "child": "2.0"})
    assert evaluate_plugin_compatibility(data, graph).status == "compatible"


def test_repository_provenance_only_matches_literal_github():
    assert repository_identity("git@github.com:sparksq/sparkrun-coldsnap-plugin.git") == repository_identity(
        "https://github.com/sparksq/sparkrun-coldsnap-plugin"
    )
    assert repository_identity("https://github.com/fork/sparkrun-coldsnap-plugin") != repository_identity(
        "https://github.com/sparksq/sparkrun-coldsnap-plugin"
    )


def test_startup_uses_external_tier_without_compatibility_audit(repository, manager):
    """Fresh-process loading must preserve namespace origin, resources, and registry trust."""
    root, commit, _git = repository
    commit(
        body="""
from sparkrun.core.registry_defaults import register_default_registry
from sparkrun.core.registry import RegistryEntry

def register(v):
    register_default_registry(RegistryEntry(name="fixture-recipes", url="https://example.invalid/recipes", subpath="recipes", trusted=True), owner=__name__)
"""
    )
    manager.add(str(root), trust=True)
    script = """
import json
import sparkrun.core.plugin_compatibility as compatibility
compatibility.evaluate_plugin_compatibility = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("startup audit"))
from sparkrun.core.bootstrap import init_sparkrun
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.plugin_inventory import list_plugins
from sparkrun.core.registry_defaults import iter_declared_registries
import sparkrun.plugins
before = list(sparkrun.plugins.__path__)
config = SparkrunConfig()
init_sparkrun(config=config)
import sparkrun.plugins.managedfixture as plugin
from importlib.resources import files
assert files(plugin).joinpath("data.txt").read_text() == "1.0"
assert list(sparkrun.plugins.__path__) == before
rows = [p for p in list_plugins(config) if p.name == "managedfixture"]
assert len(rows) == 1 and rows[0].managed and rows[0].source == "external" and rows[0].loaded
entry = next(d for d in iter_declared_registries() if d.entry.name == "fixture-recipes")
assert entry.tier.value == "out-of-tree"
assert entry.effective_entry().trusted is False
print("loaded")
"""
    import sys
    from sparkrun.core.application_profile import child_environment

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, **child_environment(config_path=manager.config.config_path), "SPARKRUN_NO_EXTERNAL_PLUGINS": "0"},
    )
    assert result.returncode == 0, result.stderr
    assert "loaded" in result.stdout


def test_unknown_gate_is_installed_disabled_until_enabled(repository, manager, monkeypatch):
    root, commit, _git = repository
    commit(feature="plugins.fixture")
    manager.add(str(root), trust=True)
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS")
    from sparkrun.core.managed_plugins import managed_enabled

    config, records = manager.store.snapshot()
    assert not managed_enabled(records["managedfixture"], config)
    manager.remove("managedfixture")
    manager.add(str(root), trust=True, enable=True)
    config, records = manager.store.snapshot()
    assert managed_enabled(records["managedfixture"], config)
    monkeypatch.setenv("SPARKRUN_NO_EXTERNAL_PLUGINS", "1")
    assert not managed_enabled(records["managedfixture"], config)


def test_two_reference_layouts_coexist_and_do_not_cross_prune(repository, manager):
    root, commit, _git = repository
    commit(name="sparkroute", feature="gateway.sparkroute")
    manager.add(str(root), trust=True, enable=True)
    sparkroute = manager.store.snapshot()[1]["sparkroute"]
    commit(name="coldsnap", feature="plugins.coldsnap", requires=">=0.4.0,<0.5")
    manager.add(str(root), trust=True, enable=True)
    records = manager.store.snapshot()[1]
    assert records["sparkroute"] == sparkroute
    assert set(records) == {"sparkroute", "coldsnap"}
    with pytest.raises(PluginError, match="No unique"):
        manager.store.select(records, root.as_uri())
    manager.remove("coldsnap")
    assert manager.store.snapshot()[1] == {"sparkroute": sparkroute}


def test_prepared_journal_recovery_restores_prior_state(repository, manager):
    import yaml

    manager.add(str(repository[0]), trust=True)
    cfg, ledger = manager.config.config_path.read_text(), manager.store.ledger.read_text()
    after_config = cfg + "\nunrelated: partial\n"
    manager.store.journal.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "state": "prepared",
                "config_path": str(manager.config.config_path),
                "before_config": cfg,
                "before_ledger": ledger,
                "after_config": after_config,
                "after_ledger": ledger,
            }
        )
    )
    manager.config.config_path.write_text(after_config)
    with pytest.raises(PluginError, match="Interrupted plugin activation"):
        manager.store.snapshot()
    with manager.store.mutation():
        pass
    assert manager.config.config_path.read_text() == cfg
    assert manager.store.ledger.read_text() == ledger
    assert not manager.store.journal.exists()


def test_failed_atomic_write_rolls_back_and_cleans_generation(repository, manager, monkeypatch):
    import sparkrun.core.plugin_store as store_module

    root, commit, _git = repository
    manager.add(str(root), trust=True)
    before = state(manager)
    original = store_module.atomic_private_write
    failed = False

    def fail_once(path, content):
        nonlocal failed
        if path == manager.store.ledger and not failed:
            failed = True
            raise OSError("injected disk failure")
        return original(path, content)

    monkeypatch.setattr(store_module, "atomic_private_write", fail_once)
    commit("1.1")
    with pytest.raises(OSError, match="disk failure"):
        manager.update("managedfixture")
    assert state(manager) == before
    assert len(generations(manager)) == 1
    assert not manager.store.journal.exists()


def test_corrupt_history_fails_rollback_and_symlink_store_cannot_delete_outside(repository, manager, tmp_path):
    root, commit, _git = repository
    manager.add(str(root), trust=True)
    old = state(manager)["active_generation"]
    commit("1.1")
    manager.update("managedfixture")
    before = state(manager)
    path = manager.store.generation_path("managedfixture", old)
    (path / "payload/sparkrun/plugins/managedfixture/helper.py").write_text("broken")
    with pytest.raises(PluginError, match="modified or is incomplete"):
        manager.rollback("managedfixture")
    assert state(manager) == before
    import shutil

    shutil.rmtree(path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "important").write_text("keep")
    path.symlink_to(outside, target_is_directory=True)
    result = manager.remove("managedfixture")
    assert "cleanup pending" in result["warning"]
    assert (outside / "important").read_text() == "keep"


def test_validation_timeout_preserves_old_generation(repository, manager, monkeypatch):
    from sparkrun.core.plugin_validation import validate_candidate

    root, commit, _git = repository
    manager.add(str(root), trust=True)
    before = state(manager)
    commit("1.1", body="import time\ntime.sleep(10)\n")
    monkeypatch.setattr("sparkrun.core.plugin_manager.validate_candidate", lambda *args: validate_candidate(*args, timeout=1))
    with pytest.raises(PluginError, match="timed out"):
        manager.update("managedfixture")
    assert state(manager) == before
    assert len(generations(manager)) == 1


def test_stale_config_save_preserves_new_plugin_flags(manager):
    stale = SparkrunConfig(manager.config.config_path)
    latest = SparkrunConfig(manager.config.config_path)
    latest.set("features", {"core.external_plugins": True})
    latest.save()
    stale.set("features", {"other.flag": True})
    stale.save()
    assert stale.get("features") == {"core.external_plugins": True, "other.flag": True}


def test_reconcile_default_updates_but_optout_does_no_fetch_or_write(repository, manager, monkeypatch):
    from sparkrun.core.plugin_update import reconcile_plugins

    root, commit, _git = repository
    manager.add(str(root), trust=True)
    commit("1.1")
    monkeypatch.setattr(HostSnapshot, "current", classmethod(lambda cls: manager.host))
    before = manager.store.ledger.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(PluginManager, "acquire", lambda *a: pytest.fail("opt-out fetched a plugin"))
        report = reconcile_plugins(manager.config, update=False)
    row = next(r for r in report["plugins"] if r["managed"])
    assert row["outcome"] == "availability unchecked"
    assert manager.store.ledger.read_bytes() == before
    assert not (manager.store.root / "plugins-update-report.json").exists()
    report = reconcile_plugins(manager.config)
    row = next(r for r in report["plugins"] if r["managed"])
    assert row["outcome"] == "updated"
    assert state(manager)["metadata"]["version"] == "1.1"
    assert row["compatibility_after"]["status"] == "compatible"
    # Audit includes disabled bundled plugins too.
    assert any(not r["managed"] and not r["enabled"] for r in report["plugins"])


def test_reconcile_repairs_incompatible_old_core_and_warns_on_incompatible_tip(repository, manager, monkeypatch):
    from sparkrun.core.plugin_update import reconcile_plugins

    root, commit, _git = repository
    manager.add(str(root), trust=True)
    monkeypatch.setattr(HostSnapshot, "current", classmethod(lambda cls: replace(manager.host, core_version="0.5.0")))
    commit("2.0", requires=">=0.5,<0.6")
    report = reconcile_plugins(manager.config)
    row = next(r for r in report["plugins"] if r["managed"])
    assert row["compatibility"]["status"] == "incompatible"
    assert row["compatibility_after"]["status"] == "compatible"
    before = state(manager)
    commit("3.0", requires=">=0.6")
    report = reconcile_plugins(manager.config)
    row = next(r for r in report["plugins"] if r["managed"])
    assert row["outcome"] == "failed"
    assert "incompatible" in row["error"]
    assert state(manager) == before


def test_public_cli_lifecycle_and_legacy_alias(repository, manager, monkeypatch):
    import json
    from sparkrun.cli import main

    root, commit, _git = repository
    commit(feature="plugins.fixture")
    monkeypatch.setattr(HostSnapshot, "current", classmethod(lambda cls: manager.host))
    runner = CliRunner()
    result = runner.invoke(main, ["plugins", "add", str(root), "--trust", "--enable"])
    assert result.exit_code == 0, result.output
    assert "installed" in result.output
    primary = runner.invoke(main, ["plugins", "list", "--json"])
    legacy = runner.invoke(main, ["setup", "plugins", "list", "--json"])
    assert primary.exit_code == legacy.exit_code == 0
    assert json.loads(primary.output) == json.loads(legacy.output)
    row = next(r for r in json.loads(primary.output) if r["managed"])
    assert row["compatibility"]["status"] == "compatible"
    commit("1.1", feature="plugins.fixture")
    for args in (["update", "managedfixture"], ["rollback", "managedfixture"], ["remove", "managedfixture", "--yes"]):
        result = runner.invoke(main, ["plugins", *args])
        assert result.exit_code == 0, result.output
    assert not manager.store.snapshot()[1]


def test_unknown_host_reconciliation_is_read_only(repository, manager, monkeypatch):
    from sparkrun.core.plugin_update import reconcile_plugins

    manager.add(str(repository[0]), trust=True)
    monkeypatch.setattr(HostSnapshot, "current", classmethod(lambda cls: replace(manager.host, core_version=None)))
    monkeypatch.setattr(PluginManager, "acquire", lambda *a: pytest.fail("unknown host fetched"))
    before = manager.store.ledger.read_bytes()
    report = reconcile_plugins(manager.config)
    assert "host core identity is unknown" in report["warning"]
    assert manager.store.ledger.read_bytes() == before
    assert not (manager.store.root / "plugins-update-report.json").exists()


@pytest.mark.parametrize("pin_kind", ["commit", "tag"])
@pytest.mark.parametrize("automatic", [False, True])
def test_rollback_preserves_pin_and_update_restores_it(repository, manager, pin_kind, automatic):
    root, commit, git = repository
    manager.add(str(root), trust=True)
    target = commit("1.1")
    if pin_kind == "tag":
        git("tag", "v1.1")
    manager.update("managedfixture", ref="v1.1" if pin_kind == "tag" else target)
    pinned = state(manager)["source"]
    manager.rollback("managedfixture")
    assert state(manager)["metadata"]["version"] == "1.0"
    assert state(manager)["source"] == pinned
    result = manager.update("managedfixture", automatic=automatic)
    assert result["status"] == "updated"
    assert state(manager)["revision"] == target
    assert state(manager)["source"] == pinned
    assert len(generations(manager)) == 2


def test_live_generation_lease_defers_eviction_without_exceeding_retention(repository, manager):
    from sparkrun.core.runtime_lease import runtime_lease

    root, commit, _ = repository
    first = manager.add(str(root), trust=True)["generation"]
    with runtime_lease(manager.config, kind="test-daemon"):
        commit("1.1")
        second = manager.update("managedfixture")["generation"]
        assert generations(manager) == sorted([first, second])
        commit("1.2")
        with pytest.raises(PluginError, match="in use"):
            manager.update("managedfixture")
        assert state(manager)["active_generation"] == second
        assert generations(manager) == sorted([first, second])
        with pytest.raises(PluginError, match="in use"):
            manager.remove("managedfixture")
    third = manager.update("managedfixture")["generation"]
    assert generations(manager) == sorted([second, third])
