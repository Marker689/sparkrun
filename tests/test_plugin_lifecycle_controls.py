"""Explicit lifecycle actions, plugin/capability separation and durable selection."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.plugin_lifecycle import change_feature, change_plugin
from sparkrun.core.plugin_manifest import PluginError
from sparkrun.core.plugin_policy import plugin_state
from sparkrun.proxy.config import ProxyConfig
from test_managed_plugins import repository as repository, manager as manager  # noqa: F401


def test_plugin_enable_preserves_features_and_selection(tmp_path):
    config = SparkrunConfig(tmp_path / "config.yaml")
    config.set("features", {"gateway.litellm": False})
    config.save()
    proxy = ProxyConfig(tmp_path / "proxy.yaml")
    proxy.set_proxy(gateway="another-provider", port=7001)
    proxy.save()
    assert change_plugin(config, "litellm", False)["status"] == "disabled"
    assert change_plugin(config, "litellm", True)["status"] == "enabled"
    assert SparkrunConfig(config.config_path).feature_override("gateway.litellm") is False
    assert ProxyConfig(proxy.config_path).gateway == "another-provider"
    assert ProxyConfig(proxy.config_path).port == 7001


def test_explicit_feature_enable_selects_even_when_already_on(tmp_path):
    config = SparkrunConfig(tmp_path / "config.yaml")
    for _ in range(2):
        proxy = ProxyConfig(tmp_path / "proxy.yaml")
        proxy.set_proxy(gateway="another-provider", port=7001)
        proxy.save()
        result = change_feature(config, "gateway.litellm", True)
        assert result["messages"]
        assert ProxyConfig(proxy.config_path).gateway == "litellm"
        assert ProxyConfig(proxy.config_path).port == 7001
    change_feature(config, "gateway.litellm", False)
    assert ProxyConfig(proxy.config_path).gateway == "litellm"
    change_feature(config, "gateway.litellm", None)
    assert ProxyConfig(proxy.config_path).gateway == "litellm"


def test_feature_cannot_reenable_disabled_plugin(tmp_path, monkeypatch):
    config = SparkrunConfig(tmp_path / "config.yaml")
    change_plugin(config, "litellm", False)
    before = config.config_path.read_bytes()
    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_LITELLM", "1")
    with pytest.raises(PluginError, match="disabled plugin litellm"):
        change_feature(config, "gateway.litellm", True)
    assert config.config_path.read_bytes() == before
    assert not (tmp_path / "proxy.yaml").exists()


def test_environment_and_support_gate_prevent_partial_selection(tmp_path, monkeypatch):
    config = SparkrunConfig(tmp_path / "config.yaml")
    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_LITELLM", "0")
    with pytest.raises(PluginError, match="environment override"):
        change_feature(config, "gateway.litellm", True)
    assert not config.config_path.exists()
    assert not (tmp_path / "proxy.yaml").exists()
    monkeypatch.delenv("SPARKRUN_FEATURE_GATEWAY_LITELLM")
    change_feature(config, "gateway", False)
    with pytest.raises(PluginError, match="Gateway support is disabled"):
        change_feature(config, "gateway.litellm", True)


def test_gateway_activation_restores_both_files_on_write_failure(tmp_path, monkeypatch):
    from sparkrun.core.plugin_store import PluginStore

    config = SparkrunConfig(tmp_path / "config.yaml")
    config.set("features", {"gateway.litellm": False})
    config.save()
    proxy = ProxyConfig(tmp_path / "proxy.yaml")
    proxy.set_proxy(gateway="old", port=7002)
    proxy.save()
    before = (config.config_path.read_bytes(), proxy.config_path.read_bytes())
    original = PluginStore._restore
    failed = False

    def fail_once(path, content):
        nonlocal failed
        if Path(path) == proxy.config_path and not failed:
            failed = True
            raise OSError("test publication failure")
        return original(path, content)

    monkeypatch.setattr(PluginStore, "_restore", staticmethod(fail_once))
    with pytest.raises(OSError, match="publication failure"):
        change_feature(config, "gateway.litellm", True)
    assert (config.config_path.read_bytes(), proxy.config_path.read_bytes()) == before
    assert not (tmp_path / "plugins-transaction.yaml").exists()


def test_dry_run_does_not_execute_hooks_or_change_config(tmp_path, monkeypatch):
    config = SparkrunConfig(tmp_path / "config.yaml")
    monkeypatch.setattr("sparkrun.core.plugin_lifecycle.run_hook", lambda *a, **kw: pytest.fail("hook executed"))
    assert change_feature(config, "gateway.litellm", True, dry_run=True)["status"] == "planned"
    assert change_plugin(config, "litellm", True, dry_run=True)["status"] == "planned"
    assert not config.config_path.exists()
    assert not (tmp_path / "proxy.yaml").exists()


def snippet(tmp_path, data, source):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARKRUN_FEATURE_")}
    from sparkrun.core.application_profile import child_environment

    env.update(child_environment(config_path=path))
    env.update(
        SPARKRUN_CONFIG_DIR=str(tmp_path),
        SPARKRUN_CACHE_DIR=str(tmp_path / "cache"),
        SPARKRUN_APPLICATION_CONFIG=str(path),
        SPARKRUN_NO_TELEMETRY="1",
        SPARKRUN_NO_INSTALLED_PLUGINS="1",
        SPARKRUN_NO_EXTERNAL_PLUGINS="1",
    )
    process = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def test_disabled_plugin_never_imports_and_owns_no_runtime_flags(tmp_path):
    data = snippet(
        tmp_path,
        {"plugins": {"enabled": {"litellm": False, "k8s": False}}, "features": {"gateway.litellm": True, "executor.k8s": True}},
        """
import json, sys
from sparkrun.application import initialize
from sparkrun.core.features import get_feature, is_feature_enabled
from sparkrun.proxy.gateway import list_gateways
initialize()
print(json.dumps(["sparkrun.plugins.litellm" in sys.modules, "sparkrun.plugins.k8s" in sys.modules,
                  get_feature("gateway.litellm") is None, get_feature("executor.k8s") is None, list_gateways(),
                  is_feature_enabled("gateway.litellm"), is_feature_enabled("executor.k8s")]))
""",
    )
    assert data == [False, False, True, True, [], False, False]


def test_gateway_master_flag_hides_cli_and_gates_api(tmp_path):
    data = snippet(
        tmp_path,
        {"features": {"gateway": False}},
        """
import json, sys
from click.testing import CliRunner
from sparkrun.cli import main
from sparkrun import api
runner = CliRunner()
help_result = runner.invoke(main, ['--help'])
start = runner.invoke(main, ['proxy', 'start', '--dry-run'])
stop = runner.invoke(main, ['proxy', 'stop', '--dry-run'])
try:
    api.proxy.start(api.proxy.ProxyStartOptions(dry_run=True))
except api.proxy.GatewayUnavailable as exc:
    blocked = 'Gateway support is disabled' in str(exc)
else:
    blocked = False
print(json.dumps([help_result.exit_code, any(line.strip().startswith('proxy ') for line in help_result.output.splitlines()), start.exit_code, stop.exit_code, blocked]))
""",
    )
    assert data == [0, False, 1, 0, True]


def test_plugin_policy_is_shared_and_profile_requirements_survive(tmp_path, monkeypatch):
    from sparkrun.core.application_profile import ApplicationProfile
    from sparkrun.core.plugin_policy import PluginSource

    profile = ApplicationProfile(
        id="fixture", display_name="Fixture", command="fixture", package="sparkrun", required_integrations=("important",)
    )
    config = SparkrunConfig(tmp_path / "config.yaml")
    monkeypatch.setattr(config, "_profile", profile)
    config.set("plugins.enabled", {"important": False})
    for source in PluginSource:
        state = plugin_state("important", config, source=source)
        assert not state.enabled
        assert state.required


def test_install_enable_does_not_enable_capabilities_or_select(repository, manager, monkeypatch):
    root, commit, _git = repository
    commit(
        body="""
SPARKRUN_PLUGIN_MANIFEST = {"api": 1, "features": {"gateway.fixture": {"description": "Fixture", "default": False}}}
""",
        feature="gateway.fixture",
    )
    manager.add(str(root), trust=True, enable=True)
    config = SparkrunConfig(manager.config.config_path)
    assert config.get("plugins.enabled.managedfixture") is True
    assert config.feature_override("gateway.fixture") is None
    assert ProxyConfig(config.config_path.parent / "proxy.yaml").gateway is None
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS")
    change_plugin(config, "managedfixture", False)
    commit(
        "1.1",
        body="""
SPARKRUN_PLUGIN_MANIFEST = {"api": 1, "features": {"gateway.fixture": {"description": "Fixture", "default": False}}}
""",
        feature="gateway.fixture",
    )
    manager.update("managedfixture")
    assert SparkrunConfig(config.config_path).get("plugins.enabled.managedfixture") is False


def test_explicit_hook_failure_leaves_plugin_state_unchanged(repository, manager, monkeypatch):
    root, commit, _git = repository
    commit(
        body="""
SPARKRUN_PLUGIN_MANIFEST = {"api": 1, "hooks": {"plugin.enable": "sparkrun.plugins.managedfixture:enable"}}
def enable(context):
    raise RuntimeError("deliberate enable failure")
"""
    )
    manager.add(str(root), trust=True)
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS")
    change_plugin(manager.config, "managedfixture", False)
    before = manager.config.config_path.read_bytes()
    with pytest.raises(PluginError, match="deliberate enable failure"):
        change_plugin(manager.config, "managedfixture", True)
    assert manager.config.config_path.read_bytes() == before


def test_disable_commits_despite_broken_cleanup_hook(repository, manager, monkeypatch):
    root, commit, _git = repository
    commit(
        body="""
SPARKRUN_PLUGIN_MANIFEST = {"api": 1, "hooks": {"plugin.disable": "sparkrun.plugins.managedfixture:disable"}}
def disable(context):
    raise RuntimeError("deliberate cleanup failure")
"""
    )
    manager.add(str(root), trust=True)
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS")
    result = change_plugin(manager.config, "managedfixture", False)
    assert result["status"] == "disabled"
    assert "deliberate cleanup failure" in result["warning"]
    assert SparkrunConfig(manager.config.config_path).get("plugins.enabled.managedfixture") is False


def test_management_hooks_have_correct_events_and_generations(repository, manager, monkeypatch):
    root, commit, _git = repository
    body = """
from sparkrun.plugin_lifecycle import LifecyclePlan
SPARKRUN_PLUGIN_MANIFEST = {"api": 1, "hooks": {
    "plugin.install": "sparkrun.plugins.managedfixture:record",
    "plugin.enable": "sparkrun.plugins.managedfixture:record",
    "plugin.update": "sparkrun.plugins.managedfixture:record",
    "plugin.rollback": "sparkrun.plugins.managedfixture:record",
    "plugin.remove": "sparkrun.plugins.managedfixture:record",
}}
def record(context):
    events = context.config.get("plugins", {}).get("managedfixture", {}).get("events", [])
    event = [context.event, (context.previous or {}).get("metadata", {}).get("version"),
             (context.current or {}).get("metadata", {}).get("version")]
    return LifecyclePlan(settings={"plugins.managedfixture.events": events + [event]})
"""
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS")
    commit(body=body)
    manager.add(str(root), trust=True, enable=True)
    commit("1.1", body=body)
    manager.update("managedfixture")
    manager.rollback("managedfixture")
    manager.remove("managedfixture")
    events = SparkrunConfig(manager.config.config_path).get("plugins.managedfixture.events")
    assert events == [
        ["plugin.install", None, "1.0"],
        ["plugin.enable", None, "1.0"],
        ["plugin.update", "1.0", "1.1"],
        ["plugin.rollback", "1.1", "1.0"],
        ["plugin.remove", "1.0", None],
    ]


def test_dynamic_features_can_be_managed_from_a_fresh_cli(repository, manager, monkeypatch):
    root, commit, _git = repository
    commit(
        body="""
from sparkrun.core.features import FeatureFlag
FEATURE_DEFINITIONS = [FeatureFlag("fixture.dynamic", "Dynamic feature", default=True)]
"""
    )
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS")
    manager.add(str(root), trust=True, enable=True)
    result = change_feature(manager.config, "fixture.dynamic", False)
    assert result["status"] == "disabled"
    assert SparkrunConfig(manager.config.config_path).feature_override("fixture.dynamic") is False


def test_unverified_hook_code_is_not_executed_for_disable_or_remove(repository, manager, monkeypatch):
    root, commit, _git = repository
    commit(
        body="""
SPARKRUN_PLUGIN_MANIFEST = {"api": 1, "hooks": {
    "plugin.disable": "sparkrun.plugins.managedfixture:cleanup",
    "plugin.remove": "sparkrun.plugins.managedfixture:cleanup",
}}
def cleanup(context):
    pass
"""
    )
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS")
    installed = manager.add(str(root), trust=True)
    package = (
        manager.store.generation_path("managedfixture", installed["generation"]) / "payload/sparkrun/plugins/managedfixture/__init__.py"
    )
    package.write_text(package.read_text() + '\nraise RuntimeError("unverified code executed")\n')
    result = change_plugin(manager.config, "managedfixture", False)
    assert result["warning"] and "unverified code executed" not in result["warning"]
    result = manager.remove("managedfixture")
    assert result["warning"] and "unverified code executed" not in result["warning"]


def test_concurrent_proxy_edit_is_preserved_and_selection_aborted(tmp_path, monkeypatch):
    import sparkrun.core.plugin_lifecycle as service

    config = SparkrunConfig(tmp_path / "config.yaml")
    original = service.run_hook

    def edit_proxy(*args, **kwargs):
        plan = original(*args, **kwargs)
        proxy = ProxyConfig(tmp_path / "proxy.yaml")
        proxy.set_proxy(gateway="concurrent", port=7341)
        proxy.save()
        return plan

    monkeypatch.setattr(service, "run_hook", edit_proxy)
    with pytest.raises(PluginError, match="Proxy configuration changed"):
        change_feature(config, "gateway.litellm", True)
    assert not config.config_path.exists()
    assert ProxyConfig(tmp_path / "proxy.yaml").gateway == "concurrent"
    assert ProxyConfig(tmp_path / "proxy.yaml").port == 7341


def test_directory_policy_uses_declared_canonical_identity(tmp_path, monkeypatch):
    from sparkrun.core.plugin_inventory import list_plugins
    from sparkrun.core.external_plugins import directory_plugin_identities

    config = SparkrunConfig(tmp_path / "config.yaml")
    directory = tmp_path / "external"
    directory.mkdir()
    (directory / "unrelated_module.py").write_text("SPARKRUN_PLUGIN_API_VERSION = 1\n")
    (directory / "plugin.toml").write_text('name = "canonical-fixture"\nmodule = "unrelated_module"\n')
    config.set("plugins.paths", [str(directory)])
    config.set("plugins.enabled", {"canonical-fixture": False})
    config.set("features", {"core.external_plugins": True})
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS")
    plugin = next(p for p in list_plugins(config) if p.name == "unrelated_module")
    assert plugin.plugin_id == "canonical-fixture"
    assert plugin.enabled is False
    assert directory_plugin_identities([directory], config) == {}


def test_directory_plugins_honor_the_external_loading_gate(tmp_path, monkeypatch):
    """Directory plugins are external: the policy re-check must see
    ``core.external_plugins``. They were evaluated under a "directory" source
    the policy never gated. Explicitly named paths still bypass the flag."""
    from sparkrun.core.external_plugins import directory_plugin_identities
    from sparkrun.core.plugin_policy import plugin_state

    config = SparkrunConfig(tmp_path / "config.yaml")
    directory = tmp_path / "external"
    directory.mkdir()
    (directory / "gated_fixture.py").write_text("SPARKRUN_PLUGIN_API_VERSION = 1\n")
    config.set("features", {"core.external_plugins": False})
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS")
    assert directory_plugin_identities([directory], config) == {}
    assert set(directory_plugin_identities([directory], config, explicit=True)) == {(directory.resolve(), "gated_fixture")}
    with pytest.raises(ValueError):
        plugin_state("gated-fixture", config, source="directory")


def test_legacy_loading_alias_honors_explicit_channel(tmp_path):
    from sparkrun.core.features import is_feature_enabled

    config = SparkrunConfig(tmp_path / "config.yaml")
    assert is_feature_enabled("integration.k8s", config=config, channel="alpha", env={})
    assert not is_feature_enabled("integration.k8s", config=config, channel="stable", env={})
    config.set("plugins.enabled", {"k8s": False})
    assert not is_feature_enabled("integration.k8s", config=config, channel="alpha", env={})


def test_lifecycle_plan_cannot_write_host_loading_policy(tmp_path, monkeypatch):
    import sparkrun.core.plugin_lifecycle as lifecycle

    config = SparkrunConfig(tmp_path / "config.yaml")
    monkeypatch.setattr(lifecycle, "_run_worker", lambda *a, **kw: {"settings": {"plugins.enabled.litellm": False}})
    with pytest.raises(PluginError, match="reserved"):
        lifecycle.run_hook(config, "enabled", {"declarations": {"hooks": {"plugin.enable": "fixture:enable"}}}, "plugin.enable")
    assert not config.config_path.exists()
