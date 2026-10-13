"""Plugin source must not change registration atomicity or inventory truth."""

from types import ModuleType

import pytest

from test_external_plugins import clean_sys as clean_sys


@pytest.mark.parametrize("source", ["directory", "bundled", "installed"])
@pytest.mark.parametrize("failure_phase", ["import", "hook", "api"])
def test_failed_plugin_rolls_back_and_independent_plugin_still_loads(tmp_path, monkeypatch, clean_sys, source, failure_phase):
    import importlib
    from types import SimpleNamespace
    from sparkrun.core.bootstrap import get_variables
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.external_plugins import load_external_plugins, loaded_plugin_module
    from sparkrun.core.in_tree_plugins import load_in_tree_plugins, IN_TREE_PLUGIN_FEATURES
    from sparkrun.core.installed_plugins import load_installed_plugins, installed_plugin_inventory
    from sparkrun.core.features import get_feature, register_feature, FeatureFlag
    from sparkrun.core.cli_registry import registered_cli_commands
    from sparkrun.core.run_handlers import registered_run_handlers
    from sparkrun.core.setup_steps import all_setup_steps
    from sparkrun.transports import list_transports

    v = get_variables()
    before_good = v.get("RECOVERY_GOOD_VALUE", 0)
    config = SparkrunConfig()
    package = tmp_path / "recovery_plugins"
    package.mkdir()
    (package / "__init__.py").write_text("")
    body = """
from sparkrun.core.features import FeatureFlag, register_feature
from sparkrun.core.run_handlers import RunHandler, register_run_handler
from sparkrun.core.setup_steps import SetupStep, register_setup_step
from sparkrun.core.cli_registry import register_cli_command
from sparkrun.transports.base import Transport
SPARKRUN_PLUGIN_API_VERSION = 1
class RecoveryTransport(Transport):
    transport_name = "recovery_bad"
def contribute():
    register_feature(FeatureFlag("test.recovery_bad", "bad", default=True))
    register_run_handler(RunHandler("recovery_bad", lambda *a, **kw: None))
    register_setup_step(SetupStep("recovery_bad", "bad", feature_flag="test.recovery_bad"))
    register_cli_command(lambda: None, name="recovery_bad")
def register(v):
    contribute()
    v.set("RECOVERY_BAD_VALUE", True)
    raise RuntimeError("bad registration")
"""
    if failure_phase == "import":
        body += '\ncontribute()\nraise RuntimeError("bad import")\n'
    if failure_phase == "api":
        body = body.replace("SPARKRUN_PLUGIN_API_VERSION = 1", "") + "\ncontribute()\n"
    (package / "recovery_bad.py").write_text(body)
    (package / "recovery_good.py").write_text("""
SPARKRUN_PLUGIN_API_VERSION = 1
def register(v):
    v.set("RECOVERY_GOOD_VALUE", v.get("RECOVERY_GOOD_VALUE", 0) + 1)
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    if source == "directory":
        config._data["plugins"] = {"paths": [str(package)]}
        loaded = load_external_plugins(v, paths=[package])
        prefix = ""
        assert loaded == ["recovery_good"]
    elif source == "bundled":
        gates = dict(IN_TREE_PLUGIN_FEATURES)
        for name in ("recovery_bad", "recovery_good"):
            flag = "test.gate_" + name
            register_feature(FeatureFlag(flag, flag, default=True))
            gates[name] = flag
        monkeypatch.setattr("sparkrun.core.in_tree_plugins.IN_TREE_PLUGIN_FEATURES", gates)
        monkeypatch.setattr("sparkrun.core.in_tree_plugins.IN_TREE_PLUGIN_PACKAGE", "recovery_plugins")
        loaded = load_in_tree_plugins(v, package="recovery_plugins")
        prefix = "recovery_plugins."
        assert loaded == ["recovery_good"]
    else:
        monkeypatch.delenv("SPARKRUN_NO_INSTALLED_PLUGINS")
        monkeypatch.setattr("sparkrun.core.installed_plugins._attempted", set())
        entries = [
            SimpleNamespace(
                name=name,
                value="recovery_plugins." + name,
                dist=SimpleNamespace(name=name, version="1"),
                load=lambda name=name: importlib.import_module("recovery_plugins." + name),
            )
            for name in ("recovery_bad", "recovery_good")
        ]
        monkeypatch.setattr("sparkrun.core.installed_plugins.entry_points", lambda **_: entries)
        config = SparkrunConfig()
        config._data["integrations"] = {name: True for name in ("recovery_bad", "recovery_good")}
        load_installed_plugins(v, config=config)
        rows = {row.name: row for row in installed_plugin_inventory()}
        assert not rows["recovery_bad"].loaded and rows["recovery_bad"].failure
        assert rows["recovery_good"].loaded
        prefix = "recovery_plugins."
    from sparkrun.core.plugin_inventory import list_plugins

    inventory = {row.name: row for row in list_plugins(config, v)}
    bad, good = inventory["recovery_bad"], inventory["recovery_good"]
    assert not bad.loaded
    if failure_phase == "api":
        assert "missing API declaration" in bad.failure
        assert "declare SPARKRUN_PLUGIN_API_VERSION = 1" in bad.failure
    else:
        assert bad.failure == "RuntimeError: bad " + ("import" if failure_phase == "import" else "registration")
    assert good.loaded and good.failure is None
    assert bad.to_dict()["failure"] == bad.failure
    assert loaded_plugin_module(prefix + "recovery_bad") is None
    assert loaded_plugin_module(prefix + "recovery_good") is not None
    assert get_feature("test.recovery_bad") is None
    assert "recovery_bad" not in registered_run_handlers(SparkrunConfig())
    assert "recovery_bad" not in {step.key for step in all_setup_steps()}
    assert "recovery_bad" not in {spec.name for spec in registered_cli_commands()}
    assert "recovery_bad" not in list_transports(v)
    assert not v.get("RECOVERY_BAD_VALUE")
    assert v.get("RECOVERY_GOOD_VALUE") == before_good + 1
    # Registering the successfully loaded module again is a no-op.
    from sparkrun.core.external_plugins import load_plugin_module

    assert load_plugin_module(loaded_plugin_module(prefix + "recovery_good"), v)
    assert v.get("RECOVERY_GOOD_VALUE") == before_good + 1


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_direct_module_registration_restores_state_on_failure(monkeypatch, error):
    from sparkrun.core.bootstrap import get_variables
    from sparkrun.core.external_plugins import load_plugin_module, loaded_plugin_module
    from sparkrun.core.run_handlers import RunHandler, register_run_handler, registered_run_handlers

    from sparkrun.core.config import SparkrunConfig

    v = get_variables()
    monkeypatch.setattr("sparkrun.core.run_handlers._RUN_HANDLERS", {})
    module = ModuleType("recovery_direct")
    module.SPARKRUN_PLUGIN_API_VERSION = 1

    def register(v):
        register_run_handler(RunHandler("recovery_direct", lambda *a, **kw: None))
        raise error("failed")

    module.register = register
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            load_plugin_module(module, v)
    else:
        assert not load_plugin_module(module, v)
    assert not registered_run_handlers(SparkrunConfig())
    assert loaded_plugin_module(module.__name__) is None


@pytest.mark.parametrize("declaration", ["missing", None, True, 1.0, "1", 0, 1, 2])
def test_installed_plugin_api_is_integer_and_independent_of_profile_api(monkeypatch, clean_sys, declaration):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from sparkrun.core.bootstrap import get_variables
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.features import FeatureFlag, register_feature, get_feature
    from sparkrun.core.installed_plugins import load_installed_plugins, installed_plugin_inventory
    from sparkrun.core.registration import PLUGIN_API_VERSION

    v = get_variables()
    from sparkrun.core import features

    monkeypatch.setattr(features, "FEATURE_FLAGS", dict(features.FEATURE_FLAGS))
    monkeypatch.setattr("sparkrun.core.application_profile.APPLICATION_PROFILE_API_VERSION", 999)
    monkeypatch.delenv("SPARKRUN_NO_INSTALLED_PLUGINS")
    monkeypatch.setattr("sparkrun.core.installed_plugins._attempted", set())
    module = ModuleType("recovery_version")
    if declaration != "missing":
        module.SPARKRUN_PLUGIN_API_VERSION = declaration
    module.register = Mock()

    def load():
        register_feature(FeatureFlag("test.version_import", "version import", default=True))
        return module

    entry = SimpleNamespace(name="version-test", value=module.__name__, dist=SimpleNamespace(name="version-test", version="0.1"), load=load)
    monkeypatch.setattr("sparkrun.core.installed_plugins.entry_points", lambda **kw: [entry])
    config = SparkrunConfig()
    config._data["integrations"] = {"version-test": True}
    load_installed_plugins(v, config=config)
    row = installed_plugin_inventory()[0]
    accepted = type(declaration) is int and declaration == PLUGIN_API_VERSION
    assert row.loaded is accepted
    assert (get_feature("test.version_import") is not None) is accepted
    if accepted:
        module.register.assert_called_once_with(v)
        assert row.failure is None
    else:
        module.register.assert_not_called()
        assert "API" in row.failure


@pytest.mark.parametrize("declaration", [None, True, 1.0, "1", 0, 1, 2])
def test_direct_module_requires_compatible_api(declaration):
    from unittest.mock import Mock
    from sparkrun.core.bootstrap import get_variables
    from sparkrun.core.external_plugins import load_plugin_module

    module = ModuleType("module_api_declaration")
    module.register = Mock()
    if declaration is not None:
        module.SPARKRUN_PLUGIN_API_VERSION = declaration
    accepted = type(declaration) is int and declaration == 1
    assert load_plugin_module(module, get_variables()) is accepted
    assert module.register.call_count == int(accepted)


@pytest.mark.parametrize("source", ["directory", "bundled", "installed"])
@pytest.mark.parametrize("invalid", ["missing", "cycle"])
def test_invalid_setup_graph_rolls_back_before_plugin_is_reported_loaded(tmp_path, monkeypatch, clean_sys, source, invalid):
    import importlib
    from types import SimpleNamespace
    from sparkrun.core.bootstrap import get_variables
    from sparkrun.core.external_plugins import load_external_plugins, loaded_plugin_module
    from sparkrun.core.in_tree_plugins import load_in_tree_plugins, IN_TREE_PLUGIN_FEATURES
    from sparkrun.core.installed_plugins import (
        load_installed_plugins,
        installed_plugin_inventory,
        require_integrations,
        RequiredIntegrationError,
    )
    from sparkrun.core.application_profile import get_application_profile
    from sparkrun.core.features import FeatureFlag, register_feature, get_feature
    from sparkrun.core.setup_steps import all_setup_steps
    from sparkrun.api.setup import run_setup_steps, SetupActionContext, run_setup_undo, SetupManifest
    from test_setup_steps import state_context
    from dataclasses import replace

    v = get_variables()
    package = tmp_path / "graph_plugins"
    package.mkdir()
    (package / "__init__.py").write_text("")
    dependency = "graph_bad" if invalid == "cycle" else "graph_missing"
    (package / "graph_bad.py").write_text(
        """
from sparkrun.core.features import FeatureFlag, register_feature
from sparkrun.core.setup_steps import SetupStep, register_setup_step
SPARKRUN_PLUGIN_API_VERSION = 1
def register(v):
    register_feature(FeatureFlag("setup.steps.graph_bad", "bad", default=False))
    register_setup_step(SetupStep("graph_bad", "bad", requires=(%r,), feature_flag="setup.steps.graph_bad"))
"""
        % dependency
    )
    (package / "graph_good.py").write_text("""
SPARKRUN_PLUGIN_API_VERSION = 1
def register(v):
    v.set("GRAPH_GOOD_LOADED", True)
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    if source == "directory":
        assert load_external_plugins(v, paths=[package]) == ["graph_good"]
        prefix = ""
    elif source == "bundled":
        gates = dict(IN_TREE_PLUGIN_FEATURES)
        for name in ("graph_bad", "graph_good"):
            flag = "test.gate_" + name
            register_feature(FeatureFlag(flag, flag, default=True))
            gates[name] = flag
        monkeypatch.setattr("sparkrun.core.in_tree_plugins.IN_TREE_PLUGIN_FEATURES", gates)
        assert load_in_tree_plugins(v, package="graph_plugins") == ["graph_good"]
        prefix = "graph_plugins."
    else:
        monkeypatch.delenv("SPARKRUN_NO_INSTALLED_PLUGINS", raising=False)
        monkeypatch.setattr("sparkrun.core.installed_plugins._attempted", set())
        monkeypatch.setattr(
            "sparkrun.core.installed_plugins.get_application_profile",
            lambda: replace(get_application_profile(), required_integrations=("graph-bad",)),
        )
        eps = [
            SimpleNamespace(
                name=name.replace("_", "-"),
                value="graph_plugins." + name,
                dist=None,
                load=lambda name=name: importlib.import_module("graph_plugins." + name),
            )
            for name in ("graph_bad", "graph_good")
        ]
        monkeypatch.setattr("sparkrun.core.installed_plugins.entry_points", lambda **kw: eps)
        from sparkrun.core.config import SparkrunConfig

        cfg = SparkrunConfig()
        cfg.set("integrations", {"graph-bad": True, "graph-good": True})
        cfg._profile = replace(get_application_profile(), required_integrations=("graph-bad",))
        load_installed_plugins(v, config=cfg)
        rows = installed_plugin_inventory()
        assert not rows[0].loaded and rows[0].failure and rows[1].loaded
        with pytest.raises(RequiredIntegrationError, match="graph-bad"):
            require_integrations()
        prefix = "graph_plugins."
    assert loaded_plugin_module(prefix + "graph_bad") is None
    assert get_feature("setup.steps.graph_bad") is None
    assert "graph_bad" not in {step.key for step in all_setup_steps()}
    assert v.get("GRAPH_GOOD_LOADED") is True
    state, context = state_context()
    assert (
        run_setup_steps({state.host: state}, context, SetupActionContext("tester", dry_run=True), only_steps={"docker"}).steps["docker"]
        == "ok"
    )
    empty = SetupManifest(1, "lab", "", "", "tester", [state.host])
    assert run_setup_undo(empty, SetupActionContext("tester")).complete


def test_setup_dependencies_allow_same_module_forward_references_and_loaded_providers(monkeypatch):
    from sparkrun.core.bootstrap import get_variables
    from sparkrun.core.external_plugins import load_plugin_module
    from sparkrun.core.features import FeatureFlag, register_feature
    from sparkrun.core.setup_steps import SetupStep, register_setup_step, all_setup_steps

    v = get_variables()
    provider = ModuleType("graph_provider")
    provider.SPARKRUN_PLUGIN_API_VERSION = 1
    consumer = ModuleType("graph_consumer")
    consumer.SPARKRUN_PLUGIN_API_VERSION = 1

    def register_provider(v):
        register_feature(FeatureFlag("setup.steps.graph_provider", "provider", default=False))
        # Forward reference to another step in the same module is supported.
        register_setup_step(SetupStep("graph_child", "child", requires=("graph_parent",), feature_flag="setup.steps.graph_provider"))
        register_setup_step(SetupStep("graph_parent", "parent", requires=("docker",), feature_flag="setup.steps.graph_provider"))

    def register_consumer(v):
        register_feature(FeatureFlag("setup.steps.graph_consumer", "consumer", default=False))
        register_setup_step(SetupStep("graph_consumer", "consumer", requires=("graph_child",), feature_flag="setup.steps.graph_consumer"))

    provider.register = register_provider
    consumer.register = register_consumer
    # Unknown providers fail now; there is no deferred or automatic import.
    with pytest.raises(ValueError, match="Unknown setup prerequisite"):
        load_plugin_module(consumer, v, strict=True)
    assert load_plugin_module(provider, v, strict=True)
    assert load_plugin_module(consumer, v, strict=True)
    names = [s.key for s in all_setup_steps()]
    assert names.index("docker") < names.index("graph_parent") < names.index("graph_child") < names.index("graph_consumer")


def test_directory_inventory_retains_source_failure_and_clears_on_success(tmp_path, monkeypatch, clean_sys):
    from sparkrun.core.bootstrap import get_variables
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.external_plugins import load_external_plugins
    from sparkrun.core.plugin_inventory import list_plugins
    import sys

    v = get_variables()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    name = "recovery_source_outcome"
    (first / (name + ".py")).write_text(
        "SPARKRUN_PLUGIN_API_VERSION = 1\ndef register(v):\n    if not v.get('RECOVERY_RETRY_OK'): raise RuntimeError('retry me')\n"
    )
    (second / (name + ".py")).write_text("raise AssertionError('listing must not import')\n")
    config = SparkrunConfig()
    config._data["plugins"] = {"paths": [str(first), str(second)]}
    assert load_external_plugins(v, paths=[first]) == []
    # The kill switch is still on: discovery reports the previous attempt but
    # must not execute either directory's module during listing.
    failed_module = sys.modules[name]
    rows = [row for row in list_plugins(config, v) if row.name == name]
    assert len(rows) == 2 and not any(row.enabled for row in rows)
    by_path = {row.path: row for row in rows}
    assert by_path[first].failure == "RuntimeError: retry me"
    assert by_path[second].failure is None
    assert sys.modules[name] is failed_module
    v.set("RECOVERY_RETRY_OK", True)
    try:
        assert load_external_plugins(v, paths=[first]) == [name]
        row = next(row for row in list_plugins(config, v) if row.name == name and row.path == first)
        assert row.loaded and row.failure is None
    finally:
        v.set("RECOVERY_RETRY_OK", False)
