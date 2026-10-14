"""One conflict rule for every in-process plugin registry.

Registering an equal value again is a no-op; anything else (a different value,
a different owner, or the same owner with a changed definition) raises
``PluginConflictError``. Conflicts are recorded where they are raised, so
constructing the error has no side effect on later launches.
"""

from __future__ import annotations

import pytest

from sparkrun.core.plugin_identity import PluginIdentity, plugin_registration
from sparkrun.core.registration import (
    PluginConflictError,
    claim_registration,
    recorded_plugin_conflicts,
    register_unique,
    registration_equal,
)


def _identity(name: str) -> PluginIdentity:
    return PluginIdentity(name, "tests." + name, ("external", name, None))


def test_constructing_the_error_records_nothing():
    """It used to append to the launch gate in __init__, so probing or
    formatting an error could fail every later launch in the process."""
    from sparkrun.core.installed_plugins import require_integrations

    PluginConflictError("constructed, never raised")
    assert recorded_plugin_conflicts() == ()
    require_integrations()


def test_a_raised_conflict_is_recorded_for_the_launch_gate():
    from sparkrun.core.installed_plugins import RequiredIntegrationError, require_integrations

    registry = {"k": 1}
    with pytest.raises(PluginConflictError, match="Thing 'k'"):
        register_unique(registry, "k", 2, description="Thing")
    assert registry == {"k": 1}
    with pytest.raises(RequiredIntegrationError, match="Thing 'k'"):
        require_integrations()


def test_equal_values_are_a_no_op_and_return_the_registered_value():
    registry: dict[str, tuple[int, ...]] = {}
    first = register_unique(registry, "k", (1, 2), description="Thing")
    assert register_unique(registry, "k", (1, 2), description="Thing") is first
    assert claim_registration(None, object(), description="Thing", key="k") is False


def test_callables_compare_by_definition_not_object_identity():
    """A re-imported plugin module produces new, equal function objects."""
    source = "def loader():\n    return 1\n"
    first: dict = {"__name__": "tests.reimported"}
    second: dict = {"__name__": "tests.reimported"}
    exec(compile(source, "reimported.py", "exec"), first)
    exec(compile(source, "reimported.py", "exec"), second)
    assert first["loader"] is not second["loader"]
    assert registration_equal(first["loader"], second["loader"])
    assert not registration_equal(first["loader"], lambda: 1)


def test_feature_flags_follow_the_rule():
    from sparkrun.core.features import FEATURE_FLAGS, FEATURE_OWNERS, FeatureFlag, register_feature

    flag = FeatureFlag("tests.rule", "a flag")
    try:
        with plugin_registration(_identity("feature-owner")):
            register_feature(flag)
            assert register_feature(FeatureFlag("tests.rule", "a flag")) == flag
            with pytest.raises(PluginConflictError, match="Feature flag 'tests.rule'"):
                register_feature(FeatureFlag("tests.rule", "changed"))
        with plugin_registration(_identity("feature-rival")):
            with pytest.raises(PluginConflictError, match="Feature flag 'tests.rule'"):
                register_feature(flag)
        with pytest.raises(PluginConflictError, match="Feature flag 'executor.docker'"):
            with plugin_registration(_identity("feature-squatter")):
                register_feature(FEATURE_FLAGS["executor.docker"])
    finally:
        FEATURE_FLAGS.pop("tests.rule", None)
        FEATURE_OWNERS.pop("tests.rule", None)


def test_workload_lifecycle_providers_follow_the_rule(monkeypatch):
    from sparkrun.core import workload_lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "_PROVIDERS", {})

    def matches(job):
        return True

    def control(*args, **kwargs):
        return {}

    with plugin_registration(_identity("lifecycle-fixture")):
        lifecycle.register_workload_lifecycle(lifecycle.WorkloadLifecycleProvider("lifecycle-fixture", matches, control))
        lifecycle.register_workload_lifecycle(lifecycle.WorkloadLifecycleProvider("lifecycle-fixture", matches, control))
        with pytest.raises(PluginConflictError, match="Workload lifecycle provider"):
            lifecycle.register_workload_lifecycle(lifecycle.WorkloadLifecycleProvider("lifecycle-fixture", matches, lambda *a, **kw: None))


def test_kv_strategies_cannot_replace_a_registered_name(monkeypatch):
    import sparkrun.models.kv as kv
    from sparkrun.models.kv.dense import DenseKVStrategy

    monkeypatch.setattr(kv, "_REGISTRY", list(kv._REGISTRY))
    kv.register_kv_strategy(DenseKVStrategy())  # the same strategy again: a no-op
    assert sum(s.name == "dense" for s in kv.list_kv_strategies()) == 1

    class Replacement(DenseKVStrategy):
        pass

    with pytest.raises(PluginConflictError, match="KV cache strategy 'dense'"):
        kv.register_kv_strategy(Replacement())


def test_platforms_follow_the_rule(monkeypatch):
    import sparkrun.platforms as platforms
    from sparkrun.platforms.dgx_spark import DgxSparkPlatform

    monkeypatch.setattr(platforms, "_REGISTRY", list(platforms._REGISTRY))
    platforms.register_platform(DgxSparkPlatform())  # a new instance of the same class: equal

    class Impostor(DgxSparkPlatform):
        pass

    with pytest.raises(PluginConflictError, match="Platform 'dgx-spark'"):
        platforms.register_platform(Impostor())


def test_cli_commands_follow_the_rule(monkeypatch):
    from sparkrun.core import cli_registry

    monkeypatch.setattr(cli_registry, "_CLI_EXTENSIONS", [])

    def build():
        return None

    cli_registry.register_cli_command(build, name="fixture")
    cli_registry.register_cli_command(build, name="fixture")
    with pytest.raises(PluginConflictError, match="CLI command 'fixture'.*test_registration_conflicts"):
        cli_registry.register_cli_command(lambda: None, name="fixture")
    assert len(cli_registry.registered_cli_commands()) == 1
