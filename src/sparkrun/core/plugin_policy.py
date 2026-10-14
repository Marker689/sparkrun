"""One loading policy for bundled, directory, managed and installed plugins.

Plugin state is independent of the capability flags registered by a plugin.
Old whole-plugin feature gates and installed ``integrations`` settings are
migration inputs; an explicit ``plugins.enabled`` override takes precedence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sparkrun.core.plugin_identity import normalize_plugin_id


class PluginSource(StrEnum):
    """Where a plugin comes from, as far as loading policy is concerned.

    ``EXTERNAL`` covers configured directories and managed plugins: both need
    ``core.external_plugins``. ``INSTALLED`` entry points are off unless
    selected. An unknown spelling is an error rather than a silent bypass of
    those gates.
    """

    IN_TREE = "in-tree"
    EXTERNAL = "external"
    INSTALLED = "installed"


def policy_source(kind: str) -> PluginSource:
    """Policy source for a registration identity's source kind.

    Programmatic registrations (``module``, ``application``) have no source
    gate of their own, which is the in-tree rule.
    """
    try:
        return PluginSource(kind)
    except ValueError:
        return PluginSource.IN_TREE


LEGACY_PLUGIN_FLAGS = {
    "k8s": "integration.k8s",
    "lil": "registry.lil",
    "sparkarena": "integration.arena",
    "oci-relay": "plugins.oci_relay",
    "coldsnap": "plugins.coldsnap",
}


@dataclass(frozen=True)
class PluginState:
    enabled: bool
    source: str
    reason: str | None = None
    required: bool = False


def plugin_overrides(config) -> dict[str, bool]:
    raw = config.get("plugins.enabled", {})
    if not isinstance(raw, dict) or any(type(value) is not bool for value in raw.values()):
        raise ValueError("plugins.enabled must map plugin IDs to true or false")
    result = {}
    for name, value in raw.items():
        identifier = normalize_plugin_id(name)
        if identifier in result:
            raise ValueError("Repeated canonical plugin ID: %s" % identifier)
        result[identifier] = value
    return result


def plugin_state(
    name,
    config=None,
    *,
    source: PluginSource | str = PluginSource.IN_TREE,
    legacy_feature=None,
    env=None,
    channel=None,
    external_gate: bool = True,
) -> PluginState:
    """Resolve whether plugin *name* loads.

    ``external_gate=False`` is only for explicitly named directories (a
    programmatic ``load_external_plugins(paths=...)``), which bypass the
    ``core.external_plugins`` flag by contract.
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.features import _raw_feature_enabled, feature_source
    from sparkrun.core.external_plugins import _external_plugins_disabled
    import os

    source = PluginSource(source)
    config = config if config is not None else SparkrunConfig()
    env = os.environ if env is None else env
    name = normalize_plugin_id(name)
    profile = config.profile
    required = name in {normalize_plugin_id(item) for item in profile.required_integrations}
    selected = name in {normalize_plugin_id(item) for item in (*profile.integrations, *profile.required_integrations)}
    overrides = plugin_overrides(config)
    integrations = config.get("integrations", {})
    if not isinstance(integrations, dict) or any(type(value) is not bool for value in integrations.values()):
        raise ValueError("integrations must map plugin IDs to true or false")
    integrations = {normalize_plugin_id(key): value for key, value in integrations.items()}
    legacy = LEGACY_PLUGIN_FLAGS.get(name)
    # Older plugins can still declare one loading feature. New declarations
    # contain capabilities and are independent of this migration fallback.
    legacy = legacy or legacy_feature
    if name in overrides:
        enabled, origin = overrides[name], "config"
    elif name in integrations:
        enabled, origin = integrations[name], "integrations (legacy)"
    elif selected:
        enabled, origin = True, "distribution"
    elif legacy:
        enabled = _raw_feature_enabled(legacy, config=config, env=env, channel=channel)
        origin = "legacy feature: " + feature_source(legacy, config=config, env=env, channel=channel)
    else:
        enabled, origin = source is not PluginSource.INSTALLED, "default"
    reason = None if enabled else "plugin disabled"
    if (
        source is PluginSource.EXTERNAL
        and external_gate
        and (_external_plugins_disabled() or not _raw_feature_enabled("core.external_plugins", config=config, env=env))
    ):
        enabled, reason = False, "external plugin loading disabled"
    if source is PluginSource.INSTALLED and env.get("SPARKRUN_NO_INSTALLED_PLUGINS", "").lower() in {"1", "true", "yes"}:
        enabled, reason = False, "installed plugin loading disabled"
    return PluginState(enabled, origin, reason, required)


def set_plugin_override(config, name, enabled):
    from copy import deepcopy

    overrides = deepcopy(plugin_overrides(config))
    name = normalize_plugin_id(name)
    if enabled is None:
        overrides.pop(name, None)
    else:
        overrides[name] = enabled
    config.set("plugins.enabled", overrides)
