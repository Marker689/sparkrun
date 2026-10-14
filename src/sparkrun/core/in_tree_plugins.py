"""Load first-party plugins that ship inside the sparkrun distribution.

The in-tree mate of :mod:`sparkrun.core.external_plugins`. Both walk a set of
top-level modules, register any SAF plugin subclasses they define, and run an
optional ``register(v)`` hook — the shared half is
:func:`~sparkrun.core.registration.load_and_register_plugin`. Only *where* the
modules come from differs: a package inside the wheel here, user-configured
directories there.

``sparkrun.plugins`` is for **cross-cutting integrations** — things that span
several extension points at once and are meaningful as a single removable unit:
an integration contributing a backend implementation, a hidden CLI command and
a wire protocol together, none of which is "a runtime" or "an executor". The
existing first-party packages that *do* map cleanly onto one extension point
(``runtimes``, ``transports``, ``executors``, ``schedulers``, ``builders``,
``benchmarking``) stay where they are and keep their own
``find_types_in_modules`` scan in :mod:`sparkrun.core.bootstrap`.

Plugin loading is controlled by the shared ``plugins.enabled`` policy, checked
before import. Existing whole-plugin flags remain migration fallbacks. Enabled
plugins may register separate capability flags, which cannot load a disabled
parent. ``IN_TREE_PLUGIN_FEATURES`` (derived from ``plugin_policy.KNOWN_PLUGINS``)
retains the known module inventory and legacy bindings. External loading additionally requires ``core.external_plugins``.

A broken bundled plugin is logged and skipped so unrelated commands still work.

(There is also no ``sys.path`` manipulation: these are already importable.)
"""

from __future__ import annotations

import importlib
from functools import partial
import logging
import pkgutil
from typing import TYPE_CHECKING

from sparkrun.core.plugin_policy import KNOWN_PLUGINS, application_profile_failure, declared_application_profile_api
from sparkrun.core.registration import load_and_register_plugin
from sparkrun.core.registry_defaults import DeclarationTier

if TYPE_CHECKING:
    from scitrera_app_framework import Variables

logger = logging.getLogger(__name__)

#: Package scanned for first-party plugin subpackages.
IN_TREE_PLUGIN_PACKAGE = "sparkrun.plugins"

#: Plugins shipped and tested with this Sparkrun release. Namespace membership
#: alone is not ownership: separately installed plugins can live there too.
BUNDLED_PLUGIN_NAMES = frozenset(known.module for known in KNOWN_PLUGINS.values() if known.bundled)

#: In-tree plugin module name -> the legacy feature flag gating it, derived from
#: :data:`~sparkrun.core.plugin_policy.KNOWN_PLUGINS`.
#:
#: Adding a plugin means adding a ``KNOWN_PLUGINS`` entry. Kept as data rather
#: than read off the plugin because the flag must resolve *before* the import —
#: a plugin that declared its own gate could only be consulted by importing it,
#: which is exactly what the gate is meant to avoid. SparkRoute and ColdSnap are
#: not bundled, but keep their pre-import gates for users who install their
#: packages separately into this plugin namespace. Such packages load at the
#: out-of-tree declaration tier (in_tree_declaration_tier).
IN_TREE_PLUGIN_FEATURES: dict[str, str | None] = {known.module: known.legacy_flag for known in KNOWN_PLUGINS.values()}


def plugin_application_profile_api(name: str) -> int | None:
    """The ``application_profile_api`` an in-namespace plugin declares, read without importing it."""
    from sparkrun.core.plugin_metadata import plugin_origin

    return declared_application_profile_api(plugin_origin(IN_TREE_PLUGIN_PACKAGE + "." + name))


def plugin_application_profile_failure(name: str) -> str | None:
    """Why in-namespace plugin *name* cannot load under the active application profile."""
    return application_profile_failure(name, plugin_application_profile_api(name))


def in_tree_declaration_tier(module: str) -> DeclarationTier:
    """Declaration tier for a module found in the in-tree plugin namespace.

    Only release-owned plugins at their shipped location get ``IN_TREE`` (which
    lets a registry declaration ship trusted). A separately installed package
    occupying ``sparkrun.plugins.<name>`` is third-party code like any other.
    """
    from sparkrun.core.managed_plugins import physical_origin
    from sparkrun.core.plugin_metadata import is_bundled_plugin

    return DeclarationTier.IN_TREE if is_bundled_plugin(module, physical_origin(module)) else DeclarationTier.OUT_OF_TREE


def plugin_feature_flag(name: str) -> str | None:
    """Return the feature flag gating in-tree plugin *name*, or ``None``."""
    return IN_TREE_PLUGIN_FEATURES.get(name)


def in_tree_plugin_enabled(name, v=None, *, config=None):
    from sparkrun.core.features import _gate_config
    from sparkrun.core.plugin_policy import plugin_state
    from sparkrun.core.plugin_manifest import read_declarations
    from sparkrun.core.plugin_metadata import plugin_origin

    if name not in IN_TREE_PLUGIN_FEATURES:
        return False
    module = IN_TREE_PLUGIN_PACKAGE + "." + name
    try:
        declarations = read_declarations(plugin_origin(module), module)
    except (OSError, ValueError):
        declarations = {}  # Keep metadata-only disable/recovery available.
    legacy = plugin_feature_flag(name) if not declarations else None
    return plugin_state(name, config if config is not None else _gate_config(v), legacy_feature=legacy).enabled


def iter_in_tree_plugin_names(package: str | None = None) -> list[str]:
    """Return the plugin subpackage names shipped under *package*.

    Discovery only — nothing is imported, so this answers "what ships here?"
    even for plugins whose feature flag is off. Shared with
    :func:`load_in_tree_plugins` so a listing can never name a plugin the
    loader would skip.

    *package* defaults to :data:`IN_TREE_PLUGIN_PACKAGE` resolved at call time,
    not at definition time, so a test that redirects the package redirects both
    this and the loader together.
    """
    package = package or IN_TREE_PLUGIN_PACKAGE
    try:
        root = importlib.import_module(package)
    except Exception:
        logger.exception("Could not import the in-tree plugin package %r", package)
        return []
    return [info.name for info in pkgutil.iter_modules(getattr(root, "__path__", [])) if not info.name.startswith("_")]


def load_in_tree_plugins(v: "Variables", package: str | None = None) -> list[str]:
    """Import and register every plugin under *package*.

    Args:
        v: The initialized SAF :class:`~scitrera_app_framework.Variables`.
        package: Dotted package to scan.  Overridable for tests; ``None``
            resolves :data:`IN_TREE_PLUGIN_PACKAGE` at call time.

    Returns:
        The plugin module names that loaded, in discovery order.
    """
    package = package or IN_TREE_PLUGIN_PACKAGE
    from sparkrun.core.plugin_identity import reserve_identities

    identities = in_tree_plugin_identities(v, package)
    rejected = reserve_identities(identities.values())
    identities = {name: identity for name, identity in identities.items() if identity not in rejected}
    loaded: list[str] = []
    for name in iter_in_tree_plugin_names(package):
        flag = plugin_feature_flag(name)
        if name not in IN_TREE_PLUGIN_FEATURES:
            # Skipping either way (an unregistered flag resolves off), but
            # silently — and the reason would be far from the cause. Shipping
            # an in-tree plugin means an IN_TREE_PLUGIN_FEATURES entry and a
            # flag registered in core.features.
            logger.error(
                "In-tree plugin %r is not bound to a registered feature flag (%s); "
                "add an IN_TREE_PLUGIN_FEATURES entry and register the flag in sparkrun.core.features",
                name,
                flag or "no entry",
            )
            continue
        if not in_tree_plugin_enabled(name, v):
            logger.debug("Skipping in-tree plugin %r (feature %r off)", name, flag)
            continue

        failure = plugin_application_profile_failure(name)
        if failure:
            logger.warning("Cannot load in-tree integration %s: %s", name, failure)
            continue

        dotted = "%s.%s" % (package, name)
        if name not in identities:
            continue  # Preserve the identity discovery diagnostic.
        try:
            load_and_register_plugin(
                partial(importlib.import_module, dotted),
                v,
                tier=in_tree_declaration_tier(dotted),
                source=("in-tree", dotted, None),
                identity=identities[name],
            )
        except Exception:
            logger.exception("Failed to load in-tree plugin %r", dotted)
            continue
        loaded.append(name)

    if loaded:
        logger.debug("Loaded in-tree plugins: %s", loaded)
    return loaded


def in_tree_plugin_identities(v, package=None):
    from sparkrun.core.plugin_identity import discovered_identity
    from sparkrun.core.managed_plugins import physical_origin
    from sparkrun.core.registration import record_plugin_load_failure

    package = package or IN_TREE_PLUGIN_PACKAGE
    identities = {}
    from sparkrun.core.plugin_manifest import read_declarations
    from sparkrun.core.features import declare_plugin_features

    for name in iter_in_tree_plugin_names(package):
        if name not in IN_TREE_PLUGIN_FEATURES:
            continue
        dotted = package + "." + name
        source = ("in-tree", dotted, None)
        try:
            origin = physical_origin(dotted)
            identity = discovered_identity(name, dotted, source, origin=origin)
            declare_plugin_features(identity, read_declarations(origin, dotted))
            if in_tree_plugin_enabled(name, v) and not plugin_application_profile_failure(name):
                identities[name] = identity
        except (ValueError, OSError) as exc:
            record_plugin_load_failure(source, exc)
            logger.warning("Skipping plugin %s: %s", dotted, exc)

    return identities


__all__ = [
    "IN_TREE_PLUGIN_FEATURES",
    "IN_TREE_PLUGIN_PACKAGE",
    "iter_in_tree_plugin_names",
    "load_in_tree_plugins",
    "plugin_feature_flag",
]
