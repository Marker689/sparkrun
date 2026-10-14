"""Load out-of-tree plugins from user-configured directories.

sparkrun's built-in plugins are discovered by scanning packages inside the
``sparkrun`` distribution (see :func:`sparkrun.core.bootstrap.init_sparkrun`).
This module extends that discovery to directories the user lists under
``plugins.paths`` in ``config.yaml`` — a home for executors, transports,
runtimes, schedulers, builders, and benchmarking frameworks that live outside
the open-source tree.

Duplicate top-level module names across configured directories are rejected
before importing them. An already imported module from a different directory
is also rejected; inventory reports the actual loaded origin.

For each configured directory:

1. The directory is prepended to ``sys.path``.
2. Every importable top-level module/package in it is imported.
3. The imported module is scanned for concrete subclasses of the SAF plugin
   base types and each is registered via ``register_plugin`` — exactly what
   ``init_sparkrun`` does for the built-in packages.
4. If the module defines a top-level ``register(v)`` callable, it is invoked.
   This is the escape hatch for the still-in-process registries
   (:mod:`sparkrun.platforms`, :mod:`sparkrun.orchestration.collectives`) whose
   plugins register by calling ``register_platform`` / ``get_backend`` wiring
   rather than by SAF subclass discovery.

Gated off by default on every channel behind the ``core.external_plugins``
feature flag: when it resolves off, the config-driven auto-load path returns
immediately without reading ``plugins.paths`` (let alone importing anything), so
a stock install pays zero cost and exposes zero extra surface.  Enable it with
``sparkrun setup features enable core.external_plugins`` (or
``features.core.external_plugins: true`` in ``config.yaml``).  Because the config
file and the plugin directories are user-owned, loading them is trusted by
definition — the same trust model as a pip-installed package or a pytest
plugin.  A single broken plugin logs and is skipped; it never breaks startup.

Separately, the ``SPARKRUN_NO_EXTERNAL_PLUGINS`` env var is a hard kill-switch
for the config-driven path — needed for test isolation, since pytest reads the
developer's real config (see ``_DISABLE_ENV`` below).
"""

from __future__ import annotations

from sparkrun.core.registration import enlist_registry_state, load_and_register_plugin, PluginCompatibilityError

import importlib
from functools import partial
from importlib.machinery import PathFinder
import inspect
import logging
import os
import pkgutil
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from scitrera_app_framework import ext_parse_bool, register_plugin
from scitrera_app_framework.util import find_types_in_modules

from sparkrun.core.features import FEATURE_CORE_EXTERNAL_PLUGINS

if TYPE_CHECKING:
    from scitrera_app_framework import Variables

    from sparkrun.core.registry_defaults import DeclarationTier

logger = logging.getLogger(__name__)

# Hard test/CI kill-switch for the config-driven auto-load path — set truthy to
# make it inert regardless of the ``core.external_plugins`` flag. Needed because
# tests read the developer's REAL ~/.config/sparkrun (the SAF stateful root
# isn't "ready" under pytest), so a developer who enabled the flag would
# otherwise load their real plugins mid-suite. Set by ``conftest.isolate_stateful``.
# Explicitly-passed ``paths`` (a plugin's own tests) bypass this guard.
_DISABLE_ENV = "SPARKRUN_NO_EXTERNAL_PLUGINS"


def _external_plugins_disabled() -> bool:
    return bool(ext_parse_bool(os.environ.get(_DISABLE_ENV, "")))


def _plugin_base_types() -> list[type]:
    """Return the SAF plugin base classes an external module may subclass.

    Imported lazily to avoid import cycles (this module is reached from
    ``core.bootstrap``, which the plugin packages import back).
    """
    from sparkrun.runtimes.base import RuntimePlugin
    from sparkrun.benchmarking.base import BenchmarkingPlugin
    from sparkrun.builders.base import BuilderPlugin
    from sparkrun.orchestration.executors._base import Executor
    from sparkrun.core.scheduler import Scheduler
    from sparkrun.transports.base import Transport
    from sparkrun.orchestration.telemetry._base import TelemetryProvider

    return [RuntimePlugin, BenchmarkingPlugin, BuilderPlugin, Executor, Scheduler, Transport, TelemetryProvider]


# Plugin classes that select via a ``*_name`` attribute must set it non-blank to
# be registerable; a blank value marks a shared/abstract base (mirrors the
# scheduler/transport guards in ``core.bootstrap``).
_SELECTOR_ATTRS = ("scheduler_name", "transport_name")


def _is_registerable(cls: type) -> bool:
    for attr in _SELECTOR_ATTRS:
        if hasattr(cls, attr) and not getattr(cls, attr, ""):
            return False
    return True


# Modules this process actually loaded as plugins, keyed by dotted module name.
# Recorded here — the one point both loaders pass through — so the inventory
# (``core.plugin_inventory``) reports a version only for a module *we* imported
# as a plugin. ``sys.modules`` is not a substitute: an external plugin's
# top-level name may also be importable for unrelated reasons, and reporting
# that module's ``__version__`` would attribute a stranger's version to a
# plugin sparkrun never loaded.
_LOADED_PLUGIN_MODULES: dict[str, ModuleType] = {}

enlist_registry_state(globals(), "_LOADED_PLUGIN_MODULES")
_REGISTERED_MODULES: set[tuple[int, str]] = set()

enlist_registry_state(globals(), "_REGISTERED_MODULES")


def _module_matches_directory(module: ModuleType, name: str, path: Path) -> bool:
    spec = PathFinder.find_spec(name, [str(path.resolve())])
    origin = getattr(module, "__file__", None)
    return bool(spec and spec.origin and origin and Path(spec.origin).resolve() == Path(origin).resolve())


def loaded_plugin_module(dotted: str, *, path: Path | None = None) -> "ModuleType | None":
    """Return a successfully loaded plugin, optionally restricted to its directory."""
    module = _LOADED_PLUGIN_MODULES.get(dotted)
    if module is not None and path is not None and not _module_matches_directory(module, dotted, path):
        return None
    return module


def _import_directory_plugin(name: str, path: Path, origins: list[Path]) -> ModuleType:
    if len(origins) > 1:
        raise ValueError("Plugin module %r occurs in multiple directories: %s" % (name, ", ".join(map(str, origins))))
    previous = sys.modules.get(name)
    if previous is not None and not _module_matches_directory(previous, name, path):
        raise ValueError("Plugin module %r is already imported from outside %s" % (name, path))
    module = importlib.import_module(name)
    if not _module_matches_directory(module, name, path):
        raise ValueError("Plugin module %r resolved outside %s" % (name, path))
    return module


def clear_loaded_plugin_modules() -> None:
    """Forget every recorded plugin module (test isolation)."""
    from sparkrun.core.registration import clear_plugin_load_failures

    from sparkrun.core.plugin_identity import clear_plugin_identities

    from sparkrun.core.features import clear_plugin_features
    from sparkrun.proxy.gateway import _GATEWAY_LOADERS, _GATEWAY_OWNERS, GATEWAY_FEATURE_FLAGS

    clear_plugin_features()
    for name in _GATEWAY_OWNERS:
        _GATEWAY_LOADERS.pop(name, None)
        GATEWAY_FEATURE_FLAGS.pop(name, None)
    _GATEWAY_OWNERS.clear()
    clear_plugin_identities()
    from sparkrun.diagnostics.plugin import _PLUGINS as diagnostic_plugins

    diagnostic_plugins.clear()
    clear_plugin_load_failures()
    _LOADED_PLUGIN_MODULES.clear()
    _REGISTERED_MODULES.clear()


def iter_plugin_module_names(path: Path) -> list[str]:
    """Return the importable top-level plugin module names inside *path*.

    The single definition of "what counts as a plugin module in a directory",
    shared by :func:`load_external_plugins` and the inventory, so a listing can
    never offer a name the loader would skip (or omit one it would load).
    """
    if not path.is_dir():
        return []
    return [info.name for info in pkgutil.iter_modules([str(path)]) if not info.name.startswith("_")]


def _scan_module_for_plugins(module, base: type) -> list[type]:
    """Return concrete *base* subclasses reachable from *module*.

    Scans the imported module object directly (covers single-file ``plugin.py``
    drop-ins) and, for packages, recurses into submodules via
    ``find_types_in_modules`` — which raises on a non-package, so it is only
    used when ``module`` has ``__path__``.
    """
    found: dict[str, type] = {}

    def _consider(attr) -> None:
        if inspect.isclass(attr) and issubclass(attr, base) and attr is not base and not inspect.isabstract(attr):
            found["%s.%s" % (attr.__module__, attr.__qualname__)] = attr

    for attr_name in dir(module):
        _consider(getattr(module, attr_name, None))

    if hasattr(module, "__path__") and getattr(module, "SPARKRUN_PLUGIN_SCAN_SUBMODULES", True):
        for cls in find_types_in_modules(module.__name__, base):
            _consider(cls)

    return list(found.values())


def load_plugin_module(module, v: "Variables", *, tier: "DeclarationTier | None" = None, strict: bool = False) -> bool:
    """Register everything *module* contributes: SAF subclasses, then ``register(v)``.

    Shared with :mod:`sparkrun.core.in_tree_plugins` — in-tree and out-of-tree
    plugins differ only in where their modules come from, so they must not
    differ in what counts as a registration.

    Returns True only after complete registration. Ordinary failures return
    False after rollback unless strict=True; conflicts and interrupts propagate.
    Source loaders use core.registration.load_and_register_plugin to include
    import-time contributions in the same transaction.

    Args:
        module: The imported plugin module or package.
        v: The initialized SAF :class:`~scitrera_app_framework.Variables`.
        tier: Provenance to attribute this module's registry declarations to
            (see :mod:`sparkrun.core.registry_defaults`).  Supplied by the
            *loader* because a plugin must not be able to claim its own tier —
            an in-tree tier is what lets a declaration ship trusted.  Ambient
            for the duration of the load rather than an argument, since the
            plugin makes the registration call, not us.  Defaults to
            ``OUT_OF_TREE``: least privilege.
    """
    from sparkrun.core.installed_plugins import PluginConflictError

    try:
        from sparkrun.core.plugin_identity import discovered_identity, registered_plugin_identity

        identity = registered_plugin_identity(module.__name__) if loaded_plugin_module(module.__name__) is module else None
        identity = identity or discovered_identity(
            module.__name__.split(".")[-1],
            module.__name__,
            ("module", module.__name__, getattr(module, "__file__", None)),
            origin=getattr(module, "__file__", None),
        )
        load_and_register_plugin(lambda: module, v, tier=tier, identity=identity)
    except Exception as exc:
        if strict or isinstance(exc, PluginConflictError):
            raise
        logger.exception("Plugin registration failed for %r", module.__name__)
        return False
    return True


def _register_plugin_module(module, v, *, tier=None, declarations=None) -> None:
    from sparkrun.core.registry_defaults import DeclarationTier, declaring_tier
    from sparkrun.core.features import register_feature
    from sparkrun.core.installed_plugins import claim_implementation

    if _LOADED_PLUGIN_MODULES.get(module.__name__) is module and (id(v), module.__name__) in _REGISTERED_MODULES:
        return
    from sparkrun.core.plugin_manifest import read_declarations
    from sparkrun.core.features import FeatureFlag

    declarations = declarations if declarations is not None else read_declarations(getattr(module, "__file__", None), module.__name__)
    for name, options in declarations.get("features", {}).items():
        register_feature(FeatureFlag(name=name, **options))
    for definition in getattr(module, "FEATURE_DEFINITIONS", ()):
        register_feature(definition)

    with declaring_tier(tier or DeclarationTier.OUT_OF_TREE):
        for base in _plugin_base_types():
            for cls in _scan_module_for_plugins(module, base):
                if not _is_registerable(cls):
                    continue
                claim_implementation(cls, v)
                register_plugin(cls, v=v)
                logger.debug("Registered plugin %s from %s", cls.__name__, module.__name__)
        hook = getattr(module, "register", None)
        if callable(hook):
            hook(v)
    _LOADED_PLUGIN_MODULES[module.__name__] = module
    _REGISTERED_MODULES.add((id(v), module.__name__))


def _configured_paths(v: "Variables") -> list[Path]:
    """Resolve ``plugins.paths`` from the active config root.

    Reads from ``get_config_root(v)/config.yaml`` rather than the real user
    config unconditionally, so the ``isolate_stateful`` test fixture (which
    redirects the SAF stateful root to a tmp dir) keeps external plugin loading
    inert in the test suite.
    """
    from sparkrun.core.config import SparkrunConfig, resolve_config_path

    config_path = resolve_config_path(v)
    return SparkrunConfig(config_path=config_path).external_plugin_paths


def load_external_plugins(v: "Variables", paths: "list[Path] | None" = None) -> list[str]:
    """Import and register plugins from user-configured directories.

    Args:
        v: The initialized SAF :class:`~scitrera_app_framework.Variables`.
        paths: Explicit directories to load (mainly for tests).  When ``None``,
            resolved from ``plugins.paths`` in the active config.

    Returns:
        The top-level module names that were loaded — ``[]`` when nothing is
        configured.
    """
    paths = configured_plugin_paths(v, paths)
    from sparkrun.core.features import _gate_config

    identities = directory_plugin_identities(paths, config=_gate_config(v))
    from sparkrun.core.plugin_identity import preflight_identities

    preflight_identities(identities.values())
    # Inspect all candidates before imports so duplicate module names cannot
    # silently select one directory through sys.modules or sys.path precedence.
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    candidates = {path: iter_plugin_module_names(path) for path in paths}
    origins: dict[str, list[Path]] = {}
    for path, names in candidates.items():
        for name in names:
            origins.setdefault(name, []).append(path)

    loaded: list[str] = []
    for path in paths:
        if not path.is_dir():
            logger.warning("Skipping external plugin path (not a directory): %s", path)
            continue
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
        for name in candidates[path]:
            try:
                identity = identities.get((path, name))
                if identity is None:
                    continue
                load_and_register_plugin(
                    partial(_import_directory_plugin, name, path, origins[name]), v, source=("external", name, str(path)), identity=identity
                )
            except PluginCompatibilityError as error:
                logger.warning("Skipping plugin %s", error)
                logger.debug("Plugin compatibility failure in %s from %s", name, path, exc_info=True)
                continue
            except Exception:
                logger.exception("Failed to load external plugin module %r from %s", name, path)
                continue
            loaded.append(name)
    return loaded


def directory_plugin_identities(paths, config=None):
    from sparkrun.core.plugin_identity import discovered_identity
    from sparkrun.core.registration import record_plugin_load_failure

    identities = {}
    for path in dict.fromkeys(path.resolve() for path in paths):
        for name in iter_plugin_module_names(path):
            source = ("external", name, str(path))
            try:
                spec = PathFinder.find_spec(name, [str(path)])
                identity = discovered_identity(name, name, source, origin=spec.origin if spec else None)
                from sparkrun.core.plugin_policy import plugin_state
                from sparkrun.core.features import declare_plugin_features
                from sparkrun.core.plugin_manifest import read_declarations

                declare_plugin_features(identity, read_declarations(spec.origin if spec else None, name))
                if plugin_state(identity.plugin_id, config, source="directory").enabled:
                    identities[path, name] = identity
            except (ValueError, OSError) as exc:
                record_plugin_load_failure(source, exc)
                logger.warning("Skipping plugin %s: %s", name, exc)
    return identities


def configured_plugin_paths(v, paths=None):
    if paths is None:
        # Hard test/CI override first, then the product feature flag (off by
        # default on every channel). Either short-circuits before reading
        # plugins.paths — let alone importing anything. Explicit ``paths=``
        # (programmatic / a plugin's own tests) bypass both.
        if _external_plugins_disabled():
            logger.debug("External plugin auto-load disabled via %s", _DISABLE_ENV)
            return []
        from sparkrun.core.config import SparkrunConfig, resolve_config_path
        from sparkrun.core.plugin_store import PluginStore
        from sparkrun.core.plugin_manifest import PluginError
        from sparkrun.core.features import is_feature_enabled

        try:
            config, _records = PluginStore(SparkrunConfig(resolve_config_path(v))).snapshot()
        except PluginError as exc:
            logger.warning("External plugin loading deferred: %s", exc)
            return []
        if not is_feature_enabled(FEATURE_CORE_EXTERNAL_PLUGINS.name, config=config):
            logger.debug("External plugin auto-load disabled (feature %r off)", FEATURE_CORE_EXTERNAL_PLUGINS.name)
            return []
        paths = config.external_plugin_paths

    return list(paths)
