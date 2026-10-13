"""Loader-owned identities for plugin contributions; no server dependencies."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import re
import tomllib

from sparkrun.core.registration import enlist_registry_state


def normalize_plugin_id(name: object) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,99}", name) or not name[-1].isalnum() or ".." in name:
        raise ValueError("Invalid plugin identity: %r" % name)
    result = re.sub(r"[_.-]+", "-", name.lower())
    if result in {"core", "plugins"}:
        raise ValueError("Reserved plugin identity: %s" % result)
    return result


@dataclass(frozen=True)
class PluginIdentity:
    plugin_id: str
    module: str
    source: tuple[str, str, str | None]

    def __post_init__(self):
        if normalize_plugin_id(self.plugin_id) != self.plugin_id or not self.module:
            raise ValueError("Plugin identity must be canonical and identify its module")


_IDENTITIES: dict[str, PluginIdentity] = {}
enlist_registry_state(globals(), "_IDENTITIES")
_CURRENT: ContextVar[PluginIdentity | str | None] = ContextVar("sparkrun_plugin_identity", default=None)


def current_registration_owner():
    return _CURRENT.get()


def preflight_identities(identities):
    """Check the whole selected set before reserving any identity or importing code."""
    proposed = dict(_IDENTITIES)
    for identity in identities:
        previous = proposed.get(identity.plugin_id)
        if previous is not None and previous != identity:
            from sparkrun.core.installed_plugins import PluginConflictError
            from sparkrun.core.registration import record_plugin_load_failure

            error = PluginConflictError(
                "Plugin identity %r is claimed by %r and %r" % (identity.plugin_id, previous.source, identity.source)
            )
            record_plugin_load_failure(previous.source, error)
            record_plugin_load_failure(identity.source, error)
            raise error
        proposed[identity.plugin_id] = identity
    _IDENTITIES.update(proposed)


@contextmanager
def plugin_registration(identity: PluginIdentity):
    previous = _CURRENT.get()
    if previous is not None and previous != identity:
        raise ValueError("Cannot change plugin ownership during registration")
    preflight_identities([identity])
    token = _CURRENT.set(identity)
    try:
        yield identity
    finally:
        _CURRENT.reset(token)


@contextmanager
def core_registration():
    if _CURRENT.get() not in (None, "core"):
        raise ValueError("Plugins cannot register core operations")
    token = _CURRENT.set("core")
    try:
        yield
    finally:
        _CURRENT.reset(token)


@contextmanager
def application_daemon_registration():
    """Bind embedded application operations to the active profile's stable namespace."""
    from sparkrun.core.application_profile import get_application_profile
    from sparkrun.core.registration import registry_transaction

    profile = get_application_profile()
    identity = PluginIdentity(normalize_plugin_id("app-" + profile.id), profile.package, ("application", profile.id, profile.package))
    with registry_transaction(), plugin_registration(identity):
        yield identity


def discovered_identity(name, module, source, *, origin=None, match_name=False):
    """Read nearby static metadata only when it names this exact module."""
    plugin_id = normalize_plugin_id(name)
    if origin:
        root = Path(origin).resolve().parent
        # Package-local manifests and conventional repo/{src/,}package layouts.
        for parent in (root, *list(root.parents)[:3]):
            path = parent / "plugin.toml"
            if not path.is_file():
                continue
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            if data.get("module") != module:
                continue
            declared = normalize_plugin_id(data.get("name"))
            if match_name and declared != plugin_id:
                raise ValueError("Plugin manifest identity %r disagrees with entry point %r" % (declared, name))
            plugin_id = declared
            break
    return PluginIdentity(plugin_id, module, source)


def registered_plugin_identity(module_name):
    return next((identity for identity in _IDENTITIES.values() if identity.module == module_name), None)


def clear_plugin_identities():
    """Test isolation, matching clear_loaded_plugin_modules."""
    _IDENTITIES.clear()


def preflight_plugin_sources(v, config=None):
    """Inspect every selected source before the first cross-cutting plugin import."""
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.in_tree_plugins import in_tree_plugin_identities
    from sparkrun.core.external_plugins import configured_plugin_paths, directory_plugin_identities
    from sparkrun.core.installed_plugins import _discover_installed_plugins, installed_plugin_identities
    from sparkrun.core.managed_plugins import managed_plugin_identities

    config = config or SparkrunConfig()
    selected = list(in_tree_plugin_identities(v).values())
    selected.extend(directory_plugin_identities(configured_plugin_paths(v), config=config).values())
    selected.extend(managed_plugin_identities(config).values())
    rows, _ = _discover_installed_plugins(config)
    selected.extend(installed_plugin_identities(rows).values())
    preflight_identities(selected)
