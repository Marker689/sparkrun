"""Small shared helpers for plugin registration conflicts and rollback state.

Registries enlist their module namespace and container names beside their
storage declarations. Reading through the namespace honors deliberate container
replacement (including test isolation). Registration runs before application
workers; this is not a concurrent registry mutation API.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType
from typing import Any, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from scitrera_app_framework import Variables
    from sparkrun.core.registry_defaults import DeclarationTier

# Module entry-point compatibility, independent of application profile schemas.
PLUGIN_API_VERSION = 1


class PluginCompatibilityError(ValueError):
    """A plugin needs an API update before this host can register it."""

    def __init__(self, module: str, api_version: object):
        declaration = (
            "missing API declaration"
            if api_version is None
            else "unsupported API %r (host requires %s)" % (api_version, PLUGIN_API_VERSION)
        )
        super().__init__(
            "%s: %s. Update the plugin for this host and declare SPARKRUN_PLUGIN_API_VERSION = %s."
            % (module, declaration, PLUGIN_API_VERSION)
        )


K = TypeVar("K")
V = TypeVar("V")
_SOURCES: list[tuple[dict[str, Any], str]] = []
_ACTIVE: ContextVar[tuple[_Snapshot, ...]] = ContextVar("sparkrun_registration_snapshots", default=())


def register_unique(registry: MutableMapping[K, V], key: K, value: V, *, description: str) -> None:
    """Allow identical registration; reject a second provider for the same key."""
    if key in registry and registry[key] != value:
        from sparkrun.core.installed_plugins import PluginConflictError

        raise PluginConflictError("%s %r is already registered" % (description, key))
    registry[key] = value


def enlist_registry_state(namespace: dict[str, Any], *names: str) -> None:
    """Enlist mutable registry containers, including imports during a transaction."""
    for name in names:
        value = namespace[name]
        if not isinstance(value, (dict, list, set)):
            raise TypeError("Registration state must be a dict, list, or set")
        if not any(source is namespace and key == name for source, key in _SOURCES):
            _SOURCES.append((namespace, name))
        for snapshot in _ACTIVE.get():
            snapshot.capture(value)


class _Snapshot:
    def __init__(self):
        self.saved: list[tuple[Any, Any]] = []
        self.seen: set[int] = set()

    def capture(self, value):
        # SAF's pinned adapter is intentionally confined here. Preserve live
        # Variables/plugin identities while restoring their nested containers.
        from scitrera_app_framework import Variables

        if id(value) in self.seen:
            return
        self.seen.add(id(value))
        if isinstance(value, Variables):
            for state in (value._local, value._fallback_defaults, value._type_fns, value._keys):
                self.capture(state)
        elif isinstance(value, dict):
            self.saved.append((value, dict(value)))
            for child in value.values():
                self.capture(child)
        elif isinstance(value, (list, set)):
            self.saved.append((value, value.copy()))
            for child in value:
                self.capture(child)

    def restore(self):
        for value, previous in reversed(self.saved):
            value.clear()
            if isinstance(value, list):
                value.extend(previous)
            else:
                value.update(previous)


@contextmanager
def registry_transaction(*extra_state):
    """Restore enlisted containers on failure; arbitrary plugin I/O is excluded."""
    snapshot = _Snapshot()
    for namespace, name in _SOURCES:
        snapshot.capture(namespace[name])
    for value in extra_state:
        snapshot.capture(value)
    token = _ACTIVE.set((*_ACTIVE.get(), snapshot))
    try:
        yield
    except BaseException:
        snapshot.restore()
        raise
    finally:
        _ACTIVE.reset(token)


# Diagnostic outcomes deliberately survive registration rollback. The key is
# source kind, module/entry-point name, and directory/package identity.
_LOAD_FAILURES: dict[tuple[str, str, str | None], str] = {}


def plugin_load_failure(source: tuple[str, str, str | None]) -> str | None:
    return _LOAD_FAILURES.get(source)


def record_plugin_load_failure(source: tuple[str, str, str | None], error: BaseException) -> None:
    """Record a discovery/origin failure before a module can be imported."""
    _LOAD_FAILURES[source] = format_plugin_failure(error)


def clear_plugin_load_failures() -> None:
    _LOAD_FAILURES.clear()


def format_plugin_failure(error: BaseException) -> str:
    if isinstance(error, PluginCompatibilityError):
        return str(error)
    return "%s: %s" % (type(error).__name__, error)


def load_and_register_plugin(
    loader: Callable[[], ModuleType],
    v: Variables | None,
    *,
    tier: DeclarationTier | None = None,
    source: tuple[str, str, str | None] | None = None,
) -> ModuleType:
    """Import, validate and register one module in a single transaction.

    Discovery and trust tier belong to the caller. Source-keyed failure details
    survive rollback for inventory; a successful retry clears the prior error.
    Every module declares SPARKRUN_PLUGIN_API_VERSION, regardless of source.
    Exceptions propagate after rollback; arbitrary plugin I/O is not undone.
    """
    try:
        with registry_transaction(v):
            from sparkrun.core.external_plugins import _register_plugin_module

            module = loader()
            if not isinstance(module, ModuleType):
                raise TypeError("Plugin loaders must return a module")
            api_version = getattr(module, "SPARKRUN_PLUGIN_API_VERSION", None)
            if type(api_version) is not int or api_version != PLUGIN_API_VERSION:
                raise PluginCompatibilityError(module.__name__, api_version)
            _register_plugin_module(module, v, tier=tier)
            from sparkrun.core.setup_steps import validate_setup_plans

            validate_setup_plans()  # Validate forward references after the whole module registered.
    except BaseException as error:
        if source is not None:
            _LOAD_FAILURES[source] = format_plugin_failure(error)
        raise
    if source is not None:
        _LOAD_FAILURES.pop(source, None)
    return module
