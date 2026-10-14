"""Small shared helpers for plugin registration conflicts and rollback state.

Registries enlist their module namespace and container names beside their
storage declarations. Reading through the namespace honors deliberate container
replacement (including test isolation). Registration runs before application
workers; this is not a concurrent registry mutation API.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from types import ModuleType
from typing import Any, NoReturn, TypeVar, TYPE_CHECKING

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


class PluginConflictError(RuntimeError):
    """Two registrations claim one key; import order cannot resolve it."""


K = TypeVar("K")
V = TypeVar("V")
_SOURCES: list[tuple[dict[str, Any], str]] = []
_ACTIVE: ContextVar[tuple[_Snapshot, ...]] = ContextVar("sparkrun_registration_snapshots", default=())

# Conflicts survive registration rollback: the rejected plugin is gone, but a
# launch that would have used it must not proceed as if nothing happened
# (installed_plugins.require_integrations reads this). Recorded where a conflict
# is raised, so merely constructing an error has no side effect.
_CONFLICTS: list[str] = []


def raise_plugin_conflict(message: str) -> NoReturn:
    """Record a registration conflict for launch gating, then raise it."""
    record_plugin_conflict(message)
    raise PluginConflictError(message)


def record_plugin_conflict(message: str) -> None:
    """Record a conflict that a caller contains rather than raises."""
    if message not in _CONFLICTS:
        _CONFLICTS.append(message)


def recorded_plugin_conflicts() -> tuple[str, ...]:
    return tuple(_CONFLICTS)


def clear_plugin_conflicts() -> None:
    _CONFLICTS.clear()


def _callable_identity(value: Any) -> Any:
    """Stable identity for a declared callable, across re-imports of its module.

    A re-imported plugin module creates new function objects with equal code,
    so identity is module, qualified name, code, defaults and closure contents.
    """
    if isinstance(value, partial):
        return ("partial", _callable_identity(value.func), value.args, value.keywords)
    code = getattr(value, "__code__", None)
    if code is None:
        return value
    cells = []
    for cell in getattr(value, "__closure__", None) or ():
        try:
            cells.append(cell.cell_contents)
        except ValueError:  # an empty cell
            cells.append(_EMPTY_CELL)
    return (
        "function",
        getattr(value, "__module__", None),
        getattr(value, "__qualname__", None),
        code,
        getattr(value, "__defaults__", None),
        getattr(value, "__kwdefaults__", None),
        tuple(cells),
    )


_EMPTY_CELL = object()


def registration_equal(left: Any, right: Any) -> bool:
    """Whether two registrations are the same value (the no-op case).

    Dataclasses compare field by field, sequences element-wise and callables by
    :func:`_callable_identity`; everything else uses ``==``.
    """
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        return all(registration_equal(getattr(left, f.name), getattr(right, f.name)) for f in dataclasses.fields(left))
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(registration_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, partial) or hasattr(left, "__code__"):
        return _callable_identity(left) == _callable_identity(right)
    try:
        return bool(left == right)
    except Exception:  # noqa: BLE001 - an unorderable value is simply not equal
        return False


def _describe(value: Any) -> str:
    """Name the provider behind a registration for a conflict diagnostic."""
    owner = getattr(value, "owner", None)
    owner = getattr(owner, "plugin_id", owner)
    if isinstance(owner, str) and owner:
        return owner
    if isinstance(value, tuple):
        return ", ".join(_describe(item) for item in value if item is not None) or "core"
    if isinstance(value, partial):
        return _describe(value.func)
    if isinstance(value, type) or callable(value):
        target = getattr(value, "callback", None) or value
        module, qualname = getattr(target, "__module__", None), getattr(target, "__qualname__", None)
        if module and qualname:
            return "%s.%s" % (module, qualname)
    if dataclasses.is_dataclass(value) or not isinstance(value, (str, int, float, bool)):
        return "%s.%s" % (type(value).__module__, type(value).__qualname__)
    return repr(value)


def claim_registration(
    existing: Any,
    value: Any,
    *,
    description: str,
    key: Any,
    equal: Callable[[Any, Any], bool] | None = None,
) -> bool:
    """Apply the one conflict rule every in-process plugin registry shares.

    Returns ``False`` when *key* is unclaimed (*existing* is ``None``): the
    caller stores *value*. Returns ``True`` when *value* equals the existing
    registration, which makes re-registration a no-op. Anything else, including
    a different value from the same owner, raises :class:`PluginConflictError`.
    """
    if existing is None:
        return False
    if (equal or registration_equal)(existing, value):
        return True
    raise_plugin_conflict(
        "%s %r is already registered by %s; a different registration by %s cannot replace it"
        % (description, key, _describe(existing), _describe(value))
    )


def register_unique(
    registry: MutableMapping[K, V],
    key: K,
    value: V,
    *,
    description: str,
    equal: Callable[[V, V], bool] | None = None,
) -> V:
    """Store *value* under *key* by the shared rule; return the registered value."""
    if not claim_registration(registry.get(key), value, description=description, key=key, equal=equal):
        registry[key] = value
    return registry[key]


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
    identity=None,
    declarations=None,
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

            from sparkrun.core.plugin_identity import plugin_registration

            if identity is None:
                raise ValueError("Plugin loading requires a discovery identity before import")
            with plugin_registration(identity):
                module = loader()
                if not isinstance(module, ModuleType):
                    raise TypeError("Plugin loaders must return a module")
                if module.__name__ != identity.module:
                    raise ValueError("Plugin loader returned %r; expected %r" % (module.__name__, identity.module))
                api_version = getattr(module, "SPARKRUN_PLUGIN_API_VERSION", None)
                if type(api_version) is not int or api_version != PLUGIN_API_VERSION:
                    raise PluginCompatibilityError(module.__name__, api_version)
                _register_plugin_module(module, v, tier=tier, declarations=declarations)
            from sparkrun.core.setup_steps import validate_setup_plans

            validate_setup_plans()  # Validate forward references after the whole module registered.
    except BaseException as error:
        if source is not None:
            _LOAD_FAILURES[source] = format_plugin_failure(error)
        raise
    if source is not None:
        _LOAD_FAILURES.pop(source, None)
    return module
