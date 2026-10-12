"""Framework-free daemon contracts bound to the registering plugin's identity."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
import re
from typing import Callable

from sparkrun.core.registration import enlist_registry_state
from sparkrun.core.plugin_identity import PluginIdentity, current_registration_owner, normalize_plugin_id

_LOCAL_NAME = r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*"
_PLUGIN_ID = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*"
_RESERVED_ROUTE_ROOTS = frozenset(
    {"health", "info", "ops", "jobs", "events", "streams", "maintenance", "shutdown", "snapshots", "reservations", "plugins"}
)
# Reviewed compatibility only. New plugins never reserve global legacy names.
_K8S_LEGACY = frozenset(
    {
        "ensure_kubectl",
        "cluster_info",
        "list_nodes",
        "kueue_status",
        "setup_kueue",
        "configure_service_account",
        "jobset_status",
        "stop_jobset",
    }
)


@dataclass(frozen=True)
class DaemonOp:
    name: str
    handler_loader: Callable
    request_schema: dict = field(default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": False})
    result_schema: dict = field(default_factory=dict)
    kind: str = "job"
    scope: str = "operate"
    version: int = 1
    feature_flag: str | None = None
    description: str = ""
    secret: bool = False
    owner: str | None = None
    resources: str = "auto"
    local_name: str | None = None

    def __post_init__(self):
        if not re.fullmatch(rf"(?:{_LOCAL_NAME}|plugins\.{_PLUGIN_ID}\.{_LOCAL_NAME})", self.name) or self.kind not in {
            "query",
            "job",
            "stream",
        }:
            raise ValueError("Invalid daemon operation")
        if self.resources not in {"auto", "controller", "none"}:
            raise ValueError("Invalid resource policy")
        if self.scope not in {"read", "operate", "admin"} or self.version < 1:
            raise ValueError("Invalid operation scope/version")
        if self.request_schema.get("type") != "object" or self.request_schema.get("additionalProperties") is not False:
            raise ValueError("Operation requests must declare a closed object schema")


_OPERATIONS = {}
_ALIASES = {}
_ROUTES = {}
enlist_registry_state(globals(), "_OPERATIONS", "_ALIASES", "_ROUTES")


def _owner(declared):
    owner = current_registration_owner()
    if owner is None:
        raise ValueError("Daemon registration requires a plugin loader or application_daemon_registration context")
    name = owner.plugin_id if isinstance(owner, PluginIdentity) else owner
    if declared is not None and declared != name:
        raise ValueError("Daemon owner %r does not match registering owner %r" % (declared, name))
    return owner, name


def _handler_identity(loader):
    """Stable Python handler identity across equivalent partial/lambda declarations."""
    if isinstance(loader, partial):
        return (_handler_identity(loader.func), loader.args, loader.keywords)
    if hasattr(loader, "__code__"):
        cells = tuple(cell.cell_contents for cell in (loader.__closure__ or ()))
        return (loader.__module__, loader.__qualname__, loader.__code__, loader.__defaults__, loader.__kwdefaults__, cells)
    return loader


def _register(registry, key, value, loader_field):
    previous = registry.get(key)
    if previous is not None:
        loader = getattr(value, loader_field)
        if replace(previous, **{loader_field: loader}) != value or _handler_identity(getattr(previous, loader_field)) != _handler_identity(
            loader
        ):
            raise ValueError("Daemon declaration %r is already registered by %s" % (key, previous.owner))
        return previous
    registry[key] = value
    return value


def register_daemon_op(op):
    owner, owner_name = _owner(op.owner)
    if op.local_name is not None:
        raise ValueError("Register a local declaration, not an already bound daemon operation")
    name = op.name
    if isinstance(owner, PluginIdentity):
        # Only the reviewed Kubernetes migration accepts an old global spelling.
        if owner.plugin_id == "k8s" and owner.module == "sparkrun.plugins.k8s" and name.startswith("k8s.") and name[4:] in _K8S_LEGACY:
            name = name[4:]
        if name.split(".")[0] == "plugins" or not re.fullmatch(_LOCAL_NAME, name):
            raise ValueError("Plugins must declare local operation names")
        bound = replace(op, name=f"plugins.{owner_name}.{name}", owner=owner_name, local_name=name)
    else:
        if name.split(".")[0] in _RESERVED_ROUTE_ROOTS:
            raise ValueError("Reserved core operation namespace: %s" % name)
        bound = replace(op, owner="core", local_name=name)
    if bound.name in _ALIASES:
        raise ValueError("Operation name is reserved by a legacy alias: " + bound.name)
    bound = _register(_OPERATIONS, bound.name, bound, "handler_loader")
    if isinstance(owner, PluginIdentity) and owner.plugin_id == "k8s" and owner.module == "sparkrun.plugins.k8s" and name in _K8S_LEGACY:
        alias = "k8s." + name
        if alias in _OPERATIONS or (alias in _ALIASES and _ALIASES[alias] != bound.name):
            raise ValueError("Conflicting legacy daemon operation: " + alias)
        _ALIASES[alias] = bound.name
    return bound


def register_plugin_daemon_op(op):
    if not isinstance(current_registration_owner(), PluginIdentity):
        raise ValueError("Plugin operation registration requires a plugin or application identity")
    return register_daemon_op(op)


def registered_daemon_ops():
    return tuple(_OPERATIONS[name] for name in sorted(_OPERATIONS))


def get_daemon_op(name):
    try:
        return _OPERATIONS[_ALIASES.get(name, name)]
    except KeyError:
        raise LookupError(f"Unknown operation: {name}") from None


def legacy_operation_names(op):
    return tuple(sorted(name for name, target in _ALIASES.items() if target == op.name and _OPERATIONS.get(target) is op))


def operation_http_path(op: DaemonOp) -> str | None:
    if op.kind == "stream":
        return None
    if op.owner not in (None, "core"):
        if not op.local_name:
            raise ValueError("HTTP aliases require a bound plugin operation")
        return f"/v1/plugins/{op.owner}/ops/" + op.local_name.replace(".", "/")
    return "/v1/" + op.name.replace(".", "/")


@dataclass(frozen=True)
class DaemonRoute:
    """Bounded JSON adapter. Plugins declare a relative path and local targets.

    Prefix core targets with ``core:``; cross-plugin targets use their canonical
    ``plugins.<id>.<name>``. The decoder must return a declared target spelling.
    """

    path: str
    decode_loader: Callable
    operations: tuple[str, ...]
    request_schema: dict
    owner: str | None = None
    feature_flag: str | None = None
    local_path: str | None = None
    operation_map: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.path.startswith("/"):
            valid = re.fullmatch(r"/v1/[a-z][a-z0-9_-]*(?:/[a-z0-9_][a-z0-9_-]*)*", self.path)
            if not valid or (self.local_path is None and self.path.split("/")[2] in _RESERVED_ROUTE_ROOTS):
                raise ValueError("Route must use an unreserved literal /v1 path")
        elif not re.fullmatch(r"[a-z][a-z0-9_-]*(?:/[a-z0-9_][a-z0-9_-]*)*", self.path):
            raise ValueError("Plugin routes require a literal relative path")
        if (
            not self.operations
            or self.request_schema.get("type") != "object"
            or self.request_schema.get("additionalProperties") is not False
        ):
            raise ValueError("Routes must declare allowed operations and a closed JSON request")


def _route_target(target, owner):
    if target.startswith("core:"):
        target = target[5:]
        if not re.fullmatch(_LOCAL_NAME, target) or target.split(".")[0] in _RESERVED_ROUTE_ROOTS:
            raise ValueError("Invalid core operation target")
        return target
    if target.startswith("plugins."):
        if not re.fullmatch(rf"plugins\.({_PLUGIN_ID})\.{_LOCAL_NAME}", target):
            raise ValueError("Invalid qualified plugin operation target")
        normalize_plugin_id(target.split(".")[1])
        return target
    if not re.fullmatch(_LOCAL_NAME, target):
        raise ValueError("Invalid local operation target")
    return f"plugins.{owner}.{target}" if owner != "core" else target


def register_daemon_route(route):
    owner, owner_name = _owner(route.owner)
    if route.local_path is not None or route.operation_map:
        raise ValueError("Register an unbound route declaration")
    if isinstance(owner, PluginIdentity):
        if route.path.startswith("/"):
            raise ValueError("Plugin routes require relative paths; use register_plugin_daemon_route")
        path = f"/v1/plugins/{owner_name}/routes/{route.path}"
    else:
        if not route.path.startswith("/v1/"):
            raise ValueError("Core routes require an absolute /v1 path")
        path = route.path
    mapping = tuple((target, _route_target(target, owner_name)) for target in route.operations)
    bound = replace(
        route, path=path, owner=owner_name, local_path=route.path, operations=tuple(v for _, v in mapping), operation_map=mapping
    )
    return _register(_ROUTES, path, bound, "decode_loader")


def register_plugin_daemon_route(route):
    if not isinstance(current_registration_owner(), PluginIdentity):
        raise ValueError("Plugin route registration requires a plugin or application identity")
    return register_daemon_route(route)


def resolve_route_operation(route, selected):
    try:
        return dict(route.operation_map)[selected]
    except (KeyError, TypeError):
        raise PermissionError("Adapter selected an undeclared operation") from None


def registered_daemon_routes():
    return tuple(_ROUTES[path] for path in sorted(_ROUTES))
