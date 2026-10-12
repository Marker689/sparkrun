"""Framework-free, explicit daemon operation contracts and lazy handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable
from sparkrun.core.registration import enlist_registry_state


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
    owner: str = "core"
    resources: str = "auto"

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_.]*", self.name) or self.kind not in {"query", "job", "stream"}:
            raise ValueError("Invalid daemon operation")
        if self.resources not in {"auto", "controller", "none"}:
            raise ValueError("Invalid resource policy")
        if self.scope not in {"read", "operate", "admin"} or self.version < 1:
            raise ValueError("Invalid operation scope/version")
        if self.request_schema.get("type") != "object" or self.request_schema.get("additionalProperties") is not False:
            raise ValueError("Operation requests must declare a closed object schema")


_OPERATIONS = {}
enlist_registry_state(globals(), "_OPERATIONS")


def register_daemon_op(op):
    previous = _OPERATIONS.get(op.name)
    if previous is not None:
        from dataclasses import replace

        if replace(previous, handler_loader=op.handler_loader) != op:
            raise ValueError(f"Daemon operation {op.name!r} is already registered")
        return previous
    _OPERATIONS[op.name] = op
    return op


def registered_daemon_ops():
    return tuple(_OPERATIONS[name] for name in sorted(_OPERATIONS))


def get_daemon_op(name):
    try:
        return _OPERATIONS[name]
    except KeyError:
        raise LookupError(f"Unknown operation: {name}") from None


_RESERVED_ROUTE_ROOTS = frozenset(
    {"health", "info", "ops", "jobs", "events", "streams", "maintenance", "shutdown", "snapshots", "reservations"}
)


def operation_http_path(op: DaemonOp) -> str | None:
    """Literal POST alias for a query/job; streams retain dedicated endpoints."""
    if op.kind == "stream":
        return None
    parts = op.name.split(".")
    if not all(parts) or parts[0] in _RESERVED_ROUTE_ROOTS:
        raise ValueError(f"Daemon operation {op.name!r} cannot use a malformed or reserved HTTP alias")
    return "/v1/" + "/".join(parts)


@dataclass(frozen=True)
class DaemonRoute:
    """JSON envelope adapter. Dispatch always uses the selected operation's policy.

    The lazy decoder returns ``(operation_name, arguments)``. It is a pure,
    bounded transformation, not an ASGI handler or a second execution backend.
    """

    path: str
    decode_loader: Callable
    operations: tuple[str, ...]
    request_schema: dict
    owner: str
    feature_flag: str | None = None

    def __post_init__(self):
        if not re.fullmatch(r"/v1/[a-z][a-z0-9_/-]*", self.path) or self.path.split("/")[2] in _RESERVED_ROUTE_ROOTS:
            raise ValueError("Route must use an unreserved literal /v1 path")
        if not self.operations or not self.owner or self.request_schema.get("additionalProperties") is not False:
            raise ValueError("Routes must declare allowed operations, owner, and a closed JSON request")


_ROUTES = {}
enlist_registry_state(globals(), "_ROUTES")


def register_daemon_route(route):
    from dataclasses import replace

    previous = _ROUTES.get(route.path)
    if previous is not None:
        if replace(previous, decode_loader=route.decode_loader) != route:
            raise ValueError(f"Daemon route {route.path!r} is already registered")
        return previous
    _ROUTES[route.path] = route
    return route


def registered_daemon_routes():
    return tuple(_ROUTES[path] for path in sorted(_ROUTES))
