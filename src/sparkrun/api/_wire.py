"""Versioned, closed JSON requests and explicit result DTOs. No server imports."""

from __future__ import annotations

from enum import Enum
import math
from pathlib import Path
import re
from typing import Any

from sparkrun.api._wire_fields import DTO_FIELDS
from sparkrun.core.daemon_schema import validate as validate


class WireEncodingError(TypeError, ValueError):
    """A result type is missing an explicit public codec."""


class ResultContractError(RuntimeError):
    """An operation produced a result outside its declared public contract."""

    code = "invalid_operation_result"


def check_result(op, result, *, redact=None):
    """Encode and validate once, before publishing or caching successful work."""
    try:
        result = to_wire(result)
        if redact is not None:
            result = redact(result)
        validate(result, op.result_schema, path="result", request_limits=False)
    except (ValueError, TypeError) as exc:
        raise ResultContractError("Operation %s violated its result contract" % op.name) from exc
    return result


_SECRET = re.compile(r"(^|_)(password|passwd|secret|token|api_key|private_key|authorization)($|_)", re.I)


def object_schema(properties=None, required=()) -> dict[str, Any]:
    return {"type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": False}


def reject_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET.search(key) and key not in {"api_key_env", "placement_token", "token_duration"}:
                raise ValueError("Secrets must be supplied through a credential prompt, not persisted arguments")
            reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_secrets(item)


def to_wire(value, *, depth=0) -> Any:
    if depth > 25:
        raise ValueError("Result exceeds maximum nesting depth")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return to_wire(value.value, depth=depth + 1)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_wire(v, depth=depth + 1) for v in value]
    if isinstance(value, dict):
        return {
            str(k): (
                "[redacted]"
                if _SECRET.search(str(k)) and str(k) not in {"placement_token", "api_key_env", "token_duration"}
                else to_wire(v, depth=depth + 1)
            )
            for k, v in value.items()
        }
    name = type(value).__module__ + "." + type(value).__qualname__
    if name == "sparkrun.api._models.RecipeSummary":
        return to_wire(value.to_dict(), depth=depth + 1)
    if name == "sparkrun.core.cluster_manager.ClusterStatusResult":
        return to_wire(value.to_dict(), depth=depth + 1)
    if name == "sparkrun.core.timing.Timeline":
        return to_wire(value.export(), depth=depth + 1)
    if name not in DTO_FIELDS:
        raise WireEncodingError(f"No public wire codec registered for {name}")
    result = {key: to_wire(getattr(value, key), depth=depth + 1) for key in DTO_FIELDS[name]}
    if name.endswith((".RunPlan", ".BuildPlan")):
        result["recipe"] = value.recipe.name
        result["runtime"] = value.runtime.runtime_name
        result["cluster"] = value.cluster.name if value.cluster else None
    return result


def error_body(exc):
    from sparkrun.api._errors import SparkrunError
    from sparkrun.core.coordination import ResourceBusy
    from sparkrun.daemon.jobs import Conflict

    known = isinstance(exc, (SparkrunError, ValueError, LookupError, TimeoutError, ResourceBusy, Conflict, ResultContractError))
    attrs = {
        key: to_wire(getattr(exc, key)) for key in ("hosts", "required", "reasons", "reason", "available", "code") if hasattr(exc, key)
    }
    return {
        "detail": str(exc) if known else "Operation failed; see daemon diagnostics",
        "error": {
            "type": type(exc).__name__ if known else "InternalError",
            "retryable": isinstance(exc, (ResourceBusy, TimeoutError)) or (known and getattr(exc, "retryable", False) is True),
            "attrs": attrs,
        },
    }
