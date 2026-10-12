"""Versioned, closed JSON requests and explicit result DTOs. No server imports."""

from __future__ import annotations

from enum import Enum
import math
from pathlib import Path
import re
from typing import Any

from sparkrun.api._wire_fields import DTO_FIELDS


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


def validate(value, schema, *, path="request", depth=0, request_limits=True):
    """Check declared constraints; implicit request limits do not constrain results."""
    if depth > (20 if request_limits else 25):
        raise ValueError(f"{path}: maximum nesting depth exceeded")
    if "anyOf" in schema:
        for choice in schema["anyOf"]:
            try:
                return validate(value, choice, path=path, depth=depth, request_limits=request_limits)
            except ValueError:
                pass
        raise ValueError(f"{path}: invalid value")
    kind = schema.get("type")
    matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, (list, tuple)),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": type(value) in (int, float),
        "boolean": type(value) is bool,
        "null": value is None,
    }
    if isinstance(kind, list):
        if not any(matches.get(option, False) for option in kind):
            raise ValueError(f"{path}: expected one of {kind}")
    elif kind and not matches.get(kind, False):
        raise ValueError(f"{path}: expected {kind}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        raise ValueError(f"{path}: string too short")
    if isinstance(value, str) and len(value) > schema.get("maxLength", 65536 if request_limits else math.inf):
        raise ValueError(f"{path}: string too long")
    if isinstance(value, str) and "pattern" in schema and not re.search(schema["pattern"], value):
        raise ValueError(f"{path}: invalid format")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if (
            not math.isfinite(value)
            or value < schema.get("minimum", -1e100 if request_limits else -math.inf)
            or value > schema.get("maximum", 1e100 if request_limits else math.inf)
        ):
            raise ValueError(f"{path}: number out of range")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: unsupported value")
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError(f"{path}: keys must be strings")
        properties = schema.get("properties", {})
        unknown = value.keys() - properties.keys()
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError(f"{path}: unknown fields: {', '.join(sorted(unknown))}")
        if missing := set(schema.get("required", ())) - value.keys():
            raise ValueError(f"{path}: missing fields: {', '.join(sorted(missing))}")
        for key, item in value.items():
            validate(
                item,
                properties.get(key, schema.get("additionalProperties") if isinstance(schema.get("additionalProperties"), dict) else {}),
                path=f"{path}.{key}",
                depth=depth + 1,
                request_limits=request_limits,
            )
    if isinstance(value, (tuple, list)):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path}: too few items")
        if len(value) > schema.get("maxItems", 2048 if request_limits else math.inf):
            raise ValueError(f"{path}: too many items")
        for item in value:
            validate(item, schema.get("items", {}), path=path + "[]", depth=depth + 1, request_limits=request_limits)
    return value


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
