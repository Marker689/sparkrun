"""Shared, dependency-free JSON Schema subset for daemon declarations and values."""

import json
import math
import re


_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
_ANNOTATIONS = {"title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly", "$comment"}
_COUNTS = {"minLength", "maxLength", "minItems", "maxItems"}
_KEYWORDS = (
    _ANNOTATIONS
    | _COUNTS
    | {"type", "properties", "required", "additionalProperties", "items", "anyOf", "enum", "minimum", "maximum", "pattern"}
)


def _json_equal(left, right):
    """JSON booleans are not numbers, including inside enum objects/arrays."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_equal(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def validate_schema(schema, *, path="schema", depth=0):
    """Reject malformed or unenforced contracts before publishing an operation.

    This is deliberately a subset, shared with ``validate`` below. Annotation
    keywords carry documentation only; unsupported validation keywords are errors.
    Empty schemas permit any bounded JSON value. Boolean schemas are not supported
    (except the implemented additionalProperties policy).
    """
    if depth > 100:
        raise ValueError(f"{path}: maximum schema nesting depth exceeded")
    if not isinstance(schema, dict) or any(not isinstance(key, str) for key in schema):
        raise ValueError(f"{path}: expected a schema object with string keys")
    if unknown := schema.keys() - _KEYWORDS:
        raise ValueError(f"{path}: unsupported schema keywords: {', '.join(sorted(unknown))}")
    if depth == 0:
        try:
            json.dumps(schema, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f"{path}: schema must contain finite JSON values") from exc
    for key in {"title", "description", "$comment"} & schema.keys():
        if not isinstance(schema[key], str):
            raise ValueError(f"{path}.{key}: expected a string annotation")
    for key in {"deprecated", "readOnly", "writeOnly"} & schema.keys():
        if type(schema[key]) is not bool:
            raise ValueError(f"{path}.{key}: expected a boolean annotation")
    if "examples" in schema and not isinstance(schema["examples"], list):
        raise ValueError(f"{path}.examples: expected an array annotation")
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not kinds or any(not isinstance(kind, str) or kind not in _TYPES for kind in kinds) or len(set(kinds)) != len(kinds):
            raise ValueError(f"{path}.type: expected a JSON type or unique nonempty list of types")
    for key in _COUNTS & schema.keys():
        if type(schema[key]) is not int or schema[key] < 0:
            raise ValueError(f"{path}.{key}: expected a nonnegative integer")
    for key in {"minimum", "maximum"} & schema.keys():
        if type(schema[key]) not in (int, float) or (type(schema[key]) is float and not math.isfinite(schema[key])):
            raise ValueError(f"{path}.{key}: expected a finite number")
    if "pattern" in schema:
        try:
            if not isinstance(schema["pattern"], str):
                raise TypeError()
            re.compile(schema["pattern"])
        except (TypeError, re.error) as exc:
            raise ValueError(f"{path}.pattern: invalid regular expression") from exc
    if "enum" in schema:
        values = schema["enum"]
        if (
            not isinstance(values, list)
            or not values
            or any(_json_equal(value, previous) for index, value in enumerate(values) for previous in values[:index])
        ):
            raise ValueError(f"{path}.enum: expected a unique nonempty list")
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or any(not isinstance(key, str) for key in required) or len(set(required)) != len(required):
            raise ValueError(f"{path}.required: expected unique property names")
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, dict) or any(not isinstance(key, str) for key in properties):
            raise ValueError(f"{path}.properties: expected a property map")
        for key, child in properties.items():
            validate_schema(child, path=f"{path}.properties.{key}", depth=depth + 1)
    for key in ("items", "additionalProperties"):
        if key in schema and not (key == "additionalProperties" and type(schema[key]) is bool):
            validate_schema(schema[key], path=f"{path}.{key}", depth=depth + 1)
    if "anyOf" in schema:
        choices = schema["anyOf"]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{path}.anyOf: expected a nonempty list of schemas")
        for index, child in enumerate(choices):
            validate_schema(child, path=f"{path}.anyOf[{index}]", depth=depth + 1)
    return schema


def validate(value, schema, *, path="request", depth=0, request_limits=True):
    """Check declared constraints; implicit request limits do not constrain results."""
    if depth > (20 if request_limits else 25):
        raise ValueError(f"{path}: maximum nesting depth exceeded")
    if "anyOf" in schema:
        for choice in schema["anyOf"]:
            try:
                validate(value, choice, path=path, depth=depth, request_limits=request_limits)
                break
            except ValueError:
                pass
        else:
            raise ValueError(f"{path}: invalid value")
        # The successful branch already bounded every value and child. Apply
        # explicit sibling constraints without overriding its larger declared
        # limits with implicit defaults during the second traversal.
        request_limits = False
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
            (type(value) is float and not math.isfinite(value))
            or value < schema.get("minimum", -1e100 if request_limits else -math.inf)
            or value > schema.get("maximum", 1e100 if request_limits else math.inf)
        ):
            raise ValueError(f"{path}: number out of range")
    if "enum" in schema and not any(_json_equal(value, option) for option in schema["enum"]):
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
