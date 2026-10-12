"""Declared plugin constraints match request admission and successful results."""

import pytest

from sparkrun.api._wire import ResultContractError, check_result, object_schema, validate
from sparkrun.core.daemon_registry import DaemonOp, DaemonRoute
from sparkrun.core.daemon_schema import validate_schema
from sparkrun.daemon.ops import check_request


def test_anyof_also_enforces_sibling_constraints():
    schema = object_schema({"choice": {"anyOf": [{"type": "integer"}, {"type": "string"}], "enum": [1, "one"]}}, ["choice"])
    op = DaemonOp("fixture.choose", lambda: None, request_schema=schema, result_schema=schema)
    for value in (1, "one"):
        assert check_request(op, {"choice": value}) == {"choice": value}
        assert check_result(op, {"choice": value}) == {"choice": value}
    for value in (2, "two", None, True):
        with pytest.raises(ValueError):
            check_request(op, {"choice": value})
        with pytest.raises(ResultContractError):
            check_result(op, {"choice": value})


def test_anyof_does_not_bypass_closed_objects_or_required_fields():
    schema = {**object_schema({"name": {"type": "string"}}, ["name"]), "anyOf": [{"type": "object"}, {"type": "null"}]}
    validate_schema(schema)
    assert validate({"name": "ok"}, schema) == {"name": "ok"}
    for value in ({}, {"name": "ok", "extra": True}, None):
        with pytest.raises(ValueError):
            validate(value, schema)


@pytest.mark.parametrize(
    ("value", "field"),
    [
        ("x" * 70000, {"type": "string", "maxLength": 100000}),
        ([1] * 2100, {"type": "array", "maxItems": 3000}),
        (1e101, {"type": "number", "maximum": 1e102}),
    ],
    ids=["text", "array", "number"],
)
def test_anyof_preserves_explicit_branch_limits(value, field):
    # Sibling constraints apply, but defaults must not override a branch's
    # explicit larger limit, including when visiting object children again.
    schema = {
        "anyOf": [object_schema({"value": field}), {"type": "null"}],
        "properties": {"value": {"description": "The permitted value"}},
    }
    validate_schema(schema)
    assert validate({"value": value}, schema) == {"value": value}
    schema["properties"]["value"] = {"enum": [None]}
    with pytest.raises(ValueError):
        validate({"value": value}, schema)


@pytest.mark.parametrize("keyword", ["const", "oneOf", "allOf", "$ref", "format", "patternProperties", "uniqueItems"])
@pytest.mark.parametrize("surface", ["request", "result", "route"])
def test_unsupported_nested_constraints_fail_at_declaration(keyword, surface):
    schema = object_schema({"value": {keyword: "unsupported"}})
    with pytest.raises(ValueError, match="unsupported schema keywords: " + (r"\$ref" if keyword == "$ref" else keyword)):
        if surface == "route":
            DaemonRoute("fixture", lambda: None, ("fixture",), schema)
        else:
            DaemonOp("fixture", lambda: None, **{surface + "_schema": schema})


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "not-a-type"},
        {"type": []},
        {"type": ["string", "string"]},
        {"type": [True]},
        {"properties": []},
        {"properties": {"x": False}},
        {"additionalProperties": 1},
        {"items": []},
        {"anyOf": []},
        {"anyOf": [None]},
        {"anyOf": {}},
        {"required": "x"},
        {"required": ["x", "x"]},
        {"enum": []},
        {"enum": [1, 1.0]},
        {"maxLength": -1},
        {"minItems": True},
        {"minimum": "1"},
        {"maximum": float("inf")},
        {"pattern": "["},
        {"pattern": 3},
        {"title": 3},
        {"examples": "example"},
        {"deprecated": "false"},
    ],
)
def test_malformed_declarations_fail_early(schema):
    with pytest.raises(ValueError):
        DaemonOp("fixture", lambda: None, result_schema=schema)


def test_schema_annotations_and_arbitrary_property_names_are_allowed():
    schema = object_schema({"const": {"type": "string", "description": "An ordinary property", "examples": ["one"]}})
    schema.update(title="Example", default={}, deprecated=False, readOnly=True, writeOnly=False)
    assert validate_schema(schema) == schema
    assert validate({"const": "one"}, schema) == {"const": "one"}


@pytest.mark.parametrize("value", [1, {"value": 1}, [1]])
def test_enums_distinguish_booleans_from_numbers(value):
    boolean = True if value == 1 else {"value": True} if isinstance(value, dict) else [True]
    schema = {"enum": [boolean]}
    validate_schema(schema)
    assert validate(boolean, schema) == boolean
    with pytest.raises(ValueError):
        validate(value, schema)


def test_large_request_integer_is_a_validation_error_not_overflow():
    with pytest.raises(ValueError, match="number out of range"):
        validate(10**400, {"type": "integer"})
