"""Public infrastructure envelopes shared by OpenAPI and contract tests."""

from copy import deepcopy

from sparkrun.api._wire import object_schema

S = {"type": "string"}
N = {"type": "number"}
INTEGER = {"type": "integer"}
B = {"type": "boolean"}


def nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


JOB_ACCEPTED = object_schema({"job_id": S, "events_url": S}, ["job_id", "events_url"])

ERROR = object_schema(
    {"detail": S, "error": object_schema({"type": S, "retryable": B, "attrs": {"type": "object"}}, ["type", "retryable", "attrs"])},
    ["detail"],
)
PROMPT = object_schema(
    {
        "prompt_id": S,
        "kind": {
            "type": "string",
            "description": "Built-in kinds: approval, decision, secret. Secret answers require secret-prompt permission; plugins may declare other non-secret kinds.",
        },
        "message": S,
        "default": {},
        "deadline": N,
    },
    ["prompt_id", "kind", "message", "default", "deadline"],
)
APPLICATION = object_schema(
    {name: S for name in ("id", "command", "package", "resource_namespace")}, ["id", "command", "package", "resource_namespace"]
)
CONTROLLER = object_schema(
    {"schema_version": {"type": "integer", "enum": [1]}, "application": APPLICATION, "controller_id": S},
    ["schema_version", "application", "controller_id"],
)
INFO = object_schema(
    {
        "profile": S,
        "binding": S,
        "config_path": S,
        "controller": CONTROLLER,
        "protocol": INTEGER,
        "runtime": {"type": "object"},
        "restart_required": B,
        "manager": S,
    },
    ["profile", "binding", "config_path", "controller", "protocol", "runtime", "restart_required", "manager"],
)
JOB = object_schema(
    {
        **{name: S for name in ("id", "operation", "principal", "client", "scope", "manager")},
        "version": INTEGER,
        "state": {
            "type": "string",
            "enum": ["queued", "starting", "running", "waiting_input", "cancelling", "cancelled", "succeeded", "failed", "lost"],
        },
        "created": N,
        "updated": N,
        "seq": INTEGER,
        "cancelled": nullable(N),
        "result": {},
        "error": nullable(ERROR),
        "prompt": nullable(PROMPT),
    },
    [
        "id",
        "operation",
        "version",
        "principal",
        "client",
        "scope",
        "manager",
        "state",
        "created",
        "updated",
        "seq",
        "cancelled",
        "result",
        "error",
        "prompt",
    ],
)
EVENT = {
    "type": "object",
    "properties": {"id": INTEGER, "type": S, "operation_id": S},
    "required": ["type"],
    "additionalProperties": True,
    "description": "Job SSE payload. Durable events have an integer id; gap frames require fetching the job and resuming. Fields beyond the envelope belong to the event type; structured feedback carries feedback_version.",
}


def add_infrastructure_schemas(schema, operations):
    schemas = schema["components"]["schemas"]
    job = deepcopy(JOB)
    # Successful results discriminate by both operation and contract version.
    # Historical/internal jobs retain the base envelope when no declaration matches.
    job["allOf"] = [
        {
            "if": {"properties": {"operation": {"enum": [op.name]}, "version": {"enum": [op.version]}, "state": {"enum": ["succeeded"]}}},
            "then": {"properties": {"result": {"$ref": "#/components/schemas/" + op.name + "Result"}}},
        }
        for op in operations
        if op.kind == "job"
    ]
    schemas.update(DaemonJob=job, DaemonPrompt=PROMPT, DaemonInfo=INFO, DaemonEvent=EVENT, DaemonOperationError=ERROR)
    schemas["DaemonJobSnapshot"] = object_schema(
        {"type": {"enum": ["snapshot"]}, "jobs": {"type": "array", "items": {"$ref": "#/components/schemas/DaemonJob"}}}, ["type", "jobs"]
    )
    paths = schema["paths"]
    paths["/v1/jobs/{identifier}/artifacts/diagnostics"]["get"]["responses"]["200"] = {
        "description": "Redacted NDJSON diagnostic bundle. Download after the job finishes; retained with job history.",
        "content": {"application/x-ndjson": {"schema": {"type": "string", "format": "binary"}}},
    }
    for path, method, result in (
        ("/v1/info", "get", {"$ref": "#/components/schemas/DaemonInfo"}),
        ("/v1/jobs", "get", {"type": "array", "items": {"$ref": "#/components/schemas/DaemonJob"}}),
        ("/v1/jobs/{identifier}", "get", {"$ref": "#/components/schemas/DaemonJob"}),
        ("/v1/jobs/{identifier}", "delete", {"$ref": "#/components/schemas/DaemonJob"}),
        ("/v1/jobs/{identifier}/input", "post", object_schema()),
    ):
        paths[path][method]["responses"]["200"] = {
            "description": "Successful response",
            "content": {"application/json": {"schema": result}},
        }
    paths["/v1/jobs/{identifier}/input"]["post"]["requestBody"] = {
        "required": True,
        "content": {"application/json": {"schema": object_schema({"prompt_id": S, "value": {}}, ["prompt_id", "value"])}},
    }
    for path, component in (("/v1/jobs/{identifier}/events", "DaemonEvent"), ("/v1/events", "DaemonJobSnapshot")):
        paths[path]["get"]["responses"]["200"] = {
            "description": "Server-sent events; each data frame follows the declared envelope.",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
            "x-sparkrun-event-schema": {"$ref": "#/components/schemas/" + component},
        }
