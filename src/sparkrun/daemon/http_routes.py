"""HTTP aliases and OpenAPI descriptions derived from daemon operations."""

from sparkrun.core.daemon_registry import (
    operation_http_path,
    registered_daemon_ops,
    registered_daemon_routes,
    get_daemon_op,
    legacy_operation_names,
)


JOB_ACCEPTED = {
    "type": "object",
    "properties": {"job_id": {"type": "string"}, "events_url": {"type": "string"}},
    "required": ["job_id", "events_url"],
    "additionalProperties": False,
}
ERROR = {
    "type": "object",
    "properties": {"detail": {"type": "string"}, "error": {"type": "object"}},
    "required": ["detail"],
}


def _ref(name):
    return {"$ref": "#/components/schemas/" + name}


def _content(schema):
    return {"application/json": {"schema": schema}}


def _responses(op):
    responses = {
        code: {"description": description, "content": _content(_ref("DaemonOperationError"))}
        for code, description in {
            401: "Authentication required.",
            403: "Insufficient permission, disabled feature, or rejected trust request.",
            404: "Requested resource was not found.",
            408: "Request body timed out.",
            409: "Conflicting request, exhausted admission capacity, or daemon maintenance.",
            413: "Request body exceeds the configured limit.",
            422: "Invalid JSON or operation arguments.",
            500: "Operation failed; see daemon diagnostics.",
            502: "Target hosts are unreachable.",
            504: "Operation timed out.",
        }.items()
    }
    if op.kind == "job":
        responses[202] = {
            "description": "Job accepted. Follow events_url or GET /v1/jobs/{job_id} for completion and the final result.",
            "content": _content(_ref("DaemonJobAccepted")),
        }
    else:
        responses[200] = {"description": "Operation result.", "content": _content(_ref(op.name + "Result"))}
    return responses


def _extra(op):
    extra = {
        "requestBody": {"required": True, "content": _content(_ref(op.name + "Request"))},
        "security": [{"BearerAuth": []}],
        "x-sparkrun-operation": {
            "name": op.name,
            "owner": op.owner,
            "local_name": op.local_name,
            "legacy_names": legacy_operation_names(op),
            "kind": op.kind,
            "scope": op.scope,
            "version": op.version,
            "feature_flag": op.feature_flag,
            "secret": op.secret,
            "result": _ref(op.name + "Result"),
        },
    }
    if op.kind == "job":
        extra["parameters"] = [
            {
                "name": "Idempotency-Key",
                "in": "header",
                "required": False,
                "schema": {"type": "string", "maxLength": 200},
                "description": "Retry key shared with /v1/ops/{name}; the same principal and request return the same job.",
            },
            {
                "name": "X-Sparkrun-Client",
                "in": "header",
                "required": False,
                "schema": {"type": "string"},
                "description": "Optional client label; does not affect authorization or job ownership.",
            },
        ]
    return extra


def mount_operation_routes(app, operation_handler, envelope_handler):
    """Validate all literal paths before mounting aliases and plugin adapters.

    Re-entering an application's lifespan replaces these routes and invalidates
    its OpenAPI cache, including a schema requested before plugin initialization.
    """
    ops = [(op, path) for op in registered_daemon_ops() if (path := operation_http_path(op)) is not None]
    adapters = registered_daemon_routes()
    for route in adapters:
        for name in route.operations:
            get_daemon_op(name)  # Resolve forward references before mounting any routes.
    previous = getattr(app.state, "operation_routes", ())
    existing = [route for route in app.router.routes if route not in previous]
    claimed = {}
    for path, owner in [(path, "operation " + op.name) for op, path in ops] + [
        (route.path, "adapter from " + route.owner) for route in adapters
    ]:
        if path in claimed:
            raise ValueError(f"Daemon HTTP path {path!r} is claimed by {claimed[path]} and {owner}")
        if any(route.path_regex.fullmatch(path) for route in existing if hasattr(route, "path_regex")):
            raise ValueError(f"Daemon HTTP path {path!r} for {owner} conflicts with an existing route")
        claimed[path] = owner

    app.router.routes[:] = existing
    for op, path in ops:
        behavior = "Submits a job (HTTP 202)." if op.kind == "job" else "Returns the result directly (HTTP 200)."
        app.add_api_route(
            path,
            operation_handler(op.name),
            methods=["POST"],
            operation_id="sparkrun." + op.name,
            summary=op.name,
            description=f"{op.description}\n\n{behavior} Requires {op.scope} permission. Alias of POST /v1/ops/{op.name}.",
            tags=["plugin: " + op.owner] if op.owner != "core" else [op.name.split(".")[0] if "." in op.name else "operations"],
            status_code=202 if op.kind == "job" else 200,
            responses=_responses(op),
            openapi_extra=_extra(op),
        )
    for route in adapters:
        app.add_api_route(
            route.path,
            envelope_handler(route),
            methods=["POST"],
            # Keep path separators verbatim; FastAPI's default sanitizer would
            # give a-b, a_b and a/b the same generated operation ID.
            operation_id="sparkrun:route:" + route.path,
            tags=["plugin: " + route.owner] if route.owner != "core" else ["adapters"],
            openapi_extra={
                "requestBody": {"required": True, "content": _content(route.request_schema)},
                "security": [{"BearerAuth": []}],
                "x-sparkrun-adapter": {
                    "owner": route.owner,
                    "local_path": route.local_path,
                    "operations": list(route.operations),
                },
            },
        )
    app.state.operation_routes = app.router.routes[len(existing) :]
    app.state.operation_paths = {op.name: path for op, path in ops}
    app.openapi_schema = None


def add_operation_schemas(schema, paths):
    components = schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    schemas.update(DaemonJobAccepted=JOB_ACCEPTED, DaemonOperationError=ERROR)
    components.setdefault("securitySchemes", {})["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "description": "TCP requires a daemon token. Linux Unix sockets also accept an authenticated same-user peer.",
    }
    # The ASGI boundary authenticates every endpoint except health. Describe the
    # same rule for API tools, including the generic operation/job endpoints.
    schema["security"] = [{"BearerAuth": []}]
    schema["paths"]["/v1/health"]["get"]["security"] = []
    operations = registered_daemon_ops()
    for op in operations:
        schemas[op.name + "Request"] = op.request_schema
        schemas[op.name + "Result"] = op.result_schema
    schema["x-sparkrun-operations"] = {
        op.name: {
            "request": _ref(op.name + "Request"),
            "kind": op.kind,
            "version": op.version,
            "result": _ref(op.name + "Result"),
            "http_path": paths.get(op.name),
            "owner": op.owner,
            "local_name": op.local_name,
            "legacy_names": legacy_operation_names(op),
        }
        for op in operations
    }
    return schema
