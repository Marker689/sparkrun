"""Dormant MCP stdio adapter retained for possible future reuse.

MCP is disabled: this module has no registered CLI or service entry point.
Keep the protocol implementation isolated here until support is reconsidered.
"""

from __future__ import annotations

import json
import sys
import uuid

PROTOCOL = "2025-11-25"


class Session:
    def __init__(self, client):
        self.client = client
        self.initialized = False
        self.ready = False
        self.id = uuid.uuid4().hex

    def dispatch(self, request):
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
            raise ValueError("Invalid JSON-RPC request")
        method = request["method"]
        params = request.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("Parameters must be an object")
        if method == "initialize":
            self.client.info()
            self.initialized = True
            from sparkrun import __version__

            return {
                "protocolVersion": PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "sparkrun", "version": __version__},
                "instructions": "Mutating and long operations return durable job IDs. Inspect jobs before retrying an uncertain submission. Approval and credential prompts can be answered using sparkrun daemon jobs answer.",
            }
        if method == "ping":
            return {}
        if method == "notifications/initialized" and self.initialized:
            self.ready = True
            return None
        if method.startswith("notifications/"):
            return None
        if not self.ready:
            raise ValueError("Initialize the session first")
        if method == "tools/list":
            operations = self.client.request("GET", "/v1/ops")
            tools = [
                {
                    "name": op["name"],
                    "description": op["description"],
                    "inputSchema": op["request_schema"],
                    "annotations": {
                        "readOnlyHint": op["scope"] == "read",
                        "destructiveHint": op["scope"] != "read",
                        "idempotentHint": op["kind"] == "query",
                        "openWorldHint": True,
                    },
                }
                for op in operations
            ]
            schema = {
                "type": "object",
                "properties": {"job_id": {"type": "string", "pattern": "^[a-f0-9]{32}$"}},
                "required": ["job_id"],
                "additionalProperties": False,
            }
            tools.extend(
                [
                    {
                        "name": "daemon.job",
                        "description": "Read a durable job result, state, or pending prompt.",
                        "inputSchema": schema,
                        "annotations": {"readOnlyHint": True},
                    },
                    {
                        "name": "daemon.cancel",
                        "description": "Request cancellation of a durable job.",
                        "inputSchema": schema,
                        "annotations": {"readOnlyHint": False},
                    },
                ]
            )
            return {"tools": tools}
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("Expected a tool name and argument object")
            try:
                if name in {"daemon.job", "daemon.cancel"}:
                    identifier = arguments.get("job_id", "")
                    if set(arguments) != {"job_id"} or len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
                        raise ValueError("Invalid job_id")
                    value = self.client.request("GET" if name == "daemon.job" else "DELETE", "/v1/jobs/" + identifier)
                else:
                    value = self.client.call(name, arguments, key=self.id + ":" + str(request.get("id")))
                return {"content": [{"type": "text", "text": json.dumps(value)}], "isError": False}
            except (RuntimeError, ValueError, OSError) as exc:
                return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        raise LookupError("Unsupported method")


def serve(client, *, input_stream=None, output_stream=None):
    source = input_stream or sys.stdin
    output = output_stream or sys.stdout
    session = Session(client)
    while raw := source.readline(1024 * 1024 + 1):
        request = None
        try:
            if len(raw) > 1024 * 1024 or not raw.endswith("\n"):
                raise ValueError("Message exceeds 1 MiB or is not newline terminated")
            request = json.loads(raw)
            result = session.dispatch(request)
            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
        except (ValueError, LookupError, OSError, RuntimeError) as exc:
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": {
                    "code": -32700 if isinstance(exc, json.JSONDecodeError) else -32601 if isinstance(exc, LookupError) else -32602,
                    "message": str(exc),
                },
            }
        if isinstance(request, dict) and "id" not in request:
            continue
        output.write(json.dumps(response) + "\n")
        output.flush()
