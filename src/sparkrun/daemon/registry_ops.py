"""Explicit catalog registry mutations; the action is selected by the route."""

from functools import partial

from sparkrun.api._wire import object_schema, to_wire
from sparkrun.api._wire_schemas import RESULT_SCHEMAS
from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op

REGISTRY_ACTIONS = ("add", "remove", "enable", "disable", "trust", "untrust")


def invoke(action, arguments, sctx, prompt=None):
    from sparkrun import api

    return to_wire(api.configure_registry(action, sctx=sctx, **arguments))


def _load(action):
    return partial(invoke, action)


def register_registry_ops():
    for action in REGISTRY_ACTIONS:
        fields = {"name": {"type": "string", "minLength": 1, "maxLength": 4096}}
        required = ["name"]
        if action == "add":
            fields.update(url={"type": "string", "minLength": 1, "maxLength": 4096}, subpath={"type": "string", "maxLength": 4096})
            required.append("url")
        elif action == "trust":
            fields["acknowledge_trust"] = {"type": "boolean", "enum": [True]}
            required.append("acknowledge_trust")
        name = "catalog.registries." + action
        register_daemon_op(
            DaemonOp(
                name,
                partial(_load, action),
                object_schema(fields, required),
                result_schema=RESULT_SCHEMAS[name],
                scope="admin",
                description=(
                    "Add a registry definition without cloning or granting trust."
                    if action == "add"
                    else "Grant recipe-hook trust after explicit acknowledgement."
                    if action == "trust"
                    else action.capitalize() + " a registry definition."
                ),
            )
        )
