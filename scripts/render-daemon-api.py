#!/usr/bin/env python3
"""Render the daemon reference tables from core and reference-plugin declarations.

Only registration metadata is loaded; operation handlers and servers are never run.
The external checkout is explicit because SparkRoute is not bundled with the host.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

BEGIN = "<!-- BEGIN GENERATED DAEMON OPERATIONS -->"
END = "<!-- END GENERATED DAEMON OPERATIONS -->"


def schema_type(schema):
    if "anyOf" in schema:
        return "/".join(schema_type(choice) for choice in schema["anyOf"])
    kind = schema.get("type", "any")
    if isinstance(kind, list):
        return "/".join(kind)
    if kind == "array":
        return schema_type(schema.get("items", {})) + "[]"
    return kind


def declarations(checkout):
    from sparkrun.daemon.ops import register_core_ops
    from sparkrun.core.daemon_registry import registered_daemon_ops
    from sparkrun.core.plugin_identity import PluginIdentity, plugin_registration
    from sparkrun.plugins.k8s.daemon import register as register_k8s
    import sparkrun.plugins

    register_core_ops()
    with plugin_registration(PluginIdentity("k8s", "sparkrun.plugins.k8s", ("in-tree", "sparkrun.plugins.k8s", None))):
        register_k8s()
    source = checkout.resolve() / "src/sparkrun/plugins"
    if not (source / "sparkroute/daemon.py").is_file():
        raise ValueError("Expected a matching SparkRoute plugin checkout with daemon declarations")
    sparkrun.plugins.__path__.insert(0, str(source))
    from sparkrun.plugins.sparkroute.daemon import register as register_sparkroute

    with plugin_registration(
        PluginIdentity("sparkroute", "sparkrun.plugins.sparkroute", ("external", "sparkrun.plugins.sparkroute", str(source)))
    ):
        register_sparkroute()
    return tuple(op for op in registered_daemon_ops() if op.visibility == "public")


def render(operations):
    from sparkrun.core.daemon_registry import operation_http_path

    output = BEGIN + "\n\n"
    for owner, title in (("core", "Core operations"), ("k8s", "Bundled Kubernetes plugin"), ("sparkroute", "External SparkRoute plugin")):
        output += "## " + title + "\n\n"
        if owner == "k8s":
            output += "Requires the loaded `k8s` integration and `integration.k8s`.\n\n"
        elif owner == "sparkroute":
            output += "Requires the external SparkRoute plugin and `gateway.sparkroute`; it is not bundled with Sparkrun.\n\n"
        groups = {}
        for op in sorted(operations, key=lambda item: item.name):
            if op.owner == owner:
                local = op.local_name or op.name
                group = local.split(".")[0] if "." in local else "operations"
                groups.setdefault(group, []).append(op)
        for group, items in groups.items():
            output += (
                "### "
                + group
                + "\n\n| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |\n| --- | --- | --- | --- | --- |\n"
            )
            for op in items:
                fields = [
                    "`" + name + ("!" if name in op.request_schema.get("required", []) else "") + ": " + schema_type(schema) + "`"
                    for name, schema in op.request_schema.get("properties", {}).items()
                ]
                output += (
                    f"| `{op.name}` | `{operation_http_path(op)}` | {op.kind} / {op.scope} | {op.resources} | "
                    + (", ".join(fields) or "`{}`")
                    + " |\n"
                )
            output += "\n"
    return output + END


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparkroute-checkout", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Exit nonzero when the checked-in reference is stale")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "docs/DAEMON_API.md"
    original = path.read_text()
    operations = declarations(args.sparkroute_checkout)
    start, end = original.index(BEGIN), original.index(END) + len(END)
    updated = original[:start] + render(operations) + original[end:]
    counts = {owner: sum(op.owner == owner for op in operations) for owner in ("core", "k8s", "sparkroute")}
    updated = re.sub(
        r"This snapshot contains \*\*\d+ operations:.*?\*\*\.",
        f"This snapshot contains **{len(operations)} operations: {counts['core']} core, {counts['k8s']} K8s and {counts['sparkroute']} SparkRoute**.",
        updated,
    )
    if args.check:
        if original != updated:
            raise SystemExit("Daemon API reference is stale; rerun without --check")
        print(f"Reference matches {len(operations)} operation declarations")
    else:
        path.write_text(updated)
        print(f"Rendered {len(operations)} operation declarations to {path}")


if __name__ == "__main__":
    main()
