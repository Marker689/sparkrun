"""Metadata-only Kubernetes daemon operations; handlers load only on execution."""

from functools import partial

from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op


def execute(name, arguments, sctx, prompt=None):
    from sparkrun.plugins.k8s import api
    from sparkrun.api._wire import to_wire

    result = getattr(api, name)(sctx=sctx, **arguments)
    if name == "jobset_status":
        # Kubernetes specs can contain literal credentials in pod environments.
        return {
            "name": result.get("metadata", {}).get("name"),
            "namespace": result.get("metadata", {}).get("namespace"),
            "status": to_wire(result.get("status", {})),
        }
    return to_wire(result)


def loader(name):
    return partial(execute, name)


def register():
    from sparkrun.api._wire_schemas import RESULT_SCHEMAS

    text = {"type": "string", "maxLength": 4096}
    boolean = {"type": "boolean"}
    specs = [
        ("ensure_kubectl", "admin", "version context download", "", {"download"}),
        ("cluster_info", "operate", "kubeconfig context namespace pin", "", {"pin"}),
        ("list_nodes", "operate", "kubeconfig context selector gpu_only", "", {"gpu_only"}),
        ("kueue_status", "operate", "kubeconfig context", "", set()),
        ("setup_kueue", "admin", "install kueue_version jobset_version namespace kubeconfig context dry_run", "", {"install", "dry_run"}),
        (
            "configure_service_account",
            "admin",
            "name namespace kubeconfig context create_namespace token_duration write_kubeconfig dry_run",
            "",
            {"create_namespace", "write_kubeconfig", "dry_run"},
        ),
        ("jobset_status", "operate", "name namespace kubeconfig context", "name", set()),
        ("stop_jobset", "operate", "name namespace kubeconfig context", "name", set()),
    ]
    for name, scope, names, required, bools in specs:
        properties = {key: (boolean if key in bools else text) for key in names.split()}
        register_daemon_op(
            DaemonOp(
                name,
                partial(loader, name),
                {"type": "object", "properties": properties, "required": required.split(), "additionalProperties": False},
                result_schema=RESULT_SCHEMAS["k8s." + name],
                kind="job",
                scope=scope,
                feature_flag="integration.k8s",
                owner="k8s",
                description="Kubernetes " + name.replace("_", " ") + ".",
            )
        )
