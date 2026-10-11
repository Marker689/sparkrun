"""Explicit core operation adapters and their public request schemas."""

from __future__ import annotations

from functools import partial
import inspect
from typing import Any

from sparkrun.api._wire import object_schema, to_wire, validate
from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op
from sparkrun.api._wire_schemas import RESULT_SCHEMAS

S = {"type": "string", "maxLength": 4096}
B = {"type": "boolean"}
N = {"type": "number", "minimum": 0}
INTEGER = {"type": "integer", "minimum": 0}
A = {"type": "array", "items": S, "maxItems": 512}
D = {"type": "object"}


def nullable(schema) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def fields(names) -> dict[str, Any]:
    bools = set(
        "restart persist clear_cluster now auto_discover dry_run solo trust ensure offline runtime_cache auto_port dashboard rootful sync_tuning preserve_model_perms skip_model_fan_out rebuild no_stop skip_run exit_on_first_fail export_files include_hidden include_local unique_names local_only acknowledge_trust generate full force_container keep_containers enable_ssh ephemeral proxy set_proxy_host remove checksums confirm wait_ready post_launch".split()
    )
    ints = set(
        "discover_interval discover_removal_grace_sweeps port ray_port dashboard_port init_port offset limit timeout duration parallel_pairs queue_pairs gid_index journal_lines tail interval max_workers".split()
    )
    arrays = {"hosts", "host_list", "host_filter", "extra_docker_opts", "only_steps"}
    maps = {"overrides", "executor_config", "strategy_options", "filters", "integrations", "aliases", "bench_args", "builder_context"}
    return {
        name: nullable(B if name in bools else INTEGER if name in ints else A if name in arrays else D if name in maps else S)
        for name in names.split()
    }


# Each row is a reviewed allowlist, not reflection over api.__all__.
# Long or potentially mutating reads run in workers until their query audit is complete.
SPECS = [
    ("catalog.recipes", "catalog_recipes", "query", "read", "query registry runtime local_only offset limit filters", ""),
    ("catalog.recipe", "get_recipe_details", "job", "read", "reference overrides", "reference"),
    ("catalog.registries", "list_registries", "query", "read", "", ""),
    ("catalog.clusters", "list_clusters", "query", "read", "", ""),
    ("catalog.capacity", "catalog_cluster_capacity", "job", "read", "cluster", "cluster"),
    ("catalog.import", "import_recipe", "job", "operate", "content", "content"),
    ("catalog.retain", "retain_catalog_recipe", "job", "operate", "reference", "reference"),
    ("catalog.cleanup", "cleanup_catalog_imports", "job", "operate", "", ""),
    ("catalog.refresh", "refresh_registries", "job", "operate", "", ""),
    ("catalog.registry", "configure_registry", "job", "admin", "action name url subpath acknowledge_trust", "action name"),
    ("list_jobs", "list_jobs", "query", "read", "limit", ""),
    ("status", "status", "job", "read", "hosts cluster executor", "hosts"),
    ("status_report", "status_report", "job", "read", "hosts cluster executor", "hosts"),
    ("find_running_intent", "find_running_intent", "job", "read", "intent_id hosts cluster", "intent_id hosts"),
    ("search_recipes", "search_recipes", "query", "read", "query registry runtime include_hidden include_local unique_names", ""),
    ("stop", "stop", "job", "operate", "cluster_id recipe hosts cluster overrides", ""),
    ("stop_all", "stop_all", "job", "operate", "hosts cluster dry_run", "hosts"),
    ("discover_endpoints", "discover_endpoints", "job", "read", "", ""),
    (
        "resume_benchmark",
        "resume_benchmark",
        "job",
        "operate",
        "benchmark_id dry_run integrations export_files output_file timeout exit_on_first_fail api_key_env",
        "benchmark_id",
    ),
    ("cluster.create", "create_cluster", "job", "admin", "name hosts description user", "name hosts"),
    ("cluster.update", "update_cluster", "job", "admin", "name hosts description user", "name"),
    ("cluster.delete", "delete_cluster", "job", "admin", "name", "name"),
    ("cluster.default", "set_default_cluster", "job", "admin", "name", ""),
    ("proxy.status", "proxy.status", "job", "read", "", ""),
    ("proxy.models", "proxy.models", "job", "read", "", ""),
    ("proxy.stop", "proxy.stop", "job", "operate", "dry_run", ""),
    ("proxy.sync", "proxy.sync", "job", "operate", "aliases host_filter", ""),
    ("proxy.register_loaded_model", "proxy.register_loaded_model", "job", "operate", "recipe overrides cluster", "recipe"),
    ("proxy.unregister_loaded_model", "proxy.unregister_loaded_model", "job", "operate", "recipe", "recipe"),
    ("proxy.add_alias", "proxy.add_alias", "job", "admin", "alias target", "alias target"),
    ("proxy.remove_alias", "proxy.remove_alias", "job", "admin", "alias", "alias"),
    ("proxy.list_aliases", "proxy.list_aliases", "query", "read", "", ""),
    ("proxy.list_gateways", "proxy.list_gateways", "query", "read", "", ""),
    ("proxy.resolve_gateway", "proxy.resolve_gateway", "query", "read", "name", ""),
    ("proxy.service_status", "proxy.service_status", "job", "admin", "scope journal_lines", ""),
    ("proxy.uninstall_service", "proxy.uninstall_service", "job", "admin", "scope dry_run", ""),
    ("setup.probe_ssh_access", "setup.probe_ssh_access", "job", "read", "hosts user key timeout max_workers", "hosts"),
    ("setup.ensure_local_key", "setup.ensure_local_key", "job", "admin", "preferred generate", ""),
    ("setup.mesh_ssh_keys", "setup.mesh_ssh_keys_native", "job", "admin", "hosts user key timeout dry_run", "hosts"),
]

OPTION_FIELDS = {
    "RunOptions": "recipe hosts cluster overrides strategy_options solo dry_run trust ensure scheduler transfer_mode cache_dir offline runtime_cache auto_port ray_port dashboard_port dashboard init_port executor executor_config rootful cluster_id_override transfer_interface local_cache_dir sync_tuning extra_docker_opts topology preserve_model_perms skip_model_fan_out recipe_ref",
    "BuildOptions": "recipe hosts cluster overrides dry_run trust rebuild executor executor_config cache_dir local_cache_dir transfer_mode transfer_interface topology sync_tuning preserve_model_perms skip_model_fan_out builder_context",
    "BenchmarkOptions": "recipe category framework profile bench_args hosts cluster overrides resume skip_run no_stop exit_on_first_fail timeout api_key_env integrations output_file export_files solo dry_run scheduler rootful trust offline sync_tuning extra_docker_opts",
    "ProxyStartOptions": "host port gateway host_filter cluster clear_cluster auto_discover discover_interval discover_removal_grace_sweeps restart persist dry_run",
    "ProxyServiceOptions": "scope cluster now dry_run",
}


def _resolve(target):
    from sparkrun import api

    value: Any = api
    for part in target.split("."):
        value = getattr(value, part)
    return value


def unavailable_prompt(*args, **kwargs):
    raise RuntimeError("This operation requires a worker input channel")


def invoke(target, arguments, sctx, prompt=unavailable_prompt):
    from sparkrun import api

    fn = _resolve(target)
    arguments = dict(arguments)
    parameters = inspect.signature(fn).parameters
    if target == "search_recipes":
        arguments["ensure_initialized"] = False
    if "sctx" in parameters:
        arguments["sctx"] = sctx
    if "user" in parameters and "user" not in arguments:
        arguments["user"] = sctx.config.ssh_user
    if "progress" in parameters:
        arguments["progress"] = lambda name, success: sctx.operation.emit("progress", label=name, success=success)
    if "progress_callback" in parameters:
        arguments["progress_callback"] = lambda event: sctx.operation.emit("progress", event=to_wire(event))
    if "decision_callback" in parameters:
        arguments["decision_callback"] = lambda decision: prompt("decision", decision.message, default=False)
    try:
        return to_wire(fn(**arguments))
    except api.proxy.SudoPasswordRequired:
        if target != "proxy.uninstall_service":
            raise
        arguments["sudo_password"] = prompt("secret", "Sudo password", default=None)
        return to_wire(fn(**arguments))


def invoke_options(target, model, arguments, sctx, prompt=unavailable_prompt):
    from sparkrun import api

    arguments = dict(arguments)
    confirm = arguments.pop("confirm", False)
    ready = arguments.pop("wait_ready", True)
    hooks = arguments.pop("post_launch", True)
    cls = getattr(api.proxy, model) if model.startswith("Proxy") else getattr(api, model)
    if model == "RunOptions":
        arguments.update(detached=True, follow=False, owner="daemon:" + sctx.operation.id)
    if model == "BenchmarkOptions":
        arguments["progress_callback"] = lambda event: sctx.operation.emit("progress", event=to_wire(event))
        arguments["decision_callback"] = lambda decision: prompt("decision", decision.message, default=False)
        if "resume" in arguments:
            arguments["resume"] = api.ResumeMode(arguments["resume"])
    options = cls(**arguments)
    if target == "proxy.install_service":
        from dataclasses import replace

        try:
            return to_wire(api.proxy.install_service(options, sctx=sctx))
        except api.proxy.SudoPasswordRequired:
            options = replace(options, sudo_password=prompt("secret", "Sudo password", default=None))
            return to_wire(api.proxy.install_service(options, sctx=sctx))
    if target == "run":
        plan = api.plan(options, sctx=sctx)
        sctx.operation.emit("plan", plan=to_wire(plan))
        if confirm and not prompt("approval", "Launch this plan?", default=False):
            raise InterruptedError("Launch declined")
        result = api.run(options, sctx=sctx, plan=plan)
        if result.rc:
            raise api.SparkrunError(f"Launch {result.cluster_id} failed with exit code {result.rc}")
        if not options.dry_run and not result.already_running and result.launch_result is not None:
            readiness = api.wait_ready(result, sctx=sctx) if ready else None
            if readiness is not None and not readiness.ready:
                raise api.SparkrunError("Server did not become ready: " + str(readiness.reason))
            if hooks and (plan.recipe.post_exec or plan.recipe.post_commands):
                api.post_launch(result, trust=options.trust, readiness=readiness, sctx=sctx)
        elif not options.dry_run and (ready or hooks):
            sctx.operation.emit("warning", message="Launch has no new lifecycle handle; readiness and post-launch hooks were not run.")
        return to_wire(result)
    return to_wire(_resolve(target)(options, sctx=sctx))


def _loader(target):
    return partial(invoke, target)


def _options_loader(target, model):
    return partial(invoke_options, target, model)


def register_core_ops():
    for name, target, kind, scope, names, required in SPECS:
        properties = fields(names)
        for key in required.split():
            properties[key] = properties[key]["anyOf"][0]
        if name == "catalog.import":
            properties["content"] = {"type": "string", "maxLength": 262144}
        register_daemon_op(
            DaemonOp(
                name,
                partial(_loader, target),
                object_schema(properties, required.split()),
                result_schema=RESULT_SCHEMAS[name],
                kind=kind,
                scope=scope,
                description=f"{target.replace('_', ' ').capitalize()} through the library API.",
            )
        )
    for name, model, scope in [
        ("run", "RunOptions", "operate"),
        ("plan", "RunOptions", "read"),
        ("materialize", "RunOptions", "operate"),
        ("realize_recipe", "RunOptions", "operate"),
        ("build", "BuildOptions", "operate"),
        ("plan_build", "BuildOptions", "read"),
        ("benchmark", "BenchmarkOptions", "operate"),
        ("proxy.start", "ProxyStartOptions", "operate"),
        ("proxy.install_service", "ProxyServiceOptions", "admin"),
    ]:
        properties = fields(OPTION_FIELDS[model])
        if "recipe" in properties:
            properties["recipe"] = S
        if name == "run":
            properties.update({"confirm": B, "wait_ready": B, "post_launch": B})
        register_daemon_op(
            DaemonOp(
                name,
                partial(_options_loader, name, model),
                object_schema(properties, [] if model.startswith("Proxy") else ["recipe"]),
                result_schema=RESULT_SCHEMAS[name],
                kind="job",
                scope=scope,
                description=f"{name.replace('_', ' ').capitalize()} with an operation-local context.",
            )
        )
    from sparkrun.daemon.services import watch_proxy

    register_daemon_op(
        DaemonOp(
            "proxy.watch",
            lambda: watch_proxy,
            object_schema(),
            result_schema={"type": "object", "properties": {"stopped": {"type": "string"}}, "required": ["stopped"]},
            kind="job",
            scope="admin",
            resources="none",
            description="Follow a running proxy with graceful endpoint removal; opt in via daemon.proxy_autodiscover.",
        )
    )
    from sparkrun.daemon.special_ops import register_special_ops

    register_special_ops()


def check_request(op, arguments):
    from sparkrun.api._wire import reject_secrets

    validate(arguments, op.request_schema)
    reject_secrets(arguments)
    return arguments
