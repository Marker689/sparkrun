"""Adapters for APIs whose Python callbacks and objects are operation-local."""

from __future__ import annotations

from functools import partial

from sparkrun.api._wire import object_schema, to_wire
from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op
from sparkrun.api._wire_schemas import RESULT_SCHEMAS
from sparkrun.daemon.ops import unavailable_prompt


def execute(name, arguments, sctx, prompt=unavailable_prompt):
    from sparkrun import api
    from sparkrun.api._resolve import resolve_cluster
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    data = dict(arguments)
    if name == "diagnostics.collect":
        from sparkrun.daemon.artifacts import collect

        return collect(data, sctx)
    cluster = (
        resolve_cluster(data.pop("cluster", None), data.get("hosts"), sctx=sctx) if name.startswith(("setup.", "tailscale.")) else None
    )
    if cluster is not None:
        sctx = sctx.for_cluster(cluster)
    ssh = build_ssh_kwargs(sctx.config)
    if name == "recipes.validate":
        from sparkrun.api._resolve import resolve_recipe

        return to_wire(
            api.validate_recipe(
                resolve_recipe(data["recipe"], sctx=sctx), overrides=data.get("overrides"), config=sctx.config, v=sctx.variables
            )
        )
    if name.startswith("model_cache."):
        from sparkrun.models.artifacts import ModelArtifactRequest

        management = {
            "model_cache.list": api.model_cache.inventory_cache,
            "model_cache.prune": api.model_cache.prune_cache,
            "model_cache.pin": api.model_cache.pin_revision,
            "model_cache.reconcile": api.model_cache.reconcile_claim,
            "model_cache.delete": api.model_cache.delete_model,
            "model_cache.adopt": api.model_cache.adopt_cache,
        }
        if name in management:
            data["ssh_kwargs"] = ssh
            if name == "model_cache.adopt":
                import os

                model, revision = data.pop("model", None), data.pop("revision", None)
                endpoint = data.pop("endpoint", None) or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
                if revision and not model:
                    raise ValueError("revision needs a model")
                request = ModelArtifactRequest(model, revision, endpoint=endpoint) if model else None
                return to_wire(management[name](request, endpoint=None if model else endpoint, **data))
            if name == "model_cache.delete":
                return to_wire(management[name](ModelArtifactRequest(data.pop("model"), data.pop("revision", None)), **data))
            if name in {"model_cache.pin", "model_cache.reconcile"}:
                request = ModelArtifactRequest(
                    data.pop("model"), data.pop("revision"), endpoint=data.pop("endpoint", "https://huggingface.co")
                )
                return to_wire(management[name](request, **data))
            return to_wire(management[name](**data))
        request = ModelArtifactRequest(
            **{k: data.pop(k) for k in ("model", "revision", "endpoint", "projector", "file_selection") if k in data}
        )
        fn = {
            "model_cache.inspect": api.model_cache.inspect_model,
            "model_cache.repair": api.model_cache.repair_model,
            "model_cache.resolve": api.model_cache.resolve_manifest,
            "model_cache.import": api.model_cache.import_manifest,
        }[name]
        if name == "model_cache.import":
            import json
            from sparkrun.models.artifacts import ModelArtifactManifest

            data["manifest"] = ModelArtifactManifest.from_json(json.dumps(data["manifest"]))
        if name != "model_cache.resolve":
            data["ssh_kwargs"] = ssh
        if name == "model_cache.repair":
            data["config"] = sctx.config
        return to_wire(fn(request, **data))
    if name.startswith("tailscale."):
        assert cluster is not None
        action = name.split(".")[1]
        hosts = data.pop("hosts", list(cluster.hosts))
        if action != "expose":
            data["host_list"] = hosts
        # Secrets are requested at execution, never part of the persisted request.
        if action in {"join", "down", "expose"} and data.pop("ask_sudo", False):
            data["sudo_password"] = prompt("secret", "Sudo password", default=None)
        return to_wire(getattr(api.tailscale, action)(sctx=sctx, ssh_kwargs=ssh, **data))
    if name == "setup.rdma_test":
        assert cluster is not None
        hosts = data.pop("hosts", list(cluster.hosts))
        return to_wire(api.setup.rdma_test(sctx, hosts, ssh, cluster=cluster, **data))
    if name.startswith("setup."):
        assert cluster is not None
        from sparkrun.core.setup_actions import SetupActionContext
        from sparkrun.core.setup_manifest import ManifestManager

        hosts = data.get("hosts") or list(cluster.hosts)
        dry_run = data.get("dry_run", False)
        states, context = api.setup.probe_setup_hosts(hosts, ssh_kwargs=ssh, config=sctx.config, cluster=cluster, cluster_name=cluster.name)
        if name == "setup.probe_hosts":
            return to_wire(states)
        if name == "setup.plan":
            return {host: to_wire(api.setup.build_setup_plan(state, context)) for host, state in states.items()}
        action = SetupActionContext(user=sctx.config.ssh_user, ssh_kwargs=ssh, dry_run=dry_run)
        manager = ManifestManager(sctx.cluster_manager.clusters_dir)
        callbacks = {
            "approve": lambda step, targets: bool(prompt("approval", f"Apply {step.key} to {', '.join(targets)}?", default=False)),
            "credentials": lambda: prompt("secret", "Sudo password", default=None),
            "progress_callback": lambda event: sctx.operation.emit("progress", event=to_wire(event)),
        }
        only = set(data["only_steps"]) if data.get("only_steps") else None
        if name == "setup.undo":
            if not cluster.name:
                raise ValueError("Undo requires a named cluster")
            manifest = manager.load(cluster.name, strict=True)
            if manifest is None:
                raise ValueError("No setup manifest for this cluster")
            return to_wire(api.setup.run_setup_undo(manifest, action, manifest_mgr=manager, only_steps=only, **callbacks))
        return to_wire(
            api.setup.run_setup_steps(
                states,
                context,
                action,
                cluster=cluster,
                cluster_name=cluster.name,
                manifest_mgr=manager,
                only_steps=only,
                host_credentials=lambda host: prompt("secret", f"Sudo password for {host}", default=None),
                **callbacks,
            )
        )
    raise LookupError(name)


def _load(name):
    return partial(execute, name)


def register_special_ops():
    from sparkrun.daemon.request_fields import fields

    specs = [
        ("diagnostics.collect", "read", "hosts cluster workload_id log_tail max_bytes dry_run", ""),
        ("recipes.validate", "read", "recipe overrides", "recipe"),
        ("setup.probe_hosts", "read", "hosts cluster", ""),
        ("setup.plan", "read", "hosts cluster", ""),
        ("setup.run_steps", "admin", "hosts cluster dry_run only_steps", ""),
        ("setup.undo", "admin", "cluster dry_run only_steps", "cluster"),
        (
            "setup.rdma_test",
            "operate",
            "hosts cluster suite parallel_pairs full image duration queue_pairs msg_size nccl_binary gid_index link_type force_container keep_containers dry_run",
            "",
        ),
        ("tailscale.status", "read", "hosts cluster", ""),
        ("tailscale.join", "admin", "hosts cluster tag ephemeral enable_ssh hostname dry_run ask_sudo", ""),
        ("tailscale.down", "admin", "hosts cluster remove dry_run ask_sudo", ""),
        ("tailscale.expose", "admin", "cluster proxy head_host port set_proxy_host ask_sudo", ""),
        ("model_cache.inspect", "read", "model revision endpoint projector file_selection cache_dir hosts checksums", "model cache_dir"),
        (
            "model_cache.repair",
            "operate",
            "model revision endpoint projector file_selection cache_dir hosts local_cache_dir transfer_mode checksums offline dry_run",
            "model cache_dir",
        ),
        (
            "model_cache.import",
            "operate",
            "model revision endpoint projector file_selection manifest cache_dir hosts checksums",
            "model manifest cache_dir",
        ),
        ("model_cache.resolve", "read", "model revision endpoint projector file_selection cache_dir", "model"),
        ("model_cache.list", "read", "cache_dir hosts", "cache_dir"),
        ("model_cache.prune", "admin", "cache_dir hosts dry_run min_age_days max_bytes min_free_bytes", "cache_dir"),
        ("model_cache.pin", "operate", "cache_dir hosts model revision endpoint pinned", "cache_dir model revision"),
        ("model_cache.delete", "admin", "cache_dir hosts model revision dry_run force allowed_revisions", "cache_dir model"),
        (
            "model_cache.adopt",
            "admin",
            "cache_dir hosts model revision endpoint all_models dry_run offline last_used pinned",
            "cache_dir",
        ),
        (
            "model_cache.reconcile",
            "admin",
            "cache_dir hosts model revision endpoint claim confirm_inactive",
            "cache_dir model revision claim confirm_inactive",
        ),
    ]
    for name, scope, names, required in specs:
        properties = fields(names)
        if name == "diagnostics.collect":
            properties["max_bytes"] = {"type": "integer", "minimum": 16384, "maximum": 67108864}
        if "confirm_inactive" in properties:
            properties["confirm_inactive"] = {"type": "boolean", "enum": [True]}
        register_daemon_op(
            DaemonOp(
                name,
                partial(_load, name),
                object_schema(properties, required.split()),
                result_schema=RESULT_SCHEMAS[name],
                kind="job",
                scope=scope,
                description=name.replace(".", " ").replace("_", " ").capitalize() + ".",
            )
        )
