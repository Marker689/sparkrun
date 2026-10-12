"""Reviewed mapping of public library functions to wire operations or exclusions."""

from sparkrun.daemon.ops import SPECS

API_OPERATIONS = {target: name for name, target, *_ in SPECS}
API_OPERATIONS.update(
    {
        name: name
        for name in (
            "run",
            "plan",
            "materialize",
            "realize_recipe",
            "build",
            "plan_build",
            "benchmark",
            "proxy.start",
            "proxy.install_service",
            "validate_recipe",
        )
    }
)
API_OPERATIONS.update(
    {
        "wait_ready": "run",
        "post_launch": "run",
        "schedule": "plan",
        "logs": "/v1/streams/logs",
        "open_telemetry": "/v1/streams/monitor",
        "open_live_monitor": "/v1/streams/monitor",
        "live_monitor": "/v1/streams/monitor",
        "setup.probe_setup_hosts": "setup.probe_hosts",
        "setup.check_ssh_host_identities": "setup.plan",
        "setup.resolve_setup_context": "setup.plan",
        "setup.build_setup_plan": "setup.plan",
        "setup.evaluate_host": "setup.plan",
        "setup.apply_setup_step": "setup.run_steps",
        "setup.run_setup_steps": "setup.run_steps",
        "setup.run_setup_undo": "setup.undo",
        "setup.rdma_test": "setup.rdma_test",
        "tailscale.status": "tailscale.status",
        "tailscale.join": "tailscale.join",
        "tailscale.down": "tailscale.down",
        "tailscale.expose": "tailscale.expose",
        "model_cache.inspect_model": "model_cache.inspect",
        "model_cache.resolve_manifest": "model_cache.resolve",
        "model_cache.repair_model": "model_cache.repair",
        "model_cache.import_manifest": "model_cache.import",
    }
)
EXCLUSIONS = {
    "default_sctx": "Python application binding; daemon creates an operation-local context.",
    "resolve_catalog_recipe": "Returns a live Recipe; catalog.recipe exposes a bounded preview and run resolves it server-side.",
    "resolve_recipe_filter": "Python filter helper; catalog.recipes and search_recipes accept the same filter inputs.",
    "proxy.admin_token": "Credential issuance is interactive/local; secrets are not persisted in job results.",
    "proxy.ui": "May issue a credential-bearing URL; use the local proxy ui command.",
    "setup.install_public_key_interactive": "Requires a terminal-owned ssh-copy-id/password conversation; use setup in a terminal.",
    "setup.rotate_ssh_host_keys": "Host-identity rotation has no wire operation; use the confirmed setup ssh-rotate-keys CLI or Python API.",
    "setup.build_authorized_key_script": "Script generation helper; setup runner owns execution.",
    "setup.build_collect_key_script": "Script generation helper; mesh operation owns execution.",
    "setup.build_install_keys_script": "Script generation helper; mesh operation owns execution.",
}
K8S_OPERATIONS = {
    name: "plugins.k8s." + name
    for name in (
        "ensure_kubectl",
        "cluster_info",
        "configure_service_account",
        "list_nodes",
        "kueue_status",
        "setup_kueue",
        "stop_jobset",
        "jobset_status",
    )
}
K8S_OPERATIONS.update({"logs": "/v1/streams/logs", "launch_jobset": "run", "check_feasibility": "plan"})
K8S_EXCLUSIONS = {
    "make_client": "Returns an executable Python client, not JSON.",
    "probe_nodes_fallback": "Internal hardware fallback covered by list_nodes.",
    "run_launcher_job": "Low-level arbitrary launcher specification; submit a recipe through run/build instead.",
}
