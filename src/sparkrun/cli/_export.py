"""sparkrun export command group."""

from __future__ import annotations

from sparkrun.core.application_profile import render_identity_text

import logging
import sys
import json
from pathlib import Path
from typing import NoReturn

import click
import yaml

from ._common import (
    RECIPE_NAME,
    TARGET,
    _apply_recipe_overrides,
    _get_config_and_registry,
    _get_context,
    _is_cluster_id,
    _load_recipe,
    _resolve_hosts_or_exit,
    dry_run_option,
    host_options,
    recipe_override_options,
    json_option,
    print_json,
)

# NOTE: _resolve_hosts_or_exit is retained in _export.py because export_running
# uses it conditionally and does not map cleanly to the @with_host_context
# decorator pattern.

logger = logging.getLogger(__name__)


def _apply_spark_arena_benchmarks(recipe, recipe_name: str):
    """Set metadata.spark_arena_benchmarks when the recipe comes from spark-arena."""
    from sparkrun.core.recipe import SPARK_ARENA_PREFIX

    if recipe.metadata.get("spark_arena_benchmarks"):
        return  # already set

    if recipe_name.startswith(SPARK_ARENA_PREFIX):
        from sparkrun.core.parallelism import extract_parallelism_meta

        uuid = recipe_name[len(SPARK_ARENA_PREFIX) :]
        parallelism_meta = extract_parallelism_meta(recipe.defaults or {})
        entry: dict = {"uuid": uuid}
        entry.update(parallelism_meta)
        if not parallelism_meta:
            entry["tp"] = 1
        recipe.metadata["spark_arena_benchmarks"] = [entry]


@click.group()
@click.pass_context
def export(ctx):
    """Export recipes and running workload configurations."""
    pass


@export.command("recipe")
@click.argument("recipe_name", type=RECIPE_NAME)
@json_option()
@click.option("--save", "save_path", type=click.Path(), help="Save a copy of the recipe to a file")
@click.option(
    "--keep-include",
    is_flag=True,
    help="Emit a recipe that uses include: as written, instead of the flattened, self-contained recipe",
)
@click.option(
    "--realize",
    is_flag=True,
    help="Plan against the target hosts and emit the plain recipe that launch would run (overrides resolved for this hardware)",
)
@click.option(
    "--no-pin", "no_pin", is_flag=True, help="With --realize: keep the image tag and model revision as written instead of pinning them"
)
@click.option(
    "--offline/--online",
    "offline",
    default=None,
    help="With --realize: pin what the hosts already have (resident image, cached model) instead of asking the registry and the "
    "Hub. Defaults to the cluster's offline setting.",
)
@host_options
@recipe_override_options
@click.pass_context
def export_recipe(
    ctx,
    recipe_name,
    output_json=False,
    save_path=None,
    keep_include=False,
    realize=False,
    no_pin=False,
    offline=None,
    hosts=None,
    hosts_file=None,
    cluster_name=None,
    options=(),
    tensor_parallel=None,
    pipeline_parallel=None,
    data_parallel=None,
    gpu_mem=None,
    max_model_len=None,
    image=None,
):
    """Export normalized recipe to stdout or file.

    A recipe built with include: is flattened into one self-contained recipe
    (its bases merged in) unless --keep-include is given.

    --realize probes the target hosts (--hosts / --cluster, or the default
    cluster) and places the workload exactly as run would, then emits an
    ordinary v2 recipe with the matching overrides: layers and any --tp / -o
    values baked in and the overrides: block removed. Nothing is launched.
    Platform defaults (e.g. DGX Spark env and memory utilization) are left to
    the platform, which applies them at launch.

    --realize also pins: the image becomes repo@sha256:... (through the
    recipe's builder, so an eugr :latest pins the nightly it would pull) and
    the model revision a commit. By default those are what the registry and
    the Hub name now; --offline pins what the hosts already hold instead.
    --no-pin keeps the references as written.

    Examples:

      {app_command} export recipe @lil/Qwen3.8-Flash-Next-NVFP4 --realize --tp 2

      {app_command} export recipe my-recipe --realize --cluster mylab --save realized.yaml
    """
    launch_flags = (hosts, hosts_file, cluster_name, tensor_parallel, pipeline_parallel, data_parallel, gpu_mem, max_model_len, image)
    if not realize and (options or any(flag is not None for flag in launch_flags)):
        raise click.UsageError("host and override options (--hosts, --cluster, --tp, -o, ...) only apply with --realize")
    if realize and keep_include:
        raise click.UsageError("--realize emits a flattened recipe; it cannot be combined with --keep-include")
    if not realize and (no_pin or offline is not None):
        raise click.UsageError("--no-pin and --offline only apply with --realize")
    if no_pin and offline:
        raise click.UsageError("--offline chooses where pins come from; it has no effect with --no-pin")
    if realize:
        _export_realized_recipe(
            ctx,
            recipe_name,
            output_json=output_json,
            save_path=save_path,
            pin=not no_pin,
            offline=offline,
            hosts=hosts,
            hosts_file=hosts_file,
            cluster_name=cluster_name,
            options=options,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            data_parallel=data_parallel,
            gpu_mem=gpu_mem,
            max_model_len=max_model_len,
            image=image,
        )
        return

    config, _ = _get_config_and_registry()
    recipe, recipe_path, registry_mgr = _load_recipe(config, recipe_name)

    declared = getattr(recipe, "_include_declared", None)
    if keep_include and declared is not None:
        from sparkrun.utils.yaml_helpers import LiteralBlockDumper

        text = (
            json.dumps(declared, indent=2)
            if output_json
            else yaml.dump(declared, Dumper=LiteralBlockDumper, indent=2, sort_keys=False, default_flow_style=False)
        )
        if save_path is None:
            click.echo(text)
        else:
            Path(save_path).write_text(text, encoding="utf-8")
            click.echo("Recipe saved to %s" % save_path)
        return

    # Auto-populate spark_arena_benchmarks from @spark-arena/UUID references
    _apply_spark_arena_benchmarks(recipe, recipe_name)

    if save_path is None:
        if output_json:
            print_json(recipe.to_dict())
        else:
            click.echo(recipe.export(json=False))
        return

    click.echo("Recipe saved to %s" % recipe.export(path=save_path, json=output_json))


def _export_realized_recipe(ctx, recipe_name, *, output_json, save_path, pin, offline, hosts, hosts_file, cluster_name, options, **flags):
    """``export recipe --realize``: plan like ``run``, then write the recipe that plan would launch."""
    from sparkrun import api
    from sparkrun.cli._common import _enter_offline_mode_for, resolve_host_context
    from sparkrun.utils.yaml_helpers import LiteralBlockDumper

    sctx = _get_context(ctx)
    config = sctx.config
    hctx = resolve_host_context(hosts, hosts_file, cluster_name, config, sctx.variables, sctx=sctx)
    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name, resolve=False, offline=_enter_offline_mode_for(offline, hctx))
    recipe, overrides = _apply_recipe_overrides(options, recipe=recipe, **flags)

    local_cache_dir, remote_cache_dir, _mode, _iface = hctx.cluster.resolve_transfer_config(config)
    run_options = api.RunOptions(
        recipe=recipe,
        hosts=tuple(hctx.host_list),
        cluster=hctx.cluster_name,
        overrides=dict(overrides),
        cache_dir=remote_cache_dir,
        local_cache_dir=local_cache_dir,
    )
    try:
        realized = api.realize_recipe(run_options, pin=pin, offline=offline, sctx=sctx)
    except api.SparkrunError as exc:
        click.echo("Error: %s" % exc, err=True)
        sys.exit(1)

    info = realized.recipe["metadata"]["realized_for"]
    click.echo(
        "Realized for %s (%d node(s)); %s"
        % (info.get("platform") or info.get("accelerator") or "unknown hardware", info["nodes"], hctx.describe()),
        err=True,
    )
    if realized.plan.override_resolution is not None:
        for line in realized.plan.override_resolution.describe():
            click.echo("  %s" % line, err=True)
    if realized.dropped_keys:
        click.echo("  dropped override-only keys: %s" % ", ".join(realized.dropped_keys), err=True)
    for pinned in realized.pins:
        click.echo("  pinned %s: %s -> %s (%s)" % (pinned.field, pinned.before, pinned.after, pinned.source), err=True)
    for note in realized.unpinned:
        click.echo("  not pinned: %s" % note, err=True)

    text = (
        json.dumps(realized.recipe, indent=2)
        if output_json
        else yaml.dump(realized.recipe, Dumper=LiteralBlockDumper, indent=2, sort_keys=False, default_flow_style=False)
    )
    if save_path is None:
        click.echo(text)
    else:
        Path(save_path).write_text(text, encoding="utf-8")
        click.echo("Recipe saved to %s" % save_path, err=True)


@export.command("running-recipe")
@click.argument("target", type=TARGET)
@host_options
@json_option()
@click.option("--save", "save_path", type=click.Path(), help="Save to a file")
@click.pass_context
def export_running(ctx, target, hosts, hosts_file, cluster_name, output_json, save_path):
    """Export the effective recipe from a running workload.

    TARGET can be a recipe name or a cluster ID (from {app_command} status output).
    The exported recipe includes all applied overrides baked into defaults.

    Examples:

      {app_command} export running-recipe e5f6a7b8

      {app_command} export running-recipe glm-4.7-flash-awq --cluster mylab

      {app_command} export running-recipe e5f6a7b8 --json

      {app_command} export running-recipe e5f6a7b8 --save effective-recipe.yaml
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import load_job_metadata, derive_cluster_id

    config, _ = _get_config_and_registry()

    # Resolve cluster_id
    if (cluster_id := _is_cluster_id(target)) is None:
        # Target is a recipe name — need hosts to generate cluster_id
        recipe, _recipe_path, _registry_mgr = _load_recipe(config, target)
        host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
        cluster_id = derive_cluster_id(recipe, host_list)

    # Load job metadata
    meta = load_job_metadata(cluster_id, cache_dir=str(config.cache_dir))
    if not meta:
        click.echo("Error: No job metadata found for '%s'." % target, err=True)
        click.echo("  The workload may not have been launched from this machine.", err=True)
        sys.exit(1)

    # Reconstruction strategy (priority order):
    # 1. recipe_state in metadata → full fidelity deserialization
    # 2. recipe name + overrides → load recipe, export with overrides
    # 3. recipe name + individual fields → partial override reconstruction
    recipe_state = meta.get("recipe_state")
    meta_overrides = meta.get("overrides")
    effective_image = meta.get("effective_container_image")
    # The caller's executor layer at launch (-o entrypoint='', --executor).
    executor_overrides = meta.get("executor_overrides")

    if recipe_state:
        # Best path: full-fidelity reconstruction
        recipe = Recipe._deserialize(recipe_state)
        _output_export(
            recipe, output_json, save_path, overrides=meta_overrides, container_image=effective_image, executor_overrides=executor_overrides
        )
        return

    # Fallback: load recipe by name from registries
    recipe_name = meta.get("recipe")
    if not recipe_name:
        click.echo("Error: Job metadata for '%s' has no recipe name or state." % target, err=True)
        sys.exit(1)

    try:
        recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)
    except SystemExit:
        click.echo("Error: Could not load recipe '%s' from registries." % recipe_name, err=True)
        click.echo("  The original recipe may have been removed or renamed.", err=True)
        sys.exit(1)

    if meta_overrides:
        # Good fallback: full overrides dict available
        _output_export(
            recipe, output_json, save_path, overrides=meta_overrides, container_image=effective_image, executor_overrides=executor_overrides
        )
        return

    # Legacy fallback: reconstruct partial overrides from individual metadata fields
    legacy_overrides = {}
    if meta.get("tensor_parallel") is not None:
        legacy_overrides["tensor_parallel"] = meta["tensor_parallel"]
    if meta.get("port") is not None:
        legacy_overrides["port"] = meta["port"]
    if meta.get("served_model_name") is not None:
        legacy_overrides["served_model_name"] = meta["served_model_name"]

    _output_export(
        recipe,
        output_json,
        save_path,
        overrides=legacy_overrides or None,
        container_image=effective_image,
        executor_overrides=executor_overrides,
    )


def _output_export(recipe, output_json, save_path, overrides=None, container_image=None, executor_overrides=None):
    """Export recipe to stdout or file with optional overrides baked in.

    Executor keys among *overrides* (jobs recorded before the split) move to
    ``executor_config`` rather than reaching ``defaults`` as serve flags.
    """
    from sparkrun.core.resolve import split_executor_overrides

    overrides, lifted = split_executor_overrides(overrides)
    executor_overrides = {**(executor_overrides or {}), **lifted} or None
    kwargs = {"overrides": overrides or None, "container_image": container_image, "executor_overrides": executor_overrides}
    if save_path is None:
        if output_json:
            print_json(recipe.to_dict(**kwargs))
        else:
            click.echo(recipe.export(json=False, **kwargs))
    else:
        click.echo("Recipe saved to %s" % recipe.export(path=save_path, json=output_json, **kwargs))


# ---------------------------------------------------------------------------
# systemd export (renderer over sparkrun.api.systemd_export)
# ---------------------------------------------------------------------------


@export.command("systemd")
@click.argument("target", type=TARGET)
@host_options
@recipe_override_options
@click.option("--port", type=int, default=None, help="Override service port")
@click.option("--served-model-name", default=None, help="Override served model name")
@click.option("--install", "do_install", is_flag=True, help="Deploy service to head node")
@click.option("--uninstall", "do_uninstall", is_flag=True, help="Remove service from head node")
@click.option("--start", is_flag=True, help="Start service after install (implies --install)")
@click.option("--service-name", default=None, help="Override service name (default: {resource_namespace}-<slug>)")
@click.option(
    "--pin",
    is_flag=True,
    help="Pin a mutable container tag and model revision (image digest, model commit) so every restart serves the same build",
)
@click.option(
    "--offline/--online",
    "offline",
    default=None,
    help="With --pin: pin what the cluster's hosts already hold instead of asking the registry and the Hub. "
    "Defaults to the cluster's offline setting.",
)
@click.option("--trust", is_flag=True, help="Export an untrusted recipe's hooks and isolation overrides after auditing it")
@dry_run_option
@click.pass_context
def export_systemd(
    ctx,
    target,
    hosts,
    hosts_file,
    cluster_name,
    tensor_parallel,
    pipeline_parallel,
    data_parallel,
    gpu_mem,
    max_model_len,
    options,
    image,
    port,
    served_model_name,
    do_install,
    do_uninstall,
    start,
    service_name,
    pin,
    offline,
    trust,
    dry_run,
):
    """Generate a systemd service for a {app_command} inference workload.

    TARGET can be a recipe name (with optional overrides) or a cluster_id
    (from a running workload).

    By default, prints generated artifacts (unit file, scripts) for review.
    Use --install to deploy to the head node, or --uninstall to remove.
    Rendering requires an existing remote executable; --dry-run uses placeholder
    paths. Installation can bootstrap a missing uv tool, but never replaces an
    existing environment. Use setup install on that host to repair/update it.

    The service runs against a copy of the cluster (hosts, user, executor,
    transport, scheduler), so placement is decided by run at each start.

    --pin rewrites a mutable image tag as repo@sha256:... and the model
    revision as a commit (references already pinned are kept), so restarts
    never pick up a newer build. --offline pins what every cluster host
    already holds instead of asking the registry and the Hub.

    The baked recipe is trusted on the head node, so a recipe from an
    untrusted source that runs hooks or overrides container isolation needs
    --trust.

    \b
    Examples:
      {app_command} export systemd qwen3-1.7b --cluster mylab
      {app_command} export systemd qwen3-1.7b --cluster mylab --pin
      {app_command} export systemd qwen3-1.7b --cluster mylab --install --start
      {app_command} export systemd qwen3-1.7b --cluster mylab --uninstall
      {app_command} export systemd e5f6a7b8 --install
    """
    from sparkrun import api
    from sparkrun.api import systemd_export as svc
    from sparkrun.cli._common import resolve_host_context

    if do_install and do_uninstall:
        click.echo("Error: --install and --uninstall are mutually exclusive.", err=True)
        sys.exit(1)

    if start:
        do_install = True
    if offline is not None and not pin:
        raise click.UsageError("--offline/--online choose where pins come from; they only apply with --pin")
    if do_uninstall and (pin or trust):
        raise click.UsageError("--pin and --trust apply to what gets installed, not to --uninstall")

    sctx = _get_context(ctx)
    choices = {"pin": pin, "offline": offline, "trust": trust}
    config = sctx.config

    if (cluster_id := _is_cluster_id(target)) is not None:
        # A recorded workload: its own hosts and cluster unless overridden.
        source = {}
        if hosts or hosts_file or cluster_name:
            hctx = resolve_host_context(hosts, hosts_file, cluster_name, config, sctx.variables, sctx=sctx)
            source = {"hosts": tuple(hctx.host_list), "cluster": hctx.cluster_name}
        export_options = svc.SystemdExportOptions(job=cluster_id, service_name=service_name, dry_run=dry_run, **choices, **source)
    else:
        recipe, _, _ = _load_recipe(config, target, resolve=False)
        extra_kw = {}
        if port is not None:
            extra_kw["port"] = port
        if served_model_name is not None:
            extra_kw["served_model_name"] = served_model_name
        recipe, overrides = _apply_recipe_overrides(
            options,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            data_parallel=data_parallel,
            gpu_mem=gpu_mem,
            max_model_len=max_model_len,
            image=image,
            recipe=recipe,
            **extra_kw,
        )
        hctx = resolve_host_context(hosts, hosts_file, cluster_name, config, sctx.variables, sctx=sctx)
        export_options = svc.SystemdExportOptions(
            recipe=recipe,
            overrides=dict(overrides),
            hosts=tuple(hctx.host_list),
            cluster=hctx.cluster_name,
            service_name=service_name,
            dry_run=dry_run,
            **choices,
        )

    try:
        plan = svc.plan(export_options, sctx=sctx)
    except api.JobNotFound:
        _systemd_error("No job metadata found for '%s'." % target)
    except api.SparkrunError as exc:
        _systemd_error(exc)
    for pinned in plan.pins:
        click.echo("Pinned %s: %s -> %s (%s)" % (pinned.field, pinned.before, pinned.after, pinned.source), err=True)

    if do_uninstall:
        _do_uninstall(plan)
        return
    if do_install and plan.trust_required:
        # Before any bootstrap install on the head node, not only at the file write.
        _systemd_error(plan.trust_required)

    # Rendering needs real deployment paths, but cannot authorize installation.
    host = svc.locate_host(plan)
    if host is None:
        head = plan.head_host
        if not do_install:
            raise click.ClickException(
                render_identity_text(
                    "{app_command} not found on '%s'; no installation was attempted. "
                    "Install it on the head node, or re-run with --install to deploy "
                    "(--dry-run previews artifacts with placeholder paths)." % head
                )
            )
        click.echo(render_identity_text("{app_command} not found on %s, attempting auto-install..." % head))
        try:
            bootstrap = svc.bootstrap_host(plan)
        except api.SparkrunError as exc:
            _systemd_error(exc)
        if bootstrap.dry_run:
            from sparkrun.utils.shell import quote

            click.echo(
                "Would install on %s via: uv tool install %s (only if its uv environment is absent)" % (head, quote(bootstrap.requirement))
            )
        elif bootstrap.output:
            click.echo(bootstrap.output, err=True)
        if not bootstrap.success:
            click.echo(render_identity_text("Error: Failed to install {app_command} on '%s'." % head), err=True)
            click.echo("  Use the application's guarded setup install command on the head node, then retry.", err=True)
            sys.exit(1)
        host = svc.locate_host(plan)
        if host is None:
            click.echo(render_identity_text("Error: {app_command} installed but not found on PATH on '%s'." % head), err=True)
            sys.exit(1)
        click.echo(render_identity_text("{app_command} installed on %s" % head))

    artifacts = svc.render(plan, host)

    if not do_install:
        # Review mode: display all generated artifacts
        click.echo("=" * 60)
        click.echo(render_identity_text("systemd service: {resource_namespace}-%s" % plan.slug))
        click.echo("Head node: %s" % plan.head_host)
        click.echo("Cluster hosts: %s" % ", ".join(plan.service_cluster.hosts))
        click.echo("=" * 60)
        click.echo()
        click.echo(render_identity_text("--- Unit file: /etc/systemd/system/{resource_namespace}-%s.service ---" % plan.slug))
        click.echo(artifacts.unit_text)
        click.echo("--- Baked recipe ---")
        click.echo(plan.recipe_yaml)
        click.echo("--- Cluster definition: %s ---" % plan.cluster_name)
        click.echo(plan.cluster_yaml)
        click.echo("--- Install script (user-level) ---")
        click.echo(artifacts.install_script)
        click.echo("--- Install script (sudo) ---")
        click.echo(artifacts.sudo_install_script)
        if plan.trust_required:
            click.echo("Note: %s" % plan.trust_required, err=True)
        click.echo("To deploy, re-run with --install")
        return

    _do_install(plan, artifacts, start)


def _systemd_error(exc) -> NoReturn:
    """Print an export failure (with any remote stderr) and exit 1."""
    click.echo("Error: %s" % exc, err=True)
    detail = getattr(exc, "detail", "")
    if detail:
        click.echo("  %s" % detail, err=True)
    sys.exit(1)


def _do_install(plan, artifacts, start):
    """Deploy service files and systemd unit to head node."""
    from sparkrun import api
    from sparkrun.api import systemd_export as svc

    try:
        # Step 1: User-level script (create dirs, write recipe + cluster YAML)
        click.echo("Installing service files on %s..." % plan.head_host)
        out = svc.install_files(plan, artifacts)
        if out:
            click.echo("  %s" % out)

        # Step 2: Sudo script (write unit file, daemon-reload, enable)
        click.echo("Installing systemd unit (requires sudo)...")
        sudo_password = click.prompt("[sudo] password for %s" % plan.ssh_user, hide_input=True)
        out = svc.install_unit(plan, artifacts, sudo_password=sudo_password)
        if out:
            click.echo("  %s" % out)

        # Step 3: Optionally start the service
        if start:
            click.echo("Starting %s..." % plan.service)
            svc.start(plan, sudo_password=sudo_password)
            click.echo("  %s started" % plan.service)
    except api.SparkrunError as exc:
        _systemd_error(exc)

    click.echo()
    click.echo("Service %s installed on %s." % (plan.service, plan.head_host))
    click.echo("  Status: systemctl status %s" % plan.service)
    click.echo("  Logs:   journalctl -u %s -f" % plan.service)


def _do_uninstall(plan):
    """Remove service from head node."""
    from sparkrun import api
    from sparkrun.api import systemd_export as svc

    try:
        uninstall_script = svc.render_uninstall(plan)
        if plan.dry_run:
            click.echo("[dry-run] Would uninstall %s from %s" % (plan.service, plan.head_host))
            click.echo()
            click.echo("--- Uninstall script ---")
            click.echo(uninstall_script)
            return

        click.echo("Uninstalling %s from %s (requires sudo)..." % (plan.service, plan.head_host))
        sudo_password = click.prompt("[sudo] password for %s" % plan.ssh_user, hide_input=True)
        out = svc.uninstall(plan, uninstall_script, sudo_password=sudo_password)
    except api.SparkrunError as exc:
        _systemd_error(exc)
    for line in out.splitlines():
        click.echo("  %s" % line)
    click.echo("Service %s removed from %s." % (plan.service, plan.head_host))
