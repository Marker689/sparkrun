"""Preparation-only build command; hidden with the other advanced commands."""

from __future__ import annotations

from copy import copy
import click

import sparkrun.api as api
from ._common import (
    HIDE_ADVANCED_OPTIONS,
    RECIPE_NAME,
    _apply_recipe_overrides,
    _enter_offline_mode_for,
    _get_context,
    _load_recipe,
    _parse_options,
    dry_run_option,
    host_options,
    json_option,
    print_json,
    recipe_override_options,
    resolve_host_context,
)


@click.command("build", hidden=HIDE_ADVANCED_OPTIONS)
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@recipe_override_options
@dry_run_option
@json_option()
@click.option("--executor", default=None, help="Execution environment to prepare (docker or local).")
@click.option("--cache-dir", default=None, help="Model cache directory on target hosts.")
@click.option("--local-cache-dir", default=None, help="Model download cache on the control host.")
@click.option(
    "--transfer-mode",
    default=None,
    type=click.Choice(["auto", "local", "push", "delegated", "pull"], case_sensitive=False),
    help="Resource transfer strategy; defaults to the cluster setting.",
)
@click.option("--transfer-interface", default=None, help="Network interface for resource transfers.")
@click.option("--rebuild/--no-rebuild", default=None, help="Request a builder rebuild or refresh registry images.")
@click.option(
    "--offline/--online",
    "offline",
    default=None,
    help="Prepare from images, models and recipes that already exist, with no internet access (copies inside the "
    "cluster are allowed). --online overrides a cluster set to offline.",
)
@click.option("--no-sync-tuning", is_flag=True, help="Skip registry tuning refresh; existing tuning files are still staged.")
@click.option("--trust", is_flag=True, help="Authorize recipe-owned preparation hooks and host-path overrides.")
@click.option("--builder-option", "builder_options", multiple=True, help="Builder context: KEY=VALUE (repeatable).")
@click.pass_context
def build(
    ctx,
    recipe_name,
    hosts,
    hosts_file,
    cluster_name,
    tensor_parallel,
    pipeline_parallel,
    data_parallel,
    gpu_mem,
    max_model_len,
    image,
    options,
    dry_run,
    output_json,
    executor,
    cache_dir,
    local_cache_dir,
    transfer_mode,
    transfer_interface,
    rebuild,
    offline,
    no_sync_tuning,
    trust,
    builder_options,
):
    """Build environments and distribute assets without starting inference.

    All selected hosts are prepared, even when their GPUs are occupied.
    For native recipes, prepares the venv and model cache. Model loading,
    kernel warmup, and ColdSnap capture/restore remain separate operations.

    Set SPARKRUN_ADVANCED=1 to show this command in top-level help.
    """
    sctx = _get_context(ctx)
    targets = resolve_host_context(hosts, hosts_file, cluster_name, copy(sctx.config), sctx=sctx)
    offline_early = _enter_offline_mode_for(offline, targets)
    recipe, _, _ = _load_recipe(sctx.config, recipe_name, resolve=False, retry_after_update=True, offline=offline_early)
    recipe, overrides = _apply_recipe_overrides(
        options,
        recipe=recipe,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        data_parallel=data_parallel,
        gpu_mem=gpu_mem,
        max_model_len=max_model_len,
        image=image,
    )
    request = api.BuildOptions(
        recipe=recipe,
        hosts=tuple(targets.host_list),
        cluster=targets.cluster_name,
        overrides=overrides,
        dry_run=dry_run,
        executor=executor,
        cache_dir=cache_dir,
        local_cache_dir=local_cache_dir,
        transfer_mode=transfer_mode,
        transfer_interface=transfer_interface,
        rebuild=rebuild,
        offline=offline,
        sync_tuning=not no_sync_tuning,
        trust=trust,
        builder_context=_parse_options(builder_options),
    )
    try:
        plan = api.plan_build(request, sctx=sctx)
        if not output_json:
            click.echo("Preparing: %s" % recipe.qualified_name)
            click.echo("Hosts:     %s" % ", ".join(plan.host_list))
            click.echo("Builder:   %s" % (plan.recipe.builder or "none (stage existing assets)"))
        for issue in plan.issues:
            click.echo("%s: %s" % (issue.severity.capitalize(), issue.message), err=True)
        result = api.build(request, plan=plan, sctx=sctx)
    except (api.SparkrunError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if output_json:
        print_json(result)
        return
    if result.container_image:
        click.echo("Image:     %s" % result.container_image)
    if result.environment_file:
        click.echo("Environment: %s" % result.environment_file)
    click.echo("Dry-run complete; no assets changed." if dry_run else "Build complete; assets prepared. Inference was not started.")
