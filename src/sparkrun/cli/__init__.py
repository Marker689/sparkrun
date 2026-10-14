"""sparkrun CLI — launch inference workloads on DGX Spark."""

from __future__ import annotations

import click

from sparkrun import __version__
from sparkrun.core.application_profile import get_application_profile
from .ext import PluggableGroup
from ._common import (
    RECIPE_NAME,
    RECIPE_QUERY,
    REGISTRY_NAME,
    RUNTIME_NAME,
    _setup_logging,
    _get_context as _get_context,
    install_termination_handlers,
    dry_run_option,
    host_options,
    json_option,
)
from ._adv import adv
from ._model_cache import model_cache
from ._build import build
from ._benchmark import benchmark
from ._cluster import cluster, cluster_status
from ._export import export
from ._proxy import proxy
from . import _proxy_systemd as _proxy_systemd  # noqa: E402, F401  — registers `proxy systemd`
from ._recipe import recipe, recipe_list, recipe_search, recipe_show
from ._registry import registry, registry_update as registry_update
from ._run import run
from ._setup import setup
from ._plugins import plugins
from ._daemon import daemon
from ._stop_logs import logs_cmd, stop
from ._tune import tune


def _print_version(ctx, param, value):
    """Eager callback for --version: render the channel-aware display string."""
    if not value or ctx.resilient_parsing:
        return
    try:
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.core.version import display_version

        rendered = display_version(SparkrunConfig())
    except Exception:
        rendered = __version__
    click.echo("%s, version %s" % (get_application_profile().command, rendered))
    ctx.exit()


@click.group(
    cls=PluggableGroup,
    help=get_application_profile().description,
    epilog="\n".join(link for link in (get_application_profile().documentation_url, get_application_profile().support_url) if link) or None,
)
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v detail, -vv timestamps, -vvv debug)")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all output except errors (for scripting)")
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_print_version,
    help="Show the version and exit.",
)
@click.pass_context
def main(ctx, verbose, quiet):
    """{app_command} — Launch inference workloads on NVIDIA DGX Spark systems."""
    ctx.ensure_object(dict)
    if quiet:
        verbose = -1  # sentinel: WARNING+ only
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)
    # SIGTERM/SIGHUP unwind like Ctrl-C so a killed launch tears down its ssh
    # children (and, through them, the remote session guard) instead of
    # orphaning multi-GB work on the cluster.
    install_termination_handlers()


# Register command groups and commands
main.add_command(run)
main.add_command(build)
main.add_command(model_cache)
main.add_command(stop)
main.add_command(logs_cmd)
main.add_command(setup)
main.add_command(plugins)
main.add_command(daemon)
main.add_command(tune)
main.add_command(cluster)
main.add_command(adv)
main.add_command(recipe)
main.add_command(registry)
main.add_command(benchmark)
main.add_command(export)
main.add_command(proxy)


# ---------------------------------------------------------------------------
# Top-level aliases
# ---------------------------------------------------------------------------


@main.command("list")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.argument("query", type=RECIPE_QUERY, required=False)
@click.option("--all", "-a", "show_all", is_flag=True, help="Include hidden registry recipes")
@json_option()
@click.pass_context
def list_cmd(ctx, registry, runtime, query, show_all, output_json):
    """List available recipes (alias for 'recipe list').

    QUERY may be scoped to a registry with the ``@registry`` shorthand
    (``@community`` or ``@community/qwen``).
    """
    ctx.invoke(recipe_list, registry=registry, runtime=runtime, query=query, show_all=show_all, output_json=output_json)


@main.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--no-vram", is_flag=True, help="Skip VRAM estimation")
@click.option("--tp", "--tensor-parallel", "tensor_parallel", type=int, default=None, help="Override tensor parallelism")
@click.option("--gpu-mem", type=float, default=None, help="Override GPU memory utilization (0.0-1.0)")
@click.pass_context
def show(ctx, recipe_name, no_vram, tensor_parallel, gpu_mem):
    """Show detailed recipe information (alias for 'recipe show')."""
    ctx.invoke(recipe_show, recipe_name=recipe_name, no_vram=no_vram, tensor_parallel=tensor_parallel, gpu_mem=gpu_mem)


@main.command("search")
@click.option("--registry", type=REGISTRY_NAME, default=None, help="Filter by registry name")
@click.option("--runtime", type=RUNTIME_NAME, default=None, help="Filter by runtime (e.g. vllm, sglang, llama-cpp)")
@click.argument("query", type=RECIPE_QUERY)
@click.pass_context
def search_cmd(ctx, registry, runtime, query):
    """Search for recipes by name, model, or description (alias for 'recipe search').

    QUERY may be scoped to a registry with the ``@registry`` shorthand
    (``@community`` or ``@community/qwen``).
    """
    ctx.invoke(recipe_search, registry=registry, runtime=runtime, query=query)


@main.command("status")
@host_options
@dry_run_option
@json_option()
@click.pass_context
def status(ctx, hosts, hosts_file, cluster_name, dry_run, output_json):
    """Show {app_command} containers running on cluster hosts (alias for 'cluster status')."""
    ctx.invoke(
        cluster_status,
        hosts=hosts,
        hosts_file=hosts_file,
        cluster_name=cluster_name,
        dry_run=dry_run,
        output_json=output_json,
    )


@main.command("update")
@click.option("--no-update-plugins", is_flag=True, help="Audit compatibility without fetching or updating plugins")
@click.option("--no-update-registries", is_flag=True, help="Skip updating recipe registries")
@click.option("--stable", is_flag=True, help="Switch to and update the stable channel from its configured source")
@click.option("--beta", is_flag=True, help="Switch to and update the beta channel from its configured source")
@click.option("--alpha", is_flag=True, help="Switch to and update the alpha channel from its configured source")
@click.option("--yolo", is_flag=True, help="Alias for --alpha")
@click.pass_context
def update(ctx, stable, beta, alpha, yolo, no_update_plugins=False, no_update_registries=False):
    """Update {app_command}, managed plugins, and recipe registries.

    Attempts to upgrade {app_command} via uv if it was installed that way, then
    updates compatible managed plugins and recipe registries. With no channel
    flag, updates the currently configured channel; a channel flag switches
    channels. A failed upgrade exits non-zero.

    If {app_command} was not installed via uv (e.g. pip, pipx, editable
    install), the self-upgrade step is skipped; compatible managed plugins
    and registries are still updated.
    """
    from sparkrun.cli._plugins import management_config
    from sparkrun.cli._self_update import channel_from_flags, run_update

    config = management_config(ctx)
    status = run_update(
        config,
        command=f"{get_application_profile().command} update",
        requested=channel_from_flags(stable, beta, alpha, yolo),
        update_plugins=not no_update_plugins,
        update_registries=not no_update_registries,
    )
    ctx.exit(status)
