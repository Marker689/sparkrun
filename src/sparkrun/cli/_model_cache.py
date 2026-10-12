"""Operator-visible manifest and cache migration commands."""

from dataclasses import asdict
from functools import wraps
import json
from pathlib import Path

import click

from sparkrun.api import model_cache as api
from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest, ModelArtifactRequest
from ._common import HIDE_ADVANCED_OPTIONS, _get_context


@click.group("model-cache", hidden=HIDE_ADVANCED_OPTIONS)
def model_cache():
    """Inspect, repair, retain and prune model caches on selected hosts."""


endpoint_option = click.option(
    "--endpoint",
    default="https://huggingface.co",
    envvar="HF_ENDPOINT",
    help="Model origin URL (defaults to HF_ENDPOINT or huggingface.co).",
)


def options(function):
    @wraps(function)
    def wrapped(ctx, model, revision, file_selection, projector, endpoint, cache_dir, hosts, checksums, **kwargs):
        from sparkrun.core.config import resolve_hf_cache_home
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        sctx = _get_context(ctx)
        try:
            request = ModelArtifactRequest(model, revision, endpoint, projector, file_selection)
            result = function(
                request,
                cache_dir=resolve_hf_cache_home(cache_dir),
                hosts=hosts or ("localhost",),
                checksums=checksums,
                ssh_kwargs=build_ssh_kwargs(sctx.config),
                config=sctx.config,
                **kwargs,
            )
        except (ModelArtifactError, OSError) as error:
            raise click.ClickException(str(error)) from error
        if result is not None:
            click.echo(json.dumps([asdict(report) for report in result], indent=2))
            if any(not report.complete for report in result):
                ctx.exit(1)

    for decorator in (
        click.option("--checksum", "checksums", is_flag=True, help="Hash selected files; normal inspection checks size and readability."),
        click.option("--host", "hosts", multiple=True, help="Target host (repeatable; default localhost)."),
        click.option("--cache-dir", default=None, help="HF_HOME on the target hosts."),
        endpoint_option,
        click.option("--projector", default=None),
        click.option("--file-selection", type=click.Choice(["auto", "all", "safetensors"]), default="auto", show_default=True),
        click.option("--revision", default=None),
        click.argument("model"),
    ):
        wrapped = decorator(wrapped)
    return click.pass_context(wrapped)


@model_cache.command("inspect")
@options
def inspect_cache(request, *, config, **kwargs):
    """Observe saved inventory without network access or modifying the cache."""
    return api.inspect_model(request, **kwargs)


@model_cache.command("resolve")
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Write the portable manifest here.")
@options
def resolve(request, *, output, cache_dir, **kwargs):
    """Resolve origin metadata online; download only the small weight index."""
    from sparkrun.core.config import resolve_hf_token

    manifest = api.resolve_manifest(request, token=resolve_hf_token(), cache_dir=cache_dir)
    output.write_text(manifest.to_json() + "\n")
    click.echo("Wrote %d files at revision %s to %s" % (len(manifest.files), manifest.revision, output))


@model_cache.command("import")
@click.option("--manifest", "manifest_file", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@options
def import_cache(request, *, manifest_file, config, **kwargs):
    """Import trusted expected metadata and report current completeness."""
    return api.import_manifest(request, ModelArtifactManifest.from_json(manifest_file.read_text()), **kwargs)


@model_cache.command("repair")
@click.option("--offline", is_flag=True, help="Permit cluster copies only; no origin calls or tool installation.")
@click.option("--dry-run", is_flag=True)
@click.option("--local-cache-dir", default=None)
@click.option("--transfer-mode", type=click.Choice(["local", "push", "delegated", "pull"]), default="local")
@options
def repair(request, **kwargs):
    """Repair selected paths, then validate every target against the manifest."""
    from sparkrun.core.config import resolve_hf_token

    prepared = api.repair_model(request, token=resolve_hf_token(), **kwargs)
    if prepared is None:
        click.echo("Dry run: no model metadata acquired or caches changed.")
        return None
    return tuple(binding.report for binding in prepared.bindings)


def cache_options(function):
    @wraps(function)
    def wrapped(ctx, cache_dir, hosts, **kwargs):
        from sparkrun.core.config import resolve_hf_cache_home
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        sctx = _get_context(ctx)
        try:
            reports = function(
                cache_dir=resolve_hf_cache_home(cache_dir),
                hosts=hosts or ("localhost",),
                ssh_kwargs=build_ssh_kwargs(sctx.config),
                **kwargs,
            )
        except (ValueError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps([asdict(report) for report in reports], indent=2))
        if any(not r.get("target_met", True) for r in reports):
            ctx.exit(1)

    return click.pass_context(
        click.option("--host", "hosts", multiple=True, help="Cache host; repeat for multiple hosts (default localhost).")(
            click.option("--cache-dir", help="HF_HOME on the target hosts, not its hub subdirectory.")(wrapped)
        )
    )


@model_cache.command("list")
@cache_options
def list_cache(**kwargs):
    """List cached revisions, usage, pins and workload claims without changing files.

    Untracked snapshots are reported but never eligible for pruning. Requires
    Python 3 on cache hosts; no model-origin requests or tool installation.
    """
    return api.inventory_cache(**kwargs)


@model_cache.command("prune")
@click.option("--dry-run/--apply", default=True, show_default=True, help="Preview removals, or explicitly apply the cleanup.")
@click.option(
    "--min-age-days",
    type=click.FloatRange(min=0),
    default=7,
    show_default=True,
    help="Minimum time since preparation, distribution or release.",
)
@click.option("--max-bytes", type=click.IntRange(min=0), help="Target budget for private managed snapshot bytes; excludes shared blobs.")
@click.option("--min-free-bytes", type=click.IntRange(min=0), help="Target free space on the cache filesystem in bytes.")
@cache_options
def prune_cache(**kwargs):
    """Remove eligible revisions in least-recently-used order.

    Defaults to a read-only preview. With no space target, selects all eligible
    revisions older than --min-age-days. Pins, workload claims, busy caches and
    unknown content are protected. Shared HF blobs are never deleted.
    """
    return api.prune_cache(**kwargs)


@model_cache.command("pin")
@click.argument("model")
@click.option("--revision", required=True, help="Full immutable model commit ID.")
@click.option("--remove", is_flag=True, help="Remove the retention pin instead of setting it.")
@endpoint_option
@cache_options
def pin_cache(model, revision, remove, endpoint, **kwargs):
    """Keep a tracked revision out of cleanup on each selected cache host."""
    return api.pin_revision(ModelArtifactRequest(model, revision, endpoint=endpoint), pinned=not remove, **kwargs)


@model_cache.command("reconcile")
@click.argument("model")
@click.option("--revision", required=True, help="Full immutable model commit ID.")
@click.option("--claim", required=True, help="Exact outstanding claim ID shown by model-cache list.")
@click.option("--confirm-inactive", is_flag=True, help="Confirm all workloads/launches sharing this claim have stopped.")
@endpoint_option
@cache_options
def reconcile_cache(model, revision, claim, confirm_inactive, endpoint, **kwargs):
    """Release an interrupted workload's claim after inspecting its hosts.

    Normal stop releases claims automatically. A dead controller or an old
    timestamp does not prove inactivity. Shared-cache claims may cover several
    hosts; inspect all of them before confirming. Does not delete model files.
    """
    if not confirm_inactive:
        raise click.ClickException("Inspect every host in the claim, stop its workloads, then pass --confirm-inactive")
    return api.reconcile_claim(ModelArtifactRequest(model, revision, endpoint=endpoint), claim=claim, confirm_inactive=True, **kwargs)
