"""Operator-visible manifest and cache migration commands."""

from dataclasses import asdict
from functools import wraps
import json
from pathlib import Path
import sys

import click

from sparkrun.api import model_cache as api
from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest, ModelArtifactRequest
from sparkrun.utils.cli_formatters import format_bytes, format_cache_validation_reports, format_model_cache_reports
from ._common import HIDE_ADVANCED_OPTIONS, _get_cluster_manager, _get_context, host_options, json_option, resolve_host_context


@click.group("model-cache", hidden=HIDE_ADVANCED_OPTIONS)
def model_cache():
    """Inspect, repair, retain and prune model caches on cluster hosts.

    Targets the hosts of --cluster, --hosts or --hosts-file, else the default
    cluster, else config default_hosts, else localhost.
    """


def _resolve_targets(sctx, *, legacy_hosts, hosts, hosts_file, cluster_name, cache_dir):
    """Resolve target hosts and their HF home like every other cluster command.

    Returns ``(host_list, cache_dir, banner)``. ``--host`` (repeatable, legacy)
    is merged into ``--hosts``. With no host source at all the control
    machine's own cache (localhost) is the target, as before clusters were
    supported. The cache dir is ``--cache-dir`` → the cluster's ``cache_dir`` →
    the head's HF home probed over SSH (what a launch would use), so a remote
    host is never pointed at the control machine's home directory.
    """
    from sparkrun.core.launcher import resolve_effective_cache_dir
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    config = sctx.config
    hosts = ",".join([*legacy_hosts, *([hosts] if hosts else [])]) or None
    if not (hosts or hosts_file or cluster_name or config.default_hosts or _get_cluster_manager(sctx=sctx).get_default()):
        host_list, cluster_cache, banner = ["localhost"], None, "Hosts: localhost (no cluster configured)"
    else:
        hctx = resolve_host_context(hosts, hosts_file, cluster_name, config, sctx=sctx)
        host_list, cluster_cache, banner = hctx.host_list, hctx.cluster.cache_dir, hctx.describe()
    cache_dir = resolve_effective_cache_dir(cache_dir or cluster_cache, host_list, build_ssh_kwargs(config), config)
    return host_list, cache_dir, banner


def _target_options(function):
    """``--cluster/--hosts/--hosts-file``, the hidden legacy ``--host`` and ``--cache-dir``."""
    function = click.option("--cache-dir", default=None, help="HF_HOME on the target hosts, not its hub subdirectory.")(function)
    function = click.option("--host", "legacy_hosts", multiple=True, hidden=True, help="Target host (repeatable; legacy).")(function)
    return host_options(function)


endpoint_option = click.option(
    "--endpoint",
    default="https://huggingface.co",
    envvar="HF_ENDPOINT",
    help="Model origin URL (defaults to HF_ENDPOINT or huggingface.co).",
)


def _pop_targets(kwargs):
    return {key: kwargs.pop(key) for key in ("legacy_hosts", "hosts", "hosts_file", "cluster_name", "cache_dir")}


def options(function):
    @wraps(function)
    def wrapped(ctx, model, revision, file_selection, projector, endpoint, checksums, output_json, **kwargs):
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        sctx = _get_context(ctx)
        try:
            host_list, cache_dir, banner = _resolve_targets(sctx, **_pop_targets(kwargs))
            if not output_json:
                click.echo(banner)
            request = ModelArtifactRequest(model, revision, endpoint, projector, file_selection)
            result = function(
                request,
                cache_dir=cache_dir,
                hosts=tuple(host_list),
                checksums=checksums,
                ssh_kwargs=build_ssh_kwargs(sctx.config),
                config=sctx.config,
                **kwargs,
            )
        except (ModelArtifactError, OSError) as error:
            raise click.ClickException(str(error)) from error
        if result is not None:
            if output_json:
                click.echo(json.dumps([asdict(report) for report in result], indent=2))
            else:
                click.echo(format_cache_validation_reports(result))
            if any(not report.complete for report in result):
                ctx.exit(1)

    for decorator in (
        json_option("Print the per-cache reports as JSON."),
        click.option("--checksum", "checksums", is_flag=True, help="Hash selected files; normal inspection checks size and readability."),
        _target_options,
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
@click.argument("model")
@click.option("--revision", default=None)
@click.option("--file-selection", type=click.Choice(["auto", "all", "safetensors"]), default="auto", show_default=True)
@click.option("--projector", default=None)
@endpoint_option
@click.option("--cache-dir", default=None, help="HF_HOME on this control machine, used for the weight index download.")
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Write the portable manifest here.")
def resolve(model, revision, file_selection, projector, endpoint, cache_dir, output):
    """Resolve origin metadata online; download only the small weight index.

    Runs on the control machine only; import the manifest onto cluster hosts.
    """
    from sparkrun.core.config import resolve_hf_cache_home, resolve_hf_token

    try:
        request = ModelArtifactRequest(model, revision, endpoint, projector, file_selection)
        manifest = api.resolve_manifest(request, token=resolve_hf_token(), cache_dir=resolve_hf_cache_home(cache_dir))
        output.write_text(manifest.to_json() + "\n")
    except (ModelArtifactError, OSError) as error:
        raise click.ClickException(str(error)) from error
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


def cache_options(action):
    """Shared host/cache options for retention commands; *action* selects the summary footer."""
    return lambda function: _cache_options(function, action)


def _cache_options(function, action):
    @wraps(function)
    def wrapped(ctx, output_json, **kwargs):
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        sctx = _get_context(ctx)
        show_all = kwargs.pop("show_all", False)
        try:
            host_list, cache_dir, banner = _resolve_targets(sctx, **_pop_targets(kwargs))
            if not output_json:
                click.echo(banner)
            reports = function(
                cache_dir=cache_dir,
                hosts=tuple(host_list),
                ssh_kwargs=build_ssh_kwargs(sctx.config),
                **kwargs,
            )
        except (ValueError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc
        if output_json:
            click.echo(json.dumps([asdict(report) for report in reports], indent=2))
        else:
            click.echo(format_model_cache_reports(reports, action=action, show_untracked=show_all))
        if any(not r.get("target_met", True) for r in reports):
            ctx.exit(1)

    return click.pass_context(_target_options(json_option("Print the per-cache reports as JSON.")(wrapped)))


@model_cache.command("list")
@click.option("--all", "show_all", is_flag=True, help="Also list untracked snapshots (always included in --json).")
@cache_options("inventory")
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
@click.option(
    "--max-bytes",
    type=click.IntRange(min=0),
    help="Target budget for private managed snapshot bytes, including blobs only that snapshot uses.",
)
@click.option("--min-free-bytes", type=click.IntRange(min=0), help="Target free space on the cache filesystem in bytes.")
@cache_options("prune")
def prune_cache(**kwargs):
    """Remove eligible revisions in least-recently-used order.

    Defaults to a read-only preview. With no space target, selects all eligible
    revisions older than --min-age-days. Pins, workload claims, busy caches and
    unknown content are protected. Blobs another revision still uses are kept.
    """
    return api.prune_cache(**kwargs)


@model_cache.command("pin")
@click.argument("model")
@click.option("--revision", required=True, help="Full immutable model commit ID.")
@click.option("--remove", is_flag=True, help="Remove the retention pin instead of setting it.")
@endpoint_option
@cache_options("pin")
def pin_cache(model, revision, remove, endpoint, **kwargs):
    """Keep a tracked revision out of cleanup on each selected cache host."""
    return api.pin_revision(ModelArtifactRequest(model, revision, endpoint=endpoint), pinned=not remove, **kwargs)


@model_cache.command("reconcile")
@click.argument("model")
@click.option("--revision", required=True, help="Full immutable model commit ID.")
@click.option("--claim", required=True, help="Exact outstanding claim ID shown by model-cache list.")
@click.option("--confirm-inactive", is_flag=True, help="Confirm all workloads/launches sharing this claim have stopped.")
@endpoint_option
@cache_options("reconcile")
def reconcile_cache(model, revision, claim, confirm_inactive, endpoint, **kwargs):
    """Release an interrupted workload's claim after inspecting its hosts.

    Normal stop releases claims automatically. A dead controller or an old
    timestamp does not prove inactivity. Shared-cache claims may cover several
    hosts; inspect all of them before confirming. Does not delete model files.
    """
    if not confirm_inactive:
        raise click.ClickException("Inspect every host in the claim, stop its workloads, then pass --confirm-inactive")
    return api.reconcile_claim(ModelArtifactRequest(model, revision, endpoint=endpoint), claim=claim, confirm_inactive=True, **kwargs)


def _interactive() -> bool:
    return sys.stdin.isatty()


@model_cache.command("delete")
@click.argument("model")
@click.option("--revision", default=None, help="Delete only this commit (or ref, e.g. main); default deletes every cached revision.")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted and stop.")
@click.option("--yes", "-y", "assume_yes", is_flag=True, help="Delete without asking for confirmation.")
@click.option("--force", is_flag=True, help="Also delete revisions with outstanding workload claims (likely in use).")
@click.pass_context
@_target_options
@json_option("Print the per-cache reports as JSON.")
def delete_cache(ctx, model, revision, dry_run, assume_yes, force, output_json, **targets):
    """Delete a model, or one revision of it, from the target hosts' caches.

    Unlike prune, pins, retention age and untracked status do not protect a
    revision. Outstanding workload claims do unless --force, and an active
    download always does. If any host would refuse, nothing is deleted on any
    host. Blobs are removed only when nothing else in the cache uses them.
    Saved manifests are kept, so `model-cache repair` can restore the model.
    """
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    sctx = _get_context(ctx)
    try:
        request = ModelArtifactRequest(model, revision)
        host_list, cache_dir, banner = _resolve_targets(sctx, **targets)
        common = dict(cache_dir=cache_dir, hosts=tuple(host_list), force=force, ssh_kwargs=build_ssh_kwargs(sctx.config))
        reports = api.delete_model(request, dry_run=True, **common)
        doomed = sum(1 for r in reports for e in r.entries if e.action == "would_delete")
        proceed = not dry_run and doomed and all(r.target_met for r in reports)
        if not output_json:
            click.echo(banner)
            click.echo(format_model_cache_reports(reports, action="delete"))
        if proceed and not assume_yes:
            if output_json or not _interactive():
                raise click.ClickException("Refusing to delete without confirmation; pass --yes")
            planned = sum(r.planned_bytes for r in reports)
            hosts = sum(len(r.location.hosts) if r.location else 1 for r in reports)
            where = "%d host(s)" % hosts if hosts == len(reports) else "%d cache(s) on %d host(s)" % (len(reports), hosts)
            click.confirm(
                "\nDelete %d revision(s) of %s from %s, freeing ~%s?" % (doomed, request.repo, where, format_bytes(planned)),
                abort=True,
            )
        if proceed:
            confirmed = {e.revision for r in reports for e in r.entries if e.action == "would_delete"}
            reports = api.delete_model(request, dry_run=False, allowed_revisions=confirmed, **common)
            if not output_json:
                click.echo()
                click.echo(format_model_cache_reports(reports, action="delete"))
    except (ModelArtifactError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    if output_json:
        click.echo(json.dumps([asdict(report) for report in reports], indent=2))
    if not doomed and all(r.target_met for r in reports):
        click.echo("%s%s is not cached on the target hosts." % (request.repo, "@" + revision if revision else ""), err=True)
        ctx.exit(1)
    if any(not r.target_met for r in reports):
        ctx.exit(1)
