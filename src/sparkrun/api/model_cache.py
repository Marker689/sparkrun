"""Explicit model cache inspection, inventory import and targeted repair.

Inspection is read-only and offline. Resolution contacts the model origin;
repair is the only operation here that downloads model payloads. These helpers
never start/evict workloads and do not change api.materialize's contract.

.. warning::
   **Experimental.** Function signatures and the request/report types
   (``ModelArtifactRequest``, ``ModelArtifactManifest``, ``CacheValidationReport``,
   ``CacheReport``) are core types that may change in a minor release.
"""

from __future__ import annotations

import os

from sparkrun.models.artifacts import CacheValidationReport, ModelArtifactManifest, ModelArtifactRequest
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.cache_reports import CacheReport
from sparkrun.models.manifest import check_request, fetch_inventory, manifest_path, select_manifest
from sparkrun.models.preparation import PreparedModel, prepare_model


def _ssh(ssh_kwargs, sctx):
    """SSH arguments for host IO: *sctx*'s configured ones, overridden per key.

    Without *sctx* the explicit arguments are used as given (the historical
    behaviour), so callers that already pass ``ssh_kwargs`` see no change.
    """
    if sctx is None:
        return ssh_kwargs
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    return {**build_ssh_kwargs(sctx.config), **(ssh_kwargs or {})}


def resolve_manifest(request: ModelArtifactRequest, *, token: str | None = None, cache_dir: str | None = None) -> ModelArtifactManifest:
    """Resolve authoritative metadata, including a small weight index if present."""
    return select_manifest(request, fetch_inventory(request, token=token, cache=cache_dir))


def inspect_model(
    request: ModelArtifactRequest, *, cache_dir: str, hosts=("localhost",), checksums=False, ssh_kwargs=None, session=None, sctx=None
) -> tuple[CacheValidationReport, ...]:
    reports = []
    with ModelHostIO(_ssh(ssh_kwargs, sctx), session=session) as io:
        for host in hosts:
            root = io.root(host, cache_dir)
            stored = io.read(host, str(manifest_path(root, request)))
            if stored is None:
                reports.append(
                    CacheValidationReport(
                        host,
                        "",
                        "checksum" if checksums else "metadata",
                        "unverified",
                        detail="no saved manifest; resolve or import authoritative inventory first",
                    )
                )
                continue
            manifest = ModelArtifactManifest.from_json(stored)
            check_request(manifest, request)
            reports.append(io.observe(host, manifest, root, checksums=checksums))
    return tuple(reports)


def import_manifest(
    request: ModelArtifactRequest,
    manifest: ModelArtifactManifest,
    *,
    cache_dir: str,
    hosts=("localhost",),
    checksums=False,
    ssh_kwargs=None,
    session=None,
    sctx=None,
) -> tuple[CacheValidationReport, ...]:
    """Import operator-supplied expected metadata, never infer it by listing files.

    Import can record an incomplete inventory for later repair. The returned
    observations explicitly distinguish that from a complete cache.
    """
    check_request(manifest, request)
    reports = []
    with ModelHostIO(_ssh(ssh_kwargs, sctx), session=session) as io:
        for host in hosts:
            root = io.root(host, cache_dir)
            io.lock(host, root, request.repo)
            reports.append(io.observe(host, manifest, root, checksums=checksums))
            io.publish(host, root, request.repo, str(manifest_path(root, request)), manifest.to_json())
    return tuple(reports)


def repair_model(
    request: ModelArtifactRequest,
    *,
    cache_dir: str,
    hosts=("localhost",),
    local_cache_dir=None,
    transfer_mode="local",
    checksums=False,
    offline=False,
    dry_run=False,
    token=None,
    ssh_kwargs=None,
    config=None,
    session=None,
    manifest=None,
    sctx=None,
) -> PreparedModel | None:
    """Repair selected missing/corrupt paths and revalidate all requested hosts.

    With *sctx*, ``config`` and the SSH arguments default to its configuration
    (explicit values still win, ``ssh_kwargs`` per key).
    """
    return prepare_model(
        request.model,
        list(hosts),
        cache_dir=cache_dir,
        local_cache_dir=local_cache_dir,
        revision=request.revision,
        projector=request.projector,
        file_selection=request.file_selection,
        endpoint=request.endpoint,
        transfer_mode=transfer_mode,
        checksums=checksums,
        offline=offline,
        dry_run=dry_run,
        token=token,
        ssh_kwargs=_ssh(ssh_kwargs, sctx),
        config=config if config is not None or sctx is None else sctx.config,
        session=session,
        manifest=manifest,
    )


def inventory_cache(*, cache_dir, hosts=("localhost",), ssh_kwargs=None, session=None, sctx=None) -> tuple[CacheReport, ...]:
    """Read host-local inventory, usage, pins and outstanding workload claims."""
    from sparkrun.models.cache import cache_operation

    return cache_operation(cache_dir=cache_dir, hosts=hosts, ssh_kwargs=_ssh(ssh_kwargs, sctx), session=session)


def prune_cache(
    *,
    cache_dir,
    hosts=("localhost",),
    dry_run=True,
    min_age_days=7,
    max_bytes=None,
    min_free_bytes=None,
    ssh_kwargs=None,
    session=None,
    sctx=None,
) -> tuple[CacheReport, ...]:
    """Evict eligible revisions in LRU order; preview by default.

    max_bytes budgets private managed snapshot allocations, including blobs only
    that snapshot links to (evicted with it), excluding shared blobs and untracked data. min_free_bytes measures actual filesystem space.
    Without a space target, remove every eligible revision older than min_age_days.
    """
    from sparkrun.models.cache import cache_operation, age_seconds

    return cache_operation(
        cache_dir=cache_dir,
        hosts=hosts,
        ssh_kwargs=_ssh(ssh_kwargs, sctx),
        session=session,
        action="prune",
        dry_run=dry_run,
        min_age_seconds=age_seconds(min_age_days),
        max_bytes=max_bytes,
        min_free_bytes=min_free_bytes,
    )


def pin_revision(
    request, *, cache_dir, hosts=("localhost",), pinned=True, ssh_kwargs=None, session=None, sctx=None
) -> tuple[CacheReport, ...]:
    """Pin/unpin an already tracked immutable revision on selected cache hosts."""
    from sparkrun.models.cache import cache_operation
    from sparkrun.models.usage import usage_path

    usage_path("/", request.repo, request.revision)  # Require a full immutable commit.
    return cache_operation(
        cache_dir=cache_dir,
        hosts=hosts,
        ssh_kwargs=_ssh(ssh_kwargs, sctx),
        session=session,
        action="pin",
        dry_run=False,
        repo=request.repo,
        revision=request.revision,
        endpoint=request.endpoint,
        pinned=pinned,
    )


def reconcile_claim(
    request, *, cache_dir, claim, confirm_inactive=False, hosts=("localhost",), ssh_kwargs=None, session=None, sctx=None
) -> tuple[CacheReport, ...]:
    """Release an interrupted workload claim after explicit operator inspection.

    Normal stop releases exact claims automatically. Never infer inactivity from
    age, a dead controller, an unavailable host, or missing local job metadata.
    """
    from sparkrun.models.cache import cache_operation
    from sparkrun.models.usage import usage_path

    usage_path("/", request.repo, request.revision)
    return cache_operation(
        cache_dir=cache_dir,
        hosts=hosts,
        ssh_kwargs=_ssh(ssh_kwargs, sctx),
        session=session,
        action="reconcile",
        dry_run=False,
        repo=request.repo,
        revision=request.revision,
        endpoint=request.endpoint,
        claim=claim,
        confirm_inactive=confirm_inactive,
    )


def adopt_cache(
    request: ModelArtifactRequest | None = None,
    *,
    cache_dir,
    hosts=("localhost",),
    all_models=False,
    endpoint=None,
    dry_run=True,
    offline=False,
    last_used="now",
    pinned=False,
    token=None,
    ssh_kwargs=None,
    session=None,
    sctx=None,
) -> tuple[CacheReport, ...]:
    """Start tracking snapshots downloaded outside sparkrun; preview by default.

    Adopts every untracked revision of ``request.model`` (or only
    ``request.revision``, a commit or ref), or of every model with
    *all_models*: one of the two is required, so one command cannot enroll
    everything other tools on the host also use. Online (default), each
    present file must match the model origin's inventory for that commit;
    *offline* accepts only snapshots in the Hub's blob-linked layout. Only the
    usage record is written, never a manifest. *last_used* is ``now`` (default:
    nothing becomes prune-eligible on adoption) or ``mtime`` (the snapshot's
    newest entry). *pinned* adopts and protects in one step. *endpoint* is the
    origin for *all_models* (a request carries its own); default ``HF_ENDPOINT``.

    Adopted revisions then age by sparkrun's use only: with automatic pruning
    enabled, one that other tools still use becomes a candidate (with the blobs
    only it uses) once the retention window passes. *pinned* prevents that.
    """
    from sparkrun.core.config import resolve_hf_token
    from sparkrun.models.adoption import adopt_snapshots

    if (request is None) == (not all_models):
        raise ValueError("name a model to adopt, or pass all_models (not both)")
    return adopt_snapshots(
        cache_dir=cache_dir,
        hosts=hosts,
        repo=request.repo if request is not None else None,
        revision=request.revision if request is not None else None,
        endpoint=request.endpoint if request is not None else endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        offline=offline,
        dry_run=dry_run,
        last_used=last_used,
        pinned=pinned,
        token=token if token is not None or offline else resolve_hf_token(),
        ssh_kwargs=_ssh(ssh_kwargs, sctx),
        session=session,
    )


def delete_model(
    request,
    *,
    cache_dir,
    hosts=("localhost",),
    dry_run=True,
    force=False,
    allowed_revisions=None,
    ssh_kwargs=None,
    session=None,
    sctx=None,
) -> tuple[CacheReport, ...]:
    """Delete one revision (``request.revision``), or every revision, of a model; preview by default.

    Unlike prune this ignores pins, retention age and tracking: naming the model
    is the authorization. Revisions with outstanding workload claims are refused
    unless *force*, and an active download or busy publication lock always
    refuses. Blobs are removed only when no other snapshot or repository in the
    cache references them. Saved manifests are kept, so ``repair`` can restore it.

    Applying previews every host first and deletes nothing anywhere if any host
    would refuse (``target_met`` false on its report); those preview reports are
    returned instead. Pass the revisions a user confirmed as *allowed_revisions*
    so a revision that appeared after the confirmation is refused, not deleted.
    """
    from sparkrun.models.cache import cache_operation, resolve_cache_locations

    ssh = _ssh(ssh_kwargs, sctx)
    # Applying: verify the grouping once and gate on a preview of exactly the
    # caches the apply will change. An unverified preview could have merged a
    # cache the verified apply splits off, which no refusal would then cover.
    locations = None if dry_run else resolve_cache_locations(cache_dir=cache_dir, hosts=hosts, ssh_kwargs=ssh, session=session, verify=True)
    options = dict(
        cache_dir=cache_dir,
        hosts=hosts,
        ssh_kwargs=ssh,
        session=session,
        locations=locations,
        action="delete",
        repo=request.repo,
        revision=request.revision,
        force=force,
        allowed=None if allowed_revisions is None else sorted(allowed_revisions),
    )
    preview = cache_operation(dry_run=True, **options)
    if dry_run or not all(report.target_met for report in preview):
        return preview
    return cache_operation(dry_run=False, **options)
