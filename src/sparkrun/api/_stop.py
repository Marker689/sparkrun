"""``sparkrun.api.stop`` — stop a running sparkrun workload.

Two modes:

- **By cluster_id**: provide the literal ``cluster_id`` (as returned
  by :func:`sparkrun.api.run`); the API loads the job metadata, picks
  the executor that originally launched it, and stops its native controller
  resource or tears down observed container names on each host. Without
  a discovery snapshot, it falls back to candidate names.
- **By recipe+hosts+overrides**: derive the same ``cluster_id`` the
  launcher would have produced and dispatch identically.  Useful for
  ``sparkrun stop <recipe>`` semantics.

Returns a :class:`StopResult` summarizing how many containers were
removed and any per-host errors.
"""

from __future__ import annotations

from sparkrun.core.coordination import operation_boundary

import logging
from typing import TYPE_CHECKING

from sparkrun.api._errors import JobNotFound, SparkrunError
from sparkrun.api._models import StopResult

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.cluster_status import ClusterStatus
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)


@operation_boundary(mutation=True)
def stop(
    *,
    workload_id: str | None = None,
    recipe: "str | Recipe | None" = None,
    hosts: list[str] | tuple[str, ...] | None = None,
    overrides: dict | None = None,
    cluster: "str | ClusterDefinition | None" = None,
    cache_dir: str | None = None,
    discovered: "ClusterStatus | None" = None,
    dry_run: bool = False,
    sctx: "SparkrunContext | None" = None,
) -> StopResult:
    """Stop a running sparkrun workload.

    Either ``workload_id`` *or* (``recipe`` + a host source) is required.
    When both are provided, ``workload_id`` wins.

    Args:
        workload_id: The workload's id (``RunResult.workload_id``).
        dry_run: Resolve the target, hosts and executor exactly as a real stop
            would, then return without tearing anything down
            (``result.dry_run`` is true, ``containers_removed`` is 0).
            Discovery still contacts the hosts.
        discovered: Optional status snapshot for exact per-host container names.
            Reuses discovery during eviction instead of inferring ranks from
            the surviving host count. The snapshot must cover the requested
            hosts and match the resolved executor/destination; missing container
            detail is an error, not permission to fall back to guessed names.
        cache_dir: Explicit metadata root, otherwise the supplied or current
            application's configured cache. Resolved before job lookup.
        sctx: Optional shared :class:`SparkrunContext` for chained
            api calls (registry/cluster manager + config sharing).
    """
    from sparkrun.api._resolve import (
        discover_cluster_id_by_intent,
        scope_operation,
        resolve_cluster,
        resolve_cluster_for_job,
        resolve_recipe,
    )
    from sparkrun.orchestration.executor import resolve_executor
    from sparkrun.orchestration.job_metadata import (
        generate_intent_id,
        load_job_metadata,
    )

    from sparkrun.core.config import resolve_configured_cache_dir

    cluster_id = workload_id
    cache_dir = str(resolve_configured_cache_dir(cache_dir, config=sctx.config if sctx is not None else None))

    # Derive cluster_id from recipe+hosts when not given explicitly.
    if not cluster_id:
        if recipe is None:
            raise SparkrunError("api.stop requires workload_id or recipe+hosts")
        cluster_def = resolve_cluster(cluster, hosts, sctx=sctx)
        resolved_recipe = resolve_recipe(recipe, sctx=sctx)
        from sparkrun.api._resolve import resolve_operation_target

        cluster_def, _ = resolve_operation_target(recipe=resolved_recipe, cluster=cluster_def, sctx=sctx)
        intent_id = generate_intent_id(resolved_recipe, overrides=overrides)
        target_hosts = list(cluster_def.hosts)

        # Status-driven discovery: ask the executor what's running on
        # the supplied hosts and filter for cluster_ids matching the
        # computed intent.  Load-aware schedulers may have placed the
        # workload on a different host set than ``hosts`` — that's the
        # whole point of separating intent from placement.  Here we
        # accept that the *user's* host scope is the authoritative
        # discovery range.
        cluster_id = discover_cluster_id_by_intent(
            intent_id,
            target_hosts,
            cluster_def=cluster_def,
            cache_dir=cache_dir,
            sctx=sctx,
        )
        meta = load_job_metadata(cluster_id, cache_dir=cache_dir)
        # Discovery ran against the *invocation's* cluster; teardown runs
        # against the *job's*, which the metadata may name even when this
        # invocation didn't (see ``resolve_cluster_for_job``).  Hosts stay as
        # resolved above — only the connection identity is recovered.
        if meta:
            cluster_def = resolve_cluster_for_job(cluster, target_hosts, meta=meta, sctx=sctx)
    else:
        # cluster_id given — load metadata to recover hosts/executor.
        meta = load_job_metadata(cluster_id, cache_dir=cache_dir)
        if meta is None and not hosts and cluster is None:
            raise JobNotFound("No job metadata found for cluster_id %r and no hosts provided" % cluster_id)

        # Determine target hosts: explicit > metadata > error.
        if hosts:
            target_hosts = list(hosts)
        elif meta and meta.get("hosts"):
            target_hosts = list(meta["hosts"])
        elif cluster is not None:
            target_hosts = list(resolve_cluster(cluster, sctx=sctx).hosts)
        else:
            target_hosts = []

        if not target_hosts:
            raise JobNotFound("No hosts known for cluster_id %r" % cluster_id)

        # Now that we know the hosts, build a cluster definition for
        # downstream consumers.  With no explicit cluster this is where the
        # job's own recorded cluster (SSH user, executor pin, transport) is
        # recovered — the alternative is an anonymous definition that
        # connects as the control node's login (issue #277).
        cluster_def = resolve_cluster_for_job(cluster, target_hosts, meta=meta, sctx=sctx)

    # Provider refresh may update connection/executor settings. Resolve only
    # after that refresh; SSH transports are a no-op here.
    sctx, _ = scope_operation(cluster_def, sctx=sctx)

    # Resolve the executor — prefer recipe-encoded selection from metadata
    # so we use the same executor that launched the workload.
    from sparkrun.core._executor_destination import metadata_executor_overrides

    cli_overrides = metadata_executor_overrides(meta)

    if not (cli_overrides or {}).get("executor"):
        # No metadata (or none naming an executor) — ask the cluster what is
        # actually running instead of defaulting.  Without this a job whose
        # metadata is gone gets the *default* executor, and a teardown aimed
        # at the wrong substrate exits 0 having done nothing: docker
        # truthfully reports no such container, so the workload survives and
        # its record is dropped as a confirmed stop.  Metadata does go
        # missing — an interrupted launch, a manually cleared cache, or the
        # very bug this guards against.
        discovered_exec = _discover_executor_name(
            cluster_id,
            target_hosts,
            cluster_def=cluster_def,
            sctx=sctx,
            discovered=discovered,
        )
        if discovered_exec:
            cli_overrides = dict(cli_overrides or {})
            cli_overrides["executor"] = discovered_exec

    executor = resolve_executor(
        cluster=cluster_def,
        cli_overrides=cli_overrides,
        rootless=False,
        auto_user=False,
        v=sctx.variables if sctx is not None else None,
        config=sctx.config if sctx is not None else None,
    )

    from sparkrun.api._resolve import bind_job_cluster

    target = executor.resolve_target(dry_run=True)
    cluster_def = bind_job_cluster(cluster_def, target, meta)
    sctx = sctx.for_cluster(cluster_def)
    if discovered is not None and discovered.coverage:
        covered = {
            host for coverage in discovered.coverage if coverage.matches(target, ssh_user=sctx.config.ssh_user) for host in coverage.hosts
        }
        if not set(target_hosts) <= covered:
            raise SparkrunError(
                "Stop discovery does not cover the requested executor destination and hosts; refresh status before teardown."
            )

    if dry_run:
        logger.info("Dry run: would stop %s on %s", cluster_id, ", ".join(target_hosts))
        return StopResult(workload_id=cluster_id, hosts_targeted=tuple(target_hosts), containers_removed=0, dry_run=True)

    try:
        native_removed = executor.stop_workload(cluster_id, metadata=meta)
        if native_removed is not None and (type(native_removed) is not int or native_removed < 0):
            raise TypeError("Executor.stop_workload must return a nonnegative integer or None")
    except Exception as exc:
        logger.warning("Keeping job metadata for %s — native teardown failed: %s", cluster_id, exc)
        return StopResult(
            workload_id=cluster_id,
            hosts_targeted=tuple(target_hosts),
            containers_removed=0,
            errors=(str(exc),),
            hosts_failed=tuple(target_hosts),
        )
    if native_removed is None:
        removed_count, hosts_failed, errors = _stop_containers(
            cluster_id,
            target_hosts,
            cluster_def,
            executor,
            sctx,
            discovered=discovered,
        )
    else:
        removed_count, hosts_failed, errors = native_removed, (), []

    # Cleanup persistent metadata only once teardown is confirmed
    # everywhere: while a container survives, the metadata still describes
    # a live workload and is what ``stop``/``status`` use to find it again.
    if hosts_failed:
        logger.warning(
            "Keeping job metadata for %s — teardown did not confirm on: %s",
            cluster_id,
            ", ".join(hosts_failed),
        )
    else:
        try:
            from sparkrun.orchestration.workload_cleanup import finalize_workload
            from sparkrun.orchestration.primitives import build_ssh_kwargs

            finalize_workload(cluster_id, hosts=target_hosts, cache_dir=cache_dir, ssh_kwargs=build_ssh_kwargs(sctx.config))
        except Exception:
            logger.warning(
                "Could not release model cache claims or remove job metadata for %s; retry stop to reconcile", cluster_id, exc_info=True
            )

    return StopResult(
        workload_id=cluster_id,
        hosts_targeted=tuple(target_hosts),
        containers_removed=removed_count,
        errors=tuple(errors),
        hosts_failed=hosts_failed,
    )


def _stop_containers(cluster_id, target_hosts, cluster_def, executor, sctx, *, discovered=None):
    from sparkrun.api._resolve import scope_operation
    from sparkrun.orchestration.teardown import parse_teardown_removed

    if discovered is None:
        container_names = executor.enumerate_containers(cluster_id, len(target_hosts))
        host_containers = {host: list(container_names) for host in target_hosts}
    else:
        host_containers = _observed_container_names(cluster_id, target_hosts, executor.executor_name, discovered)

    # ``cleanup_containers_by_host`` is the shared teardown primitive: it
    # dispatches local-vs-SSH per host, verifies the workloads are actually
    # gone, and reports what it removed per host.  The executor resolved
    # above is threaded in so the teardown speaks the substrate that
    # launched this job — without it every stop emitted ``docker rm -f``,
    # which a ``local`` executor's native process truthfully answers "no
    # such container" to while continuing to serve.
    from sparkrun.orchestration.primitives import cleanup_containers_by_host

    sctx, ssh_kwargs = scope_operation(cluster_def, sctx=sctx, prepare=False)

    errors: list[str] = []
    try:
        results = cleanup_containers_by_host(
            host_containers,
            ssh_kwargs=ssh_kwargs,
            executor=executor,
        )
    except Exception as e:  # pragma: no cover - defensive; the primitive absorbs per-host failures
        errors.append(str(e))
        results = {}

    hosts_failed = tuple(host for host in target_hosts if not (results.get(host) and results[host].success))
    removed_count = sum(parse_teardown_removed(r.stdout) for r in results.values())
    for host in hosts_failed:
        r = results.get(host)
        detail = ((r.stderr or r.stdout).strip() if r else "") or "teardown did not confirm"
        errors.append("%s: %s" % (host, detail))

    return removed_count, hosts_failed, errors


def _observed_container_names(cluster_id, hosts, executor_name, discovered):
    """Use actual ranks on each host; never turn incomplete discovery into success."""
    names: dict[str, list[str]] = {host: [] for host in hosts}
    for entry in discovered.hosts:
        if entry.host not in names:
            continue
        if entry.host in discovered.errors:
            raise SparkrunError("Stop discovery failed on %s; refresh status before teardown." % entry.host)
        for workload in entry.workloads:
            if workload.cluster_id != cluster_id:
                continue
            if not workload.containers:
                raise SparkrunError("Stop discovery lacks container names for %s on %s." % (cluster_id, entry.host))
            for container in workload.containers:
                owner = container.executor or discovered.executor
                if owner and owner != executor_name:
                    raise SparkrunError("Stop discovery reports %s via another executor; use fresh stop_all discovery." % cluster_id)
                if not container.name:
                    raise SparkrunError("Stop discovery contains an unnamed container for %s." % cluster_id)
                if container.name not in names[entry.host]:
                    names[entry.host].append(container.name)
    missing = [host for host, containers in names.items() if not containers]
    if missing:
        raise SparkrunError("Stop discovery lacks container names for %s on: %s." % (cluster_id, ", ".join(missing)))
    return names


def _discover_executor_name(
    cluster_id: str,
    hosts: list[str],
    *,
    cluster_def,
    sctx: "SparkrunContext | None",
    discovered: "ClusterStatus | None" = None,
) -> str | None:
    """Return the executor currently reporting *cluster_id*, or ``None``.

    The live peer of the metadata lookup: ``api.status`` sweeps every executor
    on the cluster's substrate and stamps each container with the one that
    saw it, so a running workload can always name its own teardown mechanism
    even when nothing was written down.

    Returns ``None`` — "fall back to the resolution chain" — when the sweep
    fails or finds nothing.  A workload that isn't running needs no
    substrate-specific teardown, and a failed sweep must not block a stop.
    """
    snapshot = discovered
    if snapshot is None:
        try:
            from sparkrun.api._status import status

            snapshot = status(list(hosts), cluster=cluster_def, sctx=sctx)
        except Exception:
            logger.debug("Executor discovery sweep failed for %s", cluster_id, exc_info=True)
            return None

    names = {
        container.executor
        for entry in snapshot.hosts
        if entry.host in hosts
        for workload in entry.workloads
        if workload.cluster_id == cluster_id
        for container in workload.containers
        if container.executor
    }
    if len(names) > 1:
        raise SparkrunError("Workload %s spans multiple executors; use fresh stop_all discovery." % cluster_id)
    return next(iter(names), None)


__all__ = ["stop"]
