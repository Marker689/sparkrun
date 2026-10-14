"""Console-free post-launch lifecycle and endpoint discovery.

.. warning::
   **Experimental.** ``wait_ready``, ``post_launch``, ``discovery_snapshot``
   and :class:`PostLaunchResult` may change shape in a minor release while the
   daemon and gateway integrations that drive them settle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import HostsUnreachable, JobNotFound, SparkrunError, TrustRejected
from sparkrun.core.operation import operation_scope

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sparkrun.api._models import RunResult
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.launcher import LaunchResult, ServeReadiness


@dataclass(frozen=True)
class PostLaunchResult:
    """Outcome of :func:`post_launch` (experimental).

    ``stopped`` is ``True`` when the recipe's ``stop_after_post`` (or a
    failed readiness check) tore the workload down after its hooks ran.
    """

    stopped: bool


def _launch_handle(result: "RunResult | LaunchResult", action: str) -> "LaunchResult":
    launch: Any = getattr(result, "launch_result", result)
    if launch is None:
        raise ValueError("%s requires a launch result from this operation" % action)
    return launch


def wait_ready(
    result: "RunResult | LaunchResult | None" = None,
    *,
    workload_id: str | None = None,
    timeout: float | None = None,
    sctx: "SparkrunContext | None" = None,
) -> "ServeReadiness":
    """Wait until a workload's endpoint answers (experimental).

    Pass exactly one of *result* (a launch from this process) or
    *workload_id* (any recorded workload, e.g. one launched by an earlier
    process that is still starting).

    Args:
        result: The :class:`~sparkrun.api.RunResult` returned by
            :func:`sparkrun.api.run` (or its ``launch_result``).  A result
            without a launch handle (dry run, ``ensure`` skip) is refused.
        workload_id: Wait on a recorded workload instead.  Hosts, port and
            runtime come from its job metadata; the connection comes from the
            job's recorded cluster (the ``api.stop`` / ``api.logs`` rule), and
            the readiness policy from the configuration plus the recorded
            recipe.  Only the endpoint wait runs: there is no launch to
            observe startup from.
        timeout: Extra caller budget in seconds on top of the readiness
            policy (``readiness.*`` config / recipe).  ``None`` or ``0``
            adds no caller budget, matching the readiness config's
            "``0`` = unbounded" convention; the policy budgets and the
            operation's cancellation still apply.  Negative values raise
            :class:`ValueError`.
        sctx: Optional shared context; its operation's cancel event aborts
            the wait (reported as ``reason="cancelled"``).

    Returns:
        The :class:`~sparkrun.core.launcher.ServeReadiness` verdict.

    Raises:
        ValueError: Both or neither of *result* and *workload_id* were given.
        JobNotFound: *workload_id* has no metadata and no hosts.
        SparkrunError: The recorded runtime cannot be resolved.
    """
    from sparkrun.core.launcher import wait_for_serve_ready
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    if (result is None) == (workload_id is None):
        raise ValueError("wait_ready requires exactly one of result or workload_id")
    if timeout is not None and timeout < 0:
        raise ValueError("Readiness timeout must not be negative")
    sctx = resolve_sctx(sctx)
    if result is None:
        return _wait_ready_for_workload(str(workload_id), timeout=timeout, sctx=sctx)
    launch = _launch_handle(result, "Readiness")
    with operation_scope(sctx) as operation:
        operation.emit("progress", stage="readiness")
        return wait_for_serve_ready(
            launch,
            ssh_kwargs=build_ssh_kwargs(launch.config),
            timeline=sctx.timing,
            cancel=_ReadinessCancellation(operation.cancel, timeout) if timeout else operation.cancel,
            dry_run=getattr(result, "dry_run", False),
        )


def _wait_ready_for_workload(workload_id: str, *, timeout: float | None, sctx: "SparkrunContext") -> "ServeReadiness":
    """Endpoint readiness for a recorded workload, resolved like ``api.logs``."""
    import copy

    from sparkrun.api._resolve import bind_job_cluster, prepare_transport, resolve_workload_target, scope_operation
    from sparkrun.core._executor_destination import metadata_executor_overrides
    from sparkrun.core.bootstrap import get_runtime
    from sparkrun.core.config import resolve_configured_cache_dir
    from sparkrun.core.launcher import wait_for_endpoint_ready
    from sparkrun.core.readiness import resolve_readiness_settings
    from sparkrun.orchestration.executor import resolve_executor

    target = resolve_workload_target(
        workload_id=workload_id,
        recipe=None,
        hosts=None,
        overrides=None,
        cluster=None,
        cache_dir=str(resolve_configured_cache_dir(None, config=sctx.config)),
        operation="Readiness",
        sctx=sctx,
    )
    meta = target.meta or {}
    recipe = _recorded_recipe(meta)
    runtime_name = getattr(recipe, "runtime", None) or meta.get("runtime")
    if not runtime_name:
        raise JobNotFound("No runtime recorded for workload_id %r" % workload_id)
    try:
        runtime = get_runtime(runtime_name, sctx.variables)
    except ValueError as exc:
        raise SparkrunError("Cannot resolve runtime %r for workload_id %r: %s" % (runtime_name, workload_id, exc)) from exc
    if meta.get("native_resource") is None:
        prepare_transport(target.cluster)
    executor = resolve_executor(
        cluster=target.cluster,
        cli_overrides=metadata_executor_overrides(target.meta),
        rootless=False,
        auto_user=False,
        config=sctx.config,
        v=sctx.variables,
    )
    cluster = bind_job_cluster(target.cluster, executor.resolve_target(dry_run=True), target.meta)
    scoped, ssh_kwargs = scope_operation(cluster, sctx=sctx, prepare=False)
    # The runtime is a shared plugin instance; bind this job's executor to a
    # copy so the head-container name and liveness probe use the right backend.
    runtime = copy.copy(runtime)
    runtime.executor = executor
    try:
        port = int(meta.get("port") or (recipe.defaults.get("port") if recipe is not None else None) or 8000)
    except (TypeError, ValueError) as exc:
        raise SparkrunError("Workload %r records an invalid port" % workload_id) from exc
    settings = resolve_readiness_settings(config=scoped.config, recipe=recipe)
    with operation_scope(sctx) as operation:
        operation.emit("progress", stage="readiness")
        return wait_for_endpoint_ready(
            runtime=runtime,
            cluster_id=target.workload_id,
            host_list=list(target.hosts),
            is_solo=len(target.hosts) == 1,
            port=port,
            ssh_kwargs=ssh_kwargs,
            port_timeout_s=settings.port_timeout_s,
            health_timeout_s=settings.health_timeout_s,
            timeline=sctx.timing,
            cancel=_ReadinessCancellation(operation.cancel, timeout) if timeout else operation.cancel,
        )


def _recorded_recipe(meta: dict):
    """The recipe a job recorded at launch, or ``None`` when absent or unreadable."""
    state = meta.get("recipe_state")
    if not state:
        return None
    from sparkrun.core.recipe import Recipe

    try:
        return Recipe.from_state(state)
    except Exception:  # noqa: BLE001 - advisory: the recorded runtime name still identifies the job
        logger.debug("Could not rebuild the recorded recipe", exc_info=True)
        return None


def post_launch(
    result: "RunResult | LaunchResult",
    *,
    trust: bool = False,
    readiness: "ServeReadiness | None" = None,
    sctx: "SparkrunContext | None" = None,
) -> PostLaunchResult:
    """Run the recipe's post-serve lifecycle for a launch (experimental).

    Waits for readiness (unless *readiness* is supplied), runs ``post_exec``
    / ``post_commands`` and honors ``stop_after_post``.  Trust is gated only
    when the recipe actually has post hooks — the same rule
    :func:`sparkrun.api.run` applies — and reuses the verdict ``run`` already
    recorded on the launch (``LaunchResult.recipe_trusted``) before resolving
    it again.

    Raises:
        TrustRejected: The recipe has post hooks from an untrusted source and
            neither *trust* nor the launch's recorded verdict allows them.
        SparkrunError: The lifecycle failed.
        ValueError: *result* carries no launch handle.
    """
    from sparkrun.core.launcher import complete_post_launch, resolve_recipe_trust

    sctx = resolve_sctx(sctx)
    launch = _launch_handle(result, "Post-launch")
    dry_run = bool(getattr(result, "dry_run", False))
    recipe = launch.recipe
    has_hooks = bool(getattr(recipe, "post_exec", None) or getattr(recipe, "post_commands", None))
    trusted = bool(trust)
    if has_hooks and not trusted:
        # Reuse the verdict ``run`` decided for pre_exec rather than re-resolving.
        recorded = getattr(launch, "recipe_trusted", None)
        trusted = bool(recorded) if recorded is not None else resolve_recipe_trust(recipe, trust, sctx=sctx)
        if not trusted and not dry_run:
            raise TrustRejected("Post-launch hooks require explicit trust")
    with operation_scope(sctx) as operation:
        try:
            stopped = complete_post_launch(
                launch,
                launch.effective_cache_dir,
                trust=trusted,
                dry_run=dry_run,
                progress=sctx.progress,
                readiness=readiness,
                report=lambda message, **kwargs: operation.emit("progress", message=message),
            )
        except RuntimeError as exc:
            raise SparkrunError(str(exc)) from exc
    return PostLaunchResult(stopped=bool(stopped))


def discover_endpoints(*, sctx: "SparkrunContext | None" = None) -> list:
    """Metadata-only endpoint listing (internal; use :func:`discovery_snapshot`)."""
    from sparkrun.proxy.discovery import discover_endpoints as discover
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    sctx = resolve_sctx(sctx)
    return discover(sctx=sctx, ssh_kwargs=build_ssh_kwargs(sctx.config))


def discovery_snapshot(
    *,
    hosts: list[str] | tuple[str, ...] | None = None,
    cluster: "str | ClusterDefinition | None" = None,
    workload_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    metadata_only: bool = False,
    sctx: "SparkrunContext | None" = None,
) -> dict[str, Any]:
    """Credential-free endpoint observation with explicit live coverage (experimental).

    The target resolves like every other ``api`` entry point
    (:func:`~sparkrun.api._resolve.resolve_cluster`): explicit *hosts* and/or
    *cluster*, else the default cluster, else ``config.default_hosts``.  The
    cluster's full definition (SSH user, executor, transport) is used, and its
    transport is prepared before any SSH.  With no host source at all the
    snapshot is metadata-only (no live coverage), as before.

    With *metadata_only* the target is not resolved at all: every recorded
    job (narrowed by *workload_ids*) is a candidate, no status sweep runs and
    no SSH is attempted, and each candidate's endpoint is still HTTP-probed —
    only endpoints that answer are returned, with ``mode="metadata"``. For a
    caller that scopes jobs itself (by recorded cluster, fingerprint, owner)
    and must not be limited to the default cluster's hosts. *hosts* and
    *cluster* are refused alongside it, since they would be ignored.

    Each endpoint carries ``recipe_name``, the recipe reference recorded at
    launch (empty when none was recorded).

    Raises:
        ValueError: *metadata_only* is combined with *hosts* or *cluster*.
        SparkrunError: *cluster* names a cluster that cannot be loaded, or its
            transport cannot be prepared.
    """
    from sparkrun.api._resolve import resolve_cluster, scope_operation
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.proxy.discovery import discover_snapshot

    if metadata_only and (hosts or cluster is not None):
        raise ValueError("metadata_only discovery takes no hosts or cluster")
    sctx = resolve_sctx(sctx)
    definition = None
    if not metadata_only:
        try:
            definition = resolve_cluster(cluster, hosts, sctx=sctx)
        except HostsUnreachable:
            definition = None
    if definition is not None:
        sctx, ssh = scope_operation(definition, sctx=sctx)
        selected = list(definition.hosts or ())
    else:
        ssh, selected = build_ssh_kwargs(sctx.config), []
    snapshot = discover_snapshot(
        host_list=selected,
        host_filter=selected or None,
        cluster_def=definition if selected else None,
        cluster_ids=set(workload_ids) if workload_ids is not None else None,
        ssh_kwargs=ssh,
        sctx=sctx,
    )
    value = snapshot.to_dict()
    # Added here, not in ``DiscoverySnapshot.to_dict``: that projection is also
    # the ``workloads.*`` endpoint contract, whose consumers decode strictly.
    for row, endpoint in zip(value["endpoints"], snapshot.endpoints, strict=True):
        row["recipe_name"] = endpoint.recipe_name
    return value


class _ReadinessCancellation:
    """Readiness budget plus operation cancellation without cancelling its owner."""

    def __init__(self, parent, timeout):
        import time

        if timeout <= 0:
            raise ValueError("Readiness timeout must be positive")
        self.parent, self.deadline = parent, time.monotonic() + timeout

    def is_set(self):
        import time

        return self.parent.is_set() or time.monotonic() >= self.deadline

    def wait(self, timeout=None):
        import time

        remaining = max(0, self.deadline - time.monotonic())
        self.parent.wait(remaining if timeout is None else min(timeout, remaining))
        return self.is_set()
