"""Console-free post-launch lifecycle and endpoint discovery.

.. warning::
   **Experimental.** ``wait_ready``, ``post_launch``, ``discovery_snapshot``
   and :class:`PostLaunchResult` may change shape in a minor release while the
   daemon and gateway integrations that drive them settle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import HostsUnreachable, SparkrunError, TrustRejected
from sparkrun.core.operation import operation_scope

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
    result: "RunResult | LaunchResult",
    *,
    timeout: float | None = None,
    sctx: "SparkrunContext | None" = None,
) -> "ServeReadiness":
    """Wait until a just-launched workload's endpoint answers (experimental).

    Args:
        result: The :class:`~sparkrun.api.RunResult` returned by
            :func:`sparkrun.api.run` (or its ``launch_result``).  A result
            without a launch handle (dry run, ``ensure`` skip) is refused.
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
    """
    from sparkrun.core.launcher import wait_for_serve_ready
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    if timeout is not None and timeout < 0:
        raise ValueError("Readiness timeout must not be negative")
    sctx = resolve_sctx(sctx)
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
    sctx: "SparkrunContext | None" = None,
) -> dict[str, Any]:
    """Credential-free endpoint observation with explicit live coverage (experimental).

    The target resolves like every other ``api`` entry point
    (:func:`~sparkrun.api._resolve.resolve_cluster`): explicit *hosts* and/or
    *cluster*, else the default cluster, else ``config.default_hosts``.  The
    cluster's full definition (SSH user, executor, transport) is used, and its
    transport is prepared before any SSH.  With no host source at all the
    snapshot is metadata-only (no live coverage), as before.

    Raises:
        SparkrunError: *cluster* names a cluster that cannot be loaded, or its
            transport cannot be prepared.
    """
    from sparkrun.api._resolve import resolve_cluster, scope_operation
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.proxy.discovery import discover_snapshot

    sctx = resolve_sctx(sctx)
    try:
        definition = resolve_cluster(cluster, hosts, sctx=sctx)
    except HostsUnreachable:
        definition = None
    if definition is not None:
        sctx, ssh = scope_operation(definition, sctx=sctx)
        selected = list(definition.hosts or ())
    else:
        ssh, selected = build_ssh_kwargs(sctx.config), []
    return discover_snapshot(
        host_list=selected,
        host_filter=selected or None,
        cluster_def=definition if selected else None,
        cluster_ids=set(workload_ids) if workload_ids is not None else None,
        ssh_kwargs=ssh,
        sctx=sctx,
    ).to_dict()


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
