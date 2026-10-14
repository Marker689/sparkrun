"""Console-free post-launch lifecycle and endpoint discovery."""

from __future__ import annotations

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import SparkrunError, TrustRejected
from sparkrun.core.operation import operation_scope


def wait_ready(result, *, timeout=None, sctx=None):
    from sparkrun.core.launcher import wait_for_serve_ready
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    sctx = resolve_sctx(sctx)
    launch = getattr(result, "launch_result", result)
    if launch is None:
        raise ValueError("Readiness requires a launch result from this operation")
    with operation_scope(sctx) as operation:
        operation.emit("progress", stage="readiness")
        return wait_for_serve_ready(
            launch,
            ssh_kwargs=build_ssh_kwargs(launch.config),
            timeline=sctx.timing,
            cancel=operation.cancel if timeout is None else _ReadinessCancellation(operation.cancel, timeout),
            dry_run=getattr(result, "dry_run", False),
        )


def post_launch(result, *, trust=False, readiness=None, sctx=None):
    from sparkrun.core.launcher import complete_post_launch, resolve_recipe_trust

    sctx = resolve_sctx(sctx)
    launch = getattr(result, "launch_result", result)
    if launch is None:
        raise ValueError("Post-launch requires a launch result from this operation")
    # Reuse the verdict ``run`` decided for pre_exec rather than re-resolving.
    recorded = getattr(launch, "recipe_trusted", None)
    trusted = trust or (recorded if recorded is not None else resolve_recipe_trust(launch.recipe, trust, sctx=sctx))
    dry_run = bool(getattr(result, "dry_run", False))
    if not trusted and not dry_run:
        raise TrustRejected("Post-launch hooks require explicit trust")
    with operation_scope(sctx) as operation:
        try:
            return {
                "stopped": complete_post_launch(
                    launch,
                    launch.effective_cache_dir,
                    trust=trusted,
                    dry_run=dry_run,
                    progress=sctx.progress,
                    readiness=readiness,
                    report=lambda message, **kwargs: operation.emit("progress", message=message),
                )
            }
        except RuntimeError as exc:
            raise SparkrunError(str(exc)) from exc


def discover_endpoints(*, sctx=None):
    from sparkrun.proxy.discovery import discover_endpoints as discover
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    sctx = resolve_sctx(sctx)
    return discover(sctx=sctx, ssh_kwargs=build_ssh_kwargs(sctx.config))


def discovery_snapshot(*, hosts=None, cluster=None, workload_ids=None, sctx=None):
    """Public credential-free endpoint observation with explicit live coverage."""
    from sparkrun.proxy.discovery import discover_snapshot
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    sctx = resolve_sctx(sctx)
    definition = sctx.cluster_manager.get(cluster) if cluster else None
    if cluster and definition is None:
        raise ValueError("Unknown discovery cluster")
    selected = list(hosts or (definition.hosts if definition else ()))
    ssh = build_ssh_kwargs(sctx.config)
    if definition and definition.user:
        ssh = dict(ssh, ssh_user=definition.user)
    return discover_snapshot(
        host_list=selected,
        host_filter=selected or None,
        cluster_def=definition,
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
