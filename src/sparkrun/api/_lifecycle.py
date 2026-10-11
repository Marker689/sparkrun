"""Console-free post-launch lifecycle and endpoint discovery."""

from __future__ import annotations

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import SparkrunError, TrustRejected
from sparkrun.core.operation import operation_scope


def wait_ready(result, *, sctx=None):
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
            cancel=operation.cancel,
            dry_run=getattr(result, "dry_run", False),
        )


def post_launch(result, *, trust=False, readiness=None, sctx=None):
    from sparkrun.core.launcher import complete_post_launch, resolve_recipe_trust

    sctx = resolve_sctx(sctx)
    launch = getattr(result, "launch_result", result)
    if launch is None:
        raise ValueError("Post-launch requires a launch result from this operation")
    if not resolve_recipe_trust(launch.recipe, trust, sctx=sctx):
        raise TrustRejected("Post-launch hooks require explicit trust")
    with operation_scope(sctx) as operation:
        try:
            return {
                "stopped": complete_post_launch(
                    launch,
                    launch.effective_cache_dir,
                    trust=trust,
                    dry_run=getattr(result, "dry_run", False),
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
