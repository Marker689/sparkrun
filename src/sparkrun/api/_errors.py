"""Typed exception hierarchy for the sparkrun library API.

Operational failures use this hierarchy; scheduler and executor errors are
translated at API operation boundaries. Pure model/plan validation may raise
ValueError or TypeError. Explicit application initialization exposes bootstrap
errors directly; implicit API initialization wraps them in SparkrunError.
KeyboardInterrupt and SystemExit propagate unchanged.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.api._benchmark_models import BenchmarkResult


class SparkrunError(Exception):
    """Base class for operational failures reported by the sparkrun library API.

    Callers can catch this for a generic failure path; in most cases
    callers will want to discriminate on a more specific subclass.
    """


class IntegrationUnavailable(SparkrunError):
    """Required or conflicting integrations prevent a workload from launching."""


class InsufficientCapacity(SparkrunError):
    """Cluster lacks accelerator slots for the requested parallelism.

    Surfaced when a scheduler raises
    :class:`~sparkrun.core.scheduler.InfeasibleScheduleError`.  The
    message carries the slot count seen vs requested.

    The optional :attr:`status` (a ``ClusterStatus`` snapshot),
    :attr:`host_list`, and :attr:`required` attributes are populated by
    :func:`sparkrun.api._hosts.resolve_effective_hosts` so the CLI can
    render capacity diagnostics without an extra SSH round-trip.  They
    are ``None`` / empty when the error originated elsewhere.
    """

    def __init__(
        self,
        message: str,
        *,
        status=None,
        host_list: list[str] | None = None,
        required: int | None = None,
        rejections: tuple = (),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.host_list: tuple[str, ...] = tuple(host_list or ())
        self.required = required
        self.rejections = rejections


class LayoutRequired(SparkrunError):
    """Heterogeneous cluster needs an explicit ``recipe.layout``.

    Surfaced when a scheduler raises
    :class:`~sparkrun.core.scheduler.LayoutConflictError` — i.e. the
    cluster spans multiple accelerator vendors and the auto-pack
    algorithm can't choose splits safely.
    """


class RecipeNotFound(SparkrunError):
    """Named recipe could not be resolved across configured registries."""


class InvalidRegistryFilter(SparkrunError):
    """A recipe listing was scoped to a registry that cannot be used.

    Raised by :func:`sparkrun.api.search_recipes` when the ``registry``
    filter — or the ``@registry`` scope carried by the query — names a
    registry that is unknown or disabled, or contradicts an explicit
    ``registry`` argument.  Discriminate with :attr:`reason`
    (``"unknown"`` / ``"disabled"`` / ``"conflict"``); :attr:`available`
    carries the configured registry names for the error message.

    Surfaced when :mod:`sparkrun.core.registry` raises
    :class:`~sparkrun.core.registry.RegistryFilterError`.
    """

    def __init__(self, message: str, *, registry: str = "", reason: str = "unknown", available: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.registry = registry
        self.reason = reason
        self.available = available


class HostsUnreachable(SparkrunError):
    """One or more hosts could not be reached over SSH.

    The exception carries the list of unreachable hosts in
    :attr:`hosts`.  Callers may inspect this to display per-host
    diagnostics; absent listing the message still names them.
    """

    def __init__(self, message: str, hosts: list[str] | None = None) -> None:
        super().__init__(message)
        self.hosts: tuple[str, ...] = tuple(hosts or ())


class JobNotFound(SparkrunError):
    """No running job matches the given identification (workload_id / recipe+hosts)."""


class AmbiguousWorkload(SparkrunError):
    """Multiple running workloads match the supplied recipe+hosts intent.

    Raised by :func:`sparkrun.api.stop` when the recipe path matches
    more than one running cluster (e.g. two parallel deployments of the
    same recipe on disjoint host sets).  The exception's
    :attr:`workload_ids` attribute carries the candidates so callers can
    re-invoke with an explicit ``workload_id``.
    """

    def __init__(self, message: str, workload_ids: list[str] | tuple[str, ...] | None = None) -> None:
        super().__init__(message)
        self.workload_ids: tuple[str, ...] = tuple(workload_ids or ())


class TrustRejected(SparkrunError):
    """User declined trust prompt for a third-party recipe.

    The launcher abandons the run rather than executing untrusted
    pre/post hooks.  Raised by ``api.run`` when ``options.trust=False``
    is the user's final answer.
    """


# --------------------------------------------------------------------------
# Benchmark errors
# --------------------------------------------------------------------------


class BenchmarkFailed(SparkrunError):
    """A benchmark run failed (non-zero rc, task failures, or aborted launch).

    Carries the original exit code in :attr:`exit_code` when known.
    """

    def __init__(self, message: str, exit_code: int | None = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class BenchmarkFinalizationFailed(BenchmarkFailed):
    """Validated measurements survived a later export, cleanup or notification error.

    ``stage`` identifies the primary failure; ``errors`` also retains secondary
    finalization failures. ``result`` is the completed public result. Publication
    and state failures retain the compatible BenchmarkIntegrationFailed subtype.
    """

    def __init__(self, message: str, *, stage: str, result: "BenchmarkResult") -> None:
        super().__init__(message, exit_code=1)
        self.stage = stage
        self.result = result
        self.errors = {stage: message}


class BenchmarkIntegrationFailed(BenchmarkFinalizationFailed):
    """Measurements completed, but a plugin or completion-state save failed.

    ``integration`` names the plugin, or the reserved ``<state>`` marker.
    Existing catches remain valid; catch BenchmarkFinalizationFailed to include
    export, cleanup and notification errors. Retry requires saved measurements.
    """

    def __init__(self, message: str, *, integration: str, result: "BenchmarkResult") -> None:
        super().__init__(message, stage="state" if integration == "<state>" else "integration", result=result)
        self.integration = integration


class NoResumableState(SparkrunError):
    """``ResumeMode.REQUIRED`` but no benchmark state exists for the derived id."""


class CategoryNotFound(SparkrunError):
    """Requested benchmark category has no registered frameworks."""


class AmbiguousCategoryError(SparkrunError):
    """Category has multiple frameworks; pin one via config or ``--framework``."""


class FrameworkCategoryMismatch(SparkrunError):
    """Explicit framework does not belong to the explicit category."""


__all__ = [
    "SparkrunError",
    "InsufficientCapacity",
    "LayoutRequired",
    "RecipeNotFound",
    "InvalidRegistryFilter",
    "HostsUnreachable",
    "JobNotFound",
    "AmbiguousWorkload",
    "TrustRejected",
    "BenchmarkFailed",
    "BenchmarkIntegrationFailed",
    "BenchmarkFinalizationFailed",
    "NoResumableState",
    "CategoryNotFound",
    "AmbiguousCategoryError",
    "FrameworkCategoryMismatch",
]
