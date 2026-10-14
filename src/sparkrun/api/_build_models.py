"""Preparation-only operation contracts; these do not represent serving jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sparkrun.api._models import TargetOptions

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.offline import OfflineMode
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.validation import RecipeIssue
    from sparkrun.orchestration.executor import ExecutorTarget
    from sparkrun.runtimes.base import RuntimePlugin


@dataclass(frozen=True, kw_only=True)
class BuildOptions(TargetOptions):
    """Build environments and stage assets on every selected host.

    Host selection does not use GPU occupancy or inference node-count limits.
    Builders may run temporary build/probe processes, but this operation never
    starts inference, captures snapshots, or runs launch-strategy hooks.
    Shares its target, transfer, executor and ``offline`` fields with
    :class:`~sparkrun.api.RunOptions`; only ``recipe`` may be passed positionally.
    """

    rebuild: bool | None = None
    """Request a builder rebuild / registry refresh (refused offline)."""
    builder_context: dict[str, Any] = field(default_factory=dict)
    """Builder inputs; explicit values override plugin-provided defaults."""


@dataclass(frozen=True)
class BuildPlan:
    """Resolved preparation targets, without an inference placement or job ID."""

    recipe: Recipe
    runtime: RuntimePlugin
    cluster: ClusterDefinition
    host_list: tuple[str, ...]
    executor_target: ExecutorTarget
    needs_image: bool
    options: BuildOptions = field(repr=False)
    issues: tuple[RecipeIssue, ...] = ()
    offline: OfflineMode | None = field(default=None, kw_only=True)
    """Resolved offline mode (value and deciding layer): ``BuildOptions.offline``
    > the cluster's ``offline:`` > online.  ``None`` only for a plan built
    outside :func:`sparkrun.api.plan_build`."""


@dataclass(frozen=True)
class BuildResult:
    """Completed preparation, or a dry-run preview, without starting inference.

    Empty images are normal for native executors such as local. Models lists
    the repositories selected for staging, excluding disabled resources.
    A dry run reports intended references, not verified resident content.
    """

    recipe: str
    host_list: tuple[str, ...]
    executor: str
    dry_run: bool
    container_image: str | None = None
    images_by_node: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    effective_cache_dir: str = ""
    local_cache_dir: str = ""
    environment_file: str | None = None
    timeline: dict[str, Any] = field(default_factory=dict)


__all__ = ["BuildOptions", "BuildPlan", "BuildResult"]
