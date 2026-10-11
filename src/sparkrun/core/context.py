"""Shared application context for API callers, plugins, and frontends."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from scitrera_app_framework import Variables

    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterManager
    from sparkrun.core.operation import OperationState
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.progress import LaunchProgress
    from sparkrun.core.registry import RegistryManager
    from sparkrun.core.timing import Timeline
    from sparkrun.proxy.config import ProxyConfig


@dataclass
class SparkrunContext:
    """Single session context bundling SAF Variables and SparkrunConfig.

    Obtain one through ``sparkrun.application.initialize()`` and pass it to
    API operations. CLI commands create the same context lazily. A process
    has one application/profile and canonical configuration binding; context
    reuse shares cached managers but does not imply thread-safe operations.
    """

    variables: Variables
    config: SparkrunConfig
    verbose: bool = False
    progress: LaunchProgress | None = None
    events: Callable[[dict], None] | None = None
    operation: OperationState | None = None
    timing: Timeline | None = None
    """Optional span collector spanning more than one launch.

    ``launch_inference`` creates its own when this is ``None``, so a caller
    only sets it to widen the window (e.g. to include planning) or to share
    one timeline across several launches."""

    def for_cluster(self, cluster: ClusterDefinition) -> SparkrunContext:
        """Share session services with operation-local connection settings."""
        from copy import copy

        config = self.config.for_cluster(cluster)
        if config is self.config:
            return self
        scoped = copy(self)
        scoped.config = config
        return scoped

    @property
    def application_profile(self):
        return self.config.profile

    @property
    def application_identity(self):
        from sparkrun.core.application_identity import get_application_identity

        return get_application_identity(self.application_profile)

    @cached_property
    def controller_identity(self):
        from sparkrun.core.application_identity import get_controller_identity

        return get_controller_identity(self.config)

    @cached_property
    def registry_manager(self) -> RegistryManager:
        return self.config.get_registry_manager()

    @cached_property
    def cluster_manager(self) -> ClusterManager:
        from sparkrun.core.cluster_manager import ClusterManager
        from sparkrun.core.config import get_config_root

        return ClusterManager(get_config_root(self.variables))

    @cached_property
    def proxy_config(self) -> ProxyConfig:
        return self.config.get_proxy_config()
