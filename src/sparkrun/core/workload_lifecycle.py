"""Versioned lifecycle providers, registered by the plugin loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sparkrun.core.plugin_identity import PluginIdentity, current_registration_owner
from sparkrun.core.registration import enlist_registry_state, register_unique

WORKLOAD_LIFECYCLE_API_VERSION = 1


@dataclass(frozen=True)
class WorkloadLifecycleProvider:
    name: str
    matches: Callable
    control: Callable
    version: int = WORKLOAD_LIFECYCLE_API_VERSION


_PROVIDERS = {}
enlist_registry_state(globals(), "_PROVIDERS")


def register_workload_lifecycle(provider: WorkloadLifecycleProvider):
    owner = current_registration_owner()
    if not isinstance(owner, PluginIdentity) or provider.name != owner.plugin_id:
        raise ValueError("Lifecycle providers must use their loader-assigned plugin identity")
    if provider.version != WORKLOAD_LIFECYCLE_API_VERSION:
        raise ValueError("Unsupported workload lifecycle API version")
    register_unique(_PROVIDERS, provider.name, provider, description="Workload lifecycle provider")


def registered_workload_lifecycles():
    return tuple(_PROVIDERS[key] for key in sorted(_PROVIDERS))


def lifecycle_provider(job):
    matches = [provider for provider in registered_workload_lifecycles() if provider.matches(job)]
    if len(matches) > 1:
        raise ValueError("Several providers claim this workload's lifecycle")
    return matches[0] if matches else None
