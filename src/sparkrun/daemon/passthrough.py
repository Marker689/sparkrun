"""Opt-in, read-only CLI passthrough with pre-submission discovery fallback."""

from __future__ import annotations

import os
from typing import Any


def query(name, arguments=None) -> tuple[bool, Any]:
    """Return (handled, JSON result). Never fall back after submitting a request."""
    from sparkrun.core.application_profile import env_name

    if os.environ.get(env_name("DAEMON"), "").lower() not in {"auto", "1", "true"}:
        return False, None
    from sparkrun.daemon.state import Binding
    from sparkrun.daemon.client import Client

    client = Client(binding=Binding.from_config())
    try:
        client.info()
    except (OSError, ValueError):
        return False, None
    # Only explicit read operations are eligible; never duplicate mutations.
    if name not in {"catalog.recipes", "catalog.clusters", "catalog.registries", "workloads.list", "cluster.report", "catalog.search"}:
        raise ValueError("Operation is not eligible for CLI passthrough")
    value = client.call(name, arguments or {})
    if isinstance(value, dict) and "job_id" in value:
        value = client.wait(value["job_id"])
    return True, value
