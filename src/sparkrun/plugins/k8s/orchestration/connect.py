"""Cluster connectivity probe.

:func:`probe_cluster` runs ``kubectl version -o json`` (which reports the
client version even offline, and the *server* version only when the API
is reachable) and derives a :class:`ClusterInfo` snapshot the setup / api
layers render.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .client import KubectlClient
from .errors import K8sError
from sparkrun.utils.text import excerpt

logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """Snapshot of a probe against a cluster."""

    reachable: bool
    current_context: str | None = None
    namespace: str | None = None
    server_version: str | None = None
    client_version: str | None = None
    message: str | None = None


def _current_context(client: KubectlClient) -> str | None:
    if client.context:
        return client.context
    result = client.run(["config", "current-context"])
    if result.success:
        return result.stdout.strip() or None
    return None


def probe_cluster(client: KubectlClient, *, timeout: int = 15) -> ClusterInfo:
    """Probe *client*'s target cluster.

    Never raises for an unreachable cluster — returns a
    :class:`ClusterInfo` with ``reachable=False`` and a diagnostic
    ``message`` instead, so callers can render status uniformly.
    Raises :class:`K8sError` only for a broken kubectl invocation
    (e.g. missing binary).
    """
    current_context = _current_context(client)
    result = client.run(["version", "-o", "json"], timeout=timeout)

    client_version: str | None = None
    server_version: str | None = None
    if result.stdout.strip():
        try:
            import json

            payload = json.loads(result.stdout)
            client_version = (payload.get("clientVersion") or {}).get("gitVersion")
            server_version = (payload.get("serverVersion") or {}).get("gitVersion")
        except (ValueError, AttributeError):
            logger.debug("Failed to parse kubectl version JSON")

    reachable = server_version is not None
    message = None
    if not reachable:
        message = excerpt(result.stderr, 400) or "cluster API did not report a server version"

    return ClusterInfo(
        reachable=reachable,
        current_context=current_context,
        namespace=client.namespace,
        server_version=server_version,
        client_version=client_version,
        message=message,
    )


def require_reachable(client: KubectlClient, *, timeout: int = 15) -> ClusterInfo:
    """Like :func:`probe_cluster` but raises when the cluster is unreachable."""
    info = probe_cluster(client, timeout=timeout)
    if not info.reachable:
        from .errors import ClusterUnreachableError

        raise ClusterUnreachableError(info.message or "cluster unreachable")
    return info


__all__ = ["ClusterInfo", "probe_cluster", "require_reachable", "K8sError"]
