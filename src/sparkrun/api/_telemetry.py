"""``sparkrun.api`` live-monitoring surface — telemetry + occupancy, one source.

Two axes, composed here for every monitoring client (the ``cluster monitor``
TUI, an SSE feed, a headless collector):

- **occupancy** — what runs where — ``api.status`` (cross-executor).
- **telemetry** — how loaded each host/node is — a substrate
  :class:`~sparkrun.orchestration.telemetry.TelemetryProvider`, selected by the
  cluster's status scope (host / k8s / modal).

``open_telemetry`` exposes the raw telemetry session; ``open_live_monitor`` /
``live_monitor`` compose telemetry with a background ``api.status`` poll into
substrate-agnostic :class:`~sparkrun.core.monitoring.MonitorFrame` snapshots so
a k8s cluster and a host cluster feed a client the same shape.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.monitoring import MonitorFrame
    from sparkrun.orchestration.telemetry import TelemetrySession

logger = logging.getLogger(__name__)


def _resolve_ctx(hosts, cluster, ssh_kwargs, sctx):
    """Shared setup: resolve the cluster def, scope, and ssh_kwargs."""
    from sparkrun.api._context import resolve_sctx
    from sparkrun.api._resolve import scope_operation, resolve_cluster
    from sparkrun.orchestration.executor import cluster_status_scope

    sctx = resolve_sctx(sctx)
    cluster_def = resolve_cluster(cluster, hosts, sctx=sctx)
    sctx, ssh_kwargs = scope_operation(cluster_def, sctx=sctx, ssh_kwargs=ssh_kwargs)
    scope = cluster_status_scope(cluster_def, config=sctx.config, v=sctx.variables)
    return cluster_def, scope, ssh_kwargs, sctx


def open_telemetry(
    hosts: list[str],
    *,
    cluster: "str | ClusterDefinition | None" = None,
    ssh_kwargs: dict | None = None,
    interval: int = 2,
    backend: str | None = None,
    sctx: "SparkrunContext | None" = None,
) -> "TelemetrySession | None":
    """Open a telemetry :class:`TelemetrySession` for *hosts*, or ``None``.

    Resolves the cluster's status scope and its telemetry provider (host / k8s /
    modal); returns ``None`` when no provider covers that substrate (the caller
    then has occupancy-only monitoring).  The caller drives ``snapshot()`` and
    must ``close()`` (or use it as a context manager). SSH arguments override
    cluster-scoped defaults per key; explicit None/empty values clear that key.
    """
    from sparkrun.orchestration.telemetry import get_telemetry_provider

    cluster_def, scope, ssh_kwargs, sctx = _resolve_ctx(hosts, cluster, ssh_kwargs, sctx)
    provider = get_telemetry_provider(scope, sctx.variables)
    if provider is None:
        logger.debug("No telemetry provider for scope %r; occupancy-only monitoring", scope)
        return None
    return provider.open(list(hosts), ssh_kwargs=ssh_kwargs, interval=interval, config=sctx.config, backend=backend)


class LiveMonitorSession:
    """Composes a telemetry stream with a background ``api.status`` poll.

    ``frame()`` is **non-blocking**: it reads the latest telemetry snapshot and
    the latest cached occupancy (refreshed by a background thread every
    ``status_interval`` seconds), so a client (Textual TUI) can tick it freely.
    Use as a context manager, or ``close()`` when done.
    """

    def __init__(
        self,
        hosts,
        *,
        cluster_def,
        telemetry_session,
        ssh_kwargs,
        sctx,
        status_interval,
    ) -> None:
        self._hosts = list(hosts)
        self._cluster_def = cluster_def
        self._tel = telemetry_session
        self._ssh_kwargs = ssh_kwargs
        self._sctx = sctx
        self._status_interval = max(1, int(status_interval))
        self._status = None
        self._status_error = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        from contextvars import copy_context

        self._poller = threading.Thread(target=copy_context().run, args=(self._poll_loop,), daemon=True)
        self._poller.start()

    def _poll_once(self) -> None:
        import sparkrun.api as api

        try:
            snap = api.status(self._hosts, cluster=self._cluster_def, ssh_kwargs=self._ssh_kwargs, sctx=self._sctx)
        except Exception as error:  # noqa: BLE001 - retain known workloads, but not confirmed capacity
            logger.debug("live_monitor status poll failed; keeping last snapshot", exc_info=True)
            with self._lock:
                self._status_error = "status poll failed: %s" % error
            return
        with self._lock:
            self._status = snap
            self._status_error = None

    def _poll_loop(self) -> None:
        self._poll_once()  # prime immediately so the first frame has occupancy
        while not self._stop.wait(self._status_interval):
            self._poll_once()

    def frame(self) -> "MonitorFrame":
        """Return the current combined :class:`MonitorFrame` (non-blocking)."""
        from sparkrun.core.monitoring import HostActivity, MonitorFrame

        telemetry = self._tel.snapshot() if self._tel is not None else {}
        with self._lock:
            status = self._status
            status_error = self._status_error

        observation_errors = status.observation_errors if status is not None else {}
        activities = []
        for host in self._hosts:
            tel = telemetry.get(host)
            occ = status.for_host(host) if status is not None else None
            activities.append(
                HostActivity(
                    host=host,
                    telemetry=tel.sample if tel is not None else None,
                    telemetry_error=tel.error if tel is not None else None,
                    workloads=occ.workloads if occ is not None else (),
                    used_slots=occ.used_slots if occ is not None else 0,
                    free_slots=status.free_slots(host) if status is not None and not status_error else 0,
                    status_error=status_error or observation_errors.get(host),
                )
            )
        return MonitorFrame(hosts=tuple(activities), queried_at=time.time())

    def close(self) -> None:
        self._stop.set()
        if self._tel is not None:
            try:
                self._tel.close()
            except Exception:  # noqa: BLE001
                logger.debug("telemetry session close failed", exc_info=True)

    def __enter__(self) -> "LiveMonitorSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_live_monitor(
    hosts: list[str],
    *,
    cluster: "str | ClusterDefinition | None" = None,
    ssh_kwargs: dict | None = None,
    interval: int = 2,
    status_interval: int | None = None,
    backend: str | None = None,
    sctx: "SparkrunContext | None" = None,
) -> LiveMonitorSession:
    """Open a pollable :class:`LiveMonitorSession` (telemetry + occupancy).

    *interval* is the telemetry cadence; *status_interval* the occupancy re-poll
    cadence (defaults to ``max(interval*2, 5)`` to bound SSH cost).  The TUI
    ticks ``frame()``; the ``live_monitor`` generator wraps this. SSH arguments
    use the same per-key cluster-scoped overrides as :func:`open_telemetry`.
    Frames retain partial workloads but report incomplete observations as errors
    with zero confirmed free slots.
    """
    from sparkrun.orchestration.telemetry import get_telemetry_provider

    cluster_def, scope, ssh_kwargs, sctx = _resolve_ctx(hosts, cluster, ssh_kwargs, sctx)
    provider = get_telemetry_provider(scope, sctx.variables)
    telemetry_session = None
    if provider is not None:
        telemetry_session = provider.open(list(hosts), ssh_kwargs=ssh_kwargs, interval=interval, config=sctx.config, backend=backend)
    return LiveMonitorSession(
        hosts,
        cluster_def=cluster_def,
        telemetry_session=telemetry_session,
        ssh_kwargs=ssh_kwargs,
        sctx=sctx,
        status_interval=status_interval if status_interval is not None else max(interval * 2, 5),
    )


def live_monitor(
    hosts: list[str],
    *,
    cluster: "str | ClusterDefinition | None" = None,
    ssh_kwargs: dict | None = None,
    interval: int = 2,
    status_interval: int | None = None,
    backend: str | None = None,
    sctx: "SparkrunContext | None" = None,
):
    """Yield :class:`MonitorFrame` snapshots on *interval* until the caller stops.

    Blocking generator over :class:`LiveMonitorSession`, for headless consumers
    (an SSE bridge, a logger).  Closes the session on generator teardown.  The
    wait between frames is the operation's cancel event (``sctx.operation``, or
    the current operation), so cancelling it ends the stream at once instead of
    after a full interval.
    """
    from sparkrun.api._context import resolve_sctx
    from sparkrun.core.operation import CURRENT

    sctx = resolve_sctx(sctx)
    operation = getattr(sctx, "operation", None) or CURRENT.get()
    cancel = operation.cancel if operation is not None else threading.Event()
    session = open_live_monitor(
        hosts,
        cluster=cluster,
        ssh_kwargs=ssh_kwargs,
        interval=interval,
        status_interval=status_interval,
        backend=backend,
        sctx=sctx,
    )
    try:
        while not cancel.wait(interval):
            yield session.frame()
    finally:
        session.close()


__all__ = ["open_telemetry", "open_live_monitor", "live_monitor", "LiveMonitorSession"]
