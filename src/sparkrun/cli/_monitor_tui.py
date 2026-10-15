"""Textual TUI for ``sparkrun cluster monitor``.

Driven by a :class:`~sparkrun.api.LiveMonitorSession` (``api.open_live_monitor``),
so each host row combines live resource telemetry with the workloads occupying
it from ``api.status`` — i.e. docker **and** ``local`` (and provider) workloads,
not just the docker containers the old telemetry-embedded ``docker ps`` saw.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Static

from sparkrun.core.application_profile import get_application_profile
from sparkrun.cli._monitor_logs import (
    DumpLogScreen,
    LogPane,
    LogScope,
    WorkloadList,
    workload_rows,
)
from sparkrun.core.monitoring import HostActivity, MonitorFrame, MonitorSample

if TYPE_CHECKING:
    from sparkrun.api import LiveMonitorSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BAR_WIDTH = 30


def _bar(value: float, width: int = _BAR_WIDTH) -> str:
    """Render a Unicode bar: ████████░░░░░░░░░░░░."""
    pct = max(0.0, min(value / 100.0, 1.0))
    filled = int(pct * width)
    return "█" * filled + "░" * (width - filled)


def _pct(raw: str) -> float:
    """Parse a percentage string to float, defaulting to 0."""
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


def _container_count(activity: HostActivity | None) -> int:
    """Total containers/processes occupying the host (across its workloads)."""
    if activity is None:
        return 0
    try:
        # ponytail: a duck-typed test session (Mock) has no iterable workloads
        workloads = list(activity.workloads)
    except TypeError:
        return 0
    return sum(len(w.containers) or 1 for w in workloads)


def _render_workloads(activity: Any, cache_dir: str | None = None) -> list[str]:
    return [r.text for r in workload_rows(activity, cache_dir=cache_dir)]


def _render_node_stats(host: str, activity: HostActivity | None) -> str:
    """Build the Rich-markup string for the node stats in the detail panel."""
    if activity is None:
        return f"[dim]{host}: connecting…[/dim]"

    s: MonitorSample | None = activity.telemetry

    lines: list[str] = [f"[bold]{host}[/bold]"]
    if activity.telemetry_error:
        lines.append(f"  [yellow]telemetry: {activity.telemetry_error}[/yellow]")
    if activity.status_error:
        lines.append(f"  [yellow]status: {activity.status_error}[/yellow]")

    if s is None:
        lines.append("  [dim]telemetry unavailable[/dim]")
        return "\n".join(lines)

    cpu = _pct(s.cpu_usage_pct)
    ram = _pct(s.mem_used_pct)
    gpu = _pct(s.gpu_util_pct)
    lines.extend(
        [
            "",
            f"  CPU  [cyan]{_bar(cpu)}[/cyan] {cpu:5.1f}%",
            f"  RAM  [green]{_bar(ram)}[/green] {ram:5.1f}%",
            f"  GPU  [yellow]{_bar(gpu)}[/yellow] {gpu:5.1f}%",
            "",
        ]
    )

    # Hardware details
    extras: list[str] = []
    if s.gpu_name:
        extras.append(f"GPU: {s.gpu_name}")
    if s.cpu_temp_c:
        extras.append(f"CPU temp: {s.cpu_temp_c} °C")
    if s.gpu_temp_c:
        extras.append(f"GPU temp: {s.gpu_temp_c} °C")
    if s.gpu_power_w:
        power_str = f"{s.gpu_power_w} W"
        if s.gpu_power_limit_w:
            power_str += f" / {s.gpu_power_limit_w} W"
        extras.append(f"GPU power: {power_str}")
    if s.mem_used_mb and s.mem_total_mb:
        extras.append(f"RAM: {s.mem_used_mb} / {s.mem_total_mb} MB")
    if s.gpu_mem_used_mb and s.gpu_mem_total_mb:
        extras.append(f"GPU mem: {s.gpu_mem_used_mb} / {s.gpu_mem_total_mb} MB")
    if s.gpu_encoder_pct:
        extras.append(f"Encoder: {s.gpu_encoder_pct}%")
    if s.gpu_decoder_pct:
        extras.append(f"Decoder: {s.gpu_decoder_pct}%")
    if s.gpu_fan_pct:
        extras.append(f"Fan: {s.gpu_fan_pct}%")
    if extras:
        lines.append("  " + "  │  ".join(extras))
        lines.append("")

    # Memory breakdown (when nv-monitor provides bufcache data)
    if s.mem_bufcache_mb and s.mem_used_mb and s.mem_total_mb:
        try:
            total = float(s.mem_total_mb)
            used = float(s.mem_used_mb)
            bufcache = float(s.mem_bufcache_mb)
            if total > 0:
                used_pct = used / total * 100
                buf_pct = bufcache / total * 100
                lines.append("  Memory breakdown:")
                lines.append(f"    App:    [green]{_bar(used_pct)}[/green] {used:.0f} MB ({used_pct:.1f}%)")
                lines.append(f"    Cache:  [cyan]{_bar(buf_pct)}[/cyan] {bufcache:.0f} MB ({buf_pct:.1f}%)")
                lines.append("")
        except (ValueError, TypeError):
            pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table cell formatters (telemetry columns; the Jobs column comes from occupancy)
# ---------------------------------------------------------------------------


def _cell_cpu(s: MonitorSample) -> str:
    return s.cpu_usage_pct or "-"


def _cell_ram(s: MonitorSample) -> str:
    return "%s%%" % s.mem_used_pct if s.mem_used_pct else "-"


def _cell_gpu(s: MonitorSample) -> str:
    return s.gpu_util_pct or "-"


def _cell_cpu_temp(s: MonitorSample) -> str:
    return "%s C" % s.cpu_temp_c if s.cpu_temp_c else "-"


def _cell_gpu_temp(s: MonitorSample) -> str:
    return "%s C" % s.gpu_temp_c if s.gpu_temp_c else "-"


def _cell_gpu_power(s: MonitorSample) -> str:
    return "%s W" % s.gpu_power_w if s.gpu_power_w else "-"


def _cell_swap(s: MonitorSample) -> str:
    if not s.swap_total_mb or not s.swap_used_mb:
        return "-"
    try:
        total = float(s.swap_total_mb)
        used = float(s.swap_used_mb)
        if total <= 0:
            return "0%"
        return "%.0f%%" % (used / total * 100)
    except (ValueError, TypeError):
        return "-"


def _cell_gpu_enc(s: MonitorSample) -> str:
    return s.gpu_encoder_pct or "-"


def _cell_gpu_dec(s: MonitorSample) -> str:
    return s.gpu_decoder_pct or "-"


# Ordered telemetry column definitions: (key, label, cell_fn). The leading
# Host + Jobs columns are handled separately (Jobs is occupancy-derived).
_TABLE_COLS: list[tuple[str, str, Callable[[MonitorSample], str]]] = [
    ("cpu", "CPU%", _cell_cpu),
    ("ram", "RAM%", _cell_ram),
    ("swap", "Swap%", _cell_swap),
    ("gpu", "GPU%", _cell_gpu),
    ("cpu_temp", "CPU Temp", _cell_cpu_temp),
    ("gpu_temp", "GPU Temp", _cell_gpu_temp),
    ("gpu_power", "GPU Power", _cell_gpu_power),
    ("gpu_enc", "Enc%", _cell_gpu_enc),
    ("gpu_dec", "Dec%", _cell_gpu_dec),
]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class ClusterMonitorApp(App):
    """Textual TUI for live cluster monitoring (telemetry + occupancy)."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #host-table {
        height: auto;
        max-height: 60%;
        margin: 0 1;
    }
    Screen.-logs-open #host-table,
    .-logs-open #host-table {
        max-height: 25%;
    }
    #detail-panel {
        height: 1fr;
        margin: 0 1;
        border: round $accent;
        padding: 0 1;
        overflow-y: auto;
    }
    #node-stats {
        height: auto;
    }
    #node-body {
        height: 1fr;
    }
    LogPane {
        min-height: 8;
    }
    """

    BINDINGS = [
        Binding("enter", "select_node", "Logs"),
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit", show=False),
    ]

    def __init__(
        self,
        session: "LiveMonitorSession",
        hosts: list[str],
        interval: int = 2,
        cache_dir: str | None = None,
        cluster: str | None = None,
        sctx: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._session = session
        self._hosts = list(hosts)
        self._interval = interval
        self._cache_dir = cache_dir
        self._cluster = cluster
        self._sctx = sctx
        self._selected_host: str | None = self._hosts[0] if self._hosts else None
        self._pinned_host: str | None = None
        self._frame: MonitorFrame | None = None

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield DataTable(id="host-table", cursor_type="row")
            with Vertical(id="detail-panel"):
                yield Static(id="node-stats", markup=True)
                with Vertical(id="node-body"):
                    yield WorkloadList(id="workload-list", host=self._selected_host or "")
        yield Footer()

    # -- lifecycle ----------------------------------------------------------

    def on_mount(self) -> None:
        self.title = "%s cluster monitor" % get_application_profile().command
        self.sub_title = "%d host(s) — every %ds" % (len(self._hosts), self._interval)

        table = self.query_one("#host-table", DataTable)
        table.add_column("Host", key="host")
        table.add_column("Jobs", key="jobs")
        for key, label, _ in _TABLE_COLS:
            table.add_column(label, key=key)

        for host in self._hosts:
            table.add_row(host, "-", *(["-"] * len(_TABLE_COLS)), key=host)

        # Immediate initial refresh, then tick periodically.
        self._refresh()
        self.set_interval(1.0, self._refresh)

    def on_unmount(self) -> None:
        self._session.close()

    # -- events -------------------------------------------------------------

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None:
            self._selected_host = str(event.row_key.value)
            if self._pinned_host is None:
                self._refresh_detail()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_select_node()

    def action_select_node(self) -> None:
        try:
            table = self.query_one("#host-table", DataTable)
            if not table.has_focus:
                return
        except Exception:
            return
        host = self._selected_host
        if not host:
            return
        frame = getattr(self, "_frame", None)
        activity = frame.for_host(host) if frame is not None else None
        if not activity or not activity.workloads:
            self.notify(f"No sparkrun workloads on {host}")
            return

        self._pinned_host = host
        wl = self.query_one("#workload-list", WorkloadList)
        wl.reveal()
        wl.focus()

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if getattr(event.control, "id", None) == "host-table":
            if self._is_log_open():
                self._close_log_pane(refocus_list=False)
            self._pinned_host = None
            try:
                wl = self.query_one("#workload-list", WorkloadList)
                wl.conceal()
            except Exception:
                pass
            self._refresh_detail()

    def on_workload_list_back_to_hosts(self, event: WorkloadList.BackToHosts) -> None:
        self._pinned_host = None
        wl = self.query_one("#workload-list", WorkloadList)
        wl.conceal()
        self.query_one("#host-table", DataTable).focus()

    def on_workload_list_open_log(self, event: WorkloadList.OpenLog) -> None:
        wl = self.query_one("#workload-list", WorkloadList)
        wl.display = False

        pane = LogPane(
            host=event.host,
            cluster_id=event.cluster_id,
            container=event.container,
            scope=LogScope.CONTAINER,
            frame=self._frame,
            cluster=self._cluster,
            cache_dir=self._cache_dir,
            sctx=self._sctx,
            id="log-pane",
        )
        self.query_one("#node-body", Vertical).mount(pane)
        self.add_class("-logs-open")
        self.screen.add_class("-logs-open")
        pane.focus()

    def on_log_pane_closed(self, event: LogPane.Closed) -> None:
        self._close_log_pane(refocus_list=True)

    def _close_log_pane(self, refocus_list: bool = True) -> None:
        try:
            pane = self.query_one("#log-pane", LogPane)
            pane.close()
            pane.remove()
        except Exception:
            pass
        self.remove_class("-logs-open")
        self.screen.remove_class("-logs-open")
        try:
            wl = self.query_one("#workload-list", WorkloadList)
            wl.display = True
            if refocus_list:
                wl.focus()
        except Exception:
            pass

    def _is_log_open(self) -> bool:
        try:
            return bool(self.query("#log-pane"))
        except Exception:
            return False

    def on_workload_list_request_dump(self, event: WorkloadList.RequestDump) -> None:
        self._open_dump_screen(event.host, event.cluster_id, event.container, event.scope)

    def on_log_pane_request_dump(self, event: LogPane.RequestDump) -> None:
        self._open_dump_screen(event.host, event.cluster_id, event.container, event.scope)

    def _open_dump_screen(self, host: str, cluster_id: str, container: str, scope: LogScope) -> None:
        screen = DumpLogScreen(
            host=host,
            cluster_id=cluster_id,
            container=container,
            default_scope=scope,
            frame=self._frame,
            cluster=self._cluster,
            cache_dir=self._cache_dir,
            sctx=self._sctx,
        )
        self.push_screen(screen)

    # -- refresh ------------------------------------------------------------

    def _refresh(self) -> None:
        self._frame = self._session.frame()
        self._refresh_table()
        self._refresh_detail()

    def _refresh_table(self) -> None:
        table = self.query_one("#host-table", DataTable)
        frame = getattr(self, "_frame", None)
        for host in self._hosts:
            activity = frame.for_host(host) if frame is not None else None
            if activity is None:
                continue

            # Connection status alongside the hostname.
            if activity.telemetry is None and activity.telemetry_error:
                table.update_cell(host, "host", "%s (error)" % host)
            elif activity.telemetry_error or activity.status_error:
                table.update_cell(host, "host", "%s (!)" % host)
            else:
                table.update_cell(host, "host", host)

            # Jobs from occupancy (all executors), not the telemetry stream.
            table.update_cell(host, "jobs", str(_container_count(activity)) if activity.workloads else "-")

            s = activity.telemetry
            if s is None:
                continue
            for key, _label, cell_fn in _TABLE_COLS:
                table.update_cell(host, key, cell_fn(s))

    def _refresh_detail(self) -> None:
        stats = self.query_one("#node-stats", Static)
        host = self._pinned_host or self._selected_host
        if host is None:
            stats.update("[dim]No host selected[/dim]")
            return
        frame = getattr(self, "_frame", None)
        activity = frame.for_host(host) if frame is not None else None
        stats.update(_render_node_stats(host, activity))

        try:
            wl = self.query_one("#workload-list", WorkloadList)
            rows = workload_rows(activity, cache_dir=self._cache_dir)
            wl.set_rows(rows, host=host)
        except Exception:
            pass

    def action_quit(self) -> None:
        if self._is_log_open():
            return
        self.exit()
