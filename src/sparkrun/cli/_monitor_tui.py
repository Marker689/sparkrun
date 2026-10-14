"""Textual TUI for ``sparkrun cluster monitor``.

Driven by a :class:`~sparkrun.api.LiveMonitorSession` (``api.open_live_monitor``),
so each host row combines live resource telemetry with the workloads occupying
it from ``api.status`` — i.e. docker **and** ``local`` (and provider) workloads,
not just the docker containers the old telemetry-embedded ``docker ps`` saw.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from collections.abc import Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Static

from sparkrun.core.application_profile import get_application_profile
from sparkrun.core.monitoring import HostActivity, MonitorSample

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
    return sum(len(w.containers) or 1 for w in activity.workloads)


def _render_workloads(activity: HostActivity, cache_dir: str | None) -> list[str]:
    """Render the workload list from occupancy (all executors), with metadata.

    Sourced from :attr:`HostActivity.workloads` (``api.status``) rather than the
    telemetry stream, so ``local``/provider workloads appear.  Job metadata
    (recipe/model/tp) is loaded per cluster_id; per-container name/role/status
    comes straight from the snapshot.
    """
    from sparkrun.orchestration.job_metadata import load_job_metadata

    lines: list[str] = []
    total = _container_count(activity)
    if not activity.workloads:
        lines.append("  [dim]No %s workloads running[/dim]" % get_application_profile().command)
        return lines

    lines.append(f"  [bold]Workloads ({total}):[/bold]")
    for w in activity.workloads:
        meta = load_job_metadata(w.cluster_id, cache_dir=cache_dir) or {}
        recipe = w.recipe_name or meta.get("recipe", "")
        runtime = w.runtime_name or meta.get("runtime", "")
        model = meta.get("model", "")
        tp = meta.get("tensor_parallel", "")

        rank_tag = f" [dim]×{w.ranks_on_host}[/dim]" if w.ranks_on_host > 1 else ""
        lines.append(f"    [bold]{w.cluster_id}[/bold]{rank_tag}")

        for c in w.containers:
            role_tag = f" [dim]({c.role})[/dim]" if c.role and c.role != "?" else ""
            status_tag = f" [dim]— {c.status}[/dim]" if c.status else ""
            lines.append(f"      {c.name}{role_tag}{status_tag}")

        meta_parts: list[str] = []
        if recipe:
            meta_parts.append(f"recipe=[bold]{recipe}[/bold]")
        if model:
            meta_parts.append(f"model={model}")
        if runtime:
            meta_parts.append(f"runtime={runtime}")
        if tp:
            meta_parts.append(f"tp={tp}")
        if meta_parts:
            lines.append(f"      [dim]{', '.join(meta_parts)}[/dim]")
    return lines


def _render_detail(host: str, activity: HostActivity | None, cache_dir: str | None = None) -> str:
    """Build the Rich-markup string for the detail panel."""
    if activity is None:
        return f"[dim]{host}: connecting…[/dim]"

    s: MonitorSample | None = activity.telemetry

    lines: list[str] = [f"[bold]{host}[/bold]"]
    if activity.telemetry_error:
        lines.append(f"  [yellow]telemetry: {activity.telemetry_error}[/yellow]")
    if activity.status_error:
        lines.append(f"  [yellow]status: {activity.status_error}[/yellow]")

    if s is None:
        # No telemetry yet / no telemetry provider for this substrate — still
        # show occupancy so the host isn't blank.
        lines.append("  [dim]telemetry unavailable[/dim]")
        lines.append("")
        lines.extend(_render_workloads(activity, cache_dir))
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

    lines.extend(_render_workloads(activity, cache_dir))
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
    #detail-panel {
        height: 1fr;
        margin: 0 1;
        border: round $accent;
        padding: 0 1;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit", show=False),
    ]

    def __init__(
        self,
        session: "LiveMonitorSession",
        hosts: list[str],
        interval: int = 2,
        cache_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session = session
        self._hosts = list(hosts)
        self._interval = interval
        self._cache_dir = cache_dir
        self._selected_host: str | None = self._hosts[0] if self._hosts else None

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield DataTable(id="host-table", cursor_type="row")
            yield Static(id="detail-panel", markup=True)
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

        # The session is already collecting (opened before the App); just tick.
        self.set_interval(1.0, self._refresh)

    def on_unmount(self) -> None:
        self._session.close()

    # -- events -------------------------------------------------------------

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None:
            self._selected_host = str(event.row_key.value)
            self._refresh_detail()

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
        panel = self.query_one("#detail-panel", Static)
        host = self._selected_host
        if host is None:
            panel.update("[dim]No host selected[/dim]")
            return
        frame = getattr(self, "_frame", None)
        activity = frame.for_host(host) if frame is not None else None
        panel.update(_render_detail(host, activity, cache_dir=self._cache_dir))
