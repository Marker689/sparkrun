"""Textual TUI for ``sparkrun cluster monitor``.

Driven by a :class:`~sparkrun.api.LiveMonitorSession` (``api.open_live_monitor``),
so each host row combines live resource telemetry with the workloads occupying
it from ``api.status`` — i.e. docker **and** ``local`` (and provider) workloads,
not just the docker containers the old telemetry-embedded ``docker ps`` saw.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Mapping, Sequence

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

# Sparklines are rendered as inline block glyphs inside the existing
# Static(markup=True) detail panel — textual.widgets.Sparkline would force the
# panel into a widget grid for no gain.
_SPARK_GLYPHS = "▁▂▃▄▅▆▇█"
_SPARKLINE_WIDTH = 20  # ~40 s of trend at the default 2 s record cadence
_CELL_LABEL_WIDTH = 9  # longest label ("GPU Power") keeps the grid columns lined up
_CELL_VALUE_WIDTH = 10  # "999.9 kb/s"-class values; longer ones just overflow

# Sparkline colors come from the active theme's palette, stepped away from the
# background — lighten on dark themes, darken on light ones — so they stay
# visible either way.  Five hues x three luminance steps; the grid places
# same-hue entries far apart.  (Named ANSI colors cannot be used: Textual
# resolves only what its color table knows, so "bright_red" silently falls back
# to the default foreground — the whole reason these are theme variables.)
# Minimal ANSI themes (ansi-dark/ansi-light) collapse every shade onto its base,
# so they yield 4-5 distinct colors instead of 11; that is inherent to those
# palettes, not a bug here — the full sets come from textual-dark/light and the
# third-party themes.
_SPARK_FAMILIES = ("primary", "secondary", "accent", "success", "error")
_SPARK_STEPS_DARK = ("-lighten-1", "-lighten-2", "-lighten-3")
_SPARK_STEPS_LIGHT = ("-darken-1", "-darken-2", "-darken-3")
# (family index, step index) in cell order: GPU %, GPU Temp, GPU Power, CPU %,
# CPU Temp, GPU Clock, RAM %, Net Send, Net Recv, App, Cache
_SPARK_GRID = ((0, 0), (4, 0), (2, 0), (3, 0), (1, 0), (0, 1), (3, 1), (2, 1), (4, 1), (0, 2), (1, 1))


def _spark_palette(css: Mapping[str, object], dark: bool) -> tuple[str, ...]:
    """Eleven distinct sparkline colors, in cell order, from *css* theme variables."""
    steps = _SPARK_STEPS_DARK if dark else _SPARK_STEPS_LIGHT
    try:
        return tuple(str(css[_SPARK_FAMILIES[family] + steps[step]]) for family, step in _SPARK_GRID)
    except KeyError:  # unexpected theme shape: render uncolored rather than blow up
        return ()


def _at(palette: Sequence[str], index: int) -> str | None:
    """Palette entry *index*, or None when the palette is unset/short."""
    return palette[index] if index < len(palette) else None


def _series(history: list[tuple[float, MonitorSample]] | None, field: str) -> list[float]:
    """Parsable finite float values of *field* across *history* points (skips empty/garbage)."""
    out: list[float] = []
    for _ts, sample in history or ():
        try:
            value = float(getattr(sample, field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):  # a remote "nan"/"inf" must not break the tick
            out.append(value)
    return out


def _sparkline(
    values: list[float],
    width: int = _SPARKLINE_WIDTH,
    lo: float | None = None,
    hi: float | None = None,
) -> str:
    """One block-glyph sparkline, newest right-aligned; blank (spaces) when empty.

    ``lo``/``hi`` pin the scale — percent metrics pass 0/100 so hosts stay
    comparable — and whichever end is not given uses the window min/max.  A flat
    window renders '▄' per point.
    """
    pts = list(values)[-width:]
    if not pts:
        return " " * width
    lo_eff = min(min(pts), lo) if lo is not None else min(pts)
    hi_eff = max(max(pts), hi) if hi is not None else max(pts)
    span = hi_eff - lo_eff
    if span <= 0:
        glyphs = "▄" * len(pts)
    else:
        glyphs = "".join(_SPARK_GLYPHS[min(7, max(0, round((v - lo_eff) / span * 7)))] for v in pts)
    return glyphs.rjust(width)


_RATE_UNITS = ("b/s", "kb/s", "Mb/s", "Gb/s", "Tb/s")


def _fmt_bitrate(bytes_per_s: float) -> str:
    """Network bytes/s → bits/s with an adaptive magnitude prefix (decimal steps).

    Rises and falls with the value: 60 B/s reads '480.0 b/s', 1 MB/s '8.0 Mb/s'.
    Clamps at Tb/s and reads '0.0 b/s' on an idle link.
    """
    bits = bytes_per_s * 8
    idx = 0
    while idx + 1 < len(_RATE_UNITS) and bits >= 1000.0:
        bits /= 1000.0
        idx += 1
    return f"{bits:.1f} {_RATE_UNITS[idx]}"


def _cell(
    label: str,
    sample: MonitorSample | None,
    history: list[tuple[float, MonitorSample]] | None,
    field: str,
    fmt: Callable[[float], str],
    *,
    lo: float | None = None,
    hi: float | None = None,
    color: str | None = None,
    series_field: str | None = None,
) -> str:
    """``'{label} {sparkline} {value}'`` for one grid cell.

    The value parses from the fresh frame sample (``-`` when empty/unparsable)
    because ``history()`` lags the frame by up to one record interval; the
    sparkline comes from the history series.  ``series_field`` overrides which
    column feeds the sparkline when value and trend come from different ones.
    """
    try:
        value = fmt(float(getattr(sample, field, "")))
    except (TypeError, ValueError):
        value = "-"
    spark = _sparkline(_series(history, series_field or field), lo=lo, hi=hi)
    if color:
        spark = f"[{color}]{spark}[/{color}]"
    return f"{label:<{_CELL_LABEL_WIDTH}} {spark} {value:<{_CELL_VALUE_WIDTH}}"


def _container_count(activity: HostActivity | None) -> int:
    """Total containers/processes occupying the host (across its workloads)."""
    if activity is None:
        return 0
    return sum(len(w.containers) or 1 for w in activity.workloads)


def _render_workloads(activity: Any, cache_dir: str | None = None) -> list[str]:
    return [r.text for r in workload_rows(activity, cache_dir=cache_dir)]


def _render_node_stats(
    host: str,
    activity: HostActivity | None,
    history: list[tuple[float, MonitorSample]] | None = None,
    palette: Sequence[str] = (),
) -> str:
    """Build the Rich-markup string for the node stats in the detail panel.

    One header line (host │ hostname │ RAM used/total │ CPU │ GPU (count)) then
    a 3x3 sparkline grid; an nv-monitor-only fourth row adds the App/Cache
    memory breakdown.  ``history`` is the host's ``LiveMonitorSession.history()``
    points (oldest first) — absent, every sparkline renders blank.  ``palette``
    holds the per-cell sparkline colors (see :func:`_spark_palette`); empty
    renders them uncolored.
    """
    if activity is None:
        return f"[dim]{host}: connecting…[/dim]"

    s: MonitorSample | None = activity.telemetry

    errors: list[str] = []
    if activity.telemetry_error:
        errors.append(f"  [yellow]telemetry: {activity.telemetry_error}[/yellow]")
    if activity.status_error:
        errors.append(f"  [yellow]status: {activity.status_error}[/yellow]")

    if s is None:
        return "\n".join([f"[bold]{host}[/bold]", *errors, "  [dim]telemetry unavailable[/dim]"])

    # Header line: only non-empty segments, in a fixed order.
    header: list[str] = [f"[bold]{host}[/bold]"]
    if s.hostname:
        header.append(s.hostname)
    if s.mem_used_mb and s.mem_total_mb:
        header.append(f"RAM {s.mem_used_mb}/{s.mem_total_mb} MB")
    if s.cpu_name:
        header.append(s.cpu_name)
    if s.gpu_name:
        header.append(f"{s.gpu_name} ({s.gpu_count})" if s.gpu_count else s.gpu_name)

    lines: list[str] = ["  " + " │ ".join(header), *errors, ""]

    grid = (
        (
            _cell("GPU %", s, history, "gpu_util_pct", lambda v: f"{v:.1f} %", lo=0.0, hi=100.0, color=_at(palette, 0)),
            _cell("GPU Temp", s, history, "gpu_temp_c", lambda v: f"{v:.1f} °C", color=_at(palette, 1)),
            _cell("GPU Power", s, history, "gpu_power_w", lambda v: f"{v:.2f} W", color=_at(palette, 2)),
        ),
        (
            _cell("CPU %", s, history, "cpu_usage_pct", lambda v: f"{v:.1f} %", lo=0.0, hi=100.0, color=_at(palette, 3)),
            _cell("CPU Temp", s, history, "cpu_temp_c", lambda v: f"{v:.1f} °C", color=_at(palette, 4)),
            _cell("GPU Clock", s, history, "gpu_clock_mhz", lambda v: f"{v:.0f} MHz", color=_at(palette, 5)),
        ),
        (
            _cell("RAM %", s, history, "mem_used_pct", lambda v: f"{v:.1f} %", lo=0.0, hi=100.0, color=_at(palette, 6)),
            _cell("Net Send", s, history, "net_send_bytes_s", _fmt_bitrate, color=_at(palette, 7)),
            _cell("Net Recv", s, history, "net_recv_bytes_s", _fmt_bitrate, color=_at(palette, 8)),
        ),
    )
    lines.extend("  " + "   ".join(row) for row in grid)

    # Memory breakdown row (nv-monitor backend only — the bash CSV has no bufcache).
    if s.mem_bufcache_mb and s.mem_used_mb and s.mem_total_mb:
        try:
            total = float(s.mem_total_mb)
        except (ValueError, TypeError):
            total = 0.0
        if total > 0:

            def _mem_fmt(v: float) -> str:
                return f"{v:.0f} MB ({v / total * 100:.1f}%)"

            lines.append(
                "  "
                + "   ".join(
                    (
                        _cell(
                            "App", s, history, "mem_used_mb", _mem_fmt, series_field="mem_used_pct", lo=0.0, hi=100.0, color=_at(palette, 9)
                        ),
                        _cell("Cache", s, history, "mem_bufcache_mb", _mem_fmt, color=_at(palette, 10)),
                    )
                )
            )

    lines.append("")
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
        self.query_one("#workload-list", WorkloadList).focus()

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if getattr(event.control, "id", None) == "host-table":
            if self._is_log_open():
                self._close_log_pane(refocus_list=False)
            self._pinned_host = None
            self._refresh_detail()

    def on_workload_list_back_to_hosts(self, event: WorkloadList.BackToHosts) -> None:
        self._pinned_host = None
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

        # history() is optional so duck-typed session stand-ins keep working; a
        # host outside the session's set (cannot happen in production — the same
        # host list feeds session and app) degrades to blank sparklines.
        history: list[tuple[float, MonitorSample]] = []
        history_fn = getattr(self._session, "history", None)
        if history_fn is not None:
            try:
                history = history_fn(host)
            except KeyError:
                history = []

        # Sparkline colors follow the active theme; a theme that lacks the
        # variables yields an empty palette (sparks uncolored), not a crash.
        palette: tuple[str, ...] = ()
        try:
            palette = _spark_palette(self.get_css_variables(), dark=self.current_theme.dark)
        except Exception:  # noqa: BLE001 - cosmetic only; never break the tick
            palette = ()

        stats.update(_render_node_stats(host, activity, history, palette))

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
