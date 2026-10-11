"""Tests for the resource-telemetry seam + the api live-monitoring surface.

(Distinct from ``test_telemetry.py``, which covers anonymous usage analytics.)

Covers:
- TelemetryProvider discovery / scope selection (SAF).
- HostTelemetryProvider backend selection + session snapshot (SSH mocked).
- api.open_telemetry (None when no provider for the scope).
- LiveMonitorSession / api.open_live_monitor frame composition (telemetry + occupancy).
"""

from __future__ import annotations

import time
from unittest import mock

import pytest

import sparkrun.api as api
from sparkrun.core.cluster_status import ClusterStatus, ContainerDetail, HostOccupancy, RunningWorkload
from sparkrun.core.monitoring import HostActivity, HostTelemetry, MonitorFrame, MonitorSample
from sparkrun.orchestration.logs import LogLine, LogSource
from sparkrun.orchestration.telemetry import get_telemetry_provider, list_telemetry_scopes
from sparkrun.orchestration.telemetry._base import TelemetryProvider, TelemetrySession


# --------------------------------------------------------------------------
# Provider discovery
# --------------------------------------------------------------------------


def test_host_provider_registered_and_resolvable():
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    assert "host" in list_telemetry_scopes(v)
    provider = get_telemetry_provider("host", v)
    assert provider is not None
    assert provider.scope == "host"


def test_unknown_scope_returns_none():
    from sparkrun.core.bootstrap import init_sparkrun

    v = init_sparkrun()
    # No k8s/modal telemetry provider in-tree — a substrate with no provider
    # yields None (occupancy-only monitoring), never an error.
    assert get_telemetry_provider("k8s", v) is None


def test_base_provider_open_not_implemented():
    with mock.patch.object(TelemetryProvider, "scope", "x"):
        with pytest.raises(NotImplementedError):
            TelemetryProvider().open(["h1"])


# --------------------------------------------------------------------------
# HostTelemetryProvider — backend selection + session (SSH mocked)
# --------------------------------------------------------------------------


class _FakeMonitor:
    """Stand-in for ClusterMonitor: records start/stop, exposes .states."""

    def __init__(self, hosts, ssh_kwargs, interval=2, **kw):
        from sparkrun.core.monitoring import HostMonitorState

        self.hosts = hosts
        self.interval = interval
        self.started = False
        self.stopped = False
        self.states = {h: HostMonitorState(latest=MonitorSample(hostname=h, gpu_util_pct="10")) for h in hosts}

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_host_provider_defaults_to_bash_backend(monkeypatch):
    from sparkrun.orchestration.telemetry.host import HostTelemetryProvider

    created = {}

    def _fake_cm(hosts, ssh_kwargs, interval=2, **kw):
        created["bash"] = _FakeMonitor(hosts, ssh_kwargs, interval)
        return created["bash"]

    monkeypatch.setattr("sparkrun.core.monitoring.ClusterMonitor", _fake_cm)
    session = HostTelemetryProvider().open(["h1"], ssh_kwargs={}, interval=3)
    assert created["bash"].started
    assert session.snapshot()["h1"].sample.gpu_util_pct == "10"
    session.close()
    assert created["bash"].stopped


def test_host_provider_nv_monitor_backend(monkeypatch):
    from sparkrun.orchestration.telemetry.host import HostTelemetryProvider

    created = {}

    def _fake_nv(hosts, ssh_kwargs, interval=2, **kw):
        created["nv"] = _FakeMonitor(hosts, ssh_kwargs, interval)
        return created["nv"]

    monkeypatch.setattr("sparkrun.core.monitoring.NvMonitorClusterMonitor", _fake_nv)
    HostTelemetryProvider().open(["h1"], ssh_kwargs={}, backend="nv-monitor")
    assert "nv" in created and created["nv"].started


# --------------------------------------------------------------------------
# api.open_telemetry / live-monitor composition
# --------------------------------------------------------------------------


class _FakeSession(TelemetrySession):
    def __init__(self, snap):
        self._snap = snap
        self.closed = False

    def snapshot(self):
        return self._snap

    def close(self):
        self.closed = True


class _FakeProvider:
    scope = "host"

    def __init__(self, session):
        self._session = session

    def open(self, hosts, **kw):
        return self._session


def test_open_telemetry_none_when_no_provider(monkeypatch):
    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda scope, v=None: None)
    assert api.open_telemetry(["h1"]) is None


def _drive_until(session, pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    frame = session.frame()
    while not pred(frame) and time.monotonic() < deadline:
        time.sleep(0.02)
        frame = session.frame()
    return frame


def test_live_monitor_frame_combines_telemetry_and_occupancy(monkeypatch):
    tel = {"h1": HostTelemetry(host="h1", sample=MonitorSample(hostname="h1", gpu_util_pct="42"))}
    monkeypatch.setattr(
        "sparkrun.orchestration.telemetry.get_telemetry_provider",
        lambda scope, v=None: _FakeProvider(_FakeSession(tel)),
    )
    snap = ClusterStatus(
        hosts=(HostOccupancy(host="h1", workloads=(RunningWorkload(cluster_id="sparkrun_x"),), used_slots=1, free_slots=0),),
        executor="docker",
    )
    # The background poller reads sparkrun.api.status (package attr).
    monkeypatch.setattr("sparkrun.api.status", lambda *a, **k: snap)

    session = api.open_live_monitor(["h1"], interval=1, status_interval=1)
    try:
        frame = _drive_until(session, lambda f: f.for_host("h1").used_slots == 1)
    finally:
        session.close()

    assert isinstance(frame, MonitorFrame)
    act = frame.for_host("h1")
    assert isinstance(act, HostActivity)
    assert act.telemetry.gpu_util_pct == "42"  # telemetry axis
    assert [w.cluster_id for w in act.workloads] == ["sparkrun_x"]  # occupancy axis
    assert act.used_slots == 1


def test_live_monitor_occupancy_only_when_no_telemetry(monkeypatch):
    """A substrate with no telemetry provider still yields occupancy frames."""
    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda scope, v=None: None)
    snap = ClusterStatus(hosts=(HostOccupancy(host="h1", used_slots=0, free_slots=4),), executor="docker")
    monkeypatch.setattr("sparkrun.api.status", lambda *a, **k: snap)

    session = api.open_live_monitor(["h1"], interval=1, status_interval=1)
    try:
        frame = _drive_until(session, lambda f: f.for_host("h1").free_slots == 4)
    finally:
        session.close()

    act = frame.for_host("h1")
    assert act.telemetry is None  # no telemetry provider
    assert act.free_slots == 4  # occupancy still present


def test_live_monitor_stops_promptly_when_its_operation_is_cancelled(monkeypatch):
    import threading

    from sparkrun.core.operation import OperationState

    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda scope, v=None: None)
    monkeypatch.setattr("sparkrun.api.status", lambda *a, **k: ClusterStatus(hosts=(HostOccupancy(host="h1"),), executor="docker"))
    sctx = api.default_sctx()
    sctx.operation = OperationState()
    threading.Timer(0.2, sctx.operation.cancel.set).start()
    started = time.monotonic()
    frames = list(api.live_monitor(["h1"], interval=60, sctx=sctx))
    assert frames == []  # cancelled during the first wait
    assert time.monotonic() - started < 10  # not a 60s uninterruptible sleep


# --------------------------------------------------------------------------
# TUI detail rendering — workloads sourced from occupancy (all executors)
# --------------------------------------------------------------------------


def _activity_with_local_job(telemetry=None):
    from sparkrun.core.cluster_status import ContainerDetail, RunningWorkload

    wl = RunningWorkload(
        cluster_id="sparkrun_deadbeefdeadbeef_aabbccddeeff",
        recipe_name="qwen3-1.7b-vllm",
        runtime_name="vllm",
        containers=(
            ContainerDetail(name="sparkrun_deadbeefdeadbeef_aabbccddeeff_solo", role="solo", status="Up (pid 42)", image="(local process)"),
        ),
    )
    return HostActivity(host="h1", telemetry=telemetry, workloads=(wl,), used_slots=1, free_slots=0)


def test_render_node_stats_and_workload_rows(tmp_path):
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_tui import _render_node_stats
    from sparkrun.cli._monitor_logs import workload_rows

    act = _activity_with_local_job(MonitorSample(hostname="h1", gpu_util_pct="7"))
    stats_out = _render_node_stats("h1", act)
    assert "GPU" in stats_out
    assert "Workloads" not in stats_out

    rows = workload_rows(act, cache_dir=str(tmp_path))
    texts = [r.text for r in rows]
    joined = "\n".join(texts)
    assert "sparkrun_deadbeefdeadbeef_aabbccddeeff" in joined
    assert "qwen3-1.7b-vllm" in joined
    assert "(local process)" in joined or "solo" in joined


def test_render_node_stats_shows_without_telemetry():
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_tui import _render_node_stats

    # No telemetry sample (e.g. a provider-less substrate) still renders node stats.
    out = _render_node_stats("h1", _activity_with_local_job(None))
    assert "telemetry unavailable" in out
    assert "h1" in out


def test_container_count_counts_across_workloads():
    pytest.importorskip("textual")
    from sparkrun.cli._monitor_tui import _container_count

    assert _container_count(_activity_with_local_job()) == 1
    assert _container_count(HostActivity(host="h1")) == 0


# --------------------------------------------------------------------------
# Canonical serialization (shared by CLI --json and desktop SSE)
# --------------------------------------------------------------------------


def test_serialize_activity_shape():
    from sparkrun.core.monitoring import serialize_activity

    row = serialize_activity(_activity_with_local_job(MonitorSample(hostname="h1", gpu_util_pct="5")))
    assert row["host"] == "h1"
    assert row["sample"]["gpu_util_pct"] == "5"
    assert row["used_slots"] == 1
    wl = row["workloads"][0]
    assert wl["workload_id"] == "sparkrun_deadbeefdeadbeef_aabbccddeeff"
    assert wl["recipe"] == "qwen3-1.7b-vllm"
    assert wl["containers"][0]["image"] == "(local process)"


def test_serialize_activity_no_telemetry():
    from sparkrun.core.monitoring import serialize_activity

    row = serialize_activity(HostActivity(host="h1", status_error="unreachable"))
    assert row["sample"] is None
    assert row["error"] == "unreachable"
    assert row["workloads"] == []


def test_serialize_frame_is_list_of_rows():
    from sparkrun.core.monitoring import MonitorFrame, serialize_frame

    frame = MonitorFrame(hosts=(_activity_with_local_job(), HostActivity(host="h2")), queried_at=1.0)
    rows = serialize_frame(frame)
    assert [r["host"] for r in rows] == ["h1", "h2"]


def test_format_activity_table_jobs_from_occupancy():
    from sparkrun.core.monitoring import MonitorFrame
    from sparkrun.utils.cli_formatters import format_activity_table

    frame = MonitorFrame(hosts=(_activity_with_local_job(MonitorSample(hostname="h1", cpu_usage_pct="9")),))
    out = format_activity_table(frame, ["h1"])
    assert "Jobs" in out  # header
    # The local workload's container is counted even though telemetry is docker-agnostic.
    assert "\nh1" in out and " 1" in out  # jobs=1 column
    # A None frame renders a connecting placeholder without crashing.
    assert "connecting" in format_activity_table(None, ["h1"])


# --------------------------------------------------------------------------
# ClusterMonitorApp Detail Panel Pilot Tests
# --------------------------------------------------------------------------


class _FakeLiveMonitorSession:
    def __init__(self, frame: MonitorFrame) -> None:
        self._frame = frame
        self.closed = False

    def frame(self) -> MonitorFrame:
        return self._frame

    def set_frame(self, frame: MonitorFrame) -> None:
        self._frame = frame

    def close(self) -> None:
        self.closed = True


class _DummyLogTarget:
    def __init__(self, **kwargs):
        self.sources = [LogSource(host="h1", container="c1", role="head", rank=0)]
        self.windowing_supported = True
        self.follow_cancel = None

    def follow(self, tail=200, cancel=None):
        self.follow_cancel = cancel
        yield LogLine(host="h1", container="c1", text="line from c1")

    def page(self, *args, **kwargs):
        return []

    def dump(self, *args, **kwargs):
        return 1


def test_cluster_monitor_app_detail_panel_and_node_selection(tmp_path):
    pytest.importorskip("textual")
    import asyncio
    from textual.widgets import DataTable, Static
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import WorkloadList

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(
            ContainerDetail(name="c1", role="head", status="Up", image="img"),
            ContainerDetail(name="c2", role="worker", status="Up", image="img"),
        ),
    )
    act1 = HostActivity(host="h1", telemetry=MonitorSample(hostname="h1", cpu_usage_pct="10"), workloads=(w1,))
    act2 = HostActivity(host="h2", telemetry=MonitorSample(hostname="h2", cpu_usage_pct="20"), workloads=())
    frame = MonitorFrame(hosts=(act1, act2))
    session = _FakeLiveMonitorSession(frame)

    app = ClusterMonitorApp(session, ["h1", "h2"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)

            # 1. Detail panel split: #node-stats and #workload-list
            stats = app.query_one("#node-stats", Static)
            wl = app.query_one("#workload-list", WorkloadList)
            assert stats is not None
            assert wl is not None

            # Initially h1 selected, WorkloadList is concealed (rule 0b)
            assert "h1" in str(stats.content)
            assert not wl.revealed
            assert wl.has_class("-concealed")
            assert app._pinned_host is None

            # 2. Moving table cursor to h2 updates detail panel
            table = app.query_one("#host-table", DataTable)
            table.move_cursor(row=1)
            await pilot.pause(0.05)
            assert "h2" in str(stats.content)
            assert "No sparkrun workloads running" in wl._rows[0].text

            # 3. Enter on empty host h2: notifies and keeps focus on table
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert app._pinned_host is None
            assert table.has_focus

            # 4. Move cursor back to h1 and enter -> pins host, focuses list, reveals highlight
            table.move_cursor(row=0)
            await pilot.pause(0.05)
            assert "h1" in str(stats.content)

            await pilot.press("enter")
            await pilot.pause(0.05)
            assert app._pinned_host == "h1"
            assert wl.has_focus
            assert wl.revealed is True
            assert not wl.has_class("-concealed")
            assert wl.highlighted == 1  # first container c1

            # 5. Frame update refreshes meters while pinned and preserves highlight
            act1_updated = HostActivity(host="h1", telemetry=MonitorSample(hostname="h1", cpu_usage_pct="95"), workloads=(w1,))
            session.set_frame(MonitorFrame(hosts=(act1_updated, act2)))
            app._refresh()
            await pilot.pause(0.05)
            assert "95.0%" in str(stats.content)
            assert wl.highlighted == 1

            # 6. Container disappears: highlight moves to nearest selectable row
            w1_disappeared = RunningWorkload(
                cluster_id="cid1",
                containers=(ContainerDetail(name="c2", role="worker", status="Up", image="img"),),
            )
            act1_disappeared = HostActivity(
                host="h1", telemetry=MonitorSample(hostname="h1", cpu_usage_pct="95"), workloads=(w1_disappeared,)
            )
            session.set_frame(MonitorFrame(hosts=(act1_disappeared, act2)))
            app._refresh()
            await pilot.pause(0.05)
            assert wl.highlighted == 1
            assert wl._rows[wl.highlighted].container == "c2"

    asyncio.run(_run())


def test_cluster_monitor_app_open_and_close_log_pane(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from textual.widgets import DataTable, Static
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import WorkloadList, LogPane

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )
    act1 = HostActivity(host="h1", telemetry=MonitorSample(hostname="h1", cpu_usage_pct="10"), workloads=(w1,))
    frame = MonitorFrame(hosts=(act1,))
    session = _FakeLiveMonitorSession(frame)

    dummy_target = _DummyLogTarget()
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: dummy_target)

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)
            table = app.query_one("#host-table", DataTable)
            wl = app.query_one("#workload-list", WorkloadList)
            stats = app.query_one("#node-stats", Static)

            # Select h1
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert wl.has_focus

            # Enter on c1 opens log pane
            await pilot.press("enter")
            await pilot.pause(0.1)

            # WorkloadList is hidden, LogPane is mounted
            assert wl.display is False
            log_pane = app.query_one("#log-pane", LogPane)
            assert log_pane is not None
            assert stats is not None and stats.display is True  # node stats stay visible
            assert app.has_class("-logs-open") or app.screen.has_class("-logs-open")

            # Enter inside log pane does not trigger action_select_node or break UI
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert len(app.query("#log-pane")) == 1
            assert wl.display is False
            assert not wl.has_focus

            # Node stats keep updating while log pane is open
            act1_updated = HostActivity(host="h1", telemetry=MonitorSample(hostname="h1", cpu_usage_pct="88"), workloads=(w1,))
            session.set_frame(MonitorFrame(hosts=(act1_updated,)))
            app._refresh()
            await pilot.pause(0.05)
            assert "88.0%" in str(stats.content)

            # Escape in log pane closes pane and sets its cancel event
            await pilot.press("escape")
            await pilot.pause(0.1)

            assert dummy_target.follow_cancel is not None
            assert dummy_target.follow_cancel.is_set()
            assert len(app.query("#log-pane")) == 0
            assert wl.display is True
            assert wl.has_focus
            assert wl.highlighted == 1  # c1 remains highlighted
            assert not app.has_class("-logs-open")
            assert not app.screen.has_class("-logs-open")

            # Escape in workload list unpins and returns to host table
            await pilot.press("escape")
            await pilot.pause(0.05)
            assert app._pinned_host is None
            assert not wl.revealed
            assert table.has_focus

    asyncio.run(_run())


def test_cluster_monitor_app_table_refocus_closes_log_pane(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from textual.widgets import DataTable
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import WorkloadList, LogPane

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )
    act1 = HostActivity(host="h1", workloads=(w1,))
    act2 = HostActivity(host="h2", workloads=())
    frame = MonitorFrame(hosts=(act1, act2))
    session = _FakeLiveMonitorSession(frame)

    dummy_target = _DummyLogTarget()
    monkeypatch.setattr("sparkrun.cli._monitor_logs.open_log_target", lambda *a, **kw: dummy_target)

    app = ClusterMonitorApp(session, ["h1", "h2"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause(0.05)
            table = app.query_one("#host-table", DataTable)
            wl = app.query_one("#workload-list", WorkloadList)

            # Pin h1 and open log pane
            await pilot.press("enter")  # select h1
            await pilot.pause(0.05)
            await pilot.press("enter")  # open c1 log pane
            await pilot.pause(0.1)

            log_pane = app.query_one("#log-pane", LogPane)
            assert log_pane is not None

            # User clicks table on h2 row
            await pilot.click("#host-table", offset=(3, 2))
            await pilot.pause(0.1)

            # Log pane is closed and cancelled
            assert dummy_target.follow_cancel is not None
            assert dummy_target.follow_cancel.is_set()
            assert len(app.query("#log-pane")) == 0
            assert app._pinned_host is None
            assert not wl.revealed
            assert table.has_focus

    asyncio.run(_run())


def test_cluster_monitor_app_dump_modal(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import asyncio
    from sparkrun.cli._monitor_tui import ClusterMonitorApp
    from sparkrun.cli._monitor_logs import DumpLogScreen, LogScope

    w1 = RunningWorkload(
        cluster_id="cid1",
        containers=(ContainerDetail(name="c1", role="head", status="Up", image="img"),),
    )
    act1 = HostActivity(host="h1", workloads=(w1,))
    frame = MonitorFrame(hosts=(act1,))
    session = _FakeLiveMonitorSession(frame)

    app = ClusterMonitorApp(session, ["h1"], interval=1, cache_dir=str(tmp_path))

    async def _run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.05)

            # Select h1
            await pilot.press("enter")
            await pilot.pause(0.05)

            # 'd' in workload list pushes DumpLogScreen
            await pilot.press("d")
            await pilot.pause(0.1)

            assert isinstance(app.screen, DumpLogScreen)
            assert app.screen.host == "h1"
            assert app.screen.container == "c1"
            assert app.screen.chosen_scope == LogScope.CONTAINER

            # Cancel modal
            await pilot.press("escape")
            await pilot.pause(0.05)
            assert not isinstance(app.screen, DumpLogScreen)

    asyncio.run(_run())
