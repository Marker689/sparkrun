"""api.setup CX7 planning/apply — the decisions `setup cx7` and the wizard render.

Detection, neighbor discovery and the apply are mocked at the orchestration
boundary; the planner itself runs for real.
"""

from __future__ import annotations

from unittest import mock

import pytest

from sparkrun.api.setup import (
    CX7_EVENT_DETECTED,
    CX7_EVENT_DETECTED_TOPOLOGY,
    CX7_EVENT_DETECTING_TOPOLOGY,
    CX7_EVENT_INTERFACES,
    CX7_EVENT_RING_DETECTING,
    CX7_EVENT_SAVED_TOPOLOGY,
    CX7_EVENT_SUBNETS,
    CX7_EVENT_TOPOLOGY,
    Cx7PlanError,
    apply_cx7,
    plan_cx7,
    save_cx7_topology,
)
from sparkrun.orchestration.networking import CX7HostDetection, CX7Interface, CX7Topology, CX7TopologyResult
from sparkrun.orchestration.ssh import RemoteResult

_FOUR_PORT = ("enp1s0f0np0", "enp1s0f1np1", "enP2p1s0f0np0", "enP2p1s0f1np1")


def _det(host: str, names=_FOUR_PORT, detected=True) -> CX7HostDetection:
    ifaces = [CX7Interface(name=n, ip="", prefix=0, subnet="", mtu=9000, state="up", hca="roce") for n in names]
    return CX7HostDetection(
        host=host,
        interfaces=ifaces,
        mgmt_ip="10.0.0.%d" % (10 + int(host.lstrip("h"))),
        mgmt_iface="eth0",
        used_subnets={"10.0.0.0/24"},
        sudo_ok=True,
        detected=detected,
    )


def _mgr(default=None, topology=None, fabric_interfaces=None):
    mgr = mock.Mock()
    mgr.get_default.return_value = default
    mgr.get.return_value.topology = topology
    mgr.get.return_value.fabric_interfaces = fabric_interfaces
    return mgr


def _plan(dets, **kw):
    events = []
    kw.setdefault("cluster_manager", _mgr())
    with mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=dets):
        plan = plan_cx7(list(dets), on_event=lambda kind, p: events.append(kind), **kw)
    return plan, events


def test_two_hosts_auto_plans_switch_without_discovery():
    with mock.patch("sparkrun.orchestration.networking.detect_topology") as discover:
        plan, events = _plan({h: _det(h) for h in ("h1", "h2")})
    discover.assert_not_called()
    assert plan.topology == CX7Topology.SWITCH
    assert plan.topology_ambiguous
    assert len(plan.subnets) == 2
    assert plan.configurable_count == 2
    assert events == [CX7_EVENT_DETECTED, CX7_EVENT_INTERFACES, CX7_EVENT_TOPOLOGY, CX7_EVENT_SUBNETS]


def test_explicit_direct_is_not_ambiguous():
    plan, _ = _plan({h: _det(h) for h in ("h1", "h2")}, topology="direct")
    assert plan.topology == CX7Topology.DIRECT
    assert not plan.topology_ambiguous


def test_three_two_port_hosts_on_auto_run_neighbor_discovery():
    dets = {h: _det(h) for h in ("h1", "h2", "h3")}
    with mock.patch("sparkrun.orchestration.networking.detect_topology", return_value=CX7TopologyResult(CX7Topology.SWITCH)):
        plan, events = _plan(dets)
    assert plan.topology_source == "detected"
    assert events[2:4] == [CX7_EVENT_DETECTING_TOPOLOGY, CX7_EVENT_DETECTED_TOPOLOGY]


def test_choose_topology_replaces_discovery():
    dets = {h: _det(h) for h in ("h1", "h2", "h3")}
    with mock.patch("sparkrun.orchestration.networking.detect_topology") as discover:
        plan, _ = _plan(dets, choose_topology=lambda: "switch")
    discover.assert_not_called()
    assert (plan.topology, plan.topology_source) == (CX7Topology.SWITCH, "chosen")


def test_chosen_ring_still_discovers_its_links():
    dets = {h: _det(h) for h in ("h1", "h2", "h3")}
    with mock.patch("sparkrun.orchestration.networking.detect_topology", return_value=CX7TopologyResult(CX7Topology.RING)) as discover:
        plan, events = _plan(dets, choose_topology=lambda: "ring")
    discover.assert_called_once()
    assert plan.topology == CX7Topology.RING
    assert CX7_EVENT_RING_DETECTING in events


def test_saved_topology_wins_only_when_asked():
    dets = {h: _det(h) for h in ("h1", "h2", "h3")}
    mgr = _mgr(topology="switch")
    with mock.patch("sparkrun.orchestration.networking.detect_topology") as discover:
        plan, events = _plan(dets, cluster_name="lab", cluster_manager=mgr, use_saved_topology=True)
    discover.assert_not_called()
    assert (plan.topology, plan.topology_source) == (CX7Topology.SWITCH, "saved")
    assert CX7_EVENT_SAVED_TOPOLOGY in events

    with mock.patch("sparkrun.orchestration.networking.detect_topology", return_value=CX7TopologyResult(CX7Topology.SWITCH)) as discover:
        plan, _ = _plan(dets, cluster_name="lab", cluster_manager=mgr)
    discover.assert_called_once()


def test_unknown_saved_topology_is_ignored():
    plan, _ = _plan(
        {h: _det(h) for h in ("h1", "h2")}, cluster_name="lab", cluster_manager=_mgr(topology="unknown"), use_saved_topology=True
    )
    assert plan.topology_source == "default"


def test_explicit_ring_needs_three_hosts():
    with pytest.raises(Cx7PlanError, match="exactly 3 hosts with CX7, found 2"):
        _plan({h: _det(h) for h in ("h1", "h2")}, topology="ring")


def test_explicit_ring_honors_interface_filter():
    dets = {h: _det(h) for h in ("h1", "h2", "h3")}
    with pytest.raises(Cx7PlanError, match="ring topology requires"):
        _plan(dets, topology="ring", interfaces=["*np1"])


def test_no_cx7_anywhere_raises_after_reporting_detection():
    events = []
    with mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value={"h1": _det("h1", detected=False)}):
        with pytest.raises(Cx7PlanError, match="No CX7 interfaces detected on any host"):
            plan_cx7(["h1"], cluster_manager=_mgr(), on_event=lambda kind, p: events.append((kind, p.missing_cx7)))
    assert events == [(CX7_EVENT_DETECTED, ["h1"])]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"subnet1": "192.168.11.0/24"}, "must be specified together"),
        ({"interfaces": ["*np1"], "port": 1}, "cannot be used together"),
        ({"topology": "mesh"}, "unknown topology"),
    ],
)
def test_invalid_options_raise_before_detection(kwargs, message):
    with mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts") as detect:
        with pytest.raises(Cx7PlanError, match=message):
            plan_cx7(["h1"], cluster_manager=_mgr(), **kwargs)
    detect.assert_not_called()


def test_port_selects_a_pair_present_on_every_host():
    plan, _ = _plan({"h1": _det("h1"), "h2": _det("h2")}, port=1)
    assert plan.interfaces == ["*np1"]
    assert plan.interfaces_explicit

    with pytest.raises(Cx7PlanError, match="does not select 2 interfaces on: h2"):
        _plan({"h1": _det("h1"), "h2": _det("h2", names=("enp1s0f0np0", "enP2p1s0f0np0"))}, port=1)


def test_port_out_of_range():
    with pytest.raises(Cx7PlanError, match="only 2 physical port group"):
        _plan({"h1": _det("h1")}, port=2)


def test_saved_interfaces_come_from_the_default_cluster():
    plan, _ = _plan({"h1": _det("h1")}, cluster_manager=_mgr(default="two", fabric_interfaces=["*np1"]))
    assert plan.cluster_name == "two"
    assert plan.interfaces == ["*np1"]
    assert not plan.interfaces_explicit


def _applied(plan, results, mgr):
    with mock.patch("sparkrun.orchestration.networking.apply_cx7_plan", return_value=results):
        return apply_cx7(plan, cluster_manager=mgr)


def test_partial_apply_does_not_save_topology():
    """Issue #311: 1 configured, 1 failed is not a configured fabric."""
    mgr = _mgr()
    plan, _ = _plan({h: _det(h) for h in ("h1", "h2")}, cluster_name="lab", cluster_manager=mgr)
    result = _applied(plan, [RemoteResult("h2", 255, "", "unreachable"), RemoteResult("h1", 0, "", "")], mgr)

    assert (result.configured, result.failed) == (1, 1)
    assert [r.host for r in result.results] == ["h1", "h2"]  # plan order, not apply order
    assert result.topology_save.refused and not result.topology_save.saved
    mgr.update.assert_not_called()


def test_full_apply_saves_topology_and_explicit_filter():
    mgr = _mgr()
    plan, _ = _plan({h: _det(h) for h in ("h1", "h2")}, cluster_name="lab", cluster_manager=mgr, interfaces=["*np1"])
    result = _applied(plan, [RemoteResult("h1", 0, "", ""), RemoteResult("h2", 0, "", "")], mgr)

    assert result.topology_save.saved
    assert result.interfaces_save is not None and result.interfaces_save.saved
    mgr.update.assert_any_call("lab", topology="switch")
    mgr.update.assert_any_call("lab", fabric_interfaces=["*np1"])


def test_saved_filter_is_not_written_back():
    mgr = _mgr(fabric_interfaces=["*np1"])
    plan, _ = _plan({h: _det(h) for h in ("h1", "h2")}, cluster_name="lab", cluster_manager=mgr)
    result = _applied(plan, [RemoteResult("h1", 0, "", ""), RemoteResult("h2", 0, "", "")], mgr)
    assert result.interfaces_save is None


def test_save_topology_reports_write_errors_and_skips_dry_run():
    mgr = _mgr()
    mgr.update.side_effect = RuntimeError("disk full")
    plan, _ = _plan({h: _det(h) for h in ("h1", "h2")}, cluster_name="lab", cluster_manager=mgr)

    assert save_cx7_topology(plan, cluster_manager=mgr).error == "disk full"
    mgr.update.reset_mock()
    assert not save_cx7_topology(plan, cluster_manager=mgr, dry_run=True).saved
    mgr.update.assert_not_called()


def test_save_topology_needs_a_cluster():
    plan, _ = _plan({h: _det(h) for h in ("h1", "h2")})
    save = save_cx7_topology(plan, cluster_manager=_mgr(), failed=2)
    assert save.cluster_name is None and not save.refused and not save.saved
