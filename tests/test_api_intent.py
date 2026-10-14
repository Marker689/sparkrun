"""``sparkrun.api.find_running_intent`` — placement-independent "is it up?".

``--ensure`` used to derive a cluster_id from ``(recipe, hosts)`` and look for
*that*.  A cluster_id encodes placement as well as intent, so the lookup only
ever matched a job the greedy scheduler had put on exactly the host set being
asked about.  Under an ``occupancy-*`` scheduler (random placement token) it
matched nothing, ever — ``--ensure`` launched a duplicate every time.

These tests pin the replacement property: the answer depends on the intent and
not on where the deployment landed or which scheduler placed it.
"""

from __future__ import annotations

from unittest import mock

import pytest

import sparkrun.api as api
from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload, workload_matches_intent

_INTENT = "0123456789abcdef"
_HOSTS = ["h1", "h2", "h3"]


def _status(*entries: tuple[str, tuple[RunningWorkload, ...]]) -> ClusterStatus:
    return ClusterStatus(
        hosts=tuple(HostOccupancy(host=h, workloads=w, used_slots=len(w)) for h, w in entries),
        executor="docker",
    )


def _workload(cluster_id: str, *, intent_id: str | None = None, **kw) -> RunningWorkload:
    return RunningWorkload(cluster_id=cluster_id, intent_id=intent_id, **kw)


def test_matches_regardless_of_placement_token():
    """A random (occupancy-scheduler) token doesn't hide the deployment."""
    snap = _status(("h2", (_workload("sparkrun_%s_ffff00001111" % _INTENT, intent_id=_INTENT),)))

    match = api.find_running_intent(_INTENT, _HOSTS, status=snap)

    assert match is not None
    assert match.workload_id == "sparkrun_%s_ffff00001111" % _INTENT
    assert match.hosts == ("h2",)


def test_matches_on_cluster_id_prefix_without_a_label():
    """Falls back to the cluster_id's intent prefix when no label was recovered."""
    snap = _status(("h1", (_workload("sparkrun_%s_abcd12340000" % _INTENT, intent_id=None),)))

    assert api.find_running_intent(_INTENT, _HOSTS, status=snap) is not None


def test_ignores_other_intents():
    """A different workload on the cluster is not this one."""
    snap = _status(("h1", (_workload("sparkrun_fedcba9876543210_abcd12340000", intent_id="fedcba9876543210"),)))

    assert api.find_running_intent(_INTENT, _HOSTS, status=snap) is None


def test_ignores_unidentifiable_containers():
    """A non-canonical container name is not evidence this intent is running."""
    snap = _status(("h1", (_workload("some-unrelated-container", intent_id=None),)))

    assert api.find_running_intent(_INTENT, _HOSTS, status=snap) is None


def test_collects_every_host_of_a_multi_node_deployment():
    w = _workload("sparkrun_%s_aaaa11112222" % _INTENT, intent_id=_INTENT, recipe_name="r", runtime_name="sglang")
    snap = _status(("h1", (w,)), ("h2", ()), ("h3", (w,)))

    match = api.find_running_intent(_INTENT, _HOSTS, status=snap)

    assert match.hosts == ("h1", "h3")
    assert match.recipe == "r"
    assert match.runtime == "sglang"


def test_reports_duplicate_deployments_of_one_intent():
    """Two deployments of one intent is a state the launch path avoids — say so.

    The widest one wins so the report describes the primary deployment, but
    the others are surfaced rather than silently dropped.
    """
    big = _workload("sparkrun_%s_bbbb22223333" % _INTENT, intent_id=_INTENT)
    small = _workload("sparkrun_%s_cccc33334444" % _INTENT, intent_id=_INTENT)
    snap = _status(("h1", (big,)), ("h2", (big,)), ("h3", (small,)))

    match = api.find_running_intent(_INTENT, _HOSTS, status=snap)

    assert match.workload_id == "sparkrun_%s_bbbb22223333" % _INTENT
    assert match.other_cluster_ids == ("sparkrun_%s_cccc33334444" % _INTENT,)


def test_empty_inputs_are_not_a_match():
    assert api.find_running_intent("", _HOSTS, status=_status()) is None
    assert api.find_running_intent(_INTENT, [], status=_status()) is None


def test_unreachable_cluster_counts_as_not_running():
    """Better to launch than to refuse because a probe failed."""

    def _boom(hosts, **kwargs):
        raise OSError("connection refused")

    with mock.patch.object(api, "status", _boom):
        assert api.find_running_intent(_INTENT, _HOSTS) is None


@pytest.mark.parametrize(
    "workload,expected",
    [
        (_workload("sparkrun_%s_111122223333" % _INTENT, intent_id=_INTENT), True),
        (_workload("sparkrun_%s_111122223333" % _INTENT), True),
        (_workload("sparkrun_ffffffffffffffff_111122223333", intent_id="ffffffffffffffff"), False),
        (_workload("not-a-sparkrun-name"), False),
    ],
)
def test_workload_matches_intent_predicate(workload, expected):
    """The shared predicate placement-exclusion and --ensure both depend on.

    They must agree: placement subtracts its own intent's workloads while
    ``--ensure`` looks for them.  A divergence would make ``--ensure`` decline
    to launch something placement had already decided to replace.
    """
    assert workload_matches_intent(workload, _INTENT) is expected


@pytest.mark.parametrize("prefetched", [False, True])
def test_match_respects_requested_hosts_and_order(monkeypatch, prefetched):
    inside = _workload("sparkrun_%s_bbbb22223333" % _INTENT, intent_id=_INTENT)
    outside = _workload("sparkrun_%s_aaaa11112222" % _INTENT, intent_id=_INTENT)
    # Outside deployment is wider globally, but has only one in-scope host.
    snap = _status(("h2", (inside, outside)), ("other1", (outside,)), ("h1", (inside,)), ("other2", (outside,)))
    monkeypatch.setattr(api, "status", lambda *a, **kw: snap)
    match = api.find_running_intent(_INTENT, ["h1", "missing", "h2", "h1"], status=snap if prefetched else None)
    assert match is not None
    assert match.workload_id == inside.cluster_id
    assert match.hosts == ("h1", "h2")
    assert match.other_cluster_ids == (outside.cluster_id,)


@pytest.mark.parametrize("prefetched", [False, True])
def test_match_outside_requested_hosts_is_ignored(monkeypatch, prefetched):
    snap = _status(("h1", ()), ("h2", (_workload("sparkrun_%s_aaaa11112222" % _INTENT, intent_id=_INTENT),)))
    monkeypatch.setattr(api, "status", lambda *a, **kw: snap)
    assert api.find_running_intent(_INTENT, ["h1"], status=snap if prefetched else None) is None


def test_partial_snapshot_retains_positive_match_in_requested_order():
    workload = _workload("sparkrun_%s_aaaa11112222" % _INTENT, intent_id=_INTENT)
    snap = ClusterStatus(hosts=(HostOccupancy(host="h2", workloads=(workload,)),), executor="docker", errors={"h1": "offline"})
    match = api.find_running_intent(_INTENT, ["h1", "h2"], status=snap)
    assert match is not None
    assert match.hosts == ("h2",)
