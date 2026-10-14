"""CX7 fabric configuration — console-free core of ``setup cx7`` and the wizard.

``setup cx7`` and the wizard's CX7 phase both detect the fabric, pick a
topology, choose subnets, plan addresses and apply them. They used to do it in
two copies, each with its own topology choice, saved-cluster handling and
save-after-apply rule. Those decisions live here; the commands only render and
prompt. Host resolution, sudo credentials and confirmation stay with the
caller (the setup-command convention): these take an already-resolved host
list and ``ssh_kwargs``.

Interaction is passed in rather than performed: :func:`plan_cx7` reports its
progress through ``on_event`` (so a renderer can print each step as it
happens, before a slow neighbor discovery) and asks ``choose_topology`` when a
choice is the user's to make.

Two rules here are load-bearing:

- **A partial apply is not a topology** (issue #311). :func:`apply_cx7` and
  :func:`save_cx7_topology` refuse to record the cluster's topology when any
  host failed, since everything that reads the cluster afterwards would treat
  the fabric as configured.
- **A configured cluster keeps its saved topology** when the caller asks for
  it (``use_saved_topology``): re-detecting on every clean rerun could save a
  misread over a deliberate ``setup cx7 --topology`` choice.

The apply order (the control node's own host last, then a bounded wait for
routes) is :func:`sparkrun.orchestration.networking.apply_cx7_plan`'s.

.. warning::
   **Experimental:** this module's functions and result types may change in a
   minor release.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from sparkrun.api.setup._errors import SetupFailed

if TYPE_CHECKING:
    import ipaddress

    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.orchestration.networking import CX7ClusterPlan, CX7HostDetection, CX7HostPlan, CX7Topology, CX7TopologyResult
    from sparkrun.orchestration.ssh import RemoteResult
    from sparkrun.orchestration.sudo import SudoPasswords

logger = logging.getLogger(__name__)

#: Accepted values for ``plan_cx7(topology=...)``.
CX7_TOPOLOGY_CHOICES = ("auto", "direct", "switch", "ring")

#: ``plan_cx7`` progress events, in the order they can occur. Each is passed to
#: ``on_event`` together with the plan as filled in so far.
CX7_EVENT_DETECTED = "detected"  # detections known; ``missing_cx7`` may be non-empty
CX7_EVENT_INTERFACES = "interfaces"  # interface filter resolved (may be None)
CX7_EVENT_SAVED_TOPOLOGY = "saved-topology"  # the cluster's saved topology is used
CX7_EVENT_DETECTING_TOPOLOGY = "detecting-topology"  # neighbor discovery is starting
CX7_EVENT_DETECTED_TOPOLOGY = "detected-topology"  # neighbor discovery classified the topology
CX7_EVENT_RING_DETECTING = "ring-detecting"  # ring links are being discovered
CX7_EVENT_TOPOLOGY = "topology"  # topology decided
CX7_EVENT_SUBNETS = "subnets"  # subnets selected

#: How :attr:`Cx7Plan.topology` was decided.
TOPOLOGY_DEFAULT = "default"
TOPOLOGY_REQUESTED = "requested"
TOPOLOGY_CHOSEN = "chosen"
TOPOLOGY_SAVED = "saved"
TOPOLOGY_DETECTED = "detected"


class Cx7PlanError(SetupFailed):
    """The CX7 configuration could not be planned (bad options, no fabric, no subnets)."""


@dataclass
class Cx7Plan:
    """A decided CX7 configuration: what was detected, chosen and planned.

    Filled in stages by :func:`plan_cx7`; ``on_event`` callbacks see it partly
    built (each event names the stage that just completed).
    """

    hosts: list[str]
    detections: dict[str, CX7HostDetection] = field(default_factory=dict)
    #: The cluster whose saved settings were read and where results are saved:
    #: the named cluster, else the default one, else ``None``.
    cluster_name: str | None = None
    #: Effective interface filter (names or globs), ``None`` for automatic.
    interfaces: list[str] | None = None
    #: The filter was chosen for this run (``interfaces`` / ``port``) rather
    #: than read from the cluster, so applying the plan saves it.
    interfaces_explicit: bool = False
    requested_topology: str = "auto"
    topology: CX7Topology | None = None
    topology_source: str = TOPOLOGY_DEFAULT
    topology_result: CX7TopologyResult | None = None
    subnets: list[ipaddress.IPv4Network] = field(default_factory=list)
    mtu: int = 9000
    force: bool = False
    cluster_plan: CX7ClusterPlan | None = None

    @property
    def hosts_with_cx7(self) -> dict[str, CX7HostDetection]:
        return {h: d for h, d in self.detections.items() if d.detected}

    @property
    def missing_cx7(self) -> list[str]:
        """Hosts on which no CX7 interface was detected."""
        return [h for h, d in self.detections.items() if not d.detected]

    @property
    def topology_ambiguous(self) -> bool:
        """Auto on two hosts cannot tell a direct cable from a switch at L2."""
        from sparkrun.orchestration.networking import CX7Topology

        return self.topology == CX7Topology.SWITCH and self.requested_topology == "auto" and len(self.hosts_with_cx7) <= 2

    @property
    def host_plans(self) -> list[CX7HostPlan]:
        return list(self.cluster_plan.host_plans) if self.cluster_plan else []

    @property
    def plan_failed(self) -> bool:
        """The planner refused outright (errors and no host plans)."""
        return bool(self.cluster_plan and self.cluster_plan.errors and not self.cluster_plan.host_plans)

    @property
    def all_valid(self) -> bool:
        return bool(self.cluster_plan and self.cluster_plan.all_valid)

    @property
    def configurable_count(self) -> int:
        """Hosts that need a change and have the two addresses to apply it."""
        return sum(1 for hp in self.host_plans if hp.needs_change and len(hp.assignments) >= 2)

    @property
    def changing_count(self) -> int:
        """Hosts whose plan differs from what they run (configurable or not)."""
        return sum(1 for hp in self.host_plans if hp.needs_change)

    @property
    def already_valid_count(self) -> int:
        return sum(1 for hp in self.host_plans if not hp.needs_change)

    @property
    def blocked_count(self) -> int:
        """Hosts that need a change the planner could not assign addresses for."""
        return sum(1 for hp in self.host_plans if hp.needs_change and len(hp.assignments) < 2)

    @property
    def assigned_ips(self) -> list[str]:
        return [a.ip for hp in self.host_plans for a in hp.assignments if a.ip]


@dataclass
class Cx7TopologySave:
    """Whether the plan's topology was recorded on its cluster."""

    cluster_name: str | None
    topology: str | None
    saved: bool = False
    #: Hosts that failed to apply; non-zero means the save was refused (#311).
    failed_hosts: int = 0
    error: str | None = None

    @property
    def refused(self) -> bool:
        return bool(self.cluster_name) and self.failed_hosts > 0


@dataclass
class Cx7InterfacesSave:
    """Whether an explicitly chosen interface filter was recorded on the cluster."""

    cluster_name: str
    interfaces: list[str]
    saved: bool = False
    error: str | None = None


@dataclass
class Cx7ApplyResult:
    """Outcome of :func:`apply_cx7`."""

    #: One result per configured host, in plan order.
    results: list[RemoteResult] = field(default_factory=list)
    topology_save: Cx7TopologySave | None = None
    #: ``None`` unless the plan carried an explicitly chosen filter and a cluster.
    interfaces_save: Cx7InterfacesSave | None = None

    @property
    def configured(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def failures(self) -> list[RemoteResult]:
        return [r for r in self.results if not r.success]


def validate_cx7_options(*, interfaces: Any = None, port: int | str | None = None, subnet1: str | None = None, subnet2: str | None = None):
    """Raise :class:`Cx7PlanError` for option combinations that cannot be planned.

    Cheap and side-effect free, so a caller can fail before resolving hosts.
    """
    if (subnet1 is None) != (subnet2 is None):
        raise Cx7PlanError("--subnet1 and --subnet2 must be specified together.")
    if interfaces and port is not None:
        raise Cx7PlanError("--interfaces and --port cannot be used together.")


def _default_cluster_manager() -> ClusterManager:
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import get_config_root

    return ClusterManager(get_config_root())


def _effective_cluster(cluster_name: str | None, cluster_manager) -> str | None:
    """The named cluster, else the default one (results are saved there)."""
    if cluster_name:
        return cluster_name
    if not cluster_manager:
        return None
    try:
        return cluster_manager.get_default()
    except Exception:
        logger.debug("Could not read the default cluster", exc_info=True)
        return None


def _cluster_attr(cluster_manager, cluster_name: str | None, attr: str):
    if not (cluster_manager and cluster_name):
        return None
    try:
        return getattr(cluster_manager.get(cluster_name), attr)
    except Exception:
        logger.debug("Could not read %s of cluster %s", attr, cluster_name, exc_info=True)
        return None


def _port_interfaces(hosts_with_cx7: dict[str, CX7HostDetection], port: int) -> list[str]:
    """``--port N`` → the ``*npM`` glob of the Nth physical port pair, valid on every host."""
    from sparkrun.orchestration.networking import _group_interfaces_by_port, filter_cx7_interfaces

    rep = next(iter(hosts_with_cx7.values()))
    groups = _group_interfaces_by_port(rep.interfaces)
    if port >= len(groups):
        raise Cx7PlanError("--port %d requested but only %d physical port group(s) detected." % (port, len(groups)))
    m = re.search(r"np(\d+)$", groups[port][0].name)
    if not m:
        raise Cx7PlanError("could not derive a port glob from interface '%s'; use --interfaces instead." % groups[port][0].name)
    selected = ["*np%s" % m.group(1)]
    # The glob is derived from one host; verify it yields a pair on every host
    # so a heterogeneous cluster fails here rather than deep in planning.
    bad_hosts = [h for h, d in hosts_with_cx7.items() if len(filter_cx7_interfaces(d.interfaces, selected)) < 2]
    if bad_hosts:
        raise Cx7PlanError(
            "--port %d (%s) does not select 2 interfaces on: %s. Use --interfaces with explicit names."
            % (port, selected[0], ", ".join(bad_hosts))
        )
    return selected


def plan_cx7(
    hosts: list[str],
    *,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    cluster_name: str | None = None,
    cluster_manager: ClusterManager | None = None,
    topology: str = "auto",
    use_saved_topology: bool = False,
    choose_topology: Callable[[], str] | None = None,
    interfaces: list[str] | None = None,
    port: int | None = None,
    subnet1: str | None = None,
    subnet2: str | None = None,
    mtu: int | None = None,
    force: bool = False,
    on_event: Callable[[str, Cx7Plan], None] | None = None,
) -> Cx7Plan:
    """Detect CX7 on *hosts* and decide the configuration to apply. Read-only.

    EXPERIMENTAL.

    Topology, first hit wins: the cluster's saved topology when
    *use_saved_topology*; an explicit *topology* (``ring`` is validated: exactly
    three hosts with two physical ports each); for ``auto`` on three hosts with
    two ports each, *choose_topology* (``"auto"`` / ``"switch"`` / ``"ring"``)
    when given, else neighbor discovery; otherwise ``switch``.

    Interface filter: *interfaces*, else the ``*npN`` pair of physical *port*
    (checked on every host), else the cluster's saved ``fabric_interfaces``.
    The cluster is *cluster_name*, else the default cluster.

    Raises:
        Cx7PlanError: invalid options, no CX7 on any host, an impossible ring,
            or no usable subnets.
    """
    from sparkrun.orchestration import networking as net

    validate_cx7_options(interfaces=interfaces, port=port, subnet1=subnet1, subnet2=subnet2)
    if topology not in CX7_TOPOLOGY_CHOICES:
        raise Cx7PlanError("unknown topology %r (expected one of: %s)" % (topology, ", ".join(CX7_TOPOLOGY_CHOICES)))
    if cluster_manager is None:
        cluster_manager = _default_cluster_manager()

    def emit(kind: str) -> None:
        if on_event is not None:
            on_event(kind, plan)

    plan = Cx7Plan(
        hosts=list(hosts),
        requested_topology=topology,
        mtu=net.DEFAULT_MTU if mtu is None else mtu,
        force=force,
    )
    plan.detections = net.detect_cx7_for_hosts(list(hosts), ssh_kwargs=ssh_kwargs, dry_run=dry_run)
    emit(CX7_EVENT_DETECTED)
    hosts_with_cx7 = plan.hosts_with_cx7
    if not hosts_with_cx7:
        raise Cx7PlanError("No CX7 interfaces detected on any host.")

    plan.cluster_name = _effective_cluster(cluster_name, cluster_manager)

    if interfaces:
        plan.interfaces = list(interfaces)
        plan.interfaces_explicit = True
    elif port is not None:
        plan.interfaces = _port_interfaces(hosts_with_cx7, int(port))
        plan.interfaces_explicit = True
    else:
        plan.interfaces = list(_cluster_attr(cluster_manager, plan.cluster_name, "fabric_interfaces") or []) or None
    emit(CX7_EVENT_INTERFACES)

    # ── Topology ──
    def discover() -> CX7TopologyResult:
        return net.detect_topology(plan.detections, list(hosts), ssh_kwargs=ssh_kwargs, dry_run=dry_run)

    ring_eligible = len(hosts_with_cx7) == 3 and all(
        len(net._group_interfaces_by_port(d.interfaces)) >= net.RING_MIN_PHYSICAL_PORTS for d in hosts_with_cx7.values()
    )
    plan.topology = net.CX7Topology.SWITCH
    saved = _cluster_attr(cluster_manager, plan.cluster_name, "topology") if use_saved_topology else None
    if saved in {t.value for t in net.CX7Topology} - {net.CX7Topology.UNKNOWN.value}:
        plan.topology = net.CX7Topology(saved)
        plan.topology_source = TOPOLOGY_SAVED
        emit(CX7_EVENT_SAVED_TOPOLOGY)
    elif topology == "ring":
        if len(hosts_with_cx7) != 3:
            raise Cx7PlanError("ring topology requires exactly 3 hosts with CX7, found %d" % len(hosts_with_cx7))
        for h, det in hosts_with_cx7.items():
            # Honor the interface filter so the pre-check matches what
            # plan_ring_cx7 will see, via the shared validator.
            err = net.ring_ports_error(h, net.filter_cx7_interfaces(det.interfaces, plan.interfaces))
            if err:
                raise Cx7PlanError(err)
        plan.topology = net.CX7Topology.RING
        plan.topology_source = TOPOLOGY_REQUESTED
    elif topology in ("direct", "switch"):
        plan.topology = net.CX7Topology(topology)
        plan.topology_source = TOPOLOGY_REQUESTED
    elif ring_eligible:
        choice = choose_topology() if choose_topology is not None else "auto"
        if choice in ("ring", "switch"):
            plan.topology = net.CX7Topology(choice)
            plan.topology_source = TOPOLOGY_CHOSEN
        else:
            emit(CX7_EVENT_DETECTING_TOPOLOGY)
            plan.topology_result = discover()
            plan.topology = plan.topology_result.topology
            plan.topology_source = TOPOLOGY_DETECTED
            emit(CX7_EVENT_DETECTED_TOPOLOGY)
    # else: two hosts (or no second port) cannot be told apart from a switch
    # at L2 without LLDP; `topology="direct"` overrides.

    if plan.topology == net.CX7Topology.RING and plan.topology_result is None:
        if dry_run:
            plan.topology_result = net.CX7TopologyResult(topology=net.CX7Topology.RING)
        else:
            emit(CX7_EVENT_RING_DETECTING)
            plan.topology_result = discover()
    emit(CX7_EVENT_TOPOLOGY)

    # ── Subnets + plan ──
    try:
        if plan.topology == net.CX7Topology.RING:
            plan.subnets = net.select_subnets_for_topology(plan.detections, plan.topology)
        else:
            plan.subnets = list(net.select_subnets(plan.detections, override1=subnet1, override2=subnet2))
    except RuntimeError as e:
        raise Cx7PlanError(str(e)) from e
    emit(CX7_EVENT_SUBNETS)

    if plan.topology == net.CX7Topology.RING:
        if plan.topology_result is None:
            raise Cx7PlanError("Ring topology requires detected link information")
        plan.cluster_plan = net.plan_ring_cx7(
            plan.detections, plan.topology_result, plan.subnets, mtu=plan.mtu, force=force, interfaces=plan.interfaces
        )
    else:
        plan.cluster_plan = net.plan_cluster_cx7(
            plan.detections, plan.subnets[0], plan.subnets[1], mtu=plan.mtu, force=force, interfaces=plan.interfaces
        )
    return plan


def save_cx7_topology(
    plan: Cx7Plan,
    *,
    cluster_manager: ClusterManager | None = None,
    failed: int = 0,
    dry_run: bool = False,
) -> Cx7TopologySave:
    """Record *plan*'s topology on its cluster, unless *failed* hosts make it partial.

    EXPERIMENTAL. Nothing is written under *dry_run*, without a cluster, or
    for an ``unknown`` topology. Never raises; a write error is reported in
    :attr:`Cx7TopologySave.error`.
    """
    from sparkrun.orchestration.networking import CX7Topology

    topo = plan.topology.value if plan.topology is not None else None
    result = Cx7TopologySave(cluster_name=plan.cluster_name, topology=topo, failed_hosts=failed)
    # A partial apply is not a topology: saving it would report the fabric
    # as configured to everything that reads the cluster afterwards.
    if result.refused or dry_run or not plan.cluster_name or plan.topology in (None, CX7Topology.UNKNOWN):
        return result
    try:
        mgr = cluster_manager if cluster_manager is not None else _default_cluster_manager()
        if mgr:
            mgr.update(plan.cluster_name, topology=topo)
            result.saved = True
    except Exception as e:
        result.error = str(e)
    return result


def apply_cx7(
    plan: Cx7Plan,
    *,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    sudo_password: str | None = None,
    sudo_hosts: set[str] | None = None,
    passwords: SudoPasswords | None = None,
    dispatch=None,
    cluster_manager: ClusterManager | None = None,
) -> Cx7ApplyResult:
    """Apply *plan*, then record its topology and chosen interface filter.

    EXPERIMENTAL. Credentials are the caller's to collect: *sudo_password* is
    piped to ``sudo -S`` on *sudo_hosts* only, and *passwords* re-asks a host
    whose sudo rejects it. The topology is not saved after a partial apply
    (issue #311); see :func:`save_cx7_topology`.
    """
    from sparkrun.orchestration import networking as net

    if plan.cluster_plan is None:
        raise Cx7PlanError("plan has no host assignments to apply")
    raw = net.apply_cx7_plan(
        plan.cluster_plan,
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
        sudo_password=sudo_password,
        sudo_hosts=sudo_hosts,
        passwords=passwords,
        dispatch=dispatch,
    )
    by_host = {r.host: r for r in raw}
    result = Cx7ApplyResult(results=[by_host[hp.host] for hp in plan.host_plans if hp.host in by_host])
    if cluster_manager is None and plan.cluster_name:
        cluster_manager = _default_cluster_manager()
    result.topology_save = save_cx7_topology(plan, cluster_manager=cluster_manager, failed=result.failed, dry_run=dry_run)

    # Persist an explicitly chosen interface filter so later runs reuse it.
    if plan.cluster_name and plan.interfaces_explicit and plan.interfaces:
        saved = Cx7InterfacesSave(cluster_name=plan.cluster_name, interfaces=list(plan.interfaces))
        try:
            if cluster_manager:
                cluster_manager.update(plan.cluster_name, fabric_interfaces=list(plan.interfaces))
                saved.saved = True
        except Exception as e:
            saved.error = str(e)
        result.interfaces_save = saved
    return result


__all__ = [
    "CX7_TOPOLOGY_CHOICES",
    "CX7_EVENT_DETECTED",
    "CX7_EVENT_INTERFACES",
    "CX7_EVENT_SAVED_TOPOLOGY",
    "CX7_EVENT_DETECTING_TOPOLOGY",
    "CX7_EVENT_DETECTED_TOPOLOGY",
    "CX7_EVENT_RING_DETECTING",
    "CX7_EVENT_TOPOLOGY",
    "CX7_EVENT_SUBNETS",
    "Cx7PlanError",
    "Cx7Plan",
    "Cx7TopologySave",
    "Cx7InterfacesSave",
    "Cx7ApplyResult",
    "validate_cx7_options",
    "plan_cx7",
    "save_cx7_topology",
    "apply_cx7",
]
