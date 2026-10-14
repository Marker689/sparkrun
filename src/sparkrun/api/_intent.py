"""``sparkrun.api.find_running_intent`` — "is this workload already up?"

The question ``--ensure`` asks.  It is deliberately keyed on the **intent**
(recipe + parallelism + port) rather than on a ``cluster_id``, because a
cluster_id also encodes *placement* — and placement is not stable:

* Under a status-aware scheduler (``occupancy-*``) each launch gets a random
  placement token, so no host-derived cluster_id can ever match a running job.
* Even under the deterministic (greedy) scheduler the token is a hash of a
  host set, so asking with a different host list than the launch used misses.

``--ensure`` previously derived a cluster_id from ``(recipe, hosts)`` and so
inherited both failure modes: on an occupancy cluster it *always* reported
"not running" and launched a duplicate.  Matching the intent instead answers
the question the flag is actually asking — "is this workload already serving,
wherever it happens to be?" — and gives the same answer regardless of which
scheduler placed it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.cluster_status import ClusterStatus
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentMatch:
    """A running deployment of a given launch intent."""

    intent_id: str
    workload_id: str
    """The running deployment's full ``sparkrun_<intent>_<placement>`` id."""
    hosts: tuple[str, ...] = ()
    """Hosts carrying at least one of its workloads, in query order."""
    recipe: str | None = None
    runtime: str | None = None
    started_at: float | None = None
    other_cluster_ids: tuple[str, ...] = field(default_factory=tuple)
    """Additional deployments of the same intent, if the cluster somehow holds
    more than one.  Normally empty — ``api.run`` evicts superseded deployments
    of an intent — but reported rather than hidden, since it means the cluster
    is in a state the launch path tries to avoid."""


def find_running_intent(
    intent_id: str,
    hosts: list[str] | tuple[str, ...],
    *,
    cluster=None,
    sctx: "SparkrunContext | None" = None,
    status: "ClusterStatus | None" = None,
) -> IntentMatch | None:
    """Return the deployment of *intent_id* running on *hosts*, or ``None``.

    Args:
        intent_id: Intent to look for (``RunPlan.intent_id`` /
            :func:`~sparkrun.orchestration.job_metadata.generate_intent_id`).
        hosts: Hosts to inspect — pass the cluster's **full** host list, not a
            placement's subset, or a deployment that landed elsewhere is
            missed.
        cluster: Optional :class:`ClusterDefinition` (selects the status
            substrate / executor).
        sctx: Optional shared :class:`SparkrunContext`.
        status: Pre-fetched snapshot to inspect instead of querying.  Must be
            a **raw** snapshot — one with this intent's workloads subtracted
            (as placement uses, see ``resolve_effective_hosts``'s
            ``exclude_intent_id``) would report nothing running by
            construction. Only the requested hosts are considered, in request
            order, even when the snapshot covers a larger cluster.

    Returns:
        The matching deployment with the most hosts (ties broken by
        ``cluster_id`` for determinism), or ``None`` when no positive match
        was observed. A failed or incomplete query can also yield ``None``;
        this best-effort helper does not establish verified absence. Ensure
        uses that best-effort policy. Callers requiring verified absence must
        acquire ``api.status()``, reject ``snapshot.observation_errors``, then
        pass the complete snapshot here. Partial snapshots can still supply
        positive matches.
    """
    from sparkrun.core.cluster_status import workload_matches_intent

    if not intent_id or not hosts:
        return None

    if status is None:
        import sparkrun.api as api

        try:
            status = api.status(list(hosts), cluster=cluster, sctx=sctx)
        except Exception as e:
            logger.debug("find_running_intent: status query failed, assuming not running: %s", e)
            return None
    if status is None:
        return None

    # cluster_id -> (hosts in query order, representative workload)
    by_cluster: dict[str, list] = {}
    meta: dict[str, object] = {}
    host_order = {host: index for index, host in enumerate(dict.fromkeys(hosts))}
    observations = sorted((occ for occ in status.hosts if occ.host in host_order), key=lambda occ: host_order[occ.host])
    for occ in observations:
        for w in occ.workloads:
            if not workload_matches_intent(w, intent_id):
                continue
            entry = by_cluster.setdefault(w.cluster_id, [])
            if occ.host not in entry:
                entry.append(occ.host)
            meta.setdefault(w.cluster_id, w)

    if not by_cluster:
        return None

    ranked = sorted(by_cluster.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    cluster_id, match_hosts = ranked[0]
    w = meta[cluster_id]

    return IntentMatch(
        intent_id=intent_id,
        workload_id=cluster_id,
        hosts=tuple(match_hosts),
        recipe=getattr(w, "recipe_name", None),
        runtime=getattr(w, "runtime_name", None),
        started_at=getattr(w, "started_at", None),
        other_cluster_ids=tuple(cid for cid, _ in ranked[1:]),
    )


__all__ = ["IntentMatch", "find_running_intent"]
