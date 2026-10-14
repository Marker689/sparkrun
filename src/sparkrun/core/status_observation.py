"""Recorded liveness with explicit executor/destination coverage.

A missing observation is unknown, never proof of absence. Coverage is kept per
executor so a successful Docker query cannot mask a failed local-process query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.orchestration.executor import ExecutorTarget


@dataclass(frozen=True)
class ExecutorCoverage:
    target: ExecutorTarget
    status_scope: str
    requested_hosts: frozenset[str]
    hosts: frozenset[str]
    ssh_user: str | None = None
    ssh_kwargs: Mapping | None = field(default=None, compare=False, repr=False)
    """Detached in-memory connection for teardown; never persisted in the cache."""

    def __post_init__(self):
        from sparkrun.utils.data import freeze, normalize_data

        if self.ssh_user is not None and (not isinstance(self.ssh_user, str) or not self.ssh_user):
            raise ValueError("Observation SSH user must be a nonempty string or None")
        if self.ssh_kwargs is not None:
            object.__setattr__(self, "ssh_kwargs", freeze(normalize_data(self.ssh_kwargs)))
        object.__setattr__(self, "requested_hosts", frozenset(self.requested_hosts))
        object.__setattr__(self, "hosts", frozenset(self.hosts))
        if not self.hosts <= self.requested_hosts:
            raise ValueError("Covered hosts must have been requested")

    def matches(self, target: ExecutorTarget, *, ssh_user=None) -> bool:
        from sparkrun.core._executor_destination import ExecutorDestination

        return ExecutorDestination.from_target(self.target, self.ssh_user).matches(ExecutorDestination.from_target(target, ssh_user))

    def matches_job(self, job) -> bool:
        from sparkrun.core._executor_destination import ExecutorDestination

        return ExecutorDestination.from_target(self.target, self.ssh_user).matches(
            ExecutorDestination.from_metadata(job.metadata, target=self.target)
        )

    def covers_job(self, job) -> bool:
        hosts = frozenset(job.hosts)
        return bool(hosts) and hosts <= self.hosts and self.matches_job(job)

    def to_dict(self):
        return {
            "target": self.target.to_dict(),
            "ssh_user": self.ssh_user,
            "status_scope": self.status_scope,
            "requested_hosts": sorted(self.requested_hosts),
            "hosts": sorted(self.hosts),
        }

    @classmethod
    def from_dict(cls, data):
        from sparkrun.orchestration.executor import ExecutorTarget

        def strings(key):
            values = data[key]
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError("Invalid observation hosts")
            return frozenset(values)

        target = ExecutorTarget(**data["target"])
        scope = data["status_scope"]
        hosts, requested = strings("hosts"), strings("requested_hosts")
        if not isinstance(scope, str) or not hosts <= requested:
            raise ValueError("Invalid observation coverage")
        return cls(target, scope, requested, hosts, ssh_user=data.get("ssh_user"))


@dataclass(frozen=True)
class RunningSnapshot:
    cluster_ids: frozenset[str]
    coverage: tuple[ExecutorCoverage, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "cluster_ids", frozenset(self.cluster_ids))
        object.__setattr__(self, "coverage", tuple(self.coverage))

    def _matching_coverage(self, target, ssh_user):
        matches = [c for c in self.coverage if c.matches(target, ssh_user=ssh_user)]
        # A host sweep includes peer executors. Reusing the Docker portion must
        # not reuse a local-process observation made through a different user.
        return [
            c
            for c in matches
            if all(
                not peer.target.user_scoped or peer.matches(peer.target, ssh_user=ssh_user)
                for peer in self.coverage
                if peer.status_scope == c.status_scope
            )
        ]

    def for_target(self, target: ExecutorTarget, *, ssh_user=None) -> bool:
        """Even an incomplete observation must name the requested destination."""
        return bool(self._matching_coverage(target, ssh_user))

    def covers(self, target: ExecutorTarget, hosts, *, ssh_user=None) -> bool:
        return bool(hosts) and any(set(hosts) <= c.hosts for c in self._matching_coverage(target, ssh_user))

    def matches_job(self, job) -> bool:
        return any(c.matches_job(job) for c in self.coverage)

    def confirms_absent(self, job) -> bool:
        return job.workload_id not in self.cluster_ids and any(c.covers_job(job) for c in self.coverage)

    def to_dict(self):
        return {"version": 2, "cluster_ids": sorted(self.cluster_ids), "coverage": [c.to_dict() for c in self.coverage]}

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != 2 or not isinstance(data.get("cluster_ids"), list):
            raise ValueError("Unknown running snapshot format")
        if any(not isinstance(cid, str) for cid in data["cluster_ids"]):
            raise ValueError("Invalid running snapshot IDs")
        return cls(frozenset(data["cluster_ids"]), tuple(ExecutorCoverage.from_dict(c) for c in data["coverage"]))
