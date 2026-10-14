"""Lock-host inference in ``core.coordination.operation_boundary`` (B13).

The decorator infers which hosts a mutating API call is about to touch, so it
can lock exactly those machines. Before this fix the inference skipped the
default-cluster step that ``api._resolve.resolve_cluster`` applies between
explicit hosts and ``config.default_hosts`` -- so with both a default cluster
and ``config.yaml``'s ``cluster.hosts`` configured, a flagless mutating call
(e.g. ``stop(recipe=...)``) locked ``default_hosts`` while the call itself
(via ``resolve_cluster``) actually acted on the default cluster's hosts: two
concurrent mutations holding disjoint locks over the same machines.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import yaml

from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.coordination import operation_boundary, reservation_keys


def _dummy_mutation(
    recipe=None,
    sctx=None,
    hosts=None,
    cluster=None,
    workload_id=None,
    cluster_id=None,
    options=None,
    plan=None,
    dry_run=False,
):
    return "ok"


def _capture_reserved_keys(monkeypatch):
    captured: dict = {}

    def fake_reserve_resources(root, keys):
        captured["keys"] = keys
        return nullcontext()

    monkeypatch.setattr("sparkrun.core.coordination.reserve_resources", fake_reserve_resources)
    return captured


def test_flagless_mutation_locks_the_default_cluster_not_default_hosts(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"cluster": {"hosts": ["confighost1", "confighost2"]}}))
    config = SparkrunConfig(config_path=config_path)

    mgr = ClusterManager(tmp_path)
    mgr.create("default-cluster", ["clusterhost1", "clusterhost2"])
    mgr.set_default("default-cluster")

    captured = _capture_reserved_keys(monkeypatch)
    decorated = operation_boundary(mutation=True)(_dummy_mutation)

    sctx = SimpleNamespace(config=config)
    assert decorated(recipe="some-recipe", sctx=sctx) == "ok"

    # Must match resolve_cluster's precedence: default cluster before
    # config.default_hosts, not the reverse.
    assert captured["keys"] == reservation_keys(["clusterhost1", "clusterhost2"])
    assert captured["keys"] != reservation_keys(["confighost1", "confighost2"])


def test_flagless_mutation_falls_back_to_default_hosts_without_a_default_cluster(tmp_path: Path, monkeypatch):
    """No default cluster set: config.default_hosts is still the fallback."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"cluster": {"hosts": ["confighost1", "confighost2"]}}))
    config = SparkrunConfig(config_path=config_path)

    captured = _capture_reserved_keys(monkeypatch)
    decorated = operation_boundary(mutation=True)(_dummy_mutation)

    sctx = SimpleNamespace(config=config)
    assert decorated(recipe="some-recipe", sctx=sctx) == "ok"

    assert captured["keys"] == reservation_keys(["confighost1", "confighost2"])


def test_plan_locks_candidate_hosts_not_just_the_chosen_host_list(tmp_path: Path, monkeypatch):
    """Eviction of a superseded deployment sweeps candidate_hosts, so the lock
    must cover that wider set, not only the narrower placement (host_list)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({}))
    config = SparkrunConfig(config_path=config_path)

    captured = _capture_reserved_keys(monkeypatch)
    decorated = operation_boundary(mutation=True)(_dummy_mutation)

    plan = SimpleNamespace(
        candidate_hosts=("cand1", "cand2", "cand3"),
        host_list=("cand1",),
        cluster=None,
    )
    sctx = SimpleNamespace(config=config)
    assert decorated(plan=plan, sctx=sctx) == "ok"

    assert captured["keys"] == reservation_keys(("cand1", "cand2", "cand3"))
