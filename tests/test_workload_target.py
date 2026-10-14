"""``api.stop`` and ``api.open_log_target`` address a workload the same way.

Both used to carry their own copy of "workload_id or recipe+hosts → hosts +
job metadata + the job's connection", and the copies drifted (logs prepared
the transport up to twice and kept the invocation's cluster when only the
executor differed; the two raised different errors for the same missing
hosts).  :func:`sparkrun.api._resolve.resolve_workload_target` is now the one
resolver; these tests pin its contract and that both callers go through it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import sparkrun.api as api
import sparkrun.api._resolve as resolve_mod
from sparkrun.api._errors import JobNotFound, SparkrunError
from sparkrun.api._resolve import resolve_workload_target
from sparkrun.core.application_profile import resource_name
from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.core.cluster_status import ClusterStatus, ContainerDetail, HostOccupancy, RunningWorkload
from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.job_metadata import generate_intent_id, save_job_metadata

JOB_USER = "job-user"


def _recipe() -> Recipe:
    return Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})


def _workload_id(recipe: Recipe) -> str:
    return resource_name("_%s_" % generate_intent_id(recipe)) + "111111111111"


def _cluster_manager() -> ClusterManager:
    from sparkrun.core.config import get_config_root

    return ClusterManager(get_config_root())


def _snapshot(workload_id: str, hosts: list[str]) -> ClusterStatus:
    return ClusterStatus(
        hosts=tuple(
            HostOccupancy(
                host=host,
                workloads=(
                    RunningWorkload(
                        cluster_id=workload_id,
                        containers=(ContainerDetail(name=workload_id + "_solo", role="solo", status="Up", image="img"),),
                    ),
                )
                if index == 0
                else (),
            )
            for index, host in enumerate(hosts)
        ),
        executor="docker",
    )


def _status(snapshot: ClusterStatus):
    """Answer the cross-executor discovery sweep with *snapshot*."""
    return patch("sparkrun.orchestration.executor.query_status_for_cluster", return_value=snapshot)


@pytest.fixture
def prepared(monkeypatch):
    """Record every transport preparation, by (cluster name, user)."""
    calls: list[tuple[str, str | None]] = []

    def _prepare(cluster_def, *, dry_run=False):
        calls.append((cluster_def.name, cluster_def.user))

    monkeypatch.setattr(resolve_mod, "prepare_transport", _prepare)
    return calls


# --------------------------------------------------------------------------
# By id — host precedence and errors
# --------------------------------------------------------------------------


def test_by_id_explicit_hosts_outrank_recorded_hosts(tmp_path: Path):
    recipe = _recipe()
    wid = _workload_id(recipe)
    save_job_metadata(wid, recipe, ["h1", "h2"], cache_dir=str(tmp_path))

    target = resolve_workload_target(
        workload_id=wid, recipe=None, hosts=["h2"], overrides=None, cluster=None, cache_dir=str(tmp_path), operation="op"
    )
    assert target.hosts == ["h2"]
    assert target.meta is not None
    assert target.recipe is None
    assert target.transport_prepared is False


def test_by_id_recorded_hosts_outrank_the_named_cluster(tmp_path: Path):
    """Placement is history: a named cluster supplies connection, not hosts (#277)."""
    _cluster_manager().create("lab", ["h1", "h2", "h3"], user="lab-user")
    recipe = _recipe()
    wid = _workload_id(recipe)
    save_job_metadata(wid, recipe, ["h1"], cache_dir=str(tmp_path))

    target = resolve_workload_target(
        workload_id=wid, recipe=None, hosts=None, overrides=None, cluster="lab", cache_dir=str(tmp_path), operation="op"
    )
    assert target.hosts == ["h1"]
    assert (target.cluster.name, target.cluster.user, target.cluster.hosts) == ("lab", "lab-user", ["h1"])


def test_by_id_named_cluster_supplies_hosts_only_without_metadata(tmp_path: Path):
    _cluster_manager().create("lab", ["h1", "h2"])

    target = resolve_workload_target(
        workload_id="sparkrun_x_1", recipe=None, hosts=None, overrides=None, cluster="lab", cache_dir=str(tmp_path), operation="op"
    )
    assert target.hosts == ["h1", "h2"]
    assert target.meta is None


def test_by_id_without_metadata_or_hosts_is_job_not_found(tmp_path: Path):
    with pytest.raises(JobNotFound, match="No job metadata found for workload_id"):
        resolve_workload_target(
            workload_id="sparkrun_x_1", recipe=None, hosts=None, overrides=None, cluster=None, cache_dir=str(tmp_path), operation="op"
        )


def test_by_id_with_hostless_metadata_is_job_not_found(tmp_path: Path):
    recipe = _recipe()
    wid = _workload_id(recipe)
    save_job_metadata(wid, recipe, [], cache_dir=str(tmp_path))

    with pytest.raises(JobNotFound, match="No hosts known for workload_id"):
        resolve_workload_target(
            workload_id=wid, recipe=None, hosts=None, overrides=None, cluster=None, cache_dir=str(tmp_path), operation="op"
        )


def test_neither_id_nor_recipe_names_the_operation(tmp_path: Path):
    with pytest.raises(SparkrunError, match="^api.stop requires workload_id or recipe"):
        resolve_workload_target(
            workload_id=None, recipe=None, hosts=["h1"], overrides=None, cluster=None, cache_dir=str(tmp_path), operation="api.stop"
        )


# --------------------------------------------------------------------------
# By recipe — discovery range vs. job connection
# --------------------------------------------------------------------------


def test_by_recipe_uses_the_jobs_connection_but_the_discovery_hosts(tmp_path: Path, prepared):
    """Discovery runs as the invocation; the operation runs as the job (#277).

    The recorded cluster names more hosts than the discovery range; those must
    not leak back into the target.
    """
    _cluster_manager().create("lab", ["h1", "h2", "h3"], user=JOB_USER)
    recipe = _recipe()
    wid = _workload_id(recipe)
    save_job_metadata(wid, recipe, ["h1"], cache_dir=str(tmp_path), cluster_name="lab", ssh_user=JOB_USER)

    with _status(_snapshot(wid, ["h1", "h2"])):
        target = resolve_workload_target(
            workload_id=None, recipe=recipe, hosts=["h1", "h2"], overrides=None, cluster=None, cache_dir=str(tmp_path), operation="op"
        )

    assert target.workload_id == wid
    assert target.recipe is recipe
    assert target.hosts == ["h1", "h2"]
    assert (target.cluster.name, target.cluster.user, target.cluster.hosts) == ("lab", JOB_USER, ["h1", "h2"])
    # Discovery prepared the anonymous invocation cluster, not the job's.
    assert prepared == [("", None)]
    assert target.transport_prepared is False


def test_by_recipe_same_connection_is_prepared_once(tmp_path: Path, prepared):
    recipe = _recipe()
    wid = _workload_id(recipe)
    save_job_metadata(wid, recipe, ["h1"], cache_dir=str(tmp_path))

    with _status(_snapshot(wid, ["h1"])):
        target = resolve_workload_target(
            workload_id=None, recipe=recipe, hosts=["h1"], overrides=None, cluster=None, cache_dir=str(tmp_path), operation="op"
        )

    assert prepared == [("", None)]
    assert target.transport_prepared is True


# --------------------------------------------------------------------------
# Both callers go through it, and prepare each connection once
# --------------------------------------------------------------------------


@pytest.fixture
def spy(monkeypatch):
    calls: list[dict] = []
    real = resolve_mod.resolve_workload_target

    def _spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(resolve_mod, "resolve_workload_target", _spy)
    return calls


def _job_on_lab(tmp_path: Path) -> tuple[Recipe, str]:
    _cluster_manager().create("lab", ["h1", "h2"], user=JOB_USER)
    recipe = _recipe()
    wid = _workload_id(recipe)
    save_job_metadata(wid, recipe, ["h1"], cache_dir=str(tmp_path), cluster_name="lab", ssh_user=JOB_USER)
    return recipe, wid


@pytest.mark.parametrize("by_recipe", [False, True])
def test_stop_resolves_through_the_shared_helper(tmp_path: Path, spy, prepared, by_recipe):
    recipe, wid = _job_on_lab(tmp_path)
    form = {"recipe": recipe, "hosts": ["h1"]} if by_recipe else {"workload_id": wid}

    with (
        _status(_snapshot(wid, ["h1"])),
        patch("sparkrun.api._stop._discover_executor_name", return_value=None),
    ):
        result = api.stop(cache_dir=str(tmp_path), dry_run=True, **form)

    assert len(spy) == 1 and spy[0]["operation"] == "api.stop"
    assert result.workload_id == wid
    assert result.hosts_targeted == ("h1",)
    # The job's connection is prepared exactly once (by-recipe discovery also
    # prepares the anonymous invocation cluster first).
    assert prepared.count(("lab", JOB_USER)) == 1
    assert prepared == ([("", None)] if by_recipe else []) + [("lab", JOB_USER)]


@pytest.mark.parametrize("by_recipe", [False, True])
def test_open_log_target_resolves_through_the_shared_helper(tmp_path: Path, spy, prepared, by_recipe):
    recipe, wid = _job_on_lab(tmp_path)
    form = {"recipe": recipe, "hosts": ["h1"]} if by_recipe else {"workload_id": wid}

    with _status(_snapshot(wid, ["h1"])), patch.object(DockerExecutor, "query_status", return_value=_snapshot(wid, ["h1"])):
        target = api.open_log_target(cache_dir=str(tmp_path), **form)

    assert len(spy) == 1 and spy[0]["operation"] == "Log resolution"
    assert target.workload_id == wid
    assert prepared.count(("lab", JOB_USER)) == 1
    assert prepared == ([("", None)] if by_recipe else []) + [("lab", JOB_USER)]


def test_stop_and_logs_agree_on_missing_hosts(tmp_path: Path):
    """One wording for one condition, whichever command hit it."""
    with pytest.raises(JobNotFound) as stop_err:
        api.stop(workload_id="sparkrun_x_1", cache_dir=str(tmp_path))
    with pytest.raises(JobNotFound) as logs_err:
        api.open_log_target("sparkrun_x_1", cache_dir=str(tmp_path))
    assert str(stop_err.value) == str(logs_err.value)
