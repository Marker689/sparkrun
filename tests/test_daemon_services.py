"""Read-only periodic discovery recovery must not replay mutations."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.core.config import SparkrunConfig
from sparkrun.daemon.jobs import JobStore
from sparkrun.daemon.services import Services


@pytest.fixture
def service(monkeypatch, tmp_path):
    from sparkrun.core import daemon_registry, runtime_lease

    op = SimpleNamespace(name="discovery.snapshot", version=1, scope="read")
    monkeypatch.setattr(daemon_registry, "get_daemon_op", lambda name: op)
    monkeypatch.setattr(runtime_lease, "runtime_identity", lambda config: {})
    config = SparkrunConfig(tmp_path / "config.yaml")
    config.set("daemon.discovery_interval", 5)
    store = JobStore(tmp_path / "jobs")
    return op, config, store


def lost_job(service, *, age=60, worker=None):
    op, _, store = service
    job = store.submit(op, {}, "service:daemon", "periodic service", {})
    store.finish(job, "lost", error={"detail": "worker disappeared"})
    import json

    with store.transaction() as db:
        db.execute(
            "UPDATE jobs SET created=?,updated=?,worker=? WHERE id=?",
            (time.time() - 60, time.time() - age, json.dumps(worker) if worker else None, job),
        )
    return job


@pytest.mark.parametrize("worker", [None, {"process": {"pid": 123, "created": "old"}}])
def test_lost_snapshot_replaced_once_across_service_restart(service, monkeypatch, worker):
    _, config, store = service
    old = lost_job(service, worker=worker)
    monkeypatch.setattr("sparkrun.core.process_identity.alive", lambda process: False)
    for _ in range(3):
        Services(store).tick(config, {})
    jobs = store.list()
    assert len(jobs) == 2
    assert store.get(old)["state"] == "lost"
    assert len([job for job in jobs if job["state"] == "queued"]) == 1
    assert all(job["operation"] == "discovery.snapshot" for job in jobs)


@pytest.mark.parametrize("condition", ["alive", "unknown", "backoff", "maintenance"])
def test_lost_snapshot_recovery_waits_until_safe(service, monkeypatch, condition):
    _, config, store = service
    old = lost_job(service, age=0 if condition == "backoff" else 60, worker={"process": {"pid": 123}})
    alive = Mock(side_effect=PermissionError("cannot inspect")) if condition == "unknown" else Mock(return_value=condition == "alive")
    monkeypatch.setattr("sparkrun.core.process_identity.alive", alive)
    if condition == "unknown":
        with pytest.raises(PermissionError):
            Services(store).tick(config, {})
    else:
        Services(store).tick(config, {}, maintenance=condition == "maintenance")
    assert [job["id"] for job in store.list()] == [old]


def test_mutating_lost_job_is_not_a_periodic_retry(service):
    _, config, store = service
    job = store.submit(SimpleNamespace(name="workloads.run", version=1, scope="operate"), {}, "service:daemon", "review", {})
    store.finish(job, "lost")
    Services(store).tick(config, {})
    assert store.get(job)["state"] == "lost"
    assert [row["operation"] for row in store.list()].count("workloads.run") == 1
