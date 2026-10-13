"""Native workloads use the same public log targets, readers, and daemon stream."""

from copy import deepcopy
import json
import subprocess
import sys
import threading
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from sparkrun import api
from sparkrun.core.recipe import Recipe
from sparkrun.core.operation import OperationState
from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata
from sparkrun.orchestration import logs as readers
from sparkrun.plugins.k8s.executor import K8sExecutor, K8sExecutorConfig
from sparkrun.plugins.k8s.orchestration import logs as native
from sparkrun.plugins.k8s.orchestration.client import KubectlClient
from test_k8s_log_api import log_env as log_env
from test_daemon import binding as binding

CID = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
_REAL_QUERY = native._query_json


@pytest.fixture
def saved_logs(log_env, tmp_path, monkeypatch):
    env = log_env
    env.cache = str(tmp_path / "cache")
    env.sctx.config.set("cache_dir", env.cache)
    env.controller["metadata"]["annotations"] = {"sparkrun.cluster_id": CID}
    executor = K8sExecutor(K8sExecutorConfig(k8s_namespace="test-ns", k8s_context="test-context", kubectl_path="/unused/kubectl"))
    save_job_metadata(
        CID,
        Recipe({"runtime": "sglang", "model": "test/model"}),
        ["controller-scope"],
        cache_dir=env.cache,
        executor=executor,
        native_resource={"kind": "JobSet", "name": "sample"},
    )
    env.resolve_calls = []
    from sparkrun.orchestration.executor import resolve_executor

    def resolve(**kwargs):
        env.resolve_calls.append(kwargs)
        return resolve_executor(**kwargs)

    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", resolve)
    monkeypatch.setattr(K8sExecutor, "query_status", Mock(side_effect=AssertionError("native sources own availability checks")))
    monkeypatch.setattr("sparkrun.api._resolve.prepare_transport", Mock(side_effect=AssertionError("native logs must not prepare SSH")))

    def target(**kwargs):
        return api.open_log_target(workload_id=CID, cache_dir=env.cache, sctx=env.sctx, **kwargs)

    env.target = target
    return env


def _add_worker(env, *, node="node-2"):
    env.controller["spec"]["replicatedJobs"][0]["replicas"] = 2
    job = deepcopy(env.jobs[0])
    job["metadata"].update(name="worker-job", uid="worker-job")
    job["metadata"]["labels"]["jobset.sigs.k8s.io/job-index"] = "1"
    pod = deepcopy(env.pods[0])
    pod["metadata"].update(name="worker-pod", uid="worker-pod")
    pod["metadata"]["ownerReferences"][0]["uid"] = "worker-job"
    pod["spec"]["nodeName"] = node
    env.jobs.append(job)
    env.pods.append(pod)
    return pod


def test_saved_native_target_reads_full_history_locally_and_pages(saved_logs, monkeypatch):
    env = saved_logs
    env.script.write_text(
        "import sys\ntail=int(next(a.split('=')[1] for a in sys.argv if a.startswith('--tail=')))\n"
        "rows=list(map(str, range(30)))\n"
        "for row in rows if tail < 0 else rows[-tail:] if tail else []: print(row)\n"
        "print('kubectl warning', file=sys.stderr)\n"
    )
    spawn = Mock(wraps=readers._spawn)
    monkeypatch.setattr(readers, "_spawn", spawn)
    target = env.target()
    spawn.assert_not_called()
    assert target.runtime == "sglang" and target.workload_id == CID
    (source,) = target.sources
    assert source.host == "node-1" and source.resource.name == "sample-pod"
    assert [line.text for line in target.read()] == list(map(str, range(30)))
    assert [line.text for line in target.page(source, skip=8, count=5)] == list(map(str, range(17, 22)))
    assert [line.text for line in target.page(source, skip=28, count=5)] == ["0", "1"]
    assert not target.page(source, skip=30)
    assert list(target.read(tail=0)) == []
    assert all(call.args[0][: len(env.prefix)] == env.prefix for call in spawn.call_args_list)
    overrides = env.resolve_calls[0]["cli_overrides"]
    assert overrides["k8s_namespace"] == "test-ns" and overrides["k8s_context"] == "test-context"
    assert load_job_metadata(CID, cache_dir=env.cache)


def test_multi_pod_sources_filter_without_losing_topology(saved_logs):
    env = saved_logs
    _add_worker(env, node="node-1")  # Same container name and host; source_id disambiguates.
    all_sources = env.target(scope="all").sources
    assert [(s.rank, s.role) for s in all_sources] == [(0, "head"), (1, "worker")]
    assert len({s.source_id for s in all_sources}) == 2
    assert env.target().sources == all_sources[:1]
    assert env.target(container="worker").sources == all_sources
    selected = env.target(source_id=all_sources[1].source_id)
    assert selected.sources == all_sources[1:]
    assert list(selected.read())[0].resource.name == "worker-pod"
    with pytest.raises(api.JobNotFound, match="No log sources match"):
        env.target(source_id=all_sources[1].source_id, host="elsewhere")


def test_sidecars_are_separate_sources_and_head_keeps_primary_container(saved_logs):
    env = saved_logs
    pod = env.pods[0]
    pod["spec"]["containers"].append({"name": "metrics"})
    status = deepcopy(pod["status"]["containerStatuses"][0])
    status.update(name="metrics", containerID="containerd://metrics")
    pod["status"]["containerStatuses"].append(status)
    assert [s.container for s in env.target().sources] == ["worker"]
    assert {s.container for s in env.target(scope="all").sources} == {"worker", "metrics"}
    selected = env.target(container="metrics")
    assert len(selected.sources) == 1 and selected.sources[0].role == "sidecar"
    assert list(selected.read())[0].label == "test-ns/sample-pod/metrics"


def test_controller_workload_identity_must_match_saved_metadata(saved_logs):
    saved_logs.controller["metadata"]["annotations"]["sparkrun.cluster_id"] = "other-workload"
    with pytest.raises(api.SparkrunError, match="another workload"):
        saved_logs.target()


def test_resolution_ignores_unrelated_controller_children(saved_logs):
    env = saved_logs
    unrelated = deepcopy(env.pods[0])
    unrelated["metadata"].update(name="unrelated", uid="unrelated")
    unrelated["metadata"]["ownerReferences"][0]["uid"] = "foreign-job"
    env.pods.append(unrelated)
    foreign_job = deepcopy(env.jobs[0])
    foreign_job["metadata"]["uid"] = "foreign-job"
    foreign_job["metadata"]["ownerReferences"][0]["uid"] = "old-jobset"
    env.jobs.append(foreign_job)
    assert len(env.target(scope="all").sources) == 1


@pytest.mark.parametrize("state", ["running", "terminated"])
def test_completed_container_logs_remain_readable(saved_logs, state):
    env = saved_logs
    env.pods[0]["status"]["containerStatuses"][0]["state"] = {state: {}}
    assert [line.text for line in env.target().read()] == ["out"]


@pytest.mark.parametrize("pending", ["queued", "waiting", "missing-primary"])
def test_pending_logs_report_availability_and_preserve_metadata(saved_logs, pending):
    env = saved_logs
    if pending == "queued":
        env.pods.clear()
    else:
        if pending == "missing-primary":
            _add_worker(env)
        env.pods[0]["status"]["containerStatuses"][0]["state"] = {"waiting": {"reason": "ContainerCreating"}}
    with pytest.raises(api.SparkrunError, match="No readable.*pending, queued"):
        env.target()
    if pending == "missing-primary":
        assert env.target(scope="all").sources[0].rank == 1
    assert load_job_metadata(CID, cache_dir=env.cache)


@pytest.mark.parametrize("change", ["uid", "restart", "instance", "deleted"])
def test_changed_source_requires_reopen_before_read_or_page(saved_logs, change):
    env = saved_logs
    target = env.target()
    source = target.sources[0]
    if change == "deleted":
        env.pods.clear()
    elif change == "uid":
        env.pods[0]["metadata"]["uid"] = "replacement"
    else:
        status = env.pods[0]["status"]["containerStatuses"][0]
        status["restartCount" if change == "restart" else "containerID"] = 1 if change == "restart" else "containerd://replacement"
    with pytest.raises(api.LogSourceChanged, match="reopen"):
        target.page(source, skip=0)
    with pytest.raises(api.LogSourceChanged):
        list(target.read(check=False))
    with pytest.raises(api.LogSourceChanged):
        list(target.follow(check=False))
    if change != "deleted":
        assert env.target().sources[0].source_id != source.source_id
        with pytest.raises(api.JobNotFound):
            env.target(source_id=source.source_id)


def test_restart_during_finite_page_does_not_publish_mixed_history(saved_logs, monkeypatch):
    env = saved_logs
    target = env.target()
    original = readers.read_log_command

    def read(*args, **kwargs):
        yield from original(*args, **kwargs)
        env.pods[0]["status"]["containerStatuses"][0]["restartCount"] += 1

    monkeypatch.setattr(readers, "read_log_command", read)
    with pytest.raises(api.LogSourceChanged):
        target.page(target.sources[0], skip=0)


def test_page_timeout_includes_native_identity_check(saved_logs, monkeypatch):
    env = saved_logs
    target = env.target()
    env.script.write_text("import time; time.sleep(30)")
    monkeypatch.setattr(native, "_query_json", _REAL_QUERY)
    processes = []
    spawn = readers._spawn

    def capture(*args, **kwargs):
        process = spawn(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(readers, "_spawn", capture)
    with pytest.raises(subprocess.TimeoutExpired):
        target.page(target.sources[0], skip=0, timeout=0.2)
    assert processes and all(p.poll() is not None for p in processes)


@pytest.mark.parametrize("follow", [False, True])
def test_inherited_operation_cancel_stops_silent_native_reader(saved_logs, monkeypatch, follow):
    env = saved_logs
    env.sctx.operation = OperationState()
    env.script.write_text("import time; time.sleep(30)")
    processes = []
    started = threading.Event()
    spawn = readers._spawn

    def capture(*args, **kwargs):
        p = spawn(*args, **kwargs)
        processes.append(p)
        started.set()
        return p

    monkeypatch.setattr(readers, "_spawn", capture)
    target = env.target()
    result, errors = [], []

    def consume():
        try:
            result.extend(target.read(follow=follow))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=consume)
    thread.start()
    try:
        assert started.wait(5)
    finally:
        env.sctx.operation.cancel.set()
        thread.join(5)
    assert not thread.is_alive() and not errors and not result
    assert all(p.poll() is not None and p.stdout.closed and p.stderr.closed for p in processes)


def test_daemon_native_stream_preserves_source_identity_and_filter(saved_logs, binding, monkeypatch):
    from sparkrun.daemon.auth import TokenStore
    from sparkrun.daemon.server import create_app

    env = saved_logs
    _add_worker(env)
    selected = env.target(scope="all").sources[1]
    monkeypatch.setattr("sparkrun.application.initialize", lambda **kwargs: env.sctx)
    _, token = TokenStore(binding).create("reader", scope="read")
    with TestClient(create_app(binding, start_workers=False)) as client:
        response = client.get(
            "/v1/streams/logs",
            params={"workload_id": CID, "source_id": selected.source_id, "follow": "false"},
            headers={"Authorization": "Bearer " + token},
        )
    assert response.status_code == 200
    records = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert records and all(r["type"] == "log" for r in records), records
    assert records[0]["source_id"] == selected.source_id
    assert records[0]["resource"]["name"] == "worker-pod"
    assert records[0]["host"] == "node-2" and records[0]["container"] == "worker"


def test_control_plane_query_cancellation_reaps_process(tmp_path, monkeypatch):
    # Exercise the real query helper, independent of the fake cluster fixture.
    script = tmp_path / "silent.py"
    script.write_text("import time; time.sleep(30)")
    client = KubectlClient("unused", context="ctx", namespace="ns")
    monkeypatch.setattr(client, "base_args", lambda: [sys.executable, str(script)])
    cancel = threading.Event()
    processes = []
    started = threading.Event()
    spawn = readers._spawn

    def capture(*args, **kwargs):
        proc = spawn(*args, **kwargs)
        processes.append(proc)
        started.set()
        return proc

    monkeypatch.setattr(readers, "_spawn", capture)
    result = []
    thread = threading.Thread(target=lambda: result.append(native._query_json(client, ["get", "pods"], cancel=cancel)))
    thread.start()
    try:
        assert started.wait(5)
    finally:
        cancel.set()
        thread.join(5)
    assert result == [{}] and not thread.is_alive()
    assert all(p.poll() is not None for p in processes)
