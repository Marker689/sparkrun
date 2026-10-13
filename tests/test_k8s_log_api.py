"""Native Kubernetes submissions and explicit log consumption stay console-free."""

from contextlib import closing
from copy import deepcopy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from sparkrun.api import LogLine, SparkrunError
from sparkrun.plugins.k8s import api
from sparkrun.plugins.k8s.orchestration.client import KubectlClient
from sparkrun.orchestration.ssh import RemoteResult
from test_k8s_setup import _sctx, _nodes_for, _SPARK_LABELS


@pytest.fixture
def log_env(tmp_path, monkeypatch):
    sctx = _sctx(tmp_path)
    script = tmp_path / "kubectl.py"
    script.write_text("import sys; print('out', flush=True); print('err', file=sys.stderr, flush=True)")
    client = KubectlClient("/unused/kubectl", context="test-context", namespace="test-ns")
    prefix = [sys.executable, str(script), "--context", "test-context", "-n", "test-ns"]
    monkeypatch.setattr(client, "base_args", lambda: prefix)
    labels = {"sparkrun.distribution": "sparkrun"}
    controller = {
        "metadata": {"name": "sample", "namespace": "test-ns", "uid": "controller", "labels": labels},
        "spec": {"replicatedJobs": [{"name": "gb10", "replicas": 1}]},
    }
    job = {
        "metadata": {
            "name": "sample-job",
            "namespace": "test-ns",
            "uid": "job",
            "labels": {**labels, "jobset.sigs.k8s.io/replicatedjob-name": "gb10", "jobset.sigs.k8s.io/job-index": "0"},
            "ownerReferences": [{"kind": "JobSet", "uid": "controller", "controller": True}],
        }
    }
    pod = {
        "metadata": {
            "name": "sample-pod",
            "namespace": "test-ns",
            "uid": "pod",
            "labels": labels,
            "ownerReferences": [{"kind": "Job", "uid": "job", "controller": True}],
        },
        "spec": {"nodeName": "node-1", "containers": [{"name": "worker"}]},
        "status": {
            "containerStatuses": [{"name": "worker", "state": {"running": {}}, "restartCount": 0, "containerID": "containerd://one"}]
        },
    }
    jobs, pods = [job], [pod]

    def query(client, args, **kwargs):
        if args[1] == "jobset":
            return deepcopy(controller)
        if args[1] == "job":
            return deepcopy(job)
        if args[1] in {"jobs", "pods"}:
            return deepcopy({"items": jobs if args[1] == "jobs" else pods})
        if args[1] == "pod":
            return deepcopy(next((p for p in pods if p["metadata"]["name"] == args[2]), {}))
        raise AssertionError(args)

    lookup = Mock(side_effect=query)
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.logs._query_json", lookup)
    monkeypatch.setattr("sparkrun.plugins.k8s.executor.K8sExecutor._client", lambda self: client)
    make_client = Mock(return_value=client)
    monkeypatch.setattr(api._ops, "make_client", make_client)
    return SimpleNamespace(
        sctx=sctx,
        script=script,
        client=client,
        lookup=lookup,
        make_client=make_client,
        prefix=prefix,
        controller=controller,
        jobs=jobs,
        pods=pods,
    )


@pytest.mark.parametrize("kind", ["jobset", "job"])
def test_submission_returns_before_reading_logs(log_env, monkeypatch, capfd, kind):
    from sparkrun.plugins.k8s.orchestration import launch

    env = log_env
    monkeypatch.setattr(
        "sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", lambda *a, **kw: _nodes_for([("s0", _SPARK_LABELS, 1, 1, False)])
    )
    submitted = RemoteResult(host="k8s", returncode=0, stdout="created", stderr="")
    monkeypatch.setattr(launch, "submit_jobset", lambda *a, **kw: submitted)
    monkeypatch.setattr(env.client, "run_launcher_job", lambda *a, **kw: submitted)
    reading = Mock(side_effect=AssertionError("submission must not read logs"))
    monkeypatch.setattr("sparkrun.orchestration.logs._spawn", reading)
    if kind == "jobset":
        result = api.launch_jobset(env.sctx, name="sample", rank_models=["gb10"], image="img", serve_command="serve")
        assert result.submitted
    else:
        result = api.run_launcher_job(env.sctx, name="sample", image="img", command=["true"])
        assert result.applied
    reading.assert_not_called()
    captured = capfd.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("kind", ["jobset", "job"])
def test_explicit_logs_keep_diagnostics_separate_and_read_lazily(log_env, monkeypatch, capfd, caplog, kind):
    from sparkrun.orchestration import logs as readers

    env = log_env
    spawn = Mock(wraps=readers._spawn)
    monkeypatch.setattr(readers, "_spawn", spawn)
    lines = api.logs(
        env.sctx, name="sample", kind=kind, namespace="test-ns", kubeconfig="test-config", context="test-context", tail=10, follow=True
    )
    spawn.assert_not_called()
    assert env.lookup.call_args_list[0].args[1] == ["get", kind, "sample", "-o", "json", "--ignore-not-found"]
    env.make_client.assert_called_once_with(env.sctx, namespace="test-ns", kubeconfig="test-config", context="test-context")
    with closing(lines):
        result = list(lines)
    assert [line.text for line in result] == ["out"]
    assert "err" in caplog.text
    assert all(
        isinstance(line, LogLine) and line.host == "node-1" and line.container == "worker" and line.role == "solo" for line in result
    )
    assert result[0].resource.name == "sample-pod" and result[0].resource.uid == "pod"
    assert result[0].source_id.startswith("log_")
    command = spawn.call_args.args[0]
    assert command[: len(env.prefix)] == env.prefix
    assert command[len(env.prefix) :] == ["logs", "sample-pod", "-c", "worker", "--tail=10", "--follow"]
    assert capfd.readouterr() == ("", "")


def test_closing_logs_stops_local_reader_without_deleting_job(log_env, monkeypatch, capfd):
    from sparkrun.orchestration import logs as readers

    env = log_env
    env.script.write_text("import time; print('started', flush=True); time.sleep(30)")
    processes = []
    original = readers._spawn

    def spawn(cmd, **kwargs):
        proc = original(cmd, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(readers, "_spawn", spawn)
    with closing(api.logs(env.sctx, name="sample", follow=True)) as lines:
        assert next(lines).text == "started"
        assert processes[0].poll() is None
    assert processes[0].poll() is not None
    assert processes[0].stdout.closed
    assert all(call.args[1][0] == "get" for call in env.lookup.call_args_list)
    assert capfd.readouterr() == ("", "")


@pytest.mark.parametrize(
    "resource,reason", [({}, "No Kubernetes"), ({"metadata": {"labels": {"sparkrun.distribution": "jetsonrun"}}}, "another application")]
)
def test_logs_reject_missing_or_foreign_resource_before_reading(log_env, resource, reason):
    log_env.lookup.side_effect = None
    log_env.lookup.return_value = resource
    with pytest.raises(SparkrunError, match=reason):
        api.logs(log_env.sctx, name="sample")


@pytest.mark.parametrize("kwargs", [{"name": "-unsafe"}, {"name": "a,b"}, {"name": ""}, {"kind": "pod"}, {"tail": -1}, {"tail": True}])
def test_logs_validate_selector_before_cluster_access(log_env, kwargs):
    with pytest.raises(ValueError):
        api.logs(log_env.sctx, **({"name": "sample"} | kwargs))
    log_env.make_client.assert_not_called()


def test_nonzero_log_reader_exit_is_a_library_error(log_env, capfd):
    log_env.script.write_text("import sys; print('denied', file=sys.stderr); sys.exit(7)")
    with closing(api.logs(log_env.sctx, name="sample")) as lines:
        with pytest.raises(SparkrunError, match="exit status 7"):
            next(lines)
    assert capfd.readouterr() == ("", "")


@pytest.mark.parametrize("kind", ["jobset", "job"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_follow_renders_explicit_stream_after_submission(log_env, monkeypatch, kind, dry_run):
    from sparkrun.plugins.k8s.cli import build_setup_command

    env = log_env
    monkeypatch.setattr("sparkrun.cli._common._get_context", lambda ctx: env.sctx)
    submitted = SimpleNamespace(
        name="sample", job_name="sample", namespace="test-ns", image="img", manifests_yaml="manifest", feasibility_summary="feasible"
    )
    launch = Mock(return_value=submitted)
    monkeypatch.setattr(api, "launch_jobset" if kind == "jobset" else "run_launcher_job", launch)
    events = []

    def stream():
        try:
            events.append("read")
            yield LogLine("k8s", "sample", "hello")
            raise KeyboardInterrupt
        finally:
            events.append("closed")

    logs = Mock(side_effect=lambda *a, **kw: stream())
    monkeypatch.setattr(api, "logs", logs)
    args = ["launch", "--ranks", "gb10", "--serve", "serve"] if kind == "jobset" else ["run-job", "--command", "true"]
    args += ["--name", "sample", "--image", "img", "--follow"]
    if dry_run:
        args += ["--dry-run"]
    result = CliRunner().invoke(build_setup_command(), args)
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "follow" not in launch.call_args.kwargs
    if dry_run:
        logs.assert_not_called()
        assert events == [] and "manifest" in result.output
    else:
        assert result.output.index("Reattach with:") < result.output.index("hello")
        assert logs.call_args.kwargs["kind"] == kind and logs.call_args.kwargs["follow"] is True
        assert events == ["read", "closed"]
