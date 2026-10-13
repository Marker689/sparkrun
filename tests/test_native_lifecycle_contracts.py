"""Public native run/stop contracts with real orchestration and fake Kubernetes I/O."""

from dataclasses import replace
import json

import pytest
import yaml

from sparkrun import api
from sparkrun.api._run import plan, run
from sparkrun.api._stop import stop
from sparkrun.core.run_handlers import RunHandler
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.orchestration.job_metadata import load_job_metadata, remove_job_metadata
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.plugins.k8s.orchestration.client import KubectlClient
from sparkrun.plugins.k8s.run import run_k8s
from test_benchmark_startup_collection import bench_env as bench_env
from test_run_option_contracts import run_env as run_env
from test_k8s_setup import _nodes_for, _SPARK_LABELS


@pytest.fixture
def lifecycle_env(run_env, monkeypatch, tmp_path):
    env = run_env
    env.kubeconfig = str(tmp_path / "launch-kubeconfig")
    env.target = (env.kubeconfig, "launch-context", "launch-ns")
    env.sctx.config.set("k8s", {"kubeconfig": env.kubeconfig, "context": "launch-context", "namespace": "launch-ns"})
    env.objects, env.calls = {}, []
    env.fail_delete = False
    env.survive_delete = False
    env.fail_apply = False
    env.current_context = "launch-context"
    env.options = api.RunOptions(
        recipe=env.recipe, hosts=("localhost",), solo=True, executor="k8s", executor_config={"kubectl_path": "/fake/kubectl"}
    )
    monkeypatch.setattr("sparkrun.core.run_handlers.registered_run_handlers", lambda _: {"k8s": RunHandler("k8s", run_k8s)})
    monkeypatch.setattr(
        "sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", lambda *a, **kw: _nodes_for([("gpu0", _SPARK_LABELS, 1, 1, False)])
    )
    monkeypatch.setattr(api, "stop", stop)

    def kubectl(client, args, *, input_data=None, **kwargs):
        target = (client.kubeconfig, client.context or env.current_context, client.namespace)
        env.calls.append((target, list(args)))
        objects = env.objects.setdefault(target, {})
        stdout, stderr, rc = "", "", 0
        if args[:2] == ["config", "current-context"]:
            stdout = env.current_context + "\n"
        elif args[0] == "apply":
            resource = yaml.safe_load(input_data)
            cid = resource["metadata"]["annotations"]["sparkrun.cluster_id"]
            # Metadata exists before submission, including ambiguous failure.
            assert (
                load_job_metadata(cid, cache_dir=str(env.sctx.config.cache_dir))["native_resource"]["name"] == resource["metadata"]["name"]
            )
            objects[resource["metadata"]["name"]] = resource
            if env.fail_apply:
                raise KeyboardInterrupt
            stdout = "submitted"
        elif args[:2] == ["get", "jobsets"]:
            owner = args[args.index("-l") + 1].split("=", 1)[1]
            stdout = json.dumps({"items": [r for r in objects.values() if r["metadata"]["labels"].get("sparkrun.distribution") == owner]})
        elif args[:2] == ["get", "jobset"]:
            stdout = json.dumps(objects[args[2]]) if args[2] in objects else ""
        elif args[:2] == ["delete", "jobset"]:
            assert {"--cascade=foreground", "--wait=true", "--timeout=60s"} <= set(args)
            if env.fail_delete:
                rc, stderr = 7, "delete rejected"
            elif not env.survive_delete:
                objects.pop(args[2], None)
        else:
            pytest.fail("Unexpected command: %r" % args)
        return RemoteResult(client.label, rc, stdout, stderr)

    monkeypatch.setattr(KubectlClient, "run", kubectl)
    return env


def _record(env, result):
    return load_job_metadata(result.cluster_id, cache_dir=str(env.sctx.config.cache_dir))


@pytest.mark.parametrize("target_source", ["settings", "cluster", "recipe", "caller", "current-context"])
def test_native_run_stop_by_id_recovers_exact_target_after_context_change(lifecycle_env, target_source):
    env = lifecycle_env
    options = env.options
    if target_source != "settings":
        env.sctx.config.set("k8s", {})
        target = {"kubeconfig": env.kubeconfig, "k8s_namespace": "launch-ns", "k8s_context": "launch-context"}
        if target_source == "cluster":
            options = replace(
                options, cluster=ClusterDefinition(name="ephemeral-cluster", hosts=["localhost"], executor="k8s", executor_config=target)
            )
        elif target_source == "recipe":
            env.recipe.executor_config = target
        elif target_source == "caller":
            options = replace(options, executor_config={**target, "kubectl_path": "/fake/kubectl"})
        else:
            del target["k8s_context"]
            options = replace(options, executor_config={**target, "kubectl_path": "/fake/kubectl"})
    result = run(options, sctx=env.sctx)
    saved = _record(env, result)
    assert saved["native_resource"] == {"kind": "JobSet", "name": result.metadata["k8s_jobset"]}
    assert saved["executor"] == "k8s" and saved["port"] == 8000
    assert saved["executor_config"]["k8s_context"] == "launch-context"
    assert saved["controller"] == env.sctx.controller_identity.to_dict()
    # Recreate the API context; neither the recipe object nor original cluster
    # definition is needed, and changed defaults/current-context cannot retarget stop.
    env.sctx.config.set("k8s", {"kubeconfig": "/different/config", "context": "other", "namespace": "other"})
    env.current_context = "other"
    restarted = replace(env.sctx, timing=None)
    stopped = stop(cluster_id=result.cluster_id, sctx=restarted)
    assert stopped.success and stopped.containers_removed == 1
    assert not env.objects[env.target] and _record(env, result) is None
    assert all(target == env.target for target, args in env.calls if args[0] == "delete")


@pytest.mark.parametrize("failure", ["owner", "identity", "delete", "confirmation"])
def test_native_stop_failure_keeps_metadata_and_resource(lifecycle_env, failure):
    env = lifecycle_env
    result = run(env.options, sctx=env.sctx)
    resource = env.objects[env.target][result.metadata["k8s_jobset"]]
    if failure == "owner":
        resource["metadata"]["labels"]["sparkrun.distribution"] = "another-app"
    elif failure == "identity":
        resource["metadata"]["annotations"]["sparkrun.cluster_id"] = "another-workload"
    else:
        env.fail_delete, env.survive_delete = failure == "delete", failure == "confirmation"
    stopped = stop(cluster_id=result.cluster_id, sctx=env.sctx)
    assert not stopped.success and stopped.errors and stopped.hosts_failed == ("localhost",)
    assert _record(env, result) is not None and env.objects[env.target]
    if failure in {"owner", "identity"}:
        assert not any(args[0] == "delete" for _, args in env.calls)


def test_native_stop_is_idempotent_for_absent_controller(lifecycle_env):
    env = lifecycle_env
    result = run(env.options, sctx=env.sctx)
    env.objects[env.target].clear()
    stopped = stop(cluster_id=result.cluster_id, sctx=env.sctx)
    assert stopped.success and stopped.containers_removed == 0
    assert _record(env, result) is None


def test_native_status_ensure_and_replacement_share_portable_identity(lifecycle_env):
    env = lifecycle_env
    first = run(env.options, sctx=env.sctx)
    assert first.rc == 0
    status = api.status(["localhost"], executor="k8s", sctx=env.sctx)
    assert status.running_cluster_ids() == (first.cluster_id,) and not status.errors
    reused = run(replace(env.options, ensure=True), sctx=env.sctx)
    assert reused.already_running and reused.cluster_id == first.cluster_id and reused.timeline is None
    assert reused.recipe_fingerprint == first.recipe_fingerprint
    assert sum(args[0] == "apply" for _, args in env.calls) == 1
    second_plan = plan(env.options, sctx=env.sctx)
    # Deterministic placement reuses the ID; native replacement must still
    # delete its controller before submitting the replacement.
    assert second_plan.cluster_id == first.cluster_id
    replacement = run(env.options, sctx=env.sctx, plan=second_plan)
    assert replacement.rc == 0 and not replacement.already_running
    operations = [args[0] for _, args in env.calls if args[0] in {"apply", "delete"}]
    assert operations == ["apply", "delete", "apply"]
    assert len(env.objects[env.target]) == 1


def test_native_replacement_aborts_when_old_workload_cannot_stop(lifecycle_env):
    env = lifecycle_env
    first = run(env.options, sctx=env.sctx)
    env.fail_delete = True
    with pytest.raises(api.SparkrunError, match="not confirmed"):
        run(env.options, sctx=env.sctx)
    assert sum(args[0] == "apply" for _, args in env.calls) == 1
    assert _record(env, first) is not None


def test_native_benchmark_cleanup_and_lost_metadata_recovery(lifecycle_env):
    from sparkrun.api._benchmark import _stop_inference

    env = lifecycle_env
    result = run(env.options, sctx=env.sctx)
    _stop_inference(list(result.host_list), result.cluster_id, False, sctx=env.sctx, strict=True)
    assert not env.objects[env.target] and _record(env, result) is None
    result = run(env.options, sctx=env.sctx)
    remove_job_metadata(result.cluster_id, cache_dir=str(env.sctx.config.cache_dir))
    cluster = ClusterDefinition(name="k8s", hosts=["localhost"], executor="k8s")
    stopped = stop(cluster_id=result.cluster_id, hosts=result.host_list, cluster=cluster, sctx=env.sctx)
    assert stopped.success and stopped.containers_removed == 1 and not env.objects[env.target]


def test_ambiguous_submission_remains_stoppable(lifecycle_env):
    env = lifecycle_env
    planned = plan(env.options, sctx=env.sctx)
    env.fail_apply = True
    with pytest.raises(KeyboardInterrupt):
        run(env.options, sctx=env.sctx, plan=planned)
    assert env.objects[env.target]
    assert stop(cluster_id=planned.cluster_id, sctx=env.sctx).success
    assert not env.objects[env.target]


def test_native_replacement_handles_new_placement_id(lifecycle_env):
    from sparkrun.orchestration.job_metadata import generate_cluster_id

    env = lifecycle_env
    first = run(env.options, sctx=env.sctx)
    planned = plan(env.options, sctx=env.sctx)
    planned = replace(planned, placement_token="c" * 12, cluster_id=generate_cluster_id(planned.intent_id, "c" * 12))
    second = run(env.options, sctx=env.sctx, plan=planned)
    assert first.cluster_id != second.cluster_id
    assert _record(env, first) is None and _record(env, second) is not None
    assert len(env.objects[env.target]) == 1


def test_native_status_filters_other_application_and_host_scope(lifecycle_env):
    from copy import deepcopy

    env = lifecycle_env
    launched = run(env.options, sctx=env.sctx)
    resource = env.objects[env.target][launched.metadata["k8s_jobset"]]
    foreign = deepcopy(resource)
    foreign["metadata"]["name"] = "foreign-app"
    foreign["metadata"]["labels"]["sparkrun.distribution"] = "jetsonrun"
    env.objects[env.target]["foreign-app"] = foreign
    disjoint = deepcopy(resource)
    disjoint["metadata"]["name"] = "disjoint"
    disjoint["metadata"]["annotations"]["sparkrun.hosts"] = json.dumps(["another-control-host"])
    env.objects[env.target]["disjoint"] = disjoint
    snapshot = api.status(["localhost"], executor="k8s", sctx=env.sctx)
    assert not snapshot.errors and len(snapshot.hosts[0].workloads) == 1
    assert snapshot.running_cluster_ids() == (launched.cluster_id,)


def test_native_status_failure_is_not_reported_as_empty_cluster(lifecycle_env, monkeypatch):
    from sparkrun.plugins.k8s.orchestration.errors import K8sError

    env = lifecycle_env
    monkeypatch.setattr(KubectlClient, "run_json", lambda *a, **kw: (_ for _ in ()).throw(K8sError("connection refused")))
    snapshot = api.status(["localhost"], executor="k8s", sctx=env.sctx)
    assert not snapshot.hosts and snapshot.errors == {"localhost": "connection refused"}
    with pytest.raises(api.SparkrunError, match="connection refused"):
        run(env.options, sctx=env.sctx)
    assert not any(args[0] == "apply" for _, args in env.calls)


def test_common_logs_resolve_saved_native_destination_without_guessing_pod_names(lifecycle_env, monkeypatch):
    env = lifecycle_env
    result = run(env.options, sctx=env.sctx)
    objects = env.objects[env.target]
    objects[result.metadata["k8s_jobset"]]["metadata"]["uid"] = "submitted-jobset"
    env.sctx.config.set("k8s", {"kubeconfig": "/different", "context": "changed", "namespace": "changed"})
    seen = []

    def query(client, args, **kwargs):
        seen.append((client.kubeconfig, client.context, client.namespace))
        if args[1] == "jobset":
            return client.run_json(args)
        assert args[1] in {"jobs", "pods"}
        return {"items": []}  # Controller submitted; no readable containers yet.

    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.logs._query_json", query)
    with pytest.raises(api.SparkrunError, match="pending, queued"):
        api.logs(workload_id=result.cluster_id, sctx=env.sctx)
    assert seen and all(destination == env.target for destination in seen)
    assert _record(env, result) is not None
    assert not any("logs" in args for _, args in env.calls)


@pytest.mark.parametrize("lost_metadata", [False, True])
@pytest.mark.parametrize("fail_delete", [False, True])
def test_native_stop_all_uses_controller_teardown(lifecycle_env, lost_metadata, fail_delete):
    env = lifecycle_env
    launched = run(env.options, sctx=env.sctx)
    if lost_metadata:
        remove_job_metadata(launched.cluster_id, cache_dir=str(env.sctx.config.cache_dir))
    env.fail_delete = fail_delete
    cluster = ClusterDefinition(name="native", hosts=["localhost"], executor="k8s")
    preview = api.stop_all(["localhost"], cluster=cluster, dry_run=True, sctx=env.sctx)
    assert preview.jobs_stopped == 1 and env.objects[env.target]
    assert not any(args[0] == "delete" for _, args in env.calls)
    stopped = api.stop_all(["localhost"], cluster=cluster, sctx=env.sctx)
    assert stopped.success is not fail_delete
    assert stopped.jobs_stopped == int(not fail_delete)
    assert stopped.containers_removed == int(not fail_delete)
    assert bool(env.objects[env.target]) is fail_delete
    if not lost_metadata:
        assert (_record(env, launched) is not None) is fail_delete


@pytest.mark.parametrize("count", [-1, True, "1"])
def test_invalid_native_stop_receipt_preserves_metadata(lifecycle_env, monkeypatch, count):
    env = lifecycle_env
    result = run(env.options, sctx=env.sctx)
    monkeypatch.setattr("sparkrun.plugins.k8s.executor.K8sExecutor.stop_workload", lambda *args, **kwargs: count)
    stopped = stop(cluster_id=result.cluster_id, sctx=env.sctx)
    assert not stopped.success and stopped.errors
    assert _record(env, result) is not None


def test_stop_prepares_provider_transport_before_native_teardown(lifecycle_env, monkeypatch):
    from sparkrun.plugins.k8s.executor import K8sExecutor

    env = lifecycle_env
    result = run(env.options, sctx=env.sctx)
    events = []
    monkeypatch.setattr("sparkrun.api._resolve.prepare_transport", lambda *args, **kwargs: events.append("transport"))
    original = K8sExecutor.stop_workload

    def stop_workload(self, *args, **kwargs):
        assert events == ["transport"]
        return original(self, *args, **kwargs)

    monkeypatch.setattr(K8sExecutor, "stop_workload", stop_workload)
    assert stop(cluster_id=result.cluster_id, sctx=env.sctx).success
