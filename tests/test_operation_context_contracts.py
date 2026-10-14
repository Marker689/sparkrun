"""Effective targeting and operation-local connection contracts."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from test_benchmark_startup_collection import bench_env as bench_env


def _plan(sctx, cluster, *, recipe=None):
    from sparkrun.api import RunPlan
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.bootstrap import get_runtime

    recipe = recipe or Recipe({"name": "review", "model": "test/model", "runtime": "vllm-distributed", "container": "test/image"})
    return RunPlan(
        recipe=recipe,
        runtime=get_runtime(recipe.runtime, sctx.variables),
        cluster=cluster,
        candidate_hosts=("localhost",),
        host_list=("localhost",),
        is_solo=True,
        placement=None,
        workload_id="review",
        scheduler="greedy",
    )


@pytest.mark.parametrize("source", ["cluster", "recipe"])
@pytest.mark.parametrize("caller", [None, {"k8s_namespace": "caller-ns"}])
def test_k8s_handler_preserves_resolved_target(monkeypatch, source, caller):
    from sparkrun.api import run, RunOptions
    from sparkrun.api._context import default_sctx
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.plugins.k8s import run as handler_module
    from sparkrun.plugins.k8s.executor import K8sExecutor
    from sparkrun.orchestration.executor import resolve_executor

    monkeypatch.setenv("SPARKRUN_FEATURE_INTEGRATION_K8S", "1")
    sctx = default_sctx()
    cluster = ClusterDefinition(name="lab", hosts=["localhost"], executor="k8s")
    plan = _plan(sctx, cluster)
    target = {"kubeconfig": "/tmp/review-kubeconfig", "k8s_context": "lab-context", "k8s_namespace": "lab-namespace"}
    if source == "cluster":
        cluster.executor_config = target
    else:
        plan.recipe.executor_config = target
    monkeypatch.setattr(K8sExecutor, "finalize_config", lambda *a, **kw: None)
    expected = resolve_executor(recipe=plan.recipe, cluster=cluster, runtime=plan.runtime, config=sctx.config, v=sctx.variables).config
    assert expected.k8s_context == "lab-context"
    probe = Mock(return_value="gb10")
    submit = Mock(return_value=SimpleNamespace(name="review", feasible=True))
    monkeypatch.setattr(handler_module, "_resolve_single_gpu_class", probe)
    monkeypatch.setattr("sparkrun.plugins.k8s.api._ops._launch_jobset", submit)
    run(RunOptions(recipe=plan.recipe, cluster=cluster, solo=True, dry_run=True, executor_config=caller), sctx=sctx, plan=plan)
    client = submit.call_args.args[0]
    assert client.context == "lab-context"
    assert client.namespace == submit.call_args.kwargs["namespace"] == ("caller-ns" if caller else "lab-namespace")
    assert client.kubeconfig == target["kubeconfig"]
    assert probe.call_args.args == (client,)


@pytest.mark.parametrize("failure", [PermissionError("Cannot read control-plane credentials"), KeyboardInterrupt(), SystemExit(2)])
def test_run_handler_preserves_api_error_boundary(monkeypatch, failure):
    from sparkrun.api import run, RunOptions, SparkrunError
    from sparkrun.api._context import default_sctx
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.run_handlers import RunHandler

    sctx = default_sctx()
    plan = _plan(sctx, ClusterDefinition(name="lab", hosts=["localhost"], executor="docker"))
    callback = Mock(side_effect=failure)
    monkeypatch.setattr("sparkrun.core.run_handlers.registered_run_handlers", lambda cfg: {"docker": RunHandler("docker", callback)})
    with pytest.raises(SparkrunError if isinstance(failure, Exception) else type(failure)) as caught:
        run(RunOptions(recipe=plan.recipe, solo=True, dry_run=True), sctx=sctx, plan=plan)
    if isinstance(failure, Exception):
        assert caught.value.__cause__ is failure
        assert "docker" in str(caught.value)
    else:
        assert caught.value is failure


@pytest.mark.parametrize("reverse,fail_first", [(False, False), (True, False), (False, True)])
def test_shared_context_isolates_cluster_users(monkeypatch, reverse, fail_first):
    from sparkrun.api import plan, RunOptions
    from sparkrun.api._context import default_sctx
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.recipe import Recipe

    sctx = default_sctx()
    sctx.config._data["ssh"] = {"user": "configured-user"}
    original = sctx.config.ssh_user
    assert original == "configured-user"
    observed = []

    def hosts(hosts, *a, **kw):
        observed.append(kw["sctx"].config.ssh_user)
        if fail_first and len(observed) == 1:
            raise RuntimeError("planning failed")
        return hosts, True, [], None

    monkeypatch.setattr("sparkrun.api._hosts.resolve_effective_hosts", hosts)
    recipe = Recipe({"name": "review", "model": "test/model", "runtime": "vllm-distributed", "container": "test/image"})
    clusters = [
        ClusterDefinition(name="first", hosts=["localhost"], user="cluster-user"),
        ClusterDefinition(name="second", hosts=["localhost"]),
    ]
    if reverse:
        clusters.reverse()
    for index, cluster in enumerate(clusters):
        if fail_first and index == 0:
            with pytest.raises(RuntimeError, match="planning failed"):
                plan(RunOptions(recipe=recipe, cluster=cluster, solo=True), sctx=sctx)
        else:
            plan(RunOptions(recipe=recipe, cluster=cluster, solo=True), sctx=sctx)
        assert sctx.config.ssh_user == original
    assert observed == (["configured-user", "cluster-user"] if reverse else ["cluster-user", "configured-user"])


def test_supplied_plan_and_nested_context_preserve_config(monkeypatch):
    from sparkrun.api import run, RunOptions, SparkrunError
    from sparkrun.api._context import default_sctx
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.run_handlers import RunHandler

    sctx = default_sctx()
    sctx.config.ssh_user = "caller-default"
    first = ClusterDefinition(name="first", hosts=["localhost"], user="first-user", executor="docker")
    other = ClusterDefinition(name="other", hosts=["localhost"])
    scoped = sctx.for_cluster(first)
    assert scoped.config.ssh_user == "first-user"
    assert scoped.for_cluster(other).config.ssh_user == "caller-default"
    assert scoped.variables is sctx.variables
    typed_error = SparkrunError("already translated")
    observed = []

    def launch(options, current, **kwargs):
        observed.append(current.config.ssh_user)
        raise typed_error

    monkeypatch.setattr("sparkrun.core.run_handlers.registered_run_handlers", lambda _: {"docker": RunHandler("docker", launch)})
    with pytest.raises(SparkrunError) as caught:
        run(RunOptions(recipe="unused", dry_run=True), sctx=sctx, plan=_plan(sctx, first))
    assert caught.value is typed_error
    assert observed == ["first-user"]
    assert sctx.config.ssh_user == "caller-default"


def test_benchmark_cluster_user_is_local_to_invocation(bench_env):
    from sparkrun.api import benchmark
    from sparkrun.core.cluster_manager import ClusterDefinition

    env = bench_env
    env.sctx.config.ssh_user = "configured-user"
    for user in ("benchmark-user", None):
        cluster = ClusterDefinition(name="inline", hosts=["localhost"], user=user)
        assert benchmark(replace(env.options, cluster=cluster), sctx=env.sctx).success
        assert env.run.call_args.kwargs["sctx"].config.ssh_user == (user or "configured-user")
        assert env.sctx.config.ssh_user == "configured-user"


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("failure", [None, "validation", "staging"])
def test_handler_controls_replacement_after_preparation(monkeypatch, dry_run, failure):
    from sparkrun.api import run, RunOptions, RunResult, SparkrunError
    from sparkrun.api._context import default_sctx
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.run_handlers import RunHandler

    sctx = default_sctx()
    plan = _plan(sctx, ClusterDefinition(name="lab", hosts=["localhost"], executor="docker"))
    calls = []
    replacement = Mock(side_effect=lambda **kw: (calls.append("replace"), set()))
    monkeypatch.setattr("sparkrun.api._run._evict_superseded_deployments", replacement)
    expected = RunResult(
        workload_id=plan.workload_id,
        host_list=plan.host_list,
        placement=plan.placement,
        scheduler=plan.scheduler,
        runtime=plan.runtime.runtime_name,
        executor="docker",
        started_at=0,
        dry_run=dry_run,
        is_solo=plan.is_solo,
    )

    def launch(options, current, *, plan, started_at, before_start):
        for phase in ("validation", "staging"):
            calls.append(phase)
            if failure == phase:
                raise ValueError(phase + " failed")
        if dry_run:
            assert before_start is None
        else:
            before_start()
            before_start()  # Replacement is owned once by core, even on repeated calls.
            calls.append("submit")
        return expected

    monkeypatch.setattr("sparkrun.core.run_handlers.registered_run_handlers", lambda _: {"docker": RunHandler("docker", launch)})
    options = RunOptions(recipe=plan.recipe, solo=True, dry_run=dry_run)
    if failure:
        with pytest.raises(SparkrunError, match=failure + " failed"):
            run(options, sctx=sctx, plan=plan)
        replacement.assert_not_called()
        assert "submit" not in calls
    else:
        result = run(options, sctx=sctx, plan=plan)
        assert result.workload_id == expected.workload_id and result.host_list == expected.host_list
        assert result.timeline is not None
        assert calls == (["validation", "staging"] if dry_run else ["validation", "staging", "replace", "submit"])
        assert replacement.call_count == (0 if dry_run else 1)


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("failure", ["topology", "configuration"])
def test_k8s_invalid_launch_keeps_previous_deployment(monkeypatch, dry_run, failure):
    from sparkrun.api import run, RunOptions, SparkrunError
    from sparkrun.api._context import default_sctx
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.run_handlers import RunHandler
    from sparkrun.plugins.k8s.run import run_k8s

    sctx = default_sctx()
    cluster = ClusterDefinition(name="lab", hosts=["localhost", "second"], executor="k8s")
    intent = "a" * 16
    old_id = "sparkrun_%s_%s" % (intent, "c" * 12)
    plan = replace(
        _plan(sctx, cluster),
        candidate_hosts=tuple(cluster.hosts),
        host_list=tuple(cluster.hosts),
        is_solo=failure != "topology",
        intent_id=intent,
        placement_token="b" * 12,
        workload_id="sparkrun_%s_%s" % (intent, "b" * 12),
    )
    monkeypatch.setattr("sparkrun.core.run_handlers.registered_run_handlers", lambda _: {"k8s": RunHandler("k8s", run_k8s)})
    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor_name", lambda **kw: "k8s")
    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", Mock(side_effect=ValueError("invalid kubeconfig")))
    monkeypatch.setattr(
        "sparkrun.orchestration.executor.query_status_for_cluster",
        Mock(
            return_value=SimpleNamespace(
                hosts=[SimpleNamespace(host="localhost", workloads=[SimpleNamespace(cluster_id=old_id, intent_id=intent)])]
            )
        ),
    )
    stop, submit = Mock(return_value=SimpleNamespace(hosts_failed=[])), Mock()
    monkeypatch.setattr("sparkrun.api.stop", stop)
    monkeypatch.setattr("sparkrun.plugins.k8s.api._ops._launch_jobset", submit)
    with pytest.raises(SparkrunError, match="single-pod" if failure == "topology" else "invalid kubeconfig"):
        run(RunOptions(recipe=plan.recipe, cluster=cluster, dry_run=dry_run), sctx=sctx, plan=plan)
    stop.assert_not_called()
    submit.assert_not_called()
