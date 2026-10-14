"""Public seams that replace first-party plugins' internal imports."""

from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

import sparkrun.api as api
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload
from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.job_metadata import parse_cluster_id

INTENT = "a" * 16
WORKLOAD = "sparkrun_%s_%s" % (INTENT, "1" * 12)


def _recipe(**extra):
    return Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m", **extra})


def _plan(**fields):
    values = dict(
        recipe=_recipe(),
        runtime=NS(),
        cluster=ClusterDefinition(name="lab", hosts=["h1", "h2", "h3"]),
        candidate_hosts=("h1", "h2", "h3"),
        host_list=("h1",),
        is_solo=True,
        placement=None,
        intent_id=INTENT,
        placement_token="1" * 12,
        workload_id=WORKLOAD,
    )
    values.update(fields)
    return api.RunPlan(**values)


# -- Recipe.from_state ---------------------------------------------------------


def test_recipe_from_state_round_trips_without_re_resolving():
    recipe = _recipe(defaults={"port": 8123})
    rebuilt = Recipe.from_state(recipe.__getstate__())
    assert isinstance(rebuilt, Recipe)
    assert (rebuilt.model, rebuilt.runtime, rebuilt.defaults["port"]) == ("test/m", "vllm-distributed", 8123)


def test_recipe_from_state_rejects_non_mapping():
    with pytest.raises(TypeError):
        Recipe.from_state("model: test/m")  # type: ignore[arg-type]


# -- child_environment -----------------------------------------------------------


def test_child_environment_is_public_and_is_the_core_definition():
    import sparkrun.application as application
    from sparkrun.core import application_profile

    assert "child_environment" in application.__all__
    assert application.child_environment is application_profile.child_environment
    environment = application.child_environment(config_path="/tmp/sparkrun.yaml")
    assert environment["SPARKRUN_APPLICATION_CONFIG"] == "/tmp/sparkrun.yaml"
    assert environment["SPARKRUN_APPLICATION_PROFILE"]


# -- recipe_protocols ----------------------------------------------------------


def test_recipe_protocols_resolves_runtime_only(v):
    result = api.recipe_protocols(_recipe(), sctx=api.default_sctx())
    assert isinstance(result, api.RecipeProtocols)
    assert result.protocols[0] == "openai"
    assert isinstance(result.capabilities, tuple)


def test_recipe_protocols_reports_declared_native_apis(v):
    recipe = _recipe(metadata={"native_apis": ["chat_completions", "responses", "messages"]})
    result = api.recipe_protocols(recipe, sctx=api.default_sctx())
    assert result.protocols == ("openai", "anthropic")
    assert result.capabilities == ("responses",)


def test_recipe_protocols_falls_back_to_openai_when_runtime_claims_nothing():
    from sparkrun.api._protocols import runtime_protocols

    runtime = NS(native_protocols=lambda recipe: [], native_capabilities=lambda recipe: [])
    assert runtime_protocols(runtime, _recipe()) == api.RecipeProtocols(protocols=("openai",), capabilities=())  # type: ignore[arg-type]


def test_recipe_protocols_unknown_runtime_is_a_sparkrun_error(v):
    recipe = _recipe()
    recipe.runtime = "no-such-runtime"
    with pytest.raises(api.SparkrunError):
        api.recipe_protocols(recipe, sctx=api.default_sctx())


# -- fresh_workload_id -----------------------------------------------------------


def test_fresh_workload_id_reissues_a_consistent_canonical_identity():
    plan = _plan()
    fresh = api.fresh_workload_id(plan)
    assert fresh.workload_id != plan.workload_id
    assert parse_cluster_id(fresh.workload_id) == (INTENT, fresh.placement_token)
    assert (fresh.host_list, fresh.candidate_hosts, fresh.intent_id) == (plan.host_list, plan.candidate_hosts, plan.intent_id)
    assert plan.workload_id == WORKLOAD  # the original is untouched
    assert api.fresh_workload_id(plan).workload_id != fresh.workload_id


def test_fresh_workload_id_needs_a_canonical_intent():
    with pytest.raises(ValueError):
        api.fresh_workload_id(_plan(intent_id="not-an-intent"))


# -- replace_intent_deployments ----------------------------------------------------


def _status(*rows, errors=None):
    return ClusterStatus(
        hosts=tuple(HostOccupancy(host=host, workloads=tuple(RunningWorkload(cluster_id=cid) for cid in cids)) for host, cids in rows),
        executor="docker",
        errors=dict(errors or {}),
    )


class _Stopped:
    def __init__(self, failed=()):
        self.hosts_failed = tuple(failed)


def _replace(status, *, stop_result=None, **kwargs):
    calls = []

    def fake_stop(**kw):
        calls.append(kw)
        return stop_result or _Stopped()

    with (
        patch("sparkrun.orchestration.executor.query_status_for_cluster", return_value=status) as query,
        patch.object(api, "stop", fake_stop),
    ):
        evicted = api.replace_intent_deployments(_plan(), sctx=api.default_sctx(), **kwargs)
    return evicted, calls, query


def test_replace_intent_deployments_includes_the_plans_own_id_across_its_hosts(v):
    other = "sparkrun_%s_%s" % (INTENT, "2" * 12)
    foreign = "sparkrun_%s_%s" % ("b" * 16, "3" * 12)
    elsewhere = "sparkrun_%s_%s" % (INTENT, "4" * 12)
    status = _status(("h1", [WORKLOAD, other, foreign]), ("h2", [other]), ("h3", [elsewhere]))
    evicted, calls, query = _replace(status)
    assert evicted == (WORKLOAD, other)
    assert {c["workload_id"]: c["hosts"] for c in calls} == {WORKLOAD: ["h1"], other: ["h1", "h2"]}
    assert query.call_args.args[1] == ["h1", "h2", "h3"]


def test_replace_intent_deployments_can_leave_the_current_id(v):
    evicted, _calls, _ = _replace(_status(("h1", [WORKLOAD])), include_current=False)
    assert evicted == ()


def test_replace_intent_deployments_strict_refuses_incomplete_status(v):
    with pytest.raises(api.SparkrunError, match="could not query"):
        _replace(_status(("h1", [WORKLOAD]), errors={"h2": "unreachable"}))


def test_replace_intent_deployments_strict_refuses_unconfirmed_teardown(v):
    with pytest.raises(api.SparkrunError, match="not confirmed"):
        _replace(_status(("h1", [WORKLOAD])), stop_result=_Stopped(failed=["h1"]))


def test_replace_intent_deployments_lenient_reports_and_continues(v):
    evicted, _calls, _ = _replace(_status(("h1", [WORKLOAD]), errors={"h2": "unreachable"}), strict=False)
    assert evicted == (WORKLOAD,)


# -- wait_ready(workload_id=) ------------------------------------------------------


def test_wait_ready_requires_exactly_one_target():
    with pytest.raises(ValueError, match="exactly one"):
        api.wait_ready()
    with pytest.raises(ValueError, match="exactly one"):
        api.wait_ready(NS(launch_result=NS()), workload_id=WORKLOAD)


def test_wait_ready_unknown_workload_is_job_not_found(v):
    with pytest.raises(api.JobNotFound):
        api.wait_ready(workload_id=WORKLOAD, sctx=api.default_sctx())


def test_wait_ready_for_recorded_workload_uses_its_metadata(v, monkeypatch):
    from sparkrun.orchestration.job_metadata import save_job_metadata

    sctx = api.default_sctx()
    recipe = _recipe(defaults={"port": 8123})
    save_job_metadata(WORKLOAD, recipe, ["h1", "h2"], cache_dir=str(sctx.config.cache_dir), ssh_user="alice")
    seen = {}

    def fake_wait(**kwargs):
        seen.update(kwargs)
        return NS(ready=True)

    monkeypatch.setattr("sparkrun.core.launcher.wait_for_endpoint_ready", fake_wait)
    monkeypatch.setattr("sparkrun.api._resolve.prepare_transport", lambda cluster, **kw: None)
    shared = sctx.variables

    assert api.wait_ready(workload_id=WORKLOAD, timeout=5, sctx=sctx).ready
    assert seen["cluster_id"] == WORKLOAD
    assert seen["host_list"] == ["h1", "h2"] and seen["is_solo"] is False
    assert seen["port"] == 8123
    assert seen["ssh_kwargs"]["ssh_user"] == "alice"
    assert seen["runtime"].runtime_name == "vllm-distributed"
    assert seen["runtime"].executor is not None
    from sparkrun.api._lifecycle import _ReadinessCancellation

    assert isinstance(seen["cancel"], _ReadinessCancellation)
    # The shared runtime plugin instance is not rebound to this job's executor.
    from sparkrun.core.bootstrap import get_runtime

    assert get_runtime("vllm-distributed", shared) is not seen["runtime"]


# -- discovery_snapshot(metadata_only=) -----------------------------------------------


@pytest.fixture
def snapshot_calls(monkeypatch):
    from sparkrun.proxy.discovery import DiscoveredEndpoint, DiscoverySnapshot

    calls = []
    endpoint = DiscoveredEndpoint(
        cluster_id=WORKLOAD,
        model="test/m",
        served_model_name=None,
        runtime="vllm-distributed",
        host="h1",
        port=8000,
        healthy=True,
        actual_models=["test/m"],
        recipe_name="@reg/my-recipe",
    )

    def fake(**kwargs):
        calls.append(kwargs)
        return DiscoverySnapshot(endpoints=[endpoint], hosts=list(kwargs.get("host_list") or ()))

    monkeypatch.setattr("sparkrun.proxy.discovery.discover_snapshot", fake)
    monkeypatch.setattr("sparkrun.api._resolve.prepare_transport", lambda cluster, **kw: pytest.fail("no transport in metadata mode"))
    return calls


def test_discovery_snapshot_metadata_only_skips_the_default_cluster(v, snapshot_calls):
    sctx = api.default_sctx()
    sctx.cluster_manager.create(name="lab", hosts=["h1", "h2"])
    sctx.cluster_manager.set_default("lab")
    value = api.discovery_snapshot(metadata_only=True, workload_ids={WORKLOAD}, sctx=sctx)
    call = snapshot_calls[0]
    assert call["host_list"] == [] and call["host_filter"] is None and call["cluster_def"] is None
    assert call["cluster_ids"] == {WORKLOAD}
    assert value["mode"] == "metadata"
    assert value["endpoints"][0]["recipe_name"] == "@reg/my-recipe"


def test_discovery_snapshot_metadata_only_refuses_a_target(v):
    with pytest.raises(ValueError):
        api.discovery_snapshot(metadata_only=True, hosts=["h1"])
    with pytest.raises(ValueError):
        api.discovery_snapshot(metadata_only=True, cluster="lab")


def test_discovery_snapshot_recipe_name_validates_against_the_wire_schema(v, snapshot_calls):
    from sparkrun.api._wire import validate
    from sparkrun.api._wire_schemas import RESULT_SCHEMAS

    value = api.discovery_snapshot(metadata_only=True, sctx=api.default_sctx())
    validate(value, RESULT_SCHEMAS["discovery.snapshot"])
    workload_endpoint = RESULT_SCHEMAS["workloads.ensure_ready"]["properties"]["endpoint"]
    assert "recipe_name" not in workload_endpoint["properties"]


def test_discovery_snapshot_metadata_only_is_a_daemon_argument():
    from sparkrun.core.daemon_registry import get_daemon_op
    from sparkrun.daemon.ops import check_request, register_core_ops

    register_core_ops()
    assert check_request(get_daemon_op("discovery.snapshot"), {"metadata_only": True}) == {"metadata_only": True}
