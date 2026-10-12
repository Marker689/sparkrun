"""Generic activation safety without a gateway plugin or real remote machines."""

from types import SimpleNamespace as NS
from unittest.mock import Mock
import threading
import time

import pytest

from sparkrun import api
from sparkrun.api import workloads as w
from sparkrun.core.operation import OperationState


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    ctx = NS(config=NS(config_path=tmp_path / "config.yaml"), timing=None, progress=None, operation=OperationState())
    recipe = NS(post_exec=[], post_commands=[], defaults={"port": 8000})
    binding = w.WorkloadBinding("recipe", "revision", ("lab",))
    job = NS(cluster_id="work", hosts=["one"], metadata={"recipe_fingerprint": "revision", "owner": "gateway", "cluster": "lab"})
    monkeypatch.setattr(w, "resolve_binding", lambda *a, **kw: (recipe, "revision", {}))
    monkeypatch.setattr(api, "list_jobs", lambda **kw: [])
    monkeypatch.setattr(w, "_ready", lambda *a, **kw: None)
    monkeypatch.setattr(api, "plan", Mock(return_value=NS(cluster_id="work", host_list=["one"], cluster=NS(name="lab"), recipe=recipe)))
    monkeypatch.setattr(api, "wait_ready", Mock(return_value=NS(ready=True)))
    monkeypatch.setattr(api, "post_launch", Mock(return_value={"stopped": False}))
    return ctx, recipe, binding, job


def endpoint(owner="gateway"):
    return {"cluster_id": "work", "owner": owner, "observed_at": 123.5, "actual_models": ["served"]}


def test_healthy_external_adoption_never_launches_or_takes_ownership(workflow, monkeypatch):
    ctx, _, binding, _ = workflow
    monkeypatch.setattr(w, "_ready", lambda *a, **kw: endpoint("external"))
    run = Mock()
    monkeypatch.setattr(api, "run", run)
    result = w.ensure_ready(binding, owner="gateway", sctx=ctx)
    assert result["adopted"] and result["endpoint"]["owner"] == "external"
    assert result["observed_at"] == 123.5
    run.assert_not_called()


def test_wrong_owner_and_revision_cannot_stop(workflow, monkeypatch):
    ctx, _, binding, job = workflow
    monkeypatch.setattr(api, "list_jobs", lambda **kw: [job])
    stop = Mock()
    monkeypatch.setattr(api, "stop", stop)
    with pytest.raises(w.WorkloadError, match="not owned"):
        w.stop(binding, "work", owner="someone-else", sctx=ctx)
    with pytest.raises(w.WorkloadError, match="does not match"):
        w.stop(w.WorkloadBinding("recipe", "different"), "work", owner="gateway", sctx=ctx)
    stop.assert_not_called()


def test_placement_record_precedes_launch_and_survives_interruption(workflow, monkeypatch):
    ctx, _, binding, _ = workflow

    def launch(*args, **kwargs):
        rows = w.Receipts(ctx).placements("revision", "gateway")
        assert rows[0]["placement"]["cluster_id"] == "work"
        assert rows[0]["phase"] == "launch_started"
        raise InterruptedError("lost caller")

    monkeypatch.setattr(api, "run", launch)
    with pytest.raises(InterruptedError):
        w.ensure_ready(binding, owner="gateway", sctx=ctx)
    assert w.Receipts(ctx).placements("revision", "gateway")[0]["phase"] == "launch_started"


def test_readiness_and_hooks_share_launch_context_and_actual_timestamp(workflow, monkeypatch):
    ctx, recipe, binding, _ = workflow
    recipe.post_exec = ["effect"]
    result = NS(rc=0, recipe_fingerprint="revision", cluster_id="work", launch_result=object())
    monkeypatch.setattr(api, "run", Mock(return_value=result))
    calls = 0

    def ready(*args, **kwargs):
        nonlocal calls
        calls += 1
        return endpoint() if calls > 1 else None

    monkeypatch.setattr(w, "_ready", ready)

    def post(launch, **kw):
        assert launch is result and kw["sctx"] is ctx and kw["trust"] is False
        assert w.Receipts(ctx).placements("revision", "gateway")[0]["phase"] == "hooks_started"

    monkeypatch.setattr(api, "post_launch", post)
    assert not w.ensure_ready(binding, owner="gateway", sctx=ctx)["adopted"]
    assert w.Receipts(ctx).placements("revision", "gateway")[0]["phase"] == "ready"
    assert api.wait_ready.call_args.kwargs["timeout"] <= 900


def test_uncertain_hooks_block_even_healthy_adoption_and_never_replay(workflow, monkeypatch):
    ctx, _, binding, _ = workflow
    w.Receipts(ctx).save("old", "revision", "gateway", {"cluster_id": "work", "cluster": "lab", "hosts": ["one"]}, "hooks_started")
    monkeypatch.setattr(w, "_ready", lambda *a, **kw: endpoint())
    with pytest.raises(w.WorkloadError, match="uncertain effects"):
        w.ensure_ready(binding, owner="gateway", sctx=ctx)
    api.post_launch.assert_not_called()


def test_partial_recovery_never_launches_another_workload(workflow, monkeypatch):
    ctx, _, binding, _ = workflow
    w.Receipts(ctx).save("old", "revision", "gateway", {"cluster_id": "work", "cluster": "lab", "hosts": ["one"]}, "readiness")
    monkeypatch.setattr(api, "status", lambda *a, **kw: NS(observation_errors={"one": "unreachable"}, hosts=[]))
    with pytest.raises(w.WorkloadError, match="completely observed"):
        w.ensure_ready(binding, owner="gateway", sctx=ctx)
    api.plan.assert_not_called()


def test_deadline_does_not_reset_after_launch(workflow, monkeypatch):
    ctx, _, binding, _ = workflow
    clock = [100.0]
    monkeypatch.setattr(w, "time", NS(monotonic=lambda: clock[0], time=time.time))

    def launch(*args, **kwargs):
        clock[0] += 1
        return NS(rc=0, cluster_id="work", recipe_fingerprint="revision", launch_result=object())

    monkeypatch.setattr(api, "run", launch)
    with pytest.raises(w.WorkloadError, match="readiness budget"):
        w.ensure_ready(binding, owner="gateway", timeout=0.005, sctx=ctx)
    api.wait_ready.assert_not_called()


def test_provider_must_match_exact_returned_identity(workflow, monkeypatch):
    ctx, _, binding, job = workflow
    monkeypatch.setattr(api, "list_jobs", lambda **kw: [job])
    monkeypatch.setattr(
        w, "lifecycle_provider", lambda job: NS(name="fixture", control=lambda *a, **kw: {"cluster_id": "other", "state": "running"})
    )
    with pytest.raises(w.WorkloadError, match="different or invalid"):
        w.control(binding, "work", "wake", owner="gateway", sctx=ctx)
    assert w.Receipts(ctx).state("work") == "waking"


def test_receipt_reconciliation_requires_ack_and_complete_observation(workflow, monkeypatch):
    ctx, _, binding, _ = workflow
    journal = w.Receipts(ctx)
    journal.save("old", "revision", "gateway", {"cluster_id": "work", "cluster": "lab", "hosts": ["one"]}, "hooks_started")
    with pytest.raises(w.WorkloadError, match="acknowledge"):
        w.reconcile(binding, "work", owner="gateway", sctx=ctx)
    monkeypatch.setattr(
        api, "status", lambda *a, **kw: NS(observation_errors={}, hosts=[NS(host="one")], running_cluster_ids=lambda: set())
    )
    assert w.reconcile(binding, "work", owner="gateway", acknowledge_effects=True, sctx=ctx)["state"] == "stopped"
    assert not journal.blocked_endpoints()


def test_provider_registration_requires_loader_identity(monkeypatch):
    from sparkrun.core import workload_lifecycle as lifecycle
    from sparkrun.core.plugin_identity import PluginIdentity, plugin_registration

    monkeypatch.setattr(lifecycle, "_PROVIDERS", {})
    provider = lifecycle.WorkloadLifecycleProvider("fixture", lambda job: True, lambda *a, **kw: {})
    with pytest.raises(ValueError, match="loader-assigned"):
        lifecycle.register_workload_lifecycle(provider)
    with plugin_registration(PluginIdentity("fixture", "tests.fixture", ("external", "fixture", None))):
        lifecycle.register_workload_lifecycle(provider)
    assert lifecycle.lifecycle_provider(object()) is provider


def test_readiness_timeout_does_not_cancel_owner():
    from sparkrun.api._lifecycle import _ReadinessCancellation

    owner = threading.Event()
    timeout = _ReadinessCancellation(owner, 0.001)
    assert timeout.wait(0.1)
    assert not owner.is_set()


@pytest.mark.parametrize("owner,owned,actions", [("gateway", True, ["status", "sleep", "wake"]), ("daemon", False, ["status"])])
def test_readiness_inventory_and_control_share_ownership_dto(workflow, monkeypatch, owner, owned, actions):
    ctx, _, binding, job = workflow
    job.metadata["owner"] = owner
    monkeypatch.setattr(api, "list_jobs", lambda **kw: [job])
    provider = NS(name="coldsnap", control=Mock(return_value={"cluster_id": "work", "state": "running"}))
    monkeypatch.setattr(w, "lifecycle_provider", lambda job: provider)
    monkeypatch.setattr(w, "_ready", lambda *a, **kw: {**endpoint(owner), "recipe_revision": "revision"})
    ready = w.ensure_ready(binding, owner="gateway", sctx=ctx)
    assert ready["adopted"] and ready["workload"]["owned"] is owned
    assert ready["workload"]["actions"] == actions
    status = w.control(binding, "work", "status", owner="gateway", sctx=ctx)
    assert status == w.inventory(owner="gateway", sctx=ctx)["workloads"][0]
    if not owned:
        provider.control.reset_mock()
        for action in ("sleep", "wake"):
            with pytest.raises(w.WorkloadError, match="not owned"):
                w.control(binding, "work", action, owner="gateway", sctx=ctx)
        provider.control.assert_not_called()
    assert job.metadata["owner"] == owner


def test_shared_workload_results_match_published_contracts(workflow, monkeypatch):
    from sparkrun.api._wire import validate
    from sparkrun.api._wire_schemas import RESULT_SCHEMAS
    from sparkrun.proxy.discovery import DiscoveredEndpoint, DiscoverySnapshot

    ctx, _, binding, job = workflow
    monkeypatch.setattr(api, "list_jobs", lambda **kw: [job])
    ep = DiscoveredEndpoint(
        "work", "model", None, "test", "host", 8000, True, actual_models=["model"], owner="gateway", recipe_revision="revision"
    )
    observation = DiscoverySnapshot(endpoints=[ep], observed_at=123.5).to_dict()
    monkeypatch.setattr(w, "_ready", lambda *a, **kw: {**observation["endpoints"][0], "observed_at": 123.5})
    ready = w.ensure_ready(binding, owner="gateway", sctx=ctx)
    validate(ready, RESULT_SCHEMAS["workloads.ensure_ready"])
    validate(w.inventory(owner="gateway", sctx=ctx), RESULT_SCHEMAS["workloads.inventory"])
    provider = NS(name="coldsnap", control=lambda *a, **kw: {"cluster_id": "work", "state": "running"})
    monkeypatch.setattr(w, "lifecycle_provider", lambda job: provider)
    validate(w.control(binding, "work", "status", owner="gateway", sctx=ctx), RESULT_SCHEMAS["workloads.control"])
