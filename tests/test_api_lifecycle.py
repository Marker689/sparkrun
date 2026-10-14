"""Experimental lifecycle helpers: wait_ready, post_launch, discovery_snapshot."""

from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from sparkrun import api
from sparkrun.core.operation import OperationState


def _ctx():
    return NS(config=NS(), timing=None, progress=None, operation=OperationState())


def _launch(*, post_exec=(), recorded=None):
    recipe = NS(post_exec=list(post_exec), post_commands=[])
    return NS(recipe=recipe, recipe_trusted=recorded, effective_cache_dir="/cache", config=NS())


@pytest.fixture
def lifecycle(monkeypatch):
    calls = NS(trust=[], post=[])

    def resolve_trust(recipe, trust, **kwargs):
        calls.trust.append(trust)
        return False

    def complete(launch, cache_dir, **kwargs):
        calls.post.append(kwargs)
        return True

    monkeypatch.setattr("sparkrun.core.launcher.resolve_recipe_trust", resolve_trust)
    monkeypatch.setattr("sparkrun.core.launcher.complete_post_launch", complete)
    return calls


def test_post_launch_without_hooks_needs_no_trust(lifecycle):
    result = api.post_launch(NS(launch_result=_launch(), dry_run=False), sctx=_ctx())
    assert result == api.PostLaunchResult(stopped=True)
    assert lifecycle.trust == []  # nothing to authorize, so nothing resolved
    assert lifecycle.post[0]["trust"] is False


def test_post_launch_refuses_untrusted_hooks(lifecycle):
    with pytest.raises(api.TrustRejected):
        api.post_launch(NS(launch_result=_launch(post_exec=["notify"]), dry_run=False), sctx=_ctx())
    assert lifecycle.post == []


@pytest.mark.parametrize("recorded,trust", [(True, False), (None, True)])
def test_post_launch_reuses_recorded_verdict_or_explicit_trust(lifecycle, recorded, trust):
    launch = _launch(post_exec=["notify"], recorded=recorded)
    api.post_launch(NS(launch_result=launch, dry_run=False), trust=trust, sctx=_ctx())
    assert lifecycle.trust == []  # never re-resolved
    assert lifecycle.post[0]["trust"] is True


def test_post_launch_dry_run_previews_untrusted_hooks(lifecycle):
    api.post_launch(NS(launch_result=_launch(post_exec=["notify"]), dry_run=True), sctx=_ctx())
    assert lifecycle.post[0]["dry_run"] is True and lifecycle.post[0]["trust"] is False


def test_post_launch_requires_a_launch_handle():
    with pytest.raises(ValueError, match="launch result"):
        api.post_launch(NS(launch_result=None), sctx=_ctx())


@pytest.fixture
def readiness(monkeypatch):
    seen = []
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda config: {})
    monkeypatch.setattr("sparkrun.core.launcher.wait_for_serve_ready", lambda launch, **kw: seen.append(kw) or NS(ready=True))
    return seen


@pytest.mark.parametrize("timeout", [None, 0])
def test_wait_ready_zero_or_none_adds_no_caller_budget(readiness, timeout):
    ctx = _ctx()
    assert api.wait_ready(NS(launch_result=_launch(), dry_run=False), timeout=timeout, sctx=ctx).ready
    assert readiness[0]["cancel"] is ctx.operation.cancel


def test_wait_ready_positive_timeout_bounds_without_cancelling_owner(readiness):
    from sparkrun.api._lifecycle import _ReadinessCancellation

    ctx = _ctx()
    api.wait_ready(NS(launch_result=_launch(), dry_run=False), timeout=5, sctx=ctx)
    assert isinstance(readiness[0]["cancel"], _ReadinessCancellation)
    assert not ctx.operation.cancel.is_set()


def test_wait_ready_rejects_negative_timeout(readiness):
    with pytest.raises(ValueError, match="negative"):
        api.wait_ready(NS(launch_result=_launch(), dry_run=False), timeout=-1, sctx=_ctx())


@pytest.fixture
def discovery(monkeypatch):
    seen = NS(snapshot=[], prepared=[])
    snapshot = Mock(to_dict=Mock(return_value={"endpoints": []}))
    monkeypatch.setattr("sparkrun.proxy.discovery.discover_snapshot", lambda **kw: seen.snapshot.append(kw) or snapshot)
    monkeypatch.setattr("sparkrun.api._resolve.prepare_transport", lambda cluster, **kw: seen.prepared.append(cluster))
    return seen


def test_discovery_snapshot_unknown_cluster_is_a_sparkrun_error(v, discovery):
    with pytest.raises(api.SparkrunError, match="no-such"):
        api.discovery_snapshot(cluster="no-such")
    assert discovery.snapshot == []


def test_discovery_snapshot_defaults_to_the_default_cluster_with_its_identity(v, discovery):
    sctx = api.default_sctx()
    sctx.cluster_manager.create(name="lab", hosts=["h1", "h2"], user="alice")
    sctx.cluster_manager.set_default("lab")
    assert api.discovery_snapshot(workload_ids=["w1"], sctx=sctx) == {"endpoints": []}
    call = discovery.snapshot[0]
    assert call["host_list"] == ["h1", "h2"] and call["host_filter"] == ["h1", "h2"]
    assert call["cluster_def"].name == "lab"
    assert call["ssh_kwargs"]["ssh_user"] == "alice"
    assert call["cluster_ids"] == {"w1"}
    assert [c.name for c in discovery.prepared] == ["lab"]


def test_discovery_snapshot_without_any_host_source_is_metadata_only(v, discovery):
    api.discovery_snapshot(sctx=api.default_sctx())
    call = discovery.snapshot[0]
    assert call["host_list"] == [] and call["host_filter"] is None and call["cluster_def"] is None
    assert discovery.prepared == []
