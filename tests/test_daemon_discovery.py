"""Discovery coverage, fenced owners and independent service admission."""

from test_daemon import binding as binding
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


from sparkrun.proxy.discovery import DiscoveredEndpoint, DiscoverySnapshot, discover_snapshot
from sparkrun.proxy.autodiscover import _EndpointRemovalGrace, sync_discovered
from sparkrun.proxy.discovery_service import DiscoverySession, DiscoveryOwnershipError


def endpoint(host="one", **kwargs):
    return DiscoveredEndpoint(
        cluster_id="work-" + host, model="model", served_model_name=None, runtime="test", host=host, port=8000, healthy=True, **kwargs
    )


def test_partial_coverage_preserves_unknown_hosts_without_incrementing_grace():
    grace = _EndpointRemovalGrace(2)
    one, two = endpoint(), endpoint("two")
    grace.reconcile([one, two])
    for _ in range(5):
        effective, _ = grace.reconcile([], covered_hosts={"one"})
    assert effective == [two]
    assert grace.reconcile([], covered_hosts={"two"})[0] == [two]
    assert grace.reconcile([], covered_hosts={"two"})[0] == []


def test_failed_apply_does_not_commit_grace(monkeypatch):
    from sparkrun import api

    grace = _EndpointRemovalGrace(2)
    ep = endpoint()
    grace.reconcile([ep])
    context = SimpleNamespace(proxy_config=SimpleNamespace(_load=lambda: None, aliases={}))
    apply = Mock(side_effect=RuntimeError("busy"))
    monkeypatch.setattr(api.proxy, "sync", apply)
    with pytest.raises(RuntimeError):
        sync_discovered([], grace, context)
    assert grace.reconcile([])[0] == [ep]


def test_snapshot_separates_failed_status_from_empty_cluster(monkeypatch):
    from sparkrun import api
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy

    monkeypatch.setattr(api, "list_jobs", lambda **kw: [])
    monkeypatch.setattr(api, "status", lambda *args, **kw: ClusterStatus(hosts=(HostOccupancy("one"),), errors={"two": "secret"}))
    snapshot = discover_snapshot(host_list=["one", "two"])
    assert snapshot.covered_hosts == ["one"]
    assert not snapshot.complete
    assert "secret" not in str(snapshot.to_dict())
    monkeypatch.setattr(api, "status", lambda *args, **kw: ClusterStatus(hosts=(HostOccupancy("one"),)))
    assert discover_snapshot(host_list=["one"]).complete
    assert not discover_snapshot().complete


def test_public_snapshot_omits_credentials_and_keeps_observation_timestamp():
    snapshot = DiscoverySnapshot(endpoints=[endpoint(api_key="private", plugin_items={"secret": "also-private"})], observed_at=123)
    value = snapshot.to_dict()
    assert value["observed_at"] == 123
    assert "private" not in str(value)
    assert "api_key" not in value["endpoints"][0]


def descriptor():
    return {"backend": "daemon", "interval": 30, "removal_grace_sweeps": 2, "instance_id": "instance"}


def test_only_one_backend_claim_and_old_worker_cannot_release_replacement(tmp_path):
    session = DiscoverySession(tmp_path)
    first = session.prepare(descriptor())
    with pytest.raises(DiscoveryOwnershipError):
        session.prepare({**descriptor(), "backend": "process"})
    session.stop()
    second = session.prepare(descriptor())
    session.stop(generation=first["generation"])
    assert session.read()["generation"] == second["generation"]
    assert session.read()["state"] == "starting"
    with pytest.raises(DiscoveryOwnershipError):
        session.matching(first["generation"])


def test_fence_rejects_delayed_result_after_gateway_replacement(tmp_path):
    import yaml

    session = DiscoverySession(tmp_path)
    record = session.prepare(descriptor())
    (tmp_path / "state.yaml").write_text(yaml.safe_dump({"instance_id": "replacement", "gateway": "litellm"}))
    with pytest.raises(DiscoveryOwnershipError):
        with session.applying(record["generation"]):
            pytest.fail("stale application was accepted")


def test_service_budget_cannot_starve_normal_jobs(tmp_path):
    from sparkrun.daemon.jobs import JobStore
    from sparkrun.core.daemon_registry import DaemonOp

    store = JobStore(tmp_path)
    for name in ("proxy.watch", "proxy.serve", "proxy.watch", "workloads.run"):
        store.submit(DaemonOp(name, lambda: None), {}, "test", "", {})
    assert store.claim(max_active=1, max_services=2)["operation"] == "proxy.watch"
    assert store.claim(max_active=1, max_services=2)["operation"] == "proxy.serve"
    assert store.claim(max_active=1, max_services=2)["operation"] == "workloads.run"
    assert store.claim(max_active=1, max_services=2) is None


def test_managed_replacement_preserves_logical_instance_and_controller(tmp_path):
    import os
    from sparkrun.plugins.litellm.engine import ProxyEngine
    from sparkrun.core.process_identity import process_identity

    engine = ProxyEngine(state_dir=tmp_path)
    engine._discovery_control = True
    engine._save_state(os.getpid())
    previous = engine.get_state()
    assert engine._commit_replacement(os.getpid(), None, None)
    state = engine.get_state()
    assert state["instance_id"] == previous["instance_id"]
    assert state["controller"] == process_identity()
    engine.update_autodiscover_pid(12345)
    assert engine.get_state()["instance_id"] == previous["instance_id"]


def test_real_gateway_and_watch_are_separate_services(binding, monkeypatch):
    """Cancelling discovery never owns the serving process or its replacement."""
    import textwrap
    import time
    import yaml
    from test_daemon import real_server, wait_state
    from sparkrun.core.process_identity import alive

    plugin = binding.directory / "plugins"
    plugin.mkdir()
    (plugin / "gateway_fixture.py").write_text(
        textwrap.dedent("""\
        SPARKRUN_PLUGIN_API_VERSION = 1
        def register(v):
            from sparkrun.proxy.gateway import register_gateway
            register_gateway("fixture", feature_flag="gateway.litellm", loader=loader)
        def loader():
            from sparkrun.plugins.litellm.engine import ProxyEngine
            class Fixture(ProxyEngine):
                gateway_name = "fixture"
                def _build_command(self, config_path=None):
                    import sys
                    return [sys.executable, "-c", "import time; time.sleep(120)"]
                def query_models(self):
                    return ()
            return Fixture
    """)
    )
    cache = binding.directory / "cache"
    monkeypatch.setenv("SPARKRUN_CACHE_DIR", str(cache))
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS", raising=False)
    binding.config_path.write_text(
        yaml.safe_dump(
            {"features": {"daemon": True, "core.external_plugins": True}, "plugins": {"paths": [str(plugin)]}, "cache_dir": str(cache)}
        )
    )
    state_path = cache / "proxy" / "state.yaml"
    with real_server(binding) as (_, client):
        start = client.call(
            "proxy.start", {"gateway": "fixture", "discovery_backend": "daemon", "discover_interval": 1, "host": "127.0.0.1"}
        )["job_id"]
        try:
            assert client.wait(start, timeout=30)["started"]
            state = yaml.safe_load(state_path.read_text())
            serving = state["process"]
            service_job = state["service_job"]
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                discovery = client.call("proxy.discovery.status")
                if discovery["state"] == "running":
                    break
                time.sleep(0.1)
            assert discovery["state"] == "running", discovery
            assert discovery["job_id"] != service_job
            client.call("proxy.discovery.stop")
            wait_state(client, discovery["job_id"], {"succeeded", "cancelled"})
            assert alive(serving)
            assert client.request("GET", "/v1/jobs/" + service_job)["state"] == "running"
        finally:
            stop = client.call("proxy.stop")["job_id"]
            client.wait(stop, timeout=20)
            # Wait for the dedicated supervisor, so no test worker is leaked.
            if state_path.exists():
                state = yaml.safe_load(state_path.read_text())
            else:
                state = locals().get("state", {})
            if state.get("service_job"):
                wait_state(client, state["service_job"], {"succeeded", "failed", "cancelled"})


def test_gateway_rejects_reused_pid(tmp_path, monkeypatch):
    import os
    from sparkrun.plugins.litellm.engine import ProxyEngine
    from sparkrun.core import process_identity

    engine = ProxyEngine(state_dir=tmp_path / "gateway")
    engine._save_state(os.getpid())
    assert engine.current_pid() == os.getpid()
    monkeypatch.setattr(process_identity, "process_identity", lambda pid: {"pid": pid, "created": "different"})
    assert engine.current_pid() is None


def test_old_stop_cannot_clear_new_instance_with_same_pid(tmp_path):
    import os
    from sparkrun.plugins.litellm.engine import ProxyEngine

    engine = ProxyEngine(state_dir=tmp_path / "gateway")
    engine._save_state(os.getpid())
    original = engine.get_state()["instance_id"]
    engine._save_state(os.getpid())
    engine._clear_state(expected_pid=os.getpid(), expected_instance=original)
    assert engine.state_file.exists()


@pytest.fixture(autouse=True)
def _gateway_provider(loaded_litellm_plugin):
    pass
