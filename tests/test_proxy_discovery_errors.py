"""Discovery retains safe gateway diagnostics without exposing arbitrary errors."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.api.proxy import ProxySyncResult, ProxyUpdateFailed
from sparkrun.core.operation import OperationState, operation_scope
from sparkrun.proxy import autodiscover, discovery, discovery_service
from sparkrun.proxy._supervisor import GatewayState
from sparkrun.proxy.discovery_service import DiscoverySession, run_session


@pytest.mark.parametrize("with_operation", [False, True], ids=["process", "daemon"])
@pytest.mark.parametrize("safe_error", [False, True], ids=["unexpected", "gateway"])
def test_discovery_reports_safe_failures_and_clears_them_on_recovery(tmp_path, monkeypatch, caplog, with_operation, safe_error):
    session = DiscoverySession(tmp_path)
    record = session.prepare({"backend": "daemon" if with_operation else "process", "interval": 30, "removal_grace_sweeps": 2})
    monkeypatch.setattr(GatewayState, "get_state", lambda self: {})
    monkeypatch.setattr(discovery_service, "alive", lambda process: True)
    monkeypatch.setattr(discovery, "discover_snapshot", lambda **kwargs: discovery.DiscoverySnapshot(mode="live", observed_at=123))
    detail = 'virtual model "model" owned by "sparkrun" conflicts with owner "operator"'
    error = ProxyUpdateFailed(detail) if safe_error else RuntimeError("Authorization: Bearer private-credential")
    # Even a safe outer exception must not cause its arbitrary cause to be logged.
    error.__cause__ = RuntimeError("private-cause")
    monkeypatch.setattr(
        autodiscover,
        "sync_discovered",
        Mock(side_effect=[error, error, (ProxySyncResult(proxy_running=True), 0)]),
    )
    snapshots, delays, events = [], [], []

    def wait(seconds):
        snapshots.append(session.status())
        delays.append(seconds)

    scope = operation_scope(state=OperationState(events=events.append)) if with_operation else nullcontext()
    with scope:
        run_session(session, record["generation"], SimpleNamespace(), cancelled=lambda: len(snapshots) == 3, wait=wait)

    expected = "ProxyUpdateFailed: " + detail if safe_error else "RuntimeError"
    assert [snapshot["last_error"] for snapshot in snapshots] == [expected, expected, None]
    assert [snapshot["state"] for snapshot in snapshots] == ["degraded", "degraded", "running"]
    assert snapshots[0]["last_application"] is None
    assert snapshots[2]["last_application"] is not None
    assert snapshots[2]["last_success"] == 123
    assert delays == [60, 120, 30]
    assert session.read()["state"] == "stopped"
    warnings = [record for record in caplog.records if record.name == discovery_service.__name__]
    assert len(warnings) == 1
    assert expected in warnings[0].getMessage()
    if with_operation:
        failures = [event for event in events if event.get("state") == "degraded"]
        assert len(failures) == 1
        assert failures[0]["error"] == type(error).__name__
        assert failures[0]["message"] == expected
    for secret in ("private-credential", "private-cause"):
        assert secret not in str(snapshots) + str(events) + caplog.text + session.path.read_text()


def test_routing_warning_keeps_discovery_successful_and_clears_after_correction(tmp_path, monkeypatch):
    session = DiscoverySession(tmp_path)
    record = session.prepare({"backend": "process", "interval": 30, "removal_grace_sweeps": 2})
    monkeypatch.setattr(GatewayState, "get_state", lambda self: {})
    monkeypatch.setattr(discovery_service, "alive", lambda process: True)
    monkeypatch.setattr(discovery, "discover_snapshot", lambda **kwargs: discovery.DiscoverySnapshot(mode="live", observed_at=123))
    warning = 'Discovered deployment needs a route: name "model" is reserved'
    monkeypatch.setattr(
        autodiscover,
        "sync_discovered",
        Mock(side_effect=[(ProxySyncResult(added=1, warnings=(warning,)), 0), (ProxySyncResult(), 0)]),
    )
    snapshots, delays = [], []

    def wait(seconds):
        snapshots.append(session.status())
        delays.append(seconds)

    run_session(session, record["generation"], SimpleNamespace(), cancelled=lambda: len(snapshots) == 2, wait=wait)
    assert [snapshot["state"] for snapshot in snapshots] == ["running", "running"]
    assert [snapshot["warnings"] for snapshot in snapshots] == [[warning], []]
    assert all(snapshot["last_success"] == 123 and snapshot["last_error"] is None for snapshot in snapshots)
    assert delays == [30, 30]
