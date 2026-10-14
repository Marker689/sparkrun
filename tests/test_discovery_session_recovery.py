"""A discovery claim whose owner is gone is reclaimed rather than blocking every start."""

import time

import pytest

from sparkrun.core.process_identity import process_identity
from sparkrun.proxy import discovery_service
from sparkrun.proxy.discovery_service import DiscoveryOwnershipError, DiscoverySession

DESCRIPTOR = {"backend": "process", "interval": 30, "removal_grace_sweeps": 2}


def _claim(tmp_path, **record):
    session = DiscoverySession(tmp_path)
    claimed = session.prepare(dict(DESCRIPTOR))
    claimed.update(record)
    session.write(claimed)
    return session, claimed


def test_dead_owner_is_reclaimed(tmp_path):
    dead = {"pid": 2**22 + 7, "start": "never"}
    session, old = _claim(tmp_path, state="running", process=dead)
    new = session.prepare(dict(DESCRIPTOR))
    assert new["generation"] != old["generation"]


def test_live_owner_is_never_replaced(tmp_path):
    session, _ = _claim(tmp_path, state="running", process=process_identity())
    with pytest.raises(DiscoveryOwnershipError):
        session.prepare(dict(DESCRIPTOR))


def test_unattached_claim_is_reclaimed_only_after_the_timeout(tmp_path):
    session, _ = _claim(tmp_path, state="starting", process=None)
    with pytest.raises(DiscoveryOwnershipError):
        session.prepare(dict(DESCRIPTOR))
    stale = session.read()
    stale["created"] = time.time() - discovery_service.UNATTACHED_CLAIM_TIMEOUT_S - 1
    session.write(stale)
    assert session.prepare(dict(DESCRIPTOR))["state"] == "starting"


def test_setup_failure_after_attach_releases_the_claim(tmp_path):
    session = DiscoverySession(tmp_path)
    descriptor = dict(DESCRIPTOR, initial_endpoints=[{"cluster_id": "sparkrun_x", "credential_required": True}])
    record = session.prepare(descriptor)

    class _Sctx:
        pass

    from unittest.mock import patch

    with patch("sparkrun.api.list_jobs", return_value=[]):
        with pytest.raises(ValueError, match="credentials unavailable"):
            discovery_service.run_session(session, record["generation"], _Sctx(), cancelled=lambda: True, wait=lambda s: None)
    assert session.read()["state"] == "stopped"


def test_gateway_is_stopped_when_discovery_cannot_start(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from sparkrun.plugins.litellm.engine import ProxyEngine

    engine = ProxyEngine(state_dir=tmp_path, master_key="secret")
    monkeypatch.setattr(engine, "start_autodiscover", MagicMock(side_effect=DiscoveryOwnershipError("owned")))
    stop = MagicMock(return_value=True)
    monkeypatch.setattr(engine, "stop", stop)
    child = MagicMock()
    with pytest.raises(DiscoveryOwnershipError):
        engine._start_autodiscover_or_abort(1234, {"interval": 30}, child=child)
    stop.assert_called_once()
    child.wait.assert_called()
