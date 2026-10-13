"""Model enumeration errors must survive the gateway, API, and CLI boundaries."""

from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from click.testing import CliRunner

from sparkrun import api
from sparkrun.api.proxy import _ops
from sparkrun.cli import main
from sparkrun.proxy._supervisor import GatewaySupervisor
from sparkrun.plugins.litellm.engine import ProxyEngine


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    engine = ProxyEngine(state_dir=tmp_path)
    monkeypatch.setattr(engine, "get_state", lambda: {"gateway": "litellm", "pid": 123})
    monkeypatch.setattr(engine, "is_running", lambda: True)
    monkeypatch.setattr(_ops, "_running_engine", lambda sctx=None: engine)
    monkeypatch.setattr(api.proxy, "sync", lambda **kw: api.proxy.ProxySyncResult())
    return engine


@pytest.mark.parametrize("failure", ["auth", "connection", "unsupported"])
@pytest.mark.parametrize("refresh", [False, True])
def test_failed_query_is_diagnostic_for_status_and_error_for_models(gateway, monkeypatch, failure, refresh):
    def request(*args, **kwargs):
        if failure == "auth":
            raise HTTPError("http://localhost/model/info", 401, "Unauthorized", {}, None)
        raise OSError("management connection failed")

    if failure == "unsupported":
        monkeypatch.setattr(gateway, "query_models", lambda: GatewaySupervisor.query_models(gateway))
    else:
        monkeypatch.setattr(gateway, "_api_request", request)

    snapshot = api.proxy.status()
    assert snapshot.running and snapshot.model_query_error
    assert snapshot.to_dict()["model_query_error"] == snapshot.model_query_error
    with pytest.raises(api.proxy.ProxyQueryFailed):
        api.proxy.models()
    with pytest.raises(api.proxy.ProxyQueryFailed):
        snapshot.require_models()
    for output in ([], ["--json"]):
        result = CliRunner().invoke(main, ["proxy", "models", *output, *(["--refresh"] if refresh else [])])
        assert result.exit_code == 1, result.output
        assert "Model list unavailable" in result.stderr
        assert "[]" not in result.stdout
    result = CliRunner().invoke(main, ["proxy", "status", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["model_query_error"]


@pytest.mark.parametrize("populated", [False, True])
def test_healthy_query_and_recovery_preserve_success_shape(gateway, monkeypatch, populated):
    monkeypatch.setattr(gateway, "_api_request", lambda *args: (_ for _ in ()).throw(OSError("previous failure")))
    assert api.proxy.status().model_query_error
    rows = [{"model_name": "example", "litellm_params": {"api_base": "http://worker/v1"}}] if populated else []
    monkeypatch.setattr(gateway, "_api_request", lambda *a, **kw: {"data": rows})
    models = api.proxy.models()
    assert isinstance(models, tuple) and len(models) == int(populated)
    assert api.proxy.status().model_query_error == ""
    result = CliRunner().invoke(main, ["proxy", "models", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == [m.to_dict() for m in models]


def test_stopped_gateway_does_not_query(gateway, monkeypatch):
    monkeypatch.setattr(gateway, "is_running", lambda: False)
    monkeypatch.setattr(gateway, "_api_request", lambda *a, **kw: pytest.fail("stopped gateway must not be queried"))
    assert api.proxy.models() == ()
    result = CliRunner().invoke(main, ["proxy", "models", "--json"])
    assert result.exit_code == 0 and json.loads(result.stdout) == []


def test_refresh_can_recover_a_failed_initial_query(gateway, monkeypatch):
    monkeypatch.setattr(gateway, "_api_request", lambda *a, **kw: (_ for _ in ()).throw(OSError("offline")))

    def sync(**kwargs):
        monkeypatch.setattr(gateway, "_api_request", lambda *a, **kw: {"data": [{"model_name": "recovered"}]})
        return api.proxy.ProxySyncResult()

    monkeypatch.setattr(api.proxy, "sync", sync)
    result = CliRunner().invoke(main, ["proxy", "models", "--json", "--refresh"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["model_name"] == "recovered"


@pytest.fixture(autouse=True)
def _gateway_provider(loaded_litellm_plugin):
    pass
