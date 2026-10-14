"""Managed bundles obey job authorization, closed schemas, and retention."""

import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from test_daemon import binding as binding, real_server
from test_daemon_http_routes import auth as auth
from sparkrun import api
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.context import SparkrunContext
from sparkrun.core.operation import OperationState, operation_scope
from sparkrun.daemon.artifacts import diagnostics_path
from sparkrun.daemon.auth import TokenStore
from sparkrun.daemon.server import create_app
from scitrera_app_framework import Variables


def test_bundle_download_ownership_scopes_symlinks_and_retention(binding, auth, tmp_path):
    _, other_token = TokenStore(binding).create("other", scope="read")
    other = {"Authorization": "Bearer " + other_token}
    app = create_app(binding, start_workers=False)
    with TestClient(app) as client:
        response = client.post("/v1/diagnostics/collect", headers=auth["read"], json={"hosts": ["host-a"], "dry_run": True})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        path = diagnostics_path(binding.directory, job_id)
        path.parent.mkdir(exist_ok=True)
        content = '{"_type":"bundle_complete"}\n'
        path.write_text(content)
        url = f"/v1/jobs/{job_id}/artifacts/diagnostics"
        assert client.get(url).status_code == 401
        assert client.get(url, headers=other).status_code == 403
        assert client.get(url, headers=auth["read"]).status_code == 409
        app.state.store.finish(job_id, "failed", error={"detail": "original failure"})
        response = client.get(url, headers=auth["read"])
        assert response.status_code == 200
        assert response.text == content
        assert response.headers["content-type"].startswith("application/x-ndjson")
        assert client.get(url, headers=auth["admin"]).text == content
        path.unlink()
        outside = tmp_path / "private.txt"
        outside.write_text("must not download")
        try:
            path.symlink_to(outside)
        except OSError:
            pass  # Windows may not grant symlink creation.
        else:
            assert client.get(url, headers=auth["read"]).status_code == 404
            path.unlink()
        path.write_text(content)
        app.state.store.prune(count=0)
        assert not path.exists()
        assert client.get(url, headers=auth["read"]).status_code == 404


@pytest.mark.parametrize(
    "bad",
    [
        {"output_path": "/tmp/not-allowed"},
        {"sudo_password": "not-allowed"},
        {"max_bytes": 1},
        {"max_bytes": 67108865},
        {"log_tail": -1},
        {"log_tail": 2001},
    ],
)
def test_collection_rejects_paths_secrets_and_unbounded_limits(binding, auth, bad):
    with TestClient(create_app(binding, start_workers=False)) as client:
        assert client.post("/v1/diagnostics/collect", headers=auth["admin"], json=bad).status_code == 422


def test_capture_wraps_readiness_failure_and_preserves_result_contract(binding, monkeypatch):
    from types import SimpleNamespace
    from sparkrun.core.plugin_identity import PluginIdentity, plugin_registration
    from sparkrun.daemon.ops import invoke_options
    from sparkrun.plugins import DiagnosticSection, register_plugin_diagnostics

    sctx = SparkrunContext(Variables(), SparkrunConfig(binding.config_path), operation=OperationState())
    events = []
    sctx.operation.events = events.append
    sctx.operation.secrets.add("plugin-secret")
    with plugin_registration(PluginIdentity("fixture", "fixture", ("external", "fixture", None))):
        diagnostics = register_plugin_diagnostics([DiagnosticSection("launch")])

    def run(*args, **kwargs):
        assert diagnostics.bind(sctx=kwargs["sctx"]).snapshot("launch", lambda: {"detail": "observed plugin-secret"})
        return SimpleNamespace(rc=0, already_running=False, launch_result=object())

    monkeypatch.setattr(api, "plan", lambda *a, **kw: SimpleNamespace(recipe=SimpleNamespace(post_exec=[], post_commands=[])))
    monkeypatch.setattr("sparkrun.daemon.ops.to_wire", lambda value: value)
    monkeypatch.setattr(api, "run", run)
    error = api.SparkrunError("original readiness failure")
    monkeypatch.setattr(api, "wait_ready", Mock(side_effect=error))
    with operation_scope(sctx), pytest.raises(api.SparkrunError) as caught:
        invoke_options("run", "RunOptions", {"recipe": "test", "collect_diagnostics": True}, sctx)
    assert caught.value is error
    path = diagnostics_path(binding.directory, sctx.operation.id)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1]["operation_success"] is False
    snapshot = next(row for row in records if row["_type"] == "plugin_diagnostic")
    assert snapshot["plugin"] == "fixture" and snapshot["data"] == {"detail": "observed [redacted]"}
    assert not any(row["type"].startswith("plugin_diagnostic") for row in events)
    assert "plugin-secret" not in path.read_text()
    assert events[-1]["type"] == "artifact"
    assert events[-1]["url"].endswith("/artifacts/diagnostics")


def test_false_capture_flag_is_removed_before_building_options(binding, monkeypatch):
    from sparkrun.daemon.ops import invoke_options

    result = object()
    monkeypatch.setattr(api, "benchmark", lambda options, **kw: result)
    monkeypatch.setattr("sparkrun.daemon.ops.to_wire", lambda value: value)
    sctx = SparkrunContext(Variables(), SparkrunConfig(binding.config_path), operation=OperationState())
    assert invoke_options("benchmark", "BenchmarkOptions", {"recipe": "test", "collect_diagnostics": False}, sctx) is result
    assert not diagnostics_path(binding.directory, sctx.operation.id).exists()


def test_real_worker_collects_and_serves_bundle(binding):
    with real_server(binding, extra=("--tcp", "127.0.0.1:0")) as (_, client):
        job = client.call("diagnostics.collect", {"hosts": ["host-a"], "dry_run": True})["job_id"]
        result = client.wait(job, timeout=20)
        assert result["schema_version"] == 1
        assert result["partial"] is True
        assert "output_path" not in result
        connection = client.connection()
        try:
            connection.request("GET", result["artifact"]["url"], headers=client.headers())
            response = connection.getresponse()
            assert response.status == 200
            data = response.read()
            assert len(data) == result["artifact"]["size_bytes"]
            assert json.loads(data.splitlines()[-1])["_type"] == "bundle_complete"
        finally:
            connection.close()
