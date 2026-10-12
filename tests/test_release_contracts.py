"""0.4 release regressions: lifecycle recovery and public daemon boundaries."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from test_daemon import binding as binding
from sparkrun.daemon.auth import TokenStore
from sparkrun.daemon.server import create_app
from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op, get_daemon_op
from sparkrun.core.plugin_identity import application_daemon_registration
from sparkrun.api._wire import check_result, ResultContractError, object_schema, validate


def test_invalid_query_results_are_internal_errors_not_cached(binding):
    calls = []

    def invalid(*args):
        calls.append(True)
        return {"count": "wrong"}

    with application_daemon_registration():
        register_daemon_op(
            DaemonOp(
                "broken",
                lambda: invalid,
                kind="query",
                scope="read",
                cacheable=True,
                result_schema=object_schema({"count": {"type": "integer"}}, ["count"]),
            )
        )
    headers = {"Authorization": "Bearer " + TokenStore(binding).create("test", scope="read")[1]}
    with TestClient(create_app(binding, start_workers=False), raise_server_exceptions=False) as client:
        for _ in range(2):
            response = client.post("/v1/plugins/app-sparkrun/ops/broken", json={}, headers=headers)
            assert response.status_code == 500
            assert response.json()["error"]["type"] == "ResultContractError"
        assert len(calls) == 2


def test_internal_ops_hidden_and_cannot_be_submitted_by_admin(binding):
    headers = {"Authorization": "Bearer " + TokenStore(binding).create("test", scope="admin")[1]}
    with TestClient(create_app(binding, start_workers=False)) as client:
        ops = {op["name"] for op in client.get("/v1/ops", headers=headers).json()}
        for name in ("proxy.serve", "proxy.watch"):
            assert name not in ops
            for path in ("/v1/ops/" + name, "/v1/" + name.replace(".", "/")):
                assert client.post(path, json={"generation": "test"}, headers=headers).status_code == 404
        assert client.get("/v1/jobs", headers=headers).json() == []
        info = client.get("/v1/info", headers=headers).json()
        from sparkrun.core.application_identity import get_controller_identity
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.daemon.contracts import INFO

        validate(info, INFO)
        assert info["controller"] == get_controller_identity(SparkrunConfig(binding.config_path)).to_dict()


def test_gateway_start_uses_internal_queue_without_public_submission(binding, monkeypatch):
    from sparkrun.daemon.proxy import launch_gateway
    from sparkrun.core.config import SparkrunConfig

    app = create_app(binding, start_workers=False)
    headers = {"Authorization": "Bearer " + TokenStore(binding).create("test", scope="admin")[1]}
    with TestClient(app) as client:
        info = client.get("/v1/info", headers=headers).json()
        monkeypatch.setattr("sparkrun.daemon.management.ensure_daemon", lambda _: None)
        monkeypatch.setattr("sparkrun.daemon.client.Client.request", lambda *a, **k: info)
        monkeypatch.setattr("sparkrun.daemon.client.Client.call", lambda *a, **k: pytest.fail("Internal work used public dispatch"))
        directory = binding.directory / "gateway"
        directory.mkdir()
        engine = SimpleNamespace(
            state_dir=directory,
            gateway_name="fixture",
            host="127.0.0.1",
            port=8000,
            master_key=None,
            get_state=lambda: {"service_job": app.state.store.list()[0]["id"], "service_ready": True},
            is_running=lambda: True,
        )
        job_id = launch_gateway(engine, None, None, SimpleNamespace(config=SparkrunConfig(binding.config_path)))
        job = app.state.store.get(job_id)
        assert job["operation"] == "proxy.serve" and job["principal"] == "service:daemon"
        assert json.loads((directory / "gateway-start.json").read_text())["job_id"] == job_id


def test_real_worker_marks_bad_plugin_result_failed(binding, monkeypatch):
    from test_daemon import real_server, wait_state

    plugin = binding.directory / "plugins"
    plugin.mkdir()
    (plugin / "bad_result.py").write_text("""SPARKRUN_PLUGIN_API_VERSION = 1
def execute(args, sctx, prompt): return {"count": "invalid"}
def loader(): return execute
def register(v):
    from sparkrun.plugins import DaemonOp, register_daemon_op
    from sparkrun.api._wire import object_schema
    register_daemon_op(DaemonOp("run", loader, result_schema=object_schema({"count": {"type": "integer"}}, ["count"])))
""")
    binding.config_path.write_text(
        "features:\n  daemon: true\n  core.external_plugins: true\nplugins:\n  paths:\n    - " + str(plugin) + "\n"
    )
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS", raising=False)
    with real_server(binding) as (_, client):
        job_id = client.call("plugins.bad-result.run", {})["job_id"]
        job = wait_state(client, job_id, {"failed"})
        assert job["result"] is None
        assert job["error"]["error"]["type"] == "ResultContractError"


def test_model_cache_request_rejects_null_and_keeps_origin(binding):
    from sparkrun.daemon.ops import register_core_ops, check_request

    register_core_ops()
    op = get_daemon_op("model_cache.inspect")
    base = {"model": "org/model", "cache_dir": "/hf", "endpoint": "https://mirror.example"}
    assert check_request(op, base) == base
    for value in (None, [], "host"):
        with pytest.raises(ValueError):
            check_request(op, {**base, "hosts": value})
    with pytest.raises(ResultContractError):
        check_result(op, {"incorrect": True})


def test_application_and_installed_integration_source_changes_invalidate_runtime(tmp_path, monkeypatch):
    from sparkrun.core.runtime_lease import runtime_identity
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.application_profile import SPARKRUN

    package = tmp_path / "review_app"
    package.mkdir()
    (package / "__init__.py").write_text("")
    source = package / "profile.py"
    source.write_text("value = 1\n")
    other = tmp_path / "review_plugin"
    other.mkdir()
    plugin = other / "__init__.py"
    plugin.write_text("value = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    config = SparkrunConfig(tmp_path / "config.yaml")
    monkeypatch.setattr(
        config, "_profile", replace(SPARKRUN, package="review-app", profile_ref="review_app.profile:PROFILE", update_sources={})
    )
    monkeypatch.setattr(
        "sparkrun.core.installed_plugins.discover_installed_plugins",
        lambda _: [SimpleNamespace(name="review", package="review-plugin", module="review_plugin:register", selected=True)],
    )
    first = runtime_identity(config)
    source.write_text("value = 200\n")
    second = runtime_identity(config)
    assert second != first
    plugin.write_text("value = 3000\n")
    third = runtime_identity(config)
    assert third != second
    assert third["package_stamp"] == first["package_stamp"]


@pytest.mark.skipif(__import__("os").name == "nt", reason="Linux cache-host helper")
def test_post_hook_stop_releases_claim_and_failed_stop_retains_it(tmp_path, monkeypatch):
    from test_model_cache_retention import tracked
    from test_launcher import _LifecycleRecipe, _LifecycleRuntime, _make_launch_result, _patch_lifecycle_common
    from sparkrun.core.launcher import complete_post_launch
    from sparkrun.models.preparation import prepare_model
    from sparkrun.models.usage import usage_operation, begin_workload, activate_workload, workload_claims

    manifest, _, record = tracked(tmp_path)
    runtime = _LifecycleRuntime()
    result = _make_launch_result(_LifecycleRecipe(post_commands=["echo hi"], stop_after_post=True), runtime)
    _patch_lifecycle_common(monkeypatch)
    monkeypatch.setattr("sparkrun.orchestration.hooks.build_hook_context", lambda *a, **k: {})
    monkeypatch.setattr("sparkrun.orchestration.hooks.run_post_commands", lambda *a, **k: None)
    removed = []
    with usage_operation():
        begin_workload(result.cluster_id, {})
        prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest)
        activate_workload()
        metadata = {"hosts": ["localhost"], "model_cache_claims": workload_claims()}
        monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **k: metadata)
        monkeypatch.setattr("sparkrun.orchestration.job_metadata.remove_job_metadata", lambda *a, **k: removed.append(True))
        stop = runtime.stop
        runtime.stop = lambda **kwargs: 1
        with pytest.raises(RuntimeError, match="teardown failed"):
            complete_post_launch(result, remote_cache_dir=str(tmp_path))
        assert json.loads(record.read_text())["claims"] and not removed
        runtime.stop = stop
        assert complete_post_launch(result, remote_cache_dir=str(tmp_path)) is True
    assert not json.loads(record.read_text())["claims"]
    assert removed == [True]


def test_read_queries_must_opt_into_caching(binding):
    calls = []

    def observe(*args):
        calls.append(True)
        return {"count": len(calls)}

    with application_daemon_registration():
        register_daemon_op(
            DaemonOp(
                "uncached",
                lambda: observe,
                kind="query",
                scope="read",
                result_schema=object_schema({"count": {"type": "integer"}}, ["count"]),
            )
        )
    headers = {"Authorization": "Bearer " + TokenStore(binding).create("test", scope="read")[1]}
    with TestClient(create_app(binding, start_workers=False)) as client:
        counts = [client.post("/v1/plugins/app-sparkrun/ops/uncached", json={}, headers=headers).json()["count"] for _ in range(2)]
        assert counts == [1, 2]


def test_prompt_schema_covers_benchmark_decisions_and_plugin_kinds():
    from sparkrun.daemon.contracts import PROMPT

    for kind in ("approval", "decision", "secret", "plugin-choice"):
        validate({"prompt_id": "prompt", "kind": kind, "message": "Continue?", "default": False, "deadline": 42.0}, PROMPT)


@pytest.mark.parametrize(
    "field,value,schema",
    [
        ("inventory", list(range(2049)), {"type": "array", "items": {"type": "integer"}}),
        ("content", "x" * 65537, {"type": "string"}),
    ],
)
def test_result_contract_does_not_inherit_request_size_limits(field, value, schema):
    contract = object_schema({field: schema}, [field])
    op = DaemonOp("fixture", lambda: None, result_schema=contract)
    assert check_result(op, {field: value}) == {field: value}
    with pytest.raises(ValueError):
        validate({field: value}, contract)
    bounded = {**schema, "maxItems" if field == "inventory" else "maxLength": 10}
    with pytest.raises(ResultContractError):
        check_result(replace(op, result_schema=object_schema({field: bounded}, [field])), {field: value})


def test_nullable_schema_branches_do_not_consume_result_nesting_budget():
    value, schema = "leaf", {"type": "string"}
    for _ in range(15):
        value = [value]
        schema = {"anyOf": [{"type": "array", "items": schema}, {"type": "null"}]}
    assert check_result(DaemonOp("fixture", lambda: None, result_schema=schema), value) == value
