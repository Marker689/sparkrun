"""HTTP aliases preserve operation policies, jobs and generated API contracts."""

import sys

import pytest
from fastapi.testclient import TestClient

from test_daemon import binding as binding, real_server
from sparkrun.core.daemon_registry import DaemonOp, DaemonRoute, register_daemon_op, register_daemon_route, registered_daemon_ops
from sparkrun.daemon.auth import TokenStore
from sparkrun.daemon.server import create_app


@pytest.fixture
def auth(binding, monkeypatch):
    from sparkrun.core import daemon_registry

    monkeypatch.setattr(daemon_registry, "_ROUTES", {})
    tokens = TokenStore(binding)
    return {scope: {"Authorization": "Bearer " + tokens.create(scope, scope=scope)[1]} for scope in ("read", "operate", "admin")}


@pytest.mark.parametrize("alias_first", [True, False])
def test_job_alias_shares_permissions_validation_and_idempotency(binding, auth, alias_first):
    app = create_app(binding, start_workers=False)
    paths = ["/v1/cluster/create", "/v1/ops/cluster.create"]
    if not alias_first:
        paths.reverse()
    args = {"name": "lab", "hosts": ["host-a"]}
    with TestClient(app) as client:
        for path in paths:
            assert client.post(path, json=args).status_code == 401
            for scope in ("read", "operate"):
                assert client.post(path, json=args, headers=auth[scope]).status_code == 403
        headers = {**auth["admin"], "Idempotency-Key": "create-lab", "X-Sparkrun-Client": "http-tests"}
        responses = [client.post(path, json=args, headers=headers) for path in paths]
        assert [r.status_code for r in responses] == [202, 202]
        assert responses[0].json() == responses[1].json()
        job_id = responses[0].json()["job_id"]
        assert responses[0].json()["events_url"] == f"/v1/jobs/{job_id}/events"
        row = app.state.store.get(job_id)
        assert row["operation"] == "cluster.create"
        assert row["arguments"] == args
        assert row["client"] == "http-tests"
        assert len(app.state.store.list()) == 1
        for path in paths:
            conflict = client.post(path, json={**args, "name": "other"}, headers=headers)
            assert conflict.status_code == 409
            assert client.post(path, json={**args, "unknown": True}, headers=headers).status_code == 422
            assert client.post(path, json={}, headers=headers).status_code == 422
            assert client.post(path, content="[1]", headers=headers).status_code == 422
            assert client.post(path, content=b"x" * (1024 * 1024 + 1), headers=headers).status_code == 413
        assert len(app.state.store.list()) == 1


def test_query_alias_returns_same_result_and_enforces_shared_controls(binding, auth):
    app = create_app(binding, start_workers=False)
    with TestClient(app) as client:
        paths = ["/v1/catalog/clusters", "/v1/ops/catalog.clusters"]
        responses = [client.post(path, json={}, headers=auth["read"]) for path in paths]
        assert [r.status_code for r in responses] == [200, 200]
        assert responses[0].json() == responses[1].json()
        assert not app.state.store.list()
        assert client.get(paths[0], headers=auth["read"]).status_code == 405
        assert client.post("/v1/catalog/missing", json={}, headers=auth["admin"]).status_code == 404
        for path in ("/v1/run", "/v1/ops/run"):
            assert client.post(path, json={"recipe": "example", "trust": True}, headers=auth["admin"]).status_code == 403
        app.state.coordinator.maintenance = True
        for path in paths:
            assert client.post(path, json={}, headers=auth["read"]).status_code == 409


def test_plugin_alias_honors_feature_and_secret_permissions(binding, auth, monkeypatch):
    from sparkrun.core.operation import CURRENT

    def query(args, sctx, prompt):
        return {"via_daemon": CURRENT.get().via_daemon}

    register_daemon_op(
        DaemonOp(
            "fixture.query",
            lambda: query,
            kind="query",
            scope="read",
            owner="fixture",
            feature_flag="fixture.enabled",
            secret=True,
            result_schema={"type": "object", "properties": {"via_daemon": {"type": "boolean"}}},
        )
    )
    _, token = TokenStore(binding).create("secrets", scope="read", secret=True)
    allowed = {"Authorization": "Bearer " + token}
    with TestClient(create_app(binding, start_workers=False)) as client:
        for path in ("/v1/fixture/query", "/v1/ops/fixture.query"):
            assert client.post(path, json={}, headers=allowed).status_code == 403
        assert "fixture.query" not in {op["name"] for op in client.get("/v1/ops", headers=allowed).json()}
        monkeypatch.setenv("SPARKRUN_FEATURE_FIXTURE_ENABLED", "1")
        for path in ("/v1/fixture/query", "/v1/ops/fixture.query"):
            assert client.post(path, json={}, headers=auth["admin"]).status_code == 403
            response = client.post(path, json={}, headers=allowed)
            assert response.status_code == 200
            assert response.json() == {"via_daemon": True}
        entry = next(op for op in client.get("/v1/ops", headers=allowed).json() if op["name"] == "fixture.query")
        assert entry["http_path"] == "/v1/fixture/query"


def test_openapi_describes_alias_bodies_responses_and_rebuilds_after_startup(binding, auth):
    from fastapi.openapi.models import OpenAPI

    register_daemon_op(DaemonOp("fixture.stream", lambda: None, kind="stream"))
    app = create_app(binding, start_workers=False)
    assert "/v1/cluster/create" not in app.openapi()["paths"]
    for _ in range(2):
        with TestClient(app) as client:
            assert client.get("/openapi.json").status_code == 401
            schema = client.get("/openapi.json", headers=auth["read"]).json()
            OpenAPI.model_validate(schema)
            operation_ids = []
            for item in schema["paths"].values():
                operation_ids.extend(value["operationId"] for value in item.values() if isinstance(value, dict) and "operationId" in value)
            assert len(operation_ids) == len(set(operation_ids))
            for op in registered_daemon_ops():
                path = "/v1/" + op.name.replace(".", "/")
                if op.kind == "stream":
                    assert path not in schema["paths"]
                    assert schema["x-sparkrun-operations"][op.name]["http_path"] is None
                    continue
                item = schema["paths"][path]["post"]
                assert item["operationId"] == "sparkrun." + op.name
                assert item["requestBody"] == {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/" + op.name + "Request"}}},
                }
                assert schema["components"]["schemas"][op.name + "Request"] == op.request_schema
                assert schema["components"]["schemas"][op.name + "Result"] == op.result_schema
                code, result = ("202", "DaemonJobAccepted") if op.kind == "job" else ("200", op.name + "Result")
                assert {"200", "202"} & item["responses"].keys() == {code}
                assert item["responses"][code]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/" + result}
                assert item["security"] == [{"BearerAuth": []}]
                assert item["x-sparkrun-operation"]["scope"] == op.scope
                assert item["x-sparkrun-operation"]["result"] == {"$ref": "#/components/schemas/" + op.name + "Result"}
                assert schema["x-sparkrun-operations"][op.name]["http_path"] == path
                assert len([route for route in app.routes if getattr(route, "path", None) == path]) == 1
                if op.kind == "job":
                    assert {p["name"] for p in item["parameters"]} == {"Idempotency-Key", "X-Sparkrun-Client"}
                else:
                    assert not item.get("parameters")
            assert schema["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"
            assert schema["security"] == [{"BearerAuth": []}]
            assert schema["paths"]["/v1/health"]["get"]["security"] == []
            exposed = client.get("/v1/ops", headers=auth["admin"]).json()
            assert next(op for op in exposed if op["name"] == "cluster.create")["http_path"] == "/v1/cluster/create"


@pytest.mark.parametrize("name", ["jobs", "jobs.custom", "ops.fake", "fixture..bad", "fixture."])
def test_reserved_and_malformed_aliases_fail_startup(binding, auth, name):
    register_daemon_op(DaemonOp(name, lambda: None))
    with pytest.raises(ValueError, match="reserved HTTP alias"):
        with TestClient(create_app(binding, start_workers=False)):
            pass


@pytest.mark.parametrize("conflict", ["adapter", "existing"])
def test_alias_collision_fails_before_mounting_any_operation(binding, auth, conflict):
    app = create_app(binding, start_workers=False)
    if conflict == "adapter":
        register_daemon_route(
            DaemonRoute(
                "/v1/cluster/create",
                lambda: lambda args: ("cluster.create", args),
                ("cluster.create",),
                {"type": "object", "additionalProperties": False},
                "fixture",
            )
        )
    else:
        app.add_api_route("/v1/cluster/{action}", lambda: {})
    original = list(app.routes)
    with pytest.raises(ValueError, match="Daemon HTTP path"):
        with TestClient(app):
            pass
    assert app.routes == original


@pytest.mark.parametrize(
    "extra",
    [
        ("--no-unix", "--tcp", "127.0.0.1:0"),
        pytest.param((), marks=pytest.mark.skipif(sys.platform != "linux", reason="Unix peer authentication requires Linux")),
    ],
    ids=["tcp", "unix"],
)
def test_alias_runs_real_worker_and_shares_generic_retry(binding, extra):
    with real_server(binding, extra=extra) as (_, client):
        args = {"name": "http-lab", "hosts": ["host-a"]}
        accepted = client.request("POST", "/v1/cluster/create", args, headers={"Idempotency-Key": "http-create"})
        job_id = accepted["job_id"]
        assert client.wait(job_id, timeout=15)["name"] == "http-lab"
        assert client.call("cluster.create", args, key="http-create") == accepted
        assert list(client.events(job_id))[-1]["state"] == "succeeded"
        assert client.request("POST", "/v1/catalog/clusters", {}) == client.call("catalog.clusters")
