"""Pre-release naming contracts: no aliases, distinct effects, real dispatch."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from test_daemon import binding as binding, real_server
from sparkrun.core.daemon_registry import get_daemon_op, registered_daemon_ops, operation_http_path
from sparkrun.daemon.auth import TokenStore
from sparkrun.daemon.ops import check_request, register_core_ops
from sparkrun.daemon.server import create_app


RENAMES = {
    "run": "workloads.run",
    "plan": "workloads.plan",
    "stop": "workloads.stop_matching",
    "stop_all": "workloads.stop_all",
    "find_running_intent": "workloads.find_intent",
    "list_jobs": "workloads.list",
    "status": "cluster.observe",
    "status_report": "cluster.report",
    "search_recipes": "catalog.search",
    "validate_recipe": "recipes.validate",
    "realize_recipe": "recipes.realize",
    "materialize": "workloads.materialize",
    "build": "recipes.build",
    "plan_build": "recipes.plan_build",
    "benchmark": "benchmarks.run",
    "resume_benchmark": "benchmarks.resume",
}


def test_canonical_names_replace_old_names_in_discovery_openapi_and_routes(binding):
    _, token = TokenStore(binding).create("audit", scope="admin")
    headers = {"Authorization": "Bearer " + token}
    with TestClient(create_app(binding, start_workers=False)) as client:
        advertised = {row["name"]: row for row in client.get("/v1/ops", headers=headers).json()}
        paths = client.get("/openapi.json", headers=headers).json()["paths"]
        for old, new in RENAMES.items():
            assert old not in advertised and new in advertised
            assert advertised[new]["http_path"] == "/v1/" + new.replace(".", "/")
            assert advertised[new]["http_path"] in paths
        for old in (*RENAMES, "discover_endpoints", "catalog.registry"):
            with pytest.raises(LookupError):
                get_daemon_op(old)
            for path in ("/v1/" + old.replace(".", "/"), "/v1/ops/" + old):
                assert path not in paths
                assert client.post(path, headers=headers, json={}).status_code == 404
        for stale in ("discover_endpoints", "proxy.status", "proxy.watch"):
            assert client.get("/v1/snapshots/" + stale, headers=headers).status_code == 404
        assert client.get("/v1/snapshots/discovery.snapshot", headers=headers).status_code == 200
        assert all("." in op.name for op in registered_daemon_ops() if op.owner == "core")


def test_guarded_stop_and_selector_stop_keep_distinct_contracts():
    register_core_ops()
    guarded = get_daemon_op("workloads.stop")
    selected = get_daemon_op("workloads.stop_matching")
    assert guarded.request_schema["required"] == ["recipe", "recipe_revision", "cluster_id"]
    assert selected.request_schema["required"] == []
    assert guarded.result_schema != selected.result_schema
    assert get_daemon_op("cluster.observe").result_schema != get_daemon_op("cluster.report").result_schema


def test_named_option_operations_reach_the_original_python_apis(monkeypatch):
    from sparkrun import api
    from sparkrun.core.operation import operation_scope

    register_core_ops()
    ctx = api.default_sctx()
    for name, target, model in (
        ("workloads.plan", "plan", api.RunOptions),
        ("workloads.materialize", "materialize", api.RunOptions),
        ("recipes.build", "build", api.BuildOptions),
        ("recipes.realize", "realize_recipe", api.RunOptions),
        ("benchmarks.run", "benchmark", api.BenchmarkOptions),
    ):
        fn = Mock(return_value={"target": target})
        monkeypatch.setattr(api, target, fn)
        with operation_scope(ctx):
            result = get_daemon_op(name).handler_loader()({"recipe": "fixture"}, ctx)
        assert result == {"target": target}
        assert isinstance(fn.call_args.args[0], model)
        assert fn.call_args.args[0].recipe == "fixture"
        assert fn.call_args.kwargs["sctx"] is ctx


def test_registry_actions_have_closed_action_specific_requests():
    register_core_ops()
    for action in ("add", "remove", "enable", "disable", "trust", "untrust"):
        op = get_daemon_op("catalog.registries." + action)
        assert op.scope == "admin" and op.kind == "job"
        args = {"name": "example"}
        if action == "add":
            args["url"] = "https://example.invalid/recipes.git"
        if action == "trust":
            args["acknowledge_trust"] = True
        assert check_request(op, args) == args
        for invalid in ({**args, "action": action}, {**args, "name": None}, {**args, "name": ""}):
            with pytest.raises(ValueError):
                check_request(op, invalid)
        if action not in {"add", "trust"}:
            with pytest.raises(ValueError):
                check_request(op, {**args, "acknowledge_trust": True})
    with pytest.raises(ValueError):
        check_request(get_daemon_op("catalog.registries.trust"), {"name": "example", "acknowledge_trust": False})


def test_real_registry_mutations_and_alias_retry(binding):
    register_core_ops()
    with real_server(binding) as (_, client):
        name = "naming-fixture"
        for action in ("add", "disable", "enable", "trust", "untrust", "remove"):
            arguments = {"name": name}
            if action == "add":
                arguments["url"] = "https://example.invalid/recipes.git"
            if action == "trust":
                arguments["acknowledge_trust"] = True
            op = "catalog.registries." + action
            accepted = client.call(op, arguments, key="registry-" + action)
            # Generated alias and generic dispatcher must preserve one intent.
            repeated = client.request(
                "POST", operation_http_path(get_daemon_op(op)), arguments, headers={"Idempotency-Key": "registry-" + action}
            )
            assert repeated == accepted
            result = client.wait(accepted["job_id"], timeout=15)
            entry = next((row for row in result["registries"] if row["name"] == name), None)
            if action == "remove":
                assert entry is None
            else:
                assert entry is not None
                if action in {"enable", "disable"}:
                    assert entry["enabled"] == (action == "enable")
                if action in {"trust", "untrust"}:
                    assert entry["trusted"] == (action == "trust")


def test_advisory_service_publishes_the_scoped_discovery_contract(binding, monkeypatch):
    import time
    import yaml

    cache = binding.config_path.parent / "cache"
    monkeypatch.setenv("SPARKRUN_CACHE_DIR", str(cache))
    binding.config_path.write_text(
        yaml.safe_dump({"features": {"daemon": True}, "cache_dir": str(cache), "daemon": {"discovery_interval": 5}})
    )
    with real_server(binding) as (_, client):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            value = client.request("GET", "/v1/snapshots/discovery.snapshot")
            if value["state"] == "succeeded":
                break
            assert value["state"] not in {"failed", "lost"}, value
            time.sleep(0.1)
        assert value["state"] == "succeeded", value
        assert isinstance(value["result"]["endpoints"], list)
        assert value["result"]["observed_at"] > 0
        assert value["result"]["mode"] == "metadata"
        assert value["result"]["complete"] is False
        job = client.request("GET", "/v1/jobs/" + value["job_id"])
        assert job["operation"] == "discovery.snapshot"
