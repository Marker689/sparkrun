"""Plugin ownership, independent local names, adapter policy and canonical retries."""

from dataclasses import replace
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from test_daemon import binding as binding, controlled_plugin as controlled_plugin, real_server
from sparkrun.core.daemon_registry import (
    DaemonOp,
    DaemonRoute,
    register_daemon_op,
    register_plugin_daemon_op,
    register_plugin_daemon_route,
    get_daemon_op,
    registered_daemon_ops,
    registered_daemon_routes,
)
from sparkrun.core.plugin_identity import (
    PluginIdentity,
    normalize_plugin_id,
    plugin_registration,
    core_registration,
    application_daemon_registration,
    current_registration_owner,
    preflight_identities,
    preflight_plugin_sources,
)
from sparkrun.core.registration import load_and_register_plugin
from sparkrun.daemon.auth import TokenStore
from sparkrun.daemon.jobs import JobStore, Conflict
from sparkrun.daemon.server import create_app


def identity(name="fixture", source="test"):
    return PluginIdentity(normalize_plugin_id(name), "sparkrun.plugins." + name.replace("-", "_"), (source, name, None))


def declare(v):
    pass


def module(name, hook=declare):
    result = ModuleType(name)
    result.SPARKRUN_PLUGIN_API_VERSION = 1
    result.register = hook
    return result


def write_plugin(root, plugin_id, module_name, *, fail=False):
    package = root / module_name
    package.mkdir(parents=True)
    (root / "plugin.toml").write_text(f'name = "{plugin_id}"\nmodule = "{module_name}"\n')
    (package / "__init__.py").write_text(
        """
SPARKRUN_PLUGIN_API_VERSION = 1
from sparkrun.plugins import DaemonOp, DaemonRoute, register_plugin_daemon_op, register_plugin_daemon_route
from sparkrun.core.operation import CURRENT
def query(args, sctx, prompt):
    return {"module": __name__, "via_daemon": CURRENT.get().via_daemon}
def loader(): return query
# Import-time and hook-time declarations must receive the same loader identity.
register_plugin_daemon_op(DaemonOp("status", loader, kind="query", scope="read", result_schema={"type":"object"}))
def decode(): return lambda args: ("status", args)
def core_decode(): return lambda args: ("core:cluster.create", args)
def register(v):
    register_plugin_daemon_op(DaemonOp("cluster.create", loader, result_schema={"type":"object"}))
    register_plugin_daemon_op(DaemonOp("jobs.list", loader, kind="query", scope="read", result_schema={"type":"object"}))
    register_plugin_daemon_route(DaemonRoute("compat/status", decode, ("status",), {"type":"object", "additionalProperties":False}))
    register_plugin_daemon_route(DaemonRoute("compat/create", core_decode, ("core:cluster.create",),
        {"type":"object", "properties":{"name":{"type":"string"}, "hosts":{"type":"array", "items":{"type":"string"}}}, "additionalProperties":False}))
"""
        + ('    raise RuntimeError("broken registration")\n' if fail else "")
    )
    return root


def test_two_real_directory_plugins_share_local_names_and_adapter_policy(binding, tmp_path):
    from sparkrun.core.external_plugins import load_external_plugins

    roots = [write_plugin(tmp_path / name, name, "namespace_" + name) for name in ("alpha", "beta")]
    try:
        assert load_external_plugins(None, paths=roots) == ["namespace_alpha", "namespace_beta"]
        assert current_registration_owner() is None
        tokens = TokenStore(binding)
        read = {"Authorization": "Bearer " + tokens.create("read", scope="read")[1]}
        admin = {"Authorization": "Bearer " + tokens.create("admin", scope="admin")[1]}
        with TestClient(create_app(binding, start_workers=False)) as client:
            for name in ("alpha", "beta"):
                path = f"/v1/plugins/{name}/ops/status"
                generic = f"/v1/ops/plugins.{name}.status"
                adapter = f"/v1/plugins/{name}/routes/compat/status"
                for url in (path, generic, adapter):
                    response = client.post(url, json={}, headers=read)
                    assert response.status_code == 200, response.text
                    assert response.json() == {"module": "namespace_" + name, "via_daemon": True}
                assert client.post(f"/v1/plugins/{name}/ops/jobs/list", json={}, headers=read).status_code == 200
                # The adapter lives under the plugin but must still require core admin scope.
                create = f"/v1/plugins/{name}/routes/compat/create"
                args = {"name": name, "hosts": ["host-a"]}
                assert client.post(create, json=args, headers=read).status_code == 403
                accepted = client.post(create, json=args, headers={**admin, "Idempotency-Key": name})
                retry = client.post("/v1/cluster/create", json=args, headers={**admin, "Idempotency-Key": name})
                assert accepted.status_code == retry.status_code == 202
                assert accepted.json() == retry.json()
                entry = next(op for op in client.get("/v1/ops", headers=read).json() if op["name"] == f"plugins.{name}.status")
                assert (entry["owner"], entry["local_name"], entry["http_path"]) == (name, "status", path)
            schema = client.get("/openapi.json", headers=read).json()
            for name in ("alpha", "beta"):
                alias = schema["paths"][f"/v1/plugins/{name}/ops/status"]["post"]
                assert alias["operationId"] == f"sparkrun.plugins.{name}.status"
                assert alias["tags"] == ["plugin: " + name]
                assert f"plugins.{name}.statusResult" in schema["components"]["schemas"]
            assert "/v1/alpha/status" not in schema["paths"]
    finally:
        for name in ("namespace_alpha", "namespace_beta"):
            sys.modules.pop(name, None)


def test_duplicate_normalized_directory_ids_fail_before_import(tmp_path):
    from sparkrun.core.external_plugins import load_external_plugins

    roots = [
        write_plugin(tmp_path / name, plugin_id, "namespace_" + name) for name, plugin_id in (("one", "OCI_Relay"), ("two", "oci-relay"))
    ]
    with pytest.raises(RuntimeError, match="identity 'oci-relay'.*one.*two"):
        load_external_plugins(None, paths=roots)
    assert not {"namespace_one", "namespace_two"} & sys.modules.keys()


def test_cross_source_conflict_is_detected_before_loading_any_provider(binding, tmp_path, monkeypatch):
    from sparkrun.core.config import SparkrunConfig

    root = write_plugin(tmp_path / "duplicate", "k8s", "namespace_duplicate")
    config = SparkrunConfig(binding.config_path)
    config.set("features", {"integration.k8s": True, "core.external_plugins": True})
    config.set("plugins.paths", [str(root)])
    config.save()
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS", raising=False)
    monkeypatch.setattr("sparkrun.core.external_plugins.configured_plugin_paths", lambda v: [root])
    monkeypatch.setenv("SPARKRUN_FEATURE_INTEGRATION_K8S", "1")
    with pytest.raises(RuntimeError, match="identity 'k8s'.*in-tree.*external"):
        preflight_plugin_sources(None, config)
    assert "namespace_duplicate" not in sys.modules


@pytest.mark.parametrize("source", ["in-tree", "external", "installed", "application"])
def test_identity_uses_logical_name_without_generation(source):
    assert identity("OCI_Relay", source).plugin_id == "oci-relay"
    assert identity("oci-relay", source).plugin_id == "oci-relay"


@pytest.mark.parametrize("name", ["core", "plugins", "../x", "a..b", "a/../b", "a%2fb", "_bad", "bad-", "a" * 101])
def test_invalid_plugin_ids_are_rejected(name):
    with pytest.raises(ValueError):
        normalize_plugin_id(name)


def test_registration_context_cannot_be_bypassed_and_handlers_cannot_be_replaced(binding):
    op = DaemonOp("status", lambda: declare)
    with pytest.raises(ValueError, match="requires.*context"):
        register_daemon_op(op)
    with plugin_registration(identity()):
        for owner in ("core", "other"):
            with pytest.raises(ValueError, match="owner"):
                register_daemon_op(replace(op, owner=owner))
        with pytest.raises(ValueError, match="core operations"):
            with core_registration():
                pass
        with pytest.raises(ValueError, match="local operation"):
            register_daemon_op(replace(op, name="plugins.other.status"))
        bound = register_plugin_daemon_op(op)
        assert register_daemon_op(op) is bound
        with pytest.raises(ValueError, match="already registered"):
            register_daemon_op(replace(op, handler_loader=lambda: None))
        with pytest.raises(ValueError, match="relative paths"):
            register_plugin_daemon_route(DaemonRoute("/v1/fixture", lambda: None, ("status",), op.request_schema))
    assert current_registration_owner() is None
    with pytest.raises(ValueError, match="context"):
        register_daemon_op(op)


@pytest.mark.parametrize("path", ["../bad", "a/../b", "a%2fb", "a?b", "a#b", "a/{path}", "a/*", "a//b", "a/", "a\\b"])
def test_plugin_adapter_paths_are_literal(path):
    with pytest.raises(ValueError):
        DaemonRoute(path, lambda: None, ("status",), {"type": "object", "additionalProperties": False})


def test_failed_import_rolls_back_operations_routes_and_identity(binding):
    import sparkrun.core.plugin_identity as identities

    owner = identity("k8s")

    def hook(v):
        register_plugin_daemon_op(DaemonOp("list_nodes", lambda: declare))
        register_plugin_daemon_route(DaemonRoute("nodes", lambda: None, ("list_nodes",), {"type": "object", "additionalProperties": False}))
        raise RuntimeError("fail after declarations")

    with pytest.raises(RuntimeError, match="fail after"):
        load_and_register_plugin(lambda: module(owner.module, hook), None, identity=owner)
    assert not registered_daemon_ops()
    assert not registered_daemon_routes()
    assert "k8s" not in identities._IDENTITIES
    assert current_registration_owner() is None
    with pytest.raises(LookupError):
        get_daemon_op("plugins.k8s.list_nodes")


def test_application_context_is_profile_bound_and_conflicts_with_plugin_claims(binding):
    with application_daemon_registration() as owner:
        registered = register_plugin_daemon_op(DaemonOp("status", lambda: declare))
    assert owner.plugin_id == "app-sparkrun"
    assert registered.name == "plugins.app-sparkrun.status"
    with pytest.raises(RuntimeError, match="claimed"):
        preflight_identities([identity("app-sparkrun")])


def test_k8s_discovery_and_dispatch_only_expose_canonical_names(binding):
    from sparkrun.plugins.k8s.daemon import register

    with plugin_registration(PluginIdentity("k8s", "sparkrun.plugins.k8s", ("in-tree", "sparkrun.plugins.k8s", None))):
        register()
    with pytest.raises(LookupError, match="Unknown operation"):
        get_daemon_op("k8s.list_nodes")
    headers = {"Authorization": "Bearer " + TokenStore(binding).create("admin", scope="admin")[1]}
    path = "/v1/plugins/k8s/ops/list_nodes"
    canonical = "plugins.k8s.list_nodes"
    with TestClient(create_app(binding, start_workers=False)) as client:
        operations = client.get("/v1/ops", headers=headers).json()
        assert all("legacy_names" not in op for op in operations)
        assert "k8s.list_nodes" not in {op["name"] for op in operations}
        assert next(op for op in operations if op["name"] == canonical)["http_path"] == path
        for old_path in ("/v1/ops/k8s.list_nodes", "/v1/k8s/list_nodes"):
            assert client.post(old_path, json={}, headers=headers).status_code == 404
        retry_headers = {**headers, "Idempotency-Key": "canonical"}
        accepted = client.post(path, json={}, headers=retry_headers)
        retry = client.post("/v1/ops/" + canonical, json={}, headers=retry_headers)
        assert accepted.status_code == retry.status_code == 202
        assert accepted.json() == retry.json()
        job = JobStore(binding.directory).get(accepted.json()["job_id"])
        assert job["operation"] == canonical
        schema = client.get("/openapi.json", headers=headers).json()
        assert all("legacy_names" not in op for op in schema["x-sparkrun-operations"].values())
        assert "legacy_names" not in schema["paths"][path]["post"]["x-sparkrun-operation"]


def test_canonical_retry_does_not_cross_operation_version_or_owner(binding):
    store = JobStore(binding.directory)
    with plugin_registration(identity("k8s")):
        op = register_plugin_daemon_op(DaemonOp("list_nodes", lambda: None))
    job = store.submit(op, {}, "alice", "", {}, key="retry")
    with pytest.raises(Conflict, match="different arguments"):
        store.submit(replace(op, version=2), {}, "alice", "", {}, key="retry")
    with plugin_registration(identity("other")):
        other = register_plugin_daemon_op(DaemonOp("list_nodes", lambda: None))
    with pytest.raises(Conflict, match="different arguments"):
        store.submit(other, {}, "alice", "", {}, key="retry")
    assert store.submit(op, {}, "alice", "again", {}, key="retry") == job
    store.finish(job, "succeeded")
    store.prune(age=0, count=0)
    with pytest.raises(Conflict, match="expired"):
        store.submit(op, {}, "alice", "", {}, key="retry")


def test_adapter_openapi_ids_preserve_distinct_literal_paths(binding):
    with application_daemon_registration():
        register_plugin_daemon_op(DaemonOp("status", lambda: declare))
        for path in ("a-b", "a_b", "a/b"):
            register_plugin_daemon_route(DaemonRoute(path, lambda: None, ("status",), {"type": "object", "additionalProperties": False}))
    token = TokenStore(binding).create("read", scope="read")[1]
    with TestClient(create_app(binding, start_workers=False)) as client:
        schema = client.get("/openapi.json", headers={"Authorization": "Bearer " + token}).json()
    operations = [schema["paths"][f"/v1/plugins/app-sparkrun/routes/{path}"]["post"] for path in ("a-b", "a_b", "a/b")]
    assert len({op["operationId"] for op in operations}) == 3
    assert all(op["x-sparkrun-adapter"]["operations"] == ["plugins.app-sparkrun.status"] for op in operations)
    assert all(op["requestBody"]["content"]["application/json"]["schema"]["additionalProperties"] is False for op in operations)


def test_loader_rejects_module_different_from_discovered_identity(binding):
    with pytest.raises(ValueError, match="returned.*expected"):
        load_and_register_plugin(lambda: module("different"), None, identity=identity())
    assert current_registration_owner() is None
    assert not registered_daemon_ops()


def test_real_plugin_alias_job_executes_once_and_preserves_feedback(controlled_plugin):
    with real_server(controlled_plugin, extra=("--no-unix", "--tcp", "127.0.0.1:0")) as (_, client):
        args = {"mode": "feedback"}
        accepted = client.request("POST", "/v1/plugins/daemon-fixture/ops/wait", args, headers={"Idempotency-Key": "alias-once"})
        job = accepted["job_id"]
        assert client.wait(job, timeout=15, prompt=lambda _: "private-namespace-token") == {"accepted": True}
        assert client.call("plugins.daemon-fixture.wait", args, key="alias-once") == accepted
        events = list(client.events(job))
        feedback = [event for event in events if event["type"] == "feedback"]
        assert [event["message"] for event in feedback] == ["Receiving [redacted]", "Verified"]
        assert all(event["operation_id"] == job for event in feedback)
        assert "private-namespace-token" not in json.dumps(events)
        assert (controlled_plugin.config_path.parent / "executions.txt").read_text().splitlines() == [job]


@pytest.mark.parametrize("source", ["directory", "managed", "installed"])
def test_actual_loaders_bind_the_same_manifest_namespace(binding, tmp_path, monkeypatch, source):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.external_plugins import load_external_plugins
    from sparkrun.core.installed_plugins import load_installed_plugins, installed_plugin_inventory
    from sparkrun.core.managed_plugins import load_managed_package
    from sparkrun.core.plugin_manifest import PluginMetadata

    module_name = "namespace_provider"
    root = write_plugin(tmp_path / "provider", "stable-id", module_name)
    try:
        if source == "directory":
            assert load_external_plugins(None, paths=[root]) == [module_name]
        elif source == "managed":
            metadata = PluginMetadata(
                "stable-id", module_name, "provider", "2.0", module_name, "https://example.invalid/provider", ">=0.4", ">=3.12"
            )
            load_managed_package(metadata, root / module_name, None, source_path=root / "generation-b")
        else:
            monkeypatch.syspath_prepend(str(root))
            monkeypatch.delenv("SPARKRUN_NO_INSTALLED_PLUGINS", raising=False)
            entry = SimpleNamespace(
                name="stable_id",
                value=module_name,
                dist=SimpleNamespace(name="provider", version="3.0"),
                load=lambda: importlib.import_module(module_name),
            )
            monkeypatch.setattr("sparkrun.core.installed_plugins.entry_points", lambda **_: [entry])
            config = SparkrunConfig(binding.config_path)
            config.set("integrations.stable_id", True)
            load_installed_plugins(None, config=config)
            assert installed_plugin_inventory()[0].loaded
        assert get_daemon_op("plugins.stable-id.status").owner == "stable-id"
        assert get_daemon_op("plugins.stable-id.cluster.create").local_name == "cluster.create"
        assert current_registration_owner() is None
    finally:
        sys.modules.pop(module_name, None)


def test_installed_manifest_mismatch_is_diagnosable_before_import(binding, tmp_path, monkeypatch):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.installed_plugins import load_installed_plugins, installed_plugin_inventory

    module_name = "namespace_mismatch"
    root = write_plugin(tmp_path / "mismatch", "different-id", module_name)
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.delenv("SPARKRUN_NO_INSTALLED_PLUGINS", raising=False)
    entry = SimpleNamespace(
        name="expected-id",
        value=module_name,
        dist=SimpleNamespace(name="provider", version="1.0"),
        load=lambda: pytest.fail("mismatched metadata must not import"),
    )
    monkeypatch.setattr("sparkrun.core.installed_plugins.entry_points", lambda **_: [entry])
    config = SparkrunConfig(binding.config_path)
    config.set("integrations.expected-id", True)
    load_installed_plugins(None, config=config)
    row = installed_plugin_inventory()[0]
    assert not row.loaded and "disagrees with entry point" in row.failure
    assert module_name not in sys.modules


def test_managed_namespace_survives_generation_and_version_changes(binding, monkeypatch):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.managed_plugins import managed_plugin_identities

    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS", raising=False)
    config = SparkrunConfig(binding.config_path)
    config.set("features", {"core.external_plugins": True})
    claims = []
    for generation, version in (("a" * 32, "1.0"), ("b" * 32, "2.0")):
        records = {
            "Stable_ID": {
                "active_generation": generation,
                "metadata": {"name": "Stable_ID", "module": "namespace_provider", "version": version},
            }
        }
        claims.append(managed_plugin_identities(config, records)["Stable_ID"])
    assert claims[0].plugin_id == claims[1].plugin_id == "stable-id"
    assert claims[0].source != claims[1].source


@pytest.mark.parametrize("names", [("Same_ID", "same-id"), ("same", "same")])
def test_installed_duplicate_ids_fail_before_any_import(binding, monkeypatch, names):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.installed_plugins import load_installed_plugins, installed_plugin_inventory, PluginConflictError

    entries = [
        SimpleNamespace(
            name=name,
            value="namespace_" + str(index),
            dist=SimpleNamespace(name="provider-" + str(index), version="1.0"),
            load=lambda: pytest.fail("ambiguous providers must not import"),
        )
        for index, name in enumerate(names)
    ]
    monkeypatch.setattr("sparkrun.core.installed_plugins.entry_points", lambda **_: entries)
    monkeypatch.delenv("SPARKRUN_NO_INSTALLED_PLUGINS", raising=False)
    config = SparkrunConfig(binding.config_path)
    config.set("integrations", dict.fromkeys(names, True))
    with pytest.raises(PluginConflictError, match="provider-0.*provider-1"):
        load_installed_plugins(None, config=config)
    assert all(row.failure and not row.loaded for row in installed_plugin_inventory())
