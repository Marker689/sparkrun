"""Credential-bound automation ownership across admission, retries and workers."""

import asyncio
import json
import os
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from test_daemon import binding as binding
from sparkrun.core.operation import OperationState
from sparkrun.daemon.auth import TokenStore
from sparkrun.daemon.jobs import Conflict, JobStore
from sparkrun.daemon.management import write_connection_descriptor
from sparkrun.daemon.server import Boundary, create_app
from sparkrun.daemon.worker import Worker


def test_manager_is_server_selected_and_durable_across_credential_rotation(binding):
    tokens = TokenStore(binding)
    first, token = tokens.create("gateway", manager="sparkroute", principal_id="manager:sparkroute")
    app = create_app(binding, start_workers=False)
    args = {"recipe": "recipe", "recipe_revision": "rev"}
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer " + token, "Idempotency-Key": "intent", "X-Sparkrun-Client": "other-owner"}
        assert client.get("/v1/info", headers=headers).json()["manager"] == "sparkroute"
        job = client.post("/v1/workloads/ensure_ready", json=args, headers=headers).json()["job_id"]
        saved = app.state.store.get(job)
        assert saved["manager"] == "sparkroute" and saved["arguments"] == args
        assert Worker(binding, job).operation.manager == "sparkroute"
        for spoof in ("owner", "manager", "management_identity"):
            assert client.post("/v1/workloads/ensure_ready", json={**args, spoof: "other"}, headers=headers).status_code == 422
        _, replacement = tokens.create("rotated", manager="sparkroute", principal_id="manager:sparkroute")
        tokens.revoke(first)
        assert client.post("/v1/workloads/ensure_ready", json=args, headers=headers).status_code == 401
        headers["Authorization"] = "Bearer " + replacement
        assert client.post("/v1/ops/workloads.ensure_ready", json=args, headers=headers).json()["job_id"] == job
        # An accepted operation keeps its grant after credential revocation.
        assert Worker(binding, job).operation.manager == "sparkroute"
    assert JobStore(binding.directory).get(job)["manager"] == "sparkroute"


def test_same_principal_cannot_replay_an_intent_with_another_manager(binding):
    from sparkrun.daemon.ops import register_core_ops
    from sparkrun.core.daemon_registry import get_daemon_op

    register_core_ops()
    store = JobStore(binding.directory)
    op = get_daemon_op("workloads.ensure_ready")
    store.submit(op, {"recipe": "r"}, "one-principal", "", {}, key="intent", manager="sparkroute")
    with pytest.raises(Conflict):
        store.submit(op, {"recipe": "r"}, "one-principal", "", {}, key="intent", manager="daemon")


def test_client_label_does_not_grant_management_rights(binding):
    _, token = TokenStore(binding).create("client")
    with TestClient(create_app(binding, start_workers=False)) as client:
        headers = {"Authorization": "Bearer " + token, "X-Sparkrun-Client": "sparkroute", "X-Sparkrun-Manager": "sparkroute"}
        response = client.post("/v1/workloads/inventory", json={}, headers=headers)
        assert response.status_code == 202
        job = client.get("/v1/jobs/" + response.json()["job_id"], headers=headers).json()
        assert job["manager"] == "daemon"


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="UDS peer identity is POSIX")
def test_explicit_bearer_on_uds_takes_precedence_over_local_admin(binding):
    tokens = TokenStore(binding)
    _, token = tokens.create("gateway", scope="read", manager="sparkroute")
    seen, messages = [], []

    async def app(scope, receive, send):
        seen.append(scope["sparkrun.principal"])

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "path": "/v1/info", "method": "GET", "sparkrun.peer_uid": os.getuid()}
    boundary = Boundary(app, tokens=tokens)
    asyncio.run(boundary({**scope, "headers": [(b"authorization", ("Bearer " + token).encode())]}, receive, send))
    assert seen[-1].manager == "sparkroute" and seen[-1].scope == "read" and not seen[-1].local
    asyncio.run(boundary({**scope, "headers": [(b"authorization", b"Bearer invalid")]}, receive, send))
    assert messages[0]["status"] == 401 and len(seen) == 1
    asyncio.run(boundary({**scope, "headers": []}, receive, send))
    assert seen[-1].local and seen[-1].manager == "daemon"


def test_descriptor_rotation_reuses_identity_without_accumulating_credentials(binding, tmp_path):
    path = tmp_path / "gateway" / "daemon-client.json"
    tokens = TokenStore(binding)
    identities = []
    for _ in range(2):
        write_connection_descriptor(binding, path, manager="sparkroute")
        descriptor = json.loads(path.read_text())
        from pathlib import Path

        token = Path(descriptor["token_file"]).read_text()
        principal = tokens.authenticate(token)
        identities.append((principal.id, principal.manager))
        assert "token" not in descriptor and not principal.secret
    assert identities == [("manager:sparkroute", "sparkroute")] * 2
    assert len(tokens.list()) == 1 and len(list(path.parent.glob("*.token"))) == 1


def test_token_retirement_refuses_new_admission_and_keeps_other_managers(binding):
    tokens = TokenStore(binding)
    _, token = tokens.create("gateway", manager="sparkroute")
    principal = tokens.authenticate(token)
    _, other = tokens.create("other")
    with pytest.raises(RuntimeError):
        with tokens.retire_manager("sparkroute"):
            raise RuntimeError("active jobs")
    assert tokens.authenticate(token).manager == "sparkroute"
    with tokens.retire_manager("sparkroute"):
        pass
    with pytest.raises(PermissionError):
        with tokens.admission(principal):
            pytest.fail("retired credential admitted")
    assert tokens.authenticate(other).manager == "daemon"


def test_core_adapters_use_one_ownership_policy_for_run_and_ensure(monkeypatch):
    from sparkrun import api
    from sparkrun.daemon.ops import invoke_options
    from sparkrun.daemon.workloads import invoke

    sctx = NS(operation=OperationState(via_daemon=True, manager="sparkroute"))
    ready = Mock(return_value={})
    plan = Mock(return_value={})
    monkeypatch.setattr(api.workloads, "ensure_ready", ready)
    monkeypatch.setattr(api, "plan", plan)
    invoke("ensure_ready", {"recipe": "r"}, sctx)
    invoke_options("plan", "RunOptions", {"recipe": "r"}, sctx)
    assert ready.call_args.kwargs["owner"] == "sparkroute"
    assert plan.call_args.args[0].owner == "sparkroute"


@pytest.mark.parametrize("manager", ["", "../owner", "Owner", "a" * 65, None])
def test_manager_identifier_validation(binding, manager):
    with pytest.raises(ValueError, match="Manager"):
        TokenStore(binding).create("invalid", manager=manager)


def test_inline_queries_receive_validated_manager_and_cache_by_identity(binding):
    from sparkrun.core.daemon_registry import DaemonOp, register_daemon_op
    from sparkrun.core.plugin_identity import application_daemon_registration

    with application_daemon_registration():
        register_daemon_op(
            DaemonOp("manager", lambda: lambda args, sctx, prompt: {"manager": sctx.operation.manager}, kind="query", scope="read")
        )
    tokens = TokenStore(binding)
    grants = [
        tokens.create("fixture", scope="read", principal_id="same-principal", manager=manager)[1] for manager in ("daemon", "sparkroute")
    ]
    with TestClient(create_app(binding, start_workers=False)) as client:
        for manager, token in zip(("daemon", "sparkroute"), grants, strict=True):
            value = client.post("/v1/ops/plugins.app-sparkrun.manager", json={}, headers={"Authorization": "Bearer " + token})
            assert value.status_code == 200 and value.json() == {"manager": manager}


def test_retirement_and_submission_share_an_admission_lock(binding):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    tokens = TokenStore(binding)
    _, token = tokens.create("gateway", manager="sparkroute")
    principal = tokens.authenticate(token)
    attempting = threading.Event()
    admitted = threading.Event()

    def submit():
        attempting.set()
        with pytest.raises(PermissionError):
            with tokens.admission(principal):
                admitted.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with tokens.retire_manager("sparkroute"):
            future = executor.submit(submit)
            assert attempting.wait(2)
            assert not admitted.is_set()
        future.result(timeout=5)
    assert not admitted.is_set()
