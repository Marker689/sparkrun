"""Parameter-level coverage and execution of the reviewed model-cache bindings."""

import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.api import model_cache
from sparkrun.core.daemon_registry import get_daemon_op
from sparkrun.daemon.model_cache_ops import BINDINGS
from sparkrun.daemon.ops import _resolve, check_request, register_core_ops
from test_model_artifacts import artifact, SHA


def assert_parameters_classified(binding, target):
    groups = [set(binding.forwarded.split()), set(binding.adapted_parameters), set(binding.injected), set(binding.excluded)]
    classified = set().union(*groups)
    assert sum(map(len, groups)) == len(classified), f"{binding.name}: duplicate parameter classification"
    parameters = set(inspect.signature(target).parameters)
    assert parameters == classified, (
        f"{binding.name}: unclassified parameters {parameters - classified}; stale bindings {classified - parameters}"
    )
    assert all(binding.excluded.values()), f"{binding.name}: exclusions need explanations"
    assert set(binding.injected) <= {"ssh_kwargs", "config"}
    assert set(binding.required.split()) <= set(binding.forwarded.split()) | set(binding.request_fields.split())
    assert not set(binding.forwarded.split()) & set(binding.request_fields.split())


@pytest.mark.parametrize("binding", BINDINGS, ids=lambda b: b.name)
def test_every_python_parameter_has_a_reviewed_binding(binding):
    assert_parameters_classified(binding, _resolve(binding.target))


def test_new_target_parameter_requires_review_without_becoming_a_wire_input():
    binding = next(b for b in BINDINGS if b.name == "model_cache.delete")
    target = Mock()
    signature = inspect.signature(_resolve(binding.target))
    target.__signature__ = signature.replace(
        parameters=[*signature.parameters.values(), inspect.Parameter("new_safety_option", inspect.Parameter.KEYWORD_ONLY, default=None)]
    )
    with pytest.raises(AssertionError, match="model_cache.delete: unclassified parameters.*new_safety_option"):
        assert_parameters_classified(binding, target)
    assert "new_safety_option" not in binding.request_schema()["properties"]
    with pytest.raises(AssertionError, match="stale bindings.*allowed_revisions"):
        target.__signature__ = signature.replace(parameters=[p for p in signature.parameters.values() if p.name != "allowed_revisions"])
        assert_parameters_classified(binding, target)


@pytest.mark.parametrize(
    "operation,arguments,expected",
    [
        ("list", {"cache_dir": "/cache"}, {}),
        ("prune", {"cache_dir": "/cache", "dry_run": False, "max_bytes": 0}, {"dry_run": False, "max_bytes": 0}),
        ("pin", {"cache_dir": "/cache", "model": "org/model", "revision": SHA, "pinned": False}, {"pinned": False}),
        (
            "delete",
            {"cache_dir": "/cache", "model": "org/model", "dry_run": False, "allowed_revisions": []},
            {"dry_run": False, "allowed_revisions": []},
        ),
        ("adopt", {"cache_dir": "/cache", "all_models": True, "offline": False}, {"all_models": True, "offline": False}),
        (
            "reconcile",
            {"cache_dir": "/cache", "model": "org/model", "revision": SHA, "claim": "workload", "confirm_inactive": True},
            {"claim": "workload", "confirm_inactive": True},
        ),
        ("inspect", {"cache_dir": "/cache", "model": "org/model", "checksums": False}, {"checksums": False}),
        ("repair", {"cache_dir": "/cache", "model": "org/model", "dry_run": False, "offline": True}, {"dry_run": False, "offline": True}),
        ("resolve", {"model": "org/model", "file_selection": "all"}, {}),
        ("import", {"cache_dir": "/cache", "model": "org/model", "manifest": json.loads(artifact().to_json())}, {}),
    ],
)
def test_registered_handler_preserves_arguments_and_server_injection(monkeypatch, operation, arguments, expected):
    binding = next(b for b in BINDINGS if b.name == "model_cache." + operation)
    target = Mock(return_value=())
    monkeypatch.setattr(model_cache, binding.target.split(".")[1], target)
    ssh = {"user": "daemon-user"}
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda config: ssh)
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example")
    config = object()
    register_core_ops()
    op = get_daemon_op(binding.name)
    original = dict(arguments)
    assert op.handler_loader()(check_request(op, arguments), SimpleNamespace(config=config)) == []
    kwargs = target.call_args.kwargs
    assert arguments == original
    assert all(kwargs[key] == value for key, value in expected.items())
    for field in binding.forwarded.split():
        assert (field in kwargs) == (field in arguments)
    assert not set(binding.excluded) & kwargs.keys()
    if "ssh_kwargs" in binding.injected:
        assert kwargs["ssh_kwargs"] is ssh
    if "config" in binding.injected:
        assert kwargs["config"] is config
    if "request" in kwargs:
        if operation == "adopt":
            assert kwargs["request"] is None and kwargs["endpoint"] == "https://mirror.example"
        else:
            assert kwargs["request"].model == "org/model"
            assert kwargs["request"].revision == arguments.get("revision")
    if operation == "resolve":
        assert kwargs["request"].file_selection == "all"
    if operation == "import":
        assert kwargs["manifest"] == artifact()


@pytest.mark.parametrize("operation", ["delete", "prune", "adopt"])
def test_omitting_dry_run_preserves_python_preview_default(operation):
    binding = next(b for b in BINDINGS if b.name == "model_cache." + operation)
    arguments = {"cache_dir": "/cache", "model": "org/model"} if operation != "prune" else {"cache_dir": "/cache"}
    from sparkrun import api

    bound = binding.arguments(arguments, api.default_sctx())
    assert "dry_run" not in bound
    signature = inspect.signature(_resolve(binding.target))
    call = signature.bind(**bound)
    call.apply_defaults()
    assert call.arguments["dry_run"] is True


def test_adoption_request_endpoint_is_consumed_and_bare_revision_rejected(monkeypatch):
    binding = next(b for b in BINDINGS if b.name == "model_cache.adopt")
    from sparkrun import api

    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example")
    data = binding.arguments({"cache_dir": "/cache", "model": "org/model"}, api.default_sctx())
    assert data["request"].endpoint == "https://mirror.example" and data["endpoint"] is None
    with pytest.raises(ValueError, match="revision needs a model"):
        binding.arguments({"cache_dir": "/cache", "revision": SHA}, api.default_sctx())


@pytest.mark.parametrize("key,value", [("token", "secret"), ("sctx", {}), ("session", {}), ("ssh_kwargs", {}), ("allowed_revisions", None)])
def test_delete_rejects_local_only_or_invalid_arguments(key, value):
    register_core_ops()
    with pytest.raises(ValueError):
        check_request(get_daemon_op("model_cache.delete"), {"cache_dir": "/cache", "model": "org/model", key: value})


def test_reconcile_retains_explicit_true_constraint():
    register_core_ops()
    with pytest.raises(ValueError):
        check_request(
            get_daemon_op("model_cache.reconcile"),
            {"cache_dir": "/cache", "model": "org/model", "revision": SHA, "claim": "workload", "confirm_inactive": False},
        )


def test_direct_adapter_does_not_silently_drop_unknown_arguments():
    from sparkrun.daemon.special_ops import execute

    with pytest.raises(ValueError, match="unknown fields: token"):
        execute("model_cache.delete", {"model": "org/model", "cache_dir": "/cache", "token": "untrusted"}, SimpleNamespace(config=None))
