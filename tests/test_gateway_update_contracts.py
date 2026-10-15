"""Registered providers share update errors without hiding bugs or partial saves."""

from unittest.mock import Mock

import pytest
import yaml
from click.testing import CliRunner

from sparkrun import api
from sparkrun.api.proxy import _ops
from sparkrun.core.features import FeatureFlag, register_feature
from sparkrun.proxy.contracts import GatewayOperationError, ProxyModel
from sparkrun.proxy.gateway import register_gateway
from sparkrun.proxy.supervisor import GatewaySupervisor
from test_proxy_management_initialization import prepare_application


class ContractGateway(GatewaySupervisor):
    gateway_name = "contract-test"
    wants_proxy_config = True
    failure = None
    ui_url = "http://localhost/admin"
    admin_bind_host = "localhost"
    admin_exposed = False
    admin_auth_required = True

    def __init__(self, *, proxy_config, sctx):
        super().__init__()
        self.proxy_config = proxy_config
        self.sctx = sctx

    def fail(self):
        raise self.failure

    def sync_models(self, endpoints, aliases=None):
        return self.fail()

    def sync_aliases(self, aliases):
        return self.fail()

    def register_loaded_model(self, recipe, overrides=None, cluster=None):
        return self.fail()

    def unregister_loaded_model(self, recipe):
        return self.fail()

    def admin_token(self, *, rotate=False, clear=False):
        return self.fail()

    def issue_ui_credential(self):
        return self.fail()

    def stop(self, *, dry_run=False):
        return self.fail()

    def query_models(self):
        return (ProxyModel("served"),)


def invoke_update(operation, context):
    if operation == "sync":
        return api.proxy.sync(endpoints=[], require_running=True, sctx=context)
    if operation == "add":
        return api.proxy.add_alias("friendly", "new-target", sctx=context)
    if operation == "remove":
        return api.proxy.remove_alias("friendly", sctx=context)
    if operation in {"register_loaded_model", "unregister_loaded_model"}:
        return getattr(api.proxy, operation)("fixture", sctx=context)
    if operation == "ui":
        return api.proxy.ui(issue_token=True, sctx=context)
    return getattr(api.proxy, operation)(sctx=context)


@pytest.fixture
def registered_provider(tmp_path, monkeypatch):
    prepare_application(tmp_path, monkeypatch)
    context = api.default_sctx()
    from sparkrun.core import features
    from sparkrun.proxy import gateway

    # Registration is process-global. Use fixture-owned containers so later
    # gateway-selection tests cannot see this provider or its enabled flag.
    monkeypatch.setattr(features, "FEATURE_FLAGS", dict(features.FEATURE_FLAGS))
    monkeypatch.setattr(gateway, "GATEWAY_FEATURE_FLAGS", dict(gateway.GATEWAY_FEATURE_FLAGS))
    monkeypatch.setattr(gateway, "_GATEWAY_LOADERS", dict(gateway._GATEWAY_LOADERS))
    register_feature(FeatureFlag(name="gateway.contract-test", description="contract fixture", default=True))
    register_gateway("contract-test", feature_flag="gateway.contract-test", loader=lambda: ContractGateway)
    state = tmp_path / "cache" / "proxy" / "state.yaml"
    data = yaml.safe_load(state.read_text())
    data["gateway"] = "contract-test"
    state.write_text(yaml.safe_dump(data))
    context.proxy_config.add_alias("friendly", "old-target")
    context.proxy_config.save()
    monkeypatch.setattr(_ops, "_discover", lambda **kwargs: [])
    return context


@pytest.mark.parametrize(
    "operation", ["sync", "add", "remove", "register_loaded_model", "unregister_loaded_model", "ui", "admin_token", "stop"]
)
@pytest.mark.parametrize("error_type", [GatewayOperationError, RuntimeError, NotImplementedError, KeyboardInterrupt])
def test_registered_provider_update_boundary(registered_provider, monkeypatch, operation, error_type):
    error = error_type("provider refused operation")
    monkeypatch.setattr(ContractGateway, "failure", error)
    expected = api.proxy.ProxyUpdateFailed if error_type is GatewayOperationError else error_type
    with pytest.raises(expected) as caught:
        invoke_update(operation, registered_provider)
    if error_type is GatewayOperationError:
        assert isinstance(caught.value, api.SparkrunError)
        assert caught.value.__cause__ is error
    else:
        assert caught.value is error
    if operation in {"add", "remove"}:
        saved = yaml.safe_load(registered_provider.proxy_config.config_path.read_text())
        assert saved.get("aliases", {}) == ({"friendly": "new-target"} if operation == "add" else {})


@pytest.mark.parametrize("operation", ["add", "remove"])
def test_alias_cli_reports_saved_change_and_provider_failure(registered_provider, monkeypatch, operation):
    from sparkrun.cli import main

    monkeypatch.setattr(ContractGateway, "failure", GatewayOperationError("cannot reconcile fixture"))
    args = ["proxy", "alias", operation, "friendly"] + (["new-target"] if operation == "add" else [])
    result = CliRunner().invoke(main, args, obj={"sparkrun_ctx": registered_provider})
    assert result.exit_code == 1
    assert "Alias " + ("added" if operation == "add" else "removed") in result.stdout
    assert "Error: cannot reconcile fixture" in result.stderr
    assert isinstance(result.exception, SystemExit)


def test_successful_sync_exposes_routing_warnings_in_api_and_cli(registered_provider, monkeypatch):
    from sparkrun.cli import main

    warning = "Discovered deployment is ready; its public model name is reserved"
    monkeypatch.setattr(ContractGateway, "sync_models", lambda *args: (1, 0))
    monkeypatch.setattr(ContractGateway, "discovery_warnings", lambda self: (warning,))
    result = api.proxy.sync(endpoints=[], sctx=registered_provider)
    assert result.changed and result.warnings == (warning,)
    cli = CliRunner().invoke(main, ["proxy", "sync"], obj={"sparkrun_ctx": registered_provider})
    assert cli.exit_code == 0, cli.output
    assert "Synchronized running models" in cli.output
    assert "Routing warning: " + warning in cli.output


def test_operational_construction_failure_retains_process_recovery(registered_provider, monkeypatch):
    monkeypatch.setattr(ContractGateway, "__init__", Mock(side_effect=GatewayOperationError("invalid provider configuration")))
    discovery = Mock(side_effect=AssertionError("discovered for recovery-only provider"))
    monkeypatch.setattr(_ops, "_discover", discovery)
    stop = Mock(return_value=True)
    monkeypatch.setattr(GatewaySupervisor, "stop", stop)
    status = api.proxy.status(sctx=registered_provider)
    assert status.running and status.gateway == "contract-test" and status.model_query_error
    assert api.proxy.stop(sctx=registered_provider).stopped
    with pytest.raises(api.proxy.GatewayUnavailable, match="contract-test"):
        api.proxy.sync(require_running=True, sctx=registered_provider)
    discovery.assert_not_called()
    stop.assert_called_once_with(dry_run=False)


@pytest.mark.parametrize("error_type", [RuntimeError, NotImplementedError, KeyboardInterrupt])
def test_constructor_bugs_and_interrupts_do_not_enter_recovery(registered_provider, monkeypatch, error_type):
    error = error_type("constructor bug")
    monkeypatch.setattr(ContractGateway, "__init__", Mock(side_effect=error))
    with pytest.raises(error_type) as caught:
        api.proxy.status(sctx=registered_provider)
    assert caught.value is error
