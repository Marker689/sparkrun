# SPDX-FileCopyrightText: 2026 Spark Arena
# SPDX-License-Identifier: Apache-2.0

"""Offline diagnostics contracts, runnable against source or the vendored adapter."""

import importlib
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun import api
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.context import SparkrunContext
from sparkrun.core.plugin_identity import PluginIdentity, plugin_registration
from scitrera_app_framework import Variables


@pytest.fixture
def relay(tmp_path, monkeypatch):
    module = importlib.import_module(os.environ.get("OCI_RELAY_TEST_MODULE", "sparkrun.plugins.oci_relay"))
    provider = importlib.import_module(module.__name__ + ".provider")
    diagnostics = importlib.import_module(module.__name__ + ".diagnostics")
    monkeypatch.setattr("sparkrun.core.plugin_identity._IDENTITIES", {})
    monkeypatch.setattr("sparkrun.diagnostics.plugin._PLUGINS", {})
    monkeypatch.setattr(diagnostics, "_factory", None)
    with plugin_registration(PluginIdentity("oci-relay", module.__name__, ("external", module.__name__, None))):
        diagnostics.register()
    monkeypatch.setattr("sparkrun.diagnostics.session.DiagnosticsCapture.collect_config", lambda self: None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    context = SparkrunContext(variables=Variables(), config=SparkrunConfig(config_path))
    return provider, diagnostics, context


def test_failure_snapshot_precedes_transfer_cleanup(relay, tmp_path, monkeypatch):
    provider, diagnostics, sctx = relay
    path = tmp_path / "failure.ndjson"
    primary = provider.OperationError("staging failed: Bearer credential-value")

    def close():
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        assert any(e.get("section") == "receiver_failure" for e in entries)
        return []

    runner = SimpleNamespace(architecture=Mock(side_effect=primary), close=Mock(side_effect=close))
    monkeypatch.setattr(provider, "Runner", lambda *args: runner)
    with pytest.raises(provider.OperationError) as caught:
        with api.capture_diagnostics(path, sctx=sctx):
            request = api_request()
            provider.RelayProvider()._copy(request, {}, 30)
    assert caught.value is primary
    runner.close.assert_called_once()
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    snapshots = [e for e in entries if e["_type"] == "plugin_diagnostic"]
    assert [e["section"] for e in snapshots] == ["receiver_failure", "transfer_result"]
    assert all(e["plugin"] == "oci-relay" for e in snapshots)
    assert "credential-value" not in path.read_text()
    assert entries[-1]["operation_success"] is False


def api_request():
    from sparkrun.plugins import ImageCopyRequest

    return ImageCopyRequest(image="example/image", source_host=None, targets=("receiver",), transfer_hosts=("receiver",))


def test_disabled_diagnostics_do_not_construct_payloads(relay, monkeypatch):
    provider, diagnostics, sctx = relay
    from sparkrun.diagnostics.plugin import DISABLED

    assert diagnostics.bind(sctx=sctx) is DISABLED
    monkeypatch.setattr("sparkrun.diagnostics._plugin_sink.payload_copy", Mock(side_effect=AssertionError("no payload work")))
    primary = provider.OperationError("same failure")
    runner = SimpleNamespace(architecture=Mock(side_effect=primary), close=Mock(return_value=[]))
    monkeypatch.setattr(provider, "Runner", lambda *args: runner)
    with pytest.raises(provider.OperationError) as caught:
        provider.RelayProvider()._copy(api_request(), {}, 30)
    assert caught.value is primary
    runner.close.assert_called_once()


def test_older_host_keeps_diagnostics_optional(relay, monkeypatch):
    import sparkrun.plugins

    _, diagnostics, _ = relay
    monkeypatch.delattr(sparkrun.plugins, "DIAGNOSTICS_API_VERSION")
    diagnostics.register()
    assert diagnostics.bind() is None
