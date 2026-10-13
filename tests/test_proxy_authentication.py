"""Portable LiteLLM model-list authentication through the public CLI/API."""

import pytest


@pytest.mark.parametrize("source", ["explicit", "environment", "legacy-environment"])
def test_models_cli_authenticates_to_running_litellm(tmp_path, monkeypatch, source):
    """Real loopback HTTP verifies the saved port, bearer header and CLI result."""
    import json
    import os
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from click.testing import CliRunner
    from sparkrun.cli import main
    from sparkrun.api.proxy import _ops
    from sparkrun.proxy.engine import ProxyEngine

    secret = "fixture-master-secret"
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            if self.headers.get("Authorization") != "Bearer " + secret:
                self.send_error(401)
                return
            payload = json.dumps({"data": [{"model_name": "fixture-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
        if source == "environment":
            monkeypatch.setenv("LITELLM_MASTER_KEY", secret)
        started = ProxyEngine(state_dir=tmp_path, port=server.server_port, master_key=secret if source == "explicit" else None)
        started._save_state(pid=os.getpid())
        if source == "legacy-environment":
            # Old versions inherited this variable into LiteLLM but recorded
            # master_key: null, so management must retain an env fallback.
            monkeypatch.setenv("LITELLM_MASTER_KEY", secret)
        else:
            monkeypatch.setenv("LITELLM_MASTER_KEY", "different-shell-key")
        monkeypatch.setattr(_ops, "_running_engine", lambda sctx=None: ProxyEngine(state_dir=tmp_path))
        monkeypatch.setattr(ProxyEngine, "is_running", lambda self: True)
        result = CliRunner().invoke(main, ["proxy", "models", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)[0]["model_name"] == "fixture-model"
        assert seen == [("/model/info", "Bearer " + secret)]
        assert secret not in result.output
        if source == "environment":
            # A fresh management process no longer needs the starting shell.
            monkeypatch.delenv("LITELLM_MASTER_KEY")
            assert _ops.models()[0].model_name == "fixture-model"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("key,expected", [(None, "from-env"), ("explicit", "explicit"), ("", "")])
def test_litellm_launch_credentials_are_consistent(tmp_path, monkeypatch, key, expected):
    from sparkrun.proxy.engine import ProxyEngine
    import yaml

    monkeypatch.setenv("LITELLM_MASTER_KEY", "from-env")
    engine = ProxyEngine(state_dir=tmp_path, master_key=key)
    engine.prepare_config([], {})
    config = yaml.safe_load(engine.config_path.read_text())
    assert engine.master_key == expected
    assert engine._state_payload()["master_key"] == expected
    assert config.get("general_settings", {}).get("master_key") == (expected or None)
    assert engine._build_env().get("LITELLM_MASTER_KEY") == (expected or None)
    assert engine.data_plane_authenticated is bool(expected)


def test_litellm_environment_does_not_configure_other_gateway_backends(tmp_path, monkeypatch):
    from sparkrun.proxy.config import ProxyConfig
    from sparkrun.proxy.supervisor import GatewaySupervisor

    monkeypatch.setenv("LITELLM_MASTER_KEY", "backend-private")
    assert ProxyConfig(tmp_path / "proxy.yaml").master_key is None
    assert GatewaySupervisor(state_dir=tmp_path).data_plane_authenticated is False
