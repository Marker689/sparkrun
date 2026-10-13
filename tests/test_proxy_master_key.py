"""Tests for stateless master-key auth (no DB env-vars)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestBuildLitellmConfigMasterKey:
    """Test build_litellm_config emits master_key without DB fields."""

    def test_master_key_set_emits_general_settings(self):
        """With master_key='secret', config has general_settings.master_key='secret'."""
        from sparkrun.plugins.litellm.engine import build_litellm_config

        config = build_litellm_config([], master_key="secret")

        assert "general_settings" in config
        assert config["general_settings"]["master_key"] == "secret"

    def test_master_key_set_no_database_keys(self):
        """With master_key, config has NO database_url / store_model_in_db keys."""
        from sparkrun.plugins.litellm.engine import build_litellm_config

        config = build_litellm_config([], master_key="secret")

        gen = config.get("general_settings", {})
        assert "database_url" not in gen
        assert "store_model_in_db" not in gen
        # Also check the top-level dict
        assert "database_url" not in config
        assert "store_model_in_db" not in config

    def test_master_key_none_no_general_settings(self):
        """With master_key=None, config has no general_settings.master_key."""
        from sparkrun.plugins.litellm.engine import build_litellm_config

        config = build_litellm_config([], master_key=None)

        # general_settings should be absent entirely (or contain no master_key)
        assert "master_key" not in config.get("general_settings", {})


class TestStartupEnvironmentNoDatabaseUrl:
    """Test that the litellm subprocess env never contains a database env var."""

    @pytest.fixture
    def state_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "proxy_state"
        d.mkdir()
        return d

    def _run_engine_start_capturing_env(
        self,
        state_dir: Path,
        master_key: str | None,
    ):
        """Invoke ProxyEngine.start() and capture the env passed to Popen."""
        from sparkrun.plugins.litellm.engine import ProxyEngine

        captured_envs: list[dict] = []

        def fake_popen(*args, **kwargs):
            captured_envs.append(dict(kwargs.get("env") or {}))
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = None  # still running
            mock_proc.wait.return_value = 0
            return mock_proc

        engine = ProxyEngine(
            host="127.0.0.1",
            port=14123,
            master_key=master_key,
            state_dir=state_dir,
        )

        # Write a minimal litellm config to avoid path errors
        config_path = state_dir / "litellm_config.yaml"
        config_path.write_text("model_list: []\n")

        with (
            patch("sparkrun.plugins.litellm.engine.shutil.which", return_value="/usr/bin/uvx"),
            patch("sparkrun.plugins.litellm.engine.subprocess.Popen", side_effect=fake_popen),
            patch("time.sleep", lambda *_a, **_k: None),
        ):
            rc = engine.start(config_path=config_path, foreground=False)
        assert rc == 0
        assert captured_envs, "Popen was not called"
        return captured_envs[0]

    def test_no_database_url_when_master_key_set(self, state_dir: Path):
        """ProxyEngine.start() never sets DATABASE_URL, even with a master_key."""
        env = self._run_engine_start_capturing_env(state_dir, master_key="secret")
        assert "DATABASE_URL" not in env

    def test_no_database_url_when_master_key_none(self, state_dir: Path):
        """ProxyEngine.start() never sets DATABASE_URL when master_key=None."""
        env = self._run_engine_start_capturing_env(state_dir, master_key=None)
        assert "DATABASE_URL" not in env

    def test_inherited_database_url_is_stripped(self, state_dir: Path, monkeypatch):
        """An unrelated DATABASE_URL in the operator's env must not reach litellm.

        litellm treats its mere presence as "use a database" and aborts with
        ModuleNotFoundError: No module named 'prisma'.  Someone who exports it
        for a different application would otherwise be unable to start the
        proxy at all, with a wholly unrelated error.
        """
        monkeypatch.setenv("DATABASE_URL", "postgresql://someone:else@db.internal/app")
        env = self._run_engine_start_capturing_env(state_dir, master_key="secret")
        assert "DATABASE_URL" not in env


class TestUiSupportRemoved:
    """The LiteLLM /ui is unsupported, so nothing may accept an enable_ui.

    It used to export ``DATABASE_URL=sqlite:///…``, but LiteLLM's bundled
    ``schema.prisma`` declares ``provider = "postgresql"``: the URL is
    rejected during schema validation and the proxy exits (code 3) before
    binding a port.  Serving the UI needs a PostgreSQL server plus a
    generated prisma client, neither of which sparkrun provisions.
    """

    @pytest.fixture
    def state_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "proxy_state"
        d.mkdir()
        return d

    def test_engine_rejects_enable_ui_kwarg(self, state_dir: Path):
        """ProxyEngine must not silently accept a UI request it cannot honour."""
        from sparkrun.plugins.litellm.engine import ProxyEngine

        with pytest.raises(TypeError):
            ProxyEngine(state_dir=state_dir, master_key="secret", enable_ui=True)

    def test_build_env_has_no_ui_or_db_vars(self, state_dir: Path):
        """_build_env never injects DB/UI vars — that is what broke startup."""
        from sparkrun.plugins.litellm.engine import ProxyEngine

        env = ProxyEngine(host="127.0.0.1", port=14123, master_key="secret", state_dir=state_dir)._build_env()

        assert "DATABASE_URL" not in env
        assert "UI_USERNAME" not in env
        assert "UI_PASSWORD" not in env

    def test_cli_has_no_enable_ui_option(self):
        """`proxy start` must not advertise or accept --enable-ui."""
        from click.testing import CliRunner

        from sparkrun.cli._proxy import proxy

        help_text = CliRunner().invoke(proxy, ["start", "--help"]).output
        assert "--enable-ui" not in help_text

        result = CliRunner().invoke(proxy, ["start", "--enable-ui"])
        assert result.exit_code != 0
        assert "no such option" in result.output.lower()


class TestConfigNeverEmitsDbKeys:
    """Test that build_litellm_config emits master_key and never a DB key."""

    def test_master_key_emitted(self):
        """LiteLLM config YAML carries general_settings.master_key."""
        from sparkrun.plugins.litellm.engine import build_litellm_config

        config = build_litellm_config([], master_key="secret")
        assert config["general_settings"]["master_key"] == "secret"

    def test_no_store_model_in_db(self):
        """build_litellm_config never emits store_model_in_db.

        It would make litellm require a DB-backed model store, which needs
        PostgreSQL plus a generated prisma client.
        """
        from sparkrun.plugins.litellm.engine import build_litellm_config

        for mk in (None, "secret"):
            config = build_litellm_config([], master_key=mk)
            assert "store_model_in_db" not in config
            assert "store_model_in_db" not in config.get("general_settings", {})
            assert "store_model_in_db" not in config.get("litellm_settings", {})


@pytest.fixture(autouse=True)
def _gateway_provider(loaded_litellm_plugin):
    pass
