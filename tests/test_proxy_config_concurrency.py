"""``proxy.yaml`` has two writers, and one must not erase the other.

The auto-discover daemon re-reads the config every sweep while a ``sparkrun
proxy`` command may save an alias or listener setting at any moment. A
whole-document last-writer-wins save silently discards the other's change, so
:meth:`ProxyConfig.save` locks, re-reads, and merges only the sections that
instance modified.
"""

from __future__ import annotations

import builtins
import importlib
import os
import stat
import sys
from pathlib import Path

import pytest
import yaml

from sparkrun.proxy.config import ProxyConfig


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return tmp_path / "proxy.yaml"


def _read(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------
# Section-scoped merging
# --------------------------------------------------------------------------


def test_a_concurrent_writer_does_not_lose_the_others_section(config_path: Path):
    """The failure this exists to prevent: two open handles, both save."""
    first = ProxyConfig(config_path)
    second = ProxyConfig(config_path)

    first.add_alias("fast", "qwen3-1.7b")
    first.save()

    # `second` was loaded before the alias existed. Saving its own, unrelated
    # change must not roll the file back to its stale view.
    second.set_proxy(port=4321)
    second.save()

    data = _read(config_path)
    assert data["aliases"] == {"fast": "qwen3-1.7b"}
    assert data["proxy"]["port"] == 4321


def test_an_alias_removal_is_recorded_as_a_deletion_not_an_absence(config_path: Path):
    """ "Not in my copy" is not an instruction to delete.

    The merge applies to the *newest* document, where an alias this instance
    never saw may well be present.
    """
    seed = ProxyConfig(config_path)
    seed.add_alias("fast", "model-a")
    seed.add_alias("smart", "model-b")
    seed.save()

    remover = ProxyConfig(config_path)
    other = ProxyConfig(config_path)

    other.add_alias("new", "model-c")
    other.save()

    assert remover.remove_alias("fast") is True
    remover.save()

    aliases = _read(config_path)["aliases"]
    assert "fast" not in aliases  # the explicit deletion applied
    assert aliases["smart"] == "model-b"  # untouched
    assert aliases["new"] == "model-c"  # the concurrent addition survived


def test_removing_the_last_alias_drops_the_empty_section(config_path: Path):
    cfg = ProxyConfig(config_path)
    cfg.add_alias("only", "model")
    cfg.save()

    cfg.remove_alias("only")
    cfg.save()

    assert "aliases" not in _read(config_path)


def test_a_save_with_no_pending_changes_preserves_the_in_memory_document(config_path: Path):
    """Historical whole-file semantics for a caller that mutated _data itself."""
    cfg = ProxyConfig(config_path)
    cfg._data["custom"] = {"hand": "written"}
    cfg.save()
    assert _read(config_path)["custom"] == {"hand": "written"}


def test_reload_forgets_pending_changes(config_path: Path):
    cfg = ProxyConfig(config_path)
    cfg.add_alias("fast", "model")
    cfg._load()
    cfg.save()
    assert "aliases" not in _read(config_path)


# --------------------------------------------------------------------------
# Durability / permissions
# --------------------------------------------------------------------------


def test_the_config_is_replaced_atomically_and_owner_only(config_path: Path):
    cfg = ProxyConfig(config_path)
    cfg.set_proxy(master_key="sk-secret")
    cfg.save()

    assert stat.S_IMODE(os.stat(config_path).st_mode) == 0o600
    # No temp files left behind.
    leftovers = [p.name for p in config_path.parent.iterdir() if p.name.startswith(".proxy.yaml.")]
    assert leftovers == []


def test_a_corrupt_file_is_refused_not_replaced(config_path: Path):
    """Saving over an unparseable file would drop its bind host and master key."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{{{ not yaml")

    cfg = ProxyConfig(config_path)
    cfg.set_proxy(port=4000)
    with pytest.raises(ValueError, match="could not be parsed"):
        cfg.save()

    assert config_path.read_text() == "{{{ not yaml"


# --------------------------------------------------------------------------
# Windows control node
# --------------------------------------------------------------------------


def test_the_module_imports_without_fcntl(monkeypatch):
    """``proxy.config`` is reached from SparkrunContext, i.e. every invocation.

    ``fcntl`` is POSIX-only, so a hard import here would make sparkrun
    unimportable on a Windows control node. The guard is invisible on a
    developer's machine and would rot without this.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("No module named 'fcntl'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "fcntl", raising=False)
    monkeypatch.delitem(sys.modules, "sparkrun.proxy.config", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)

    importlib.import_module("sparkrun.proxy.config")

    # Restore the real module for the rest of the session — it is process-global.
    monkeypatch.undo()
    importlib.reload(importlib.import_module("sparkrun.proxy.config"))


def test_saving_locks_through_the_shared_portable_helper(config_path: Path):
    """Proxy config delegates locking to ``utils.fs.file_lock`` (msvcrt on Windows, flock elsewhere)."""
    import sparkrun.proxy.config as config_mod

    cfg = config_mod.ProxyConfig(config_path)
    cfg.add_alias("fast", "model")
    cfg.save()

    assert _read(config_path)["aliases"] == {"fast": "model"}
    assert Path(str(config_path) + ".lock").exists()


# --------------------------------------------------------------------------
# Removal grace setting
# --------------------------------------------------------------------------


def test_removal_grace_defaults_and_clamps(config_path: Path):
    from sparkrun.proxy import DEFAULT_DISCOVER_REMOVAL_GRACE_SWEEPS

    cfg = ProxyConfig(config_path)
    assert cfg.discover_removal_grace_sweeps == DEFAULT_DISCOVER_REMOVAL_GRACE_SWEEPS

    cfg.set_proxy(discover_removal_grace_sweeps=5)
    assert cfg.discover_removal_grace_sweeps == 5

    # 0 would mean "remove before you have looked".
    cfg.set_proxy(discover_removal_grace_sweeps=0)
    assert cfg.discover_removal_grace_sweeps == 1


def test_shared_writes_preserve_gateway_namespaces(config_path: Path):
    cfg = ProxyConfig(config_path)
    namespaces = {
        "sparkroute": {"bindings": [{"recipe": "@local/model"}], "discovered_models": ["warm"]},
        "litellm": {"router_settings": {"routing_strategy": "least-busy"}},
    }
    config_path.write_text(yaml.safe_dump({"gateways": namespaces}))
    cfg.add_alias("assistant", "warm")
    cfg.set_proxy(port=4002)
    cfg.save()
    result = _read(config_path)
    assert result["gateways"] == namespaces
    assert result["aliases"] == {"assistant": "warm"}
    assert result["proxy"] == {"port": 4002}


def test_sparkroute_transport_namespace_is_scoped_to_selected_gateway(config_path: Path):
    config_path.write_text(
        yaml.safe_dump(
            {
                "proxy": {"gateway": "sparkroute"},
                "gateways": {"sparkroute": {"transport": "daemon"}, "litellm": {"transport": "invalid"}},
            }
        )
    )
    cfg = ProxyConfig(config_path)
    assert cfg.discovery_backend == "daemon"
    assert cfg.discovery_backend_for("litellm") == "process"
    assert cfg.discovery_backend_for("another") == "process"
    from sparkrun.api.proxy._ops import _configured_discovery_backend

    assert _configured_discovery_backend(cfg, "litellm") == "process"
    cfg.set_proxy(discovery_backend="process")
    assert cfg.discovery_backend_for("sparkroute") == "process"
    cfg.set_proxy(discovery_backend="daemon")
    assert cfg.discovery_backend_for("litellm") == "daemon"  # An explicit shared choice applies to both.


def test_other_gateways_ignore_invalid_sparkroute_transport(config_path: Path):
    config_path.write_text(yaml.safe_dump({"proxy": {"gateway": "litellm"}, "gateways": {"sparkroute": {"transport": "invalid"}}}))
    cfg = ProxyConfig(config_path)
    assert cfg.discovery_backend == "process"
    with pytest.raises(ValueError, match="gateways.sparkroute.transport"):
        cfg.discovery_backend_for("sparkroute")


def test_old_transport_key_does_not_select_daemon_discovery(config_path: Path):
    config_path.write_text(yaml.safe_dump({"proxy": {"gateway": "sparkroute", "gateway_transport": "daemon"}}))
    cfg = ProxyConfig(config_path)
    assert cfg.discovery_backend == "process"
