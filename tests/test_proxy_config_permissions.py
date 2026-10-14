"""Proxy config saves are owner-only (the shared atomic writer)."""

import yaml

from sparkrun.proxy.config import ProxyConfig


def test_save_writes_owner_only_file(tmp_path):
    path = tmp_path / "proxy.yaml"
    config = ProxyConfig(path)
    config.add_alias("new", "target")
    config.save()
    assert yaml.safe_load(path.read_text())["aliases"] == {"new": "target"}
    assert path.stat().st_mode & 0o077 == 0
