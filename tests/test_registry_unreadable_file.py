"""An unreadable registries.yaml is never replaced by the defaults."""

import yaml

from sparkrun.core.registry import RegistryEntry, RegistryManager

MINE = {"name": "mine", "url": "https://github.com/example/mine", "subpath": "recipes", "trusted": True}


def _manager(tmp_path, monkeypatch):
    from sparkrun.core import registry as reg_module

    monkeypatch.setattr(reg_module, "BOOTSTRAP_REGISTRY_URLS", ["https://example.com/r1"])
    discovered = [RegistryEntry(name="shipped", url="https://example.com/r1", subpath="recipes")]
    monkeypatch.setattr(RegistryManager, "_discover_manifest_entries", lambda self, url: list(discovered), raising=False)
    monkeypatch.setattr(RegistryManager, "_init_defaults_from_manifests", lambda self, urls=None: list(discovered))
    config = tmp_path / "config"
    config.mkdir()
    return RegistryManager(config, tmp_path / "cache"), config / "registries.yaml"


def test_bad_entry_is_skipped_and_the_rest_kept(tmp_path, monkeypatch):
    mgr, path = _manager(tmp_path, monkeypatch)
    path.write_text(yaml.safe_dump({"registries": [MINE, {"name": "broken", "url": "https://example.com/x"}]}))
    names = [e.name for e in mgr.list_registries()]
    assert "mine" in names and "broken" not in names
    assert "mine" in path.read_text()


def test_unparseable_file_is_not_overwritten(tmp_path, monkeypatch):
    mgr, path = _manager(tmp_path, monkeypatch)
    original = yaml.safe_dump({"registries": [MINE]}) + "  : : not yaml [\n"
    path.write_text(original)
    names = [e.name for e in mgr.list_registries()]
    assert "shipped" in names  # the run falls back to defaults...
    assert path.read_text() == original  # ...without replacing the user's file
    assert (path.parent / "registries.yaml.unreadable").read_text() == original


def test_string_false_does_not_grant_trust(tmp_path, monkeypatch):
    mgr, path = _manager(tmp_path, monkeypatch)
    path.write_text(yaml.safe_dump({"registries": [dict(MINE, trusted="false", enabled="no")]}))
    (entry,) = [e for e in mgr.list_registries() if e.name == "mine"]
    assert entry.trusted is False
    assert entry.enabled is False
