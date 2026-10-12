"""Directory imports and inventory must agree about which source executed."""

import sys

import pytest

from sparkrun.core.bootstrap import get_variables
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.external_plugins import load_external_plugins
from sparkrun.core.plugin_inventory import list_plugins
from test_external_plugins import clean_sys as clean_sys


@pytest.fixture(autouse=True)
def _recording_variables():
    v = get_variables()
    saved = {key: v.get(key) for key in ("ORIGIN_EXECUTED", "ORIGIN_INDEPENDENT")}
    for key in saved:
        v.set(key, None)
    yield
    for key, value in saved.items():
        v.set(key, value)


def _sources(tmp_path, *, package=False):
    name = "origin_contract_plugin"
    paths = [tmp_path / "first", tmp_path / "second"]
    for version, path in enumerate(paths, 1):
        target = path / name / "__init__.py" if package else path / (name + ".py")
        target.parent.mkdir(parents=True)
        target.write_text(
            'SPARKRUN_PLUGIN_API_VERSION = 1\n__version__ = %r\ndef register(v):\n    v.set("ORIGIN_EXECUTED", %r)\n'
            % (str(version), str(path))
        )
    config = SparkrunConfig()
    config._data["plugins"] = {"paths": [str(p) for p in paths]}
    return name, paths, config


@pytest.mark.parametrize("package", [False, True])
def test_duplicate_directory_names_rejected_before_import(tmp_path, clean_sys, package):
    name, paths, config = _sources(tmp_path, package=package)
    v = get_variables()
    (paths[0] / "origin_independent.py").write_text("SPARKRUN_PLUGIN_API_VERSION = 1\ndef register(v): v.set('ORIGIN_INDEPENDENT', True)\n")
    with pytest.raises(RuntimeError, match="Plugin identity.*claimed"):
        load_external_plugins(v, paths=paths)
    assert name not in sys.modules and v.get("ORIGIN_EXECUTED") is None
    assert v.get("ORIGIN_INDEPENDENT") is None
    rows = [r for r in list_plugins(config, v) if r.name == name]
    assert len(rows) == 2
    assert all(not r.loaded and "claimed" in r.failure for r in rows)
    assert {str(r.path): r.version for r in rows} == {str(path): str(i) for i, path in enumerate(paths, 1)}


@pytest.mark.parametrize("package", [False, True])
def test_loaded_origin_and_version_never_attributed_to_second_directory(tmp_path, clean_sys, package):
    name, paths, config = _sources(tmp_path, package=package)
    v = get_variables()
    assert load_external_plugins(v, paths=[paths[0]]) == [name]
    with pytest.raises(RuntimeError, match="Plugin identity.*claimed"):
        load_external_plugins(v, paths=[paths[1]])
    first, second = sorted((r for r in list_plugins(config, v) if r.name == name), key=lambda r: str(r.path))
    assert first.loaded and first.version == "1" and "claimed" in first.failure
    assert not second.loaded and second.version == "2" and "claimed" in second.failure
    assert v.get("ORIGIN_EXECUTED") == str(paths[0])
    assert load_external_plugins(v, paths=[paths[0]]) == [name]


def test_foreign_preimport_is_not_registered_as_directory_plugin(tmp_path, clean_sys, monkeypatch):
    import importlib

    name, paths, config = _sources(tmp_path)
    monkeypatch.syspath_prepend(str(paths[0]))
    foreign = importlib.import_module(name)
    v = get_variables()
    assert load_external_plugins(v, paths=[paths[1]]) == []
    assert sys.modules[name] is foreign and v.get("ORIGIN_EXECUTED") is None
    rows = [r for r in list_plugins(config, v) if r.name == name]
    assert all(not r.loaded for r in rows)
    assert {str(r.path): r.version for r in rows} == {str(path): str(i) for i, path in enumerate(paths, 1)}
    assert next(r for r in rows if r.path == paths[1]).failure
