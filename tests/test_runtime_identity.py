"""Runtime fences follow owned source without importing application/plugin code."""

from importlib.machinery import PathFinder
from importlib.util import spec_from_file_location
import sys

import pytest

from sparkrun.core.runtime_lease import _component_identity


def distribution(site, files):
    info = site / "runtime_fixture-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: runtime-fixture\nVersion: 1.0\n")
    (info / "RECORD").write_text("\n".join(str(path) + ",," for path in [*files, "runtime_fixture-1.0.dist-info/METADATA"]))


@pytest.mark.parametrize("layout", ["flat", "src", "namespace", "nested_namespace", "import_hook", "wheel"])
def test_owned_package_sources_invalidate_identity_without_imports(tmp_path, monkeypatch, layout):
    site = tmp_path / "site"
    site.mkdir()
    root = site if layout == "wheel" else tmp_path / "project"
    if layout == "src":
        root /= "src"
    module = "identity_fixture"
    if layout == "namespace":
        module = "identity_namespace." + module
    if layout == "nested_namespace":
        module = "identity_namespace.nested." + module
    package = root.joinpath(*module.split("."))
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("raise AssertionError('Runtime fingerprinting imported the plugin')\n")
    (package / "sub").mkdir()
    (package / "sub/__init__.py").write_text("")
    entry = package / "sub/entry.py"
    entry.write_text("VALUE = 1\n")
    shared = package / "common.py"
    shared.write_text("VALUE = 1\n")
    sibling = root / "unrelated.py"
    sibling.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(site))
    if layout == "import_hook":

        class EditableFinder:
            @staticmethod
            def find_spec(fullname, path=None, target=None):
                if fullname == module:
                    return spec_from_file_location(fullname, init)
                return None

        monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, EditableFinder])
        (site / "editable_finder.py").write_text("# Editable wheel finder, not the application source\n")
        distribution(site, ["editable_finder.py"])
        assert PathFinder.find_spec(module) is None
    elif layout == "wheel":
        distribution(site, [p.relative_to(site) for p in package.rglob("*.py")])
    else:
        monkeypatch.syspath_prepend(str(root))
        (site / "editable.pth").write_text(str(root))
        distribution(site, ["editable.pth"])
    reference = module + ".sub.entry:PROFILE"
    first = _component_identity("runtime-fixture", reference)
    sibling.write_text("VALUE = 12345\n")
    assert _component_identity("runtime-fixture", reference) == first
    shared.write_text("VALUE = 123456789\n")
    second = _component_identity("runtime-fixture", reference)
    assert second != first  # Includes the owning package, not just sub/entry.py.
    entry.write_text("VALUE = 12345\n")
    assert _component_identity("runtime-fixture", reference) != second
    assert not any(name == module or name.startswith(module + ".") for name in sys.modules)


@pytest.mark.parametrize("installed", [False, True])
def test_single_file_component_does_not_own_neighboring_packages(tmp_path, monkeypatch, installed):
    source = tmp_path / "identity_single.py"
    source.write_text("raise AssertionError('Do not import me')\n")
    neighbor = tmp_path / "other_package"
    neighbor.mkdir()
    other = neighbor / "__init__.py"
    other.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    package = None
    if installed:
        distribution(tmp_path, [source.name])
        package = "runtime-fixture"
    first = _component_identity(package, "identity_single:register")
    other.write_text("VALUE = 12345\n")
    assert _component_identity(package, "identity_single:register") == first
    source.write_text("raise AssertionError('Still do not import me')\n")
    assert _component_identity(package, "identity_single:register") != first
    assert "identity_single" not in sys.modules


def test_unresolvable_component_does_not_claim_source_protection():
    with pytest.raises(RuntimeError, match="Cannot locate source"):
        _component_identity(None, "not_an_installed_runtime_fixture:register")


def test_virtual_source_location_does_not_claim_file_protection(tmp_path, monkeypatch):
    class VirtualFinder:
        @staticmethod
        def find_spec(fullname, path=None, target=None):
            if fullname == "virtual_identity_fixture":
                return spec_from_file_location(fullname, tmp_path / "archive.zip" / "virtual_identity_fixture.py")
            return None

    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, VirtualFinder])
    with pytest.raises(RuntimeError, match="no trackable source location"):
        _component_identity(None, "virtual_identity_fixture:register")
