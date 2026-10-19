"""``sparkrun.plugins`` is the single documented import surface for plugin authors."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import sparkrun.plugins as surface

DOCS = Path(__file__).resolve().parents[1] / "docs"
_IMPORT = re.compile(r"^from (sparkrun(?:\.[a-z_]+)*) import (?:\(([^)]*)\)|([^(\n]+))$", re.MULTILINE)


def _documented_imports(text: str) -> list[tuple[str, list[str]]]:
    found = []
    for module, grouped, inline in _IMPORT.findall(text):
        names = [n.strip().split(" as ")[0] for n in (grouped or inline).replace("\n", ",").split(",")]
        found.append((module, [n for n in names if n]))
    return found


def test_all_is_one_sorted_literal_of_real_names():
    assert surface.__all__ == sorted(set(surface.__all__))
    assert [name for name in surface.__all__ if not hasattr(surface, name)] == []


def test_introspection_and_core_only_aliases_are_not_exported():
    for name in ("registered_daemon_ops", "register_daemon_op", "register_daemon_route"):
        assert name not in surface.__all__


def test_a_platform_plugin_can_return_a_collective_backend_from_the_surface():
    """``HardwarePlatformPlugin.collective_backend()`` is abstract; what it returns must be importable too."""
    from sparkrun.orchestration.collectives.base import CollectiveBackend

    assert surface.CollectiveBackend is CollectiveBackend
    for name in ("NcclBackend", "RcclBackend", "HcclBackend", "NoCollectiveBackend"):
        assert issubclass(getattr(surface, name), CollectiveBackend)

    class _Platform(surface.HardwarePlatformPlugin):
        platform_name = "test-single-rank"
        vendors = frozenset({"nvidia"})

        def matches(self, hw):
            return False

        def accelerator_vendor(self):
            return "nvidia"

        def collective_backend(self):
            return surface.NoCollectiveBackend("nvidia")

    assert _Platform().collective_backend().vendor == "nvidia"


def test_every_documented_plugin_import_is_exported():
    for doc in sorted(DOCS.glob("*.md")):
        for module, names in _documented_imports(doc.read_text(encoding="utf-8")):
            if module == "sparkrun.plugins":
                assert set(names) <= set(surface.__all__), (doc.name, sorted(set(names) - set(surface.__all__)))


def test_the_plugin_guide_imports_only_from_the_surface():
    modules = {module for module, _ in _documented_imports((DOCS / "PLUGINS.md").read_text(encoding="utf-8"))}
    assert modules == {"sparkrun.plugins"}


def test_the_surface_stays_console_free():
    """Every bundled plugin import loads this package, including on the API path."""
    code = "import sys, sparkrun.plugins; assert 'click' not in sys.modules, 'click imported'"
    subprocess.run([sys.executable, "-c", code], check=True)
