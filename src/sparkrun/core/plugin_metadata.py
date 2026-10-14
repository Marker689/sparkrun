"""Read plugin source declarations without executing plugin code."""

from __future__ import annotations

import ast
from importlib import metadata as distributions
from pathlib import Path

from sparkrun.core.plugin_manifest import literal_attribute


def plugin_origin(module: str, directory: Path | None = None) -> Path | None:
    if directory is not None:
        root = Path(directory).joinpath(*module.split("."))
        return root / "__init__.py" if root.is_dir() else root.with_suffix(".py")
    from sparkrun.core.managed_plugins import physical_origin

    return physical_origin(module)


def sidecar_manifest(origin: Path | None) -> dict:
    """The first readable ``plugin.toml`` / ``VENDORED.toml`` beside *origin*, else ``{}``."""
    import tomllib

    if origin is None:
        return {}
    for filename in ("plugin.toml", "VENDORED.toml"):
        try:
            return tomllib.loads((Path(origin).parent / filename).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def is_bundled_plugin(module: str, origin: Path | None) -> bool:
    """True only for a release-owned plugin at its shipped location.

    Namespace membership is not ownership: a separately installed package can
    occupy ``sparkrun.plugins.<name>`` too. The explicit release inventory and
    physical path check also work for an editable install, whose distribution
    RECORD does not list source files.
    """
    from sparkrun.core.in_tree_plugins import BUNDLED_PLUGIN_NAMES

    name = module.removeprefix("sparkrun.plugins.")
    if module != f"sparkrun.plugins.{name}" or name not in BUNDLED_PLUGIN_NAMES or origin is None:
        return False
    expected = Path(__file__).resolve().parents[1] / "plugins" / name / "__init__.py"
    return origin.resolve() == expected.resolve()


def bundled_distribution(module: str, origin: Path | None) -> distributions.Distribution | None:
    """The owning distribution of a release-owned plugin (see :func:`is_bundled_plugin`)."""
    if not is_bundled_plugin(module, origin):
        return None
    try:
        return distributions.distribution("sparkrun")
    except distributions.PackageNotFoundError:
        return None


def source_version(module: str, origin: Path | None) -> tuple[str | None, str | None]:
    """Read a literal version or an explicit bundled re-export of core's version."""
    if origin is None:
        return None, None
    version = literal_attribute(origin, "__version__")
    if isinstance(version, str) and version.strip():
        return version.strip(), "source"
    dist = bundled_distribution(module, origin)
    if dist is not None:
        try:
            tree = ast.parse(origin.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            return None, None
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "sparkrun":
                if any(alias.name == "__version__" and alias.asname in (None, "__version__") for alias in node.names):
                    return dist.version, "distribution"
    return None, None
