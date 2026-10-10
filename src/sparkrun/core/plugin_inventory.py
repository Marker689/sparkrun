"""Enumerate the plugin *modules* sparkrun knows about, loaded or not.

The console-free source behind ``sparkrun plugins list``. Bundled and
directory enumeration use their loaders' discovery helpers; installed rows use
the installed loader's metadata inventory. Listing does not import plugins or
invent a separate discovery policy.

**Scope is plugin modules** from bundled, directory, and installed sources:
in-tree subpackages of ``sparkrun.plugins`` and out-of-tree top-level modules
under ``plugins.paths``, plus installed ``sparkrun.plugins`` entry points.
Deliberately *not* every SAF extension — a runtime or
an executor shipped in core has no version distinct from sparkrun's, and
``list-runtimes`` / ``list-executors`` already enumerate those.

**Nothing here imports a plugin.** Versions come from a previously registered
module, a literal source declaration, or owning distribution metadata. Bundled
plugins that explicitly re-export Sparkrun's version use that distribution's
version; independently versioned plugins keep their own version. Undeclared or
dynamic versions remain unknown. Directory versions are always attributed to
the configured path, never a same-named module from elsewhere.

"""

from __future__ import annotations

from sparkrun.core.registration import plugin_load_failure
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scitrera_app_framework import Variables

    from sparkrun.core.config import SparkrunConfig

#: ``source`` values.
SOURCE_IN_TREE = "in-tree"
SOURCE_EXTERNAL = "external"
SOURCE_INSTALLED = "installed"

#: ``version_source`` values (``None`` when the version is unknown).
VERSION_FROM_MODULE = "module"
VERSION_FROM_DISTRIBUTION = "distribution"
VERSION_FROM_SOURCE = "source"

#: The attribute a plugin sets to declare its version.
VERSION_ATTR = "__version__"


@dataclass(frozen=True)
class PluginInfo:
    """One plugin module, whether or not it loaded this process."""

    name: str
    """Plugin name as the loader spells it (the module/subpackage name)."""

    source: str
    """:data:`SOURCE_IN_TREE`, :data:`SOURCE_EXTERNAL`, or :data:`SOURCE_INSTALLED`."""

    module: str
    """Dotted module name the loader imports."""

    enabled: bool
    """Whether the plugin's feature gate currently resolves on."""

    loaded: bool
    """Whether this process successfully registered this source, even if now disabled."""

    feature_flag: str | None = None
    """The flag gating it; ``None`` for an in-tree plugin missing its binding."""

    version: str | None = None
    """Declared version, or ``None`` for *unknown* — never a guess."""

    version_source: str | None = None
    """How :attr:`version` was obtained; ``None`` when it is unknown."""

    path: Path | None = None
    """The configured ``plugins.paths`` directory this row describes (directory plugins only)."""

    package: str | None = None
    selection_source: str | None = None
    failure: str | None = None
    required: bool = False
    managed: bool = False
    management: dict = field(default_factory=dict)

    @property
    def version_display(self) -> str:
        """Version for display, distinguishing *unknown* from *not declared*."""
        if self.version:
            return self.version
        return "unknown"

    def to_dict(self) -> dict:
        """Canonical mapping for ``--json``.

        Hand-written rather than left to ``dataclasses.asdict``, which would
        emit a ``Path`` the JSON encoder cannot serialize. ``version`` stays
        ``null`` when unknown — the string "unknown" is a *display* rendering,
        and emitting it here would be indistinguishable from a plugin that
        declared "unknown" as its version.
        """
        result = {
            "name": self.name,
            "package": self.package,
            "selected": self.enabled,
            "selection_source": self.selection_source,
            "failure": self.failure,
            "required": self.required,
            "source": self.source,
            "module": self.module,
            "enabled": self.enabled,
            "loaded": self.loaded,
            "feature_flag": self.feature_flag,
            "version": self.version,
            "version_source": self.version_source,
            "path": str(self.path) if self.path is not None else None,
        }
        if self.managed:
            result.update(managed=True, **{key: value for key, value in self.management.items() if key != "metadata"})
        return result


def _module_version(dotted: str, *, path: Path | None = None) -> tuple[str | None, str | None]:
    """Read a registered version, falling back to the exact source declaration."""
    from sparkrun.core.external_plugins import loaded_plugin_module

    module = loaded_plugin_module(dotted, path=path)
    raw = getattr(module, VERSION_ATTR, None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip(), VERSION_FROM_MODULE
    from sparkrun.core.plugin_metadata import plugin_origin, source_version

    return source_version(dotted, plugin_origin(dotted, path))


def _in_tree_plugins(v: "Variables | None") -> list[PluginInfo]:
    from sparkrun.core.external_plugins import loaded_plugin_module
    from sparkrun.core.features import feature_gate_enabled, get_feature
    from sparkrun.core.in_tree_plugins import (
        IN_TREE_PLUGIN_PACKAGE,
        iter_in_tree_plugin_names,
        plugin_feature_flag,
        plugin_application_profile_failure,
    )

    from sparkrun.core.plugin_metadata import bundled_distribution, plugin_origin

    out: list[PluginInfo] = []
    for name in iter_in_tree_plugin_names():
        dotted = "%s.%s" % (IN_TREE_PLUGIN_PACKAGE, name)
        flag = plugin_feature_flag(name)
        # An unregistered flag resolves off — same rule the loader applies, so
        # a plugin missing its IN_TREE_PLUGIN_FEATURES binding is listed as off
        # with no flag rather than omitted, which is what makes the defect
        # visible instead of silent.
        registered = flag is not None and get_feature(flag) is not None
        enabled = flag is not None and registered and feature_gate_enabled(flag, v)
        version, origin = _module_version(dotted)
        dist = bundled_distribution(dotted, plugin_origin(dotted))
        out.append(
            PluginInfo(
                name=name,
                source=SOURCE_IN_TREE,
                module=dotted,
                package="sparkrun" if dist is not None else None,
                enabled=enabled,
                loaded=loaded_plugin_module(dotted) is not None,
                feature_flag=flag if registered else None,
                failure=plugin_load_failure((SOURCE_IN_TREE, dotted, None))
                or (plugin_application_profile_failure(name) if enabled else None),
                version=version,
                version_source=origin,
            )
        )
    return out


def _external_plugins(config: "SparkrunConfig | None", v: "Variables | None") -> list[PluginInfo]:
    from sparkrun.core.external_plugins import (
        FEATURE_CORE_EXTERNAL_PLUGINS,
        _external_plugins_disabled,
        iter_plugin_module_names,
        loaded_plugin_module,
    )
    from sparkrun.core.features import feature_gate_enabled

    if config is None:
        return []

    # Enumerating directory entries imports nothing, so it is safe (and the
    # whole point of the command) to list these even with the gate off — you
    # cannot decide whether to enable the gate without seeing what it governs.
    gate = FEATURE_CORE_EXTERNAL_PLUGINS.name
    enabled = feature_gate_enabled(gate, v) and not _external_plugins_disabled()

    out: list[PluginInfo] = []
    for path in config.external_plugin_paths:
        for name in iter_plugin_module_names(path):
            version, origin = _module_version(name, path=path)
            out.append(
                PluginInfo(
                    name=name,
                    source=SOURCE_EXTERNAL,
                    module=name,
                    enabled=enabled,
                    loaded=loaded_plugin_module(name, path=path) is not None,
                    feature_flag=gate,
                    version=version,
                    version_source=origin,
                    path=path,
                    failure=plugin_load_failure((SOURCE_EXTERNAL, name, str(path.resolve()))),
                )
            )
    return out


def list_plugins(config: "SparkrunConfig | None" = None, v: "Variables | None" = None) -> list[PluginInfo]:
    """Return every known plugin module: bundled, directory, then installed.

    Args:
        config: Config supplying ``plugins.paths``. When ``None``, out-of-tree
            plugins are omitted rather than resolved from an implicit config —
            the caller owns which config is in effect.
        v: Initialized SAF variables, for feature-gate resolution.

    Returns:
        In-tree plugins sorted by name, then out-of-tree ones sorted by
        ``(path, name)`` so entries stay grouped by the directory they came
        from, followed by installed metadata rows. Loader/configuration
        validation errors propagate; listing does not import any plugin.
    """
    plugins = sorted(_in_tree_plugins(v), key=lambda p: p.name)
    plugins.extend(sorted(_external_plugins(config, v), key=lambda p: (str(p.path), p.name)))
    from sparkrun.core.installed_plugins import installed_plugin_inventory

    plugins.extend(
        PluginInfo(
            name=p.name,
            source=SOURCE_INSTALLED,
            module=p.module,
            enabled=p.selected,
            loaded=p.loaded,
            version=p.version,
            version_source=VERSION_FROM_DISTRIBUTION if p.version else None,
            package=p.package,
            selection_source=p.selection_source,
            failure=p.failure,
            required=p.required,
        )
        for p in installed_plugin_inventory(config)
    )
    if config is not None:
        plugins.extend(_managed_plugins(config))
    return plugins


def _managed_plugins(config) -> list[PluginInfo]:
    from sparkrun.core.managed_plugins import managed_enabled
    from sparkrun.core.plugin_store import PluginStore
    from sparkrun.core.external_plugins import loaded_plugin_module

    store = PluginStore(config)
    config, records = store.snapshot()
    out = []
    for name, record in sorted(records.items()):
        metadata = record["metadata"]
        path = store.generation_path(name, record["active_generation"])
        module = loaded_plugin_module(metadata["module"])
        module_file = getattr(module, "__file__", None)
        loaded = bool(module_file and Path(module_file).is_relative_to(path))
        out.append(
            PluginInfo(
                name=name,
                source=SOURCE_EXTERNAL,
                module=metadata["module"],
                enabled=managed_enabled(record, config),
                loaded=loaded,
                feature_flag=metadata.get("feature") or "core.external_plugins",
                version=metadata["version"],
                version_source="receipt",
                path=path,
                package=metadata["distribution"],
                selection_source="managed",
                managed=True,
                failure=plugin_load_failure((SOURCE_EXTERNAL, metadata["module"], str(path))),
                management={
                    "metadata": metadata,
                    "source_url": record["source"].get("url"),
                    "source_ref": record["source"].get("ref"),
                    "revision": record["revision"],
                    "active_generation": record["active_generation"],
                    "previous_generations": record["previous_generations"],
                    "retain_generations": config.get("plugins.retain_generations", 2),
                    "cleanup_pending": store.journal.exists(),
                },
            )
        )
    return out


__all__ = [
    "PluginInfo",
    "SOURCE_EXTERNAL",
    "SOURCE_INSTALLED",
    "SOURCE_IN_TREE",
    "VERSION_ATTR",
    "VERSION_FROM_DISTRIBUTION",
    "VERSION_FROM_MODULE",
    "VERSION_FROM_SOURCE",
    "list_plugins",
]
