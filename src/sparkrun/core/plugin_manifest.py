"""Static metadata for managed plugins. Reading a manifest never executes code."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
import re
import tomllib
from urllib.parse import urlsplit, urlunsplit

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version


class PluginError(ValueError):
    """An actionable plugin management failure."""


def safe_name(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,99}", value):
        raise PluginError("Invalid plugin name: %r" % value)
    return value


def safe_relative(value: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise PluginError("Plugin paths must be strings")
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or any(p in {"..", "."} for p in value.split("/")) or ":" in value:
        raise PluginError("Path must stay inside the plugin: %r" % value)
    return path


def module_parts(module: str) -> list[str]:
    if not isinstance(module, str):
        raise PluginError("Plugin module must be a string")
    parts = module.split(".")
    if not all(p.isidentifier() and not p.startswith("_") for p in parts):
        raise PluginError("Invalid plugin module: %r" % module)
    if len(parts) != 1 and not (len(parts) == 3 and parts[:2] == ["sparkrun", "plugins"]):
        raise PluginError("Supported modules are top-level packages or sparkrun.plugins.<package>")
    if module in {"sparkrun", "plugins"}:
        raise PluginError("A plugin cannot replace the host package")
    return parts


def normalize_source(value: str, *, local: bool = False) -> str:
    """Normalize identities without changing transport or retaining credentials."""
    if not isinstance(value, str):
        raise PluginError("Plugin source must be a URL or local repository path")
    if local and (Path(value).exists() or value.startswith("file://")):
        path = Path(urlsplit(value).path if value.startswith("file://") else value).expanduser().resolve()
        return path.as_uri()
    if re.fullmatch(r"git@[A-Za-z0-9.-]+:[A-Za-z0-9_./-]+", value):
        return value.rstrip("/").removesuffix(".git")
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "ssh"} or not parsed.hostname or parsed.password or parsed.query or parsed.fragment:
        raise PluginError("Use an HTTPS/SSH repository URL without credentials, query, or fragment")
    if parsed.username and not (parsed.scheme == "ssh" and parsed.username == "git"):
        raise PluginError("Keep source credentials in Git/pip authentication, not plugin URLs")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/").removesuffix(".git"), "", ""))


def repository_identity(value: str) -> str:
    """Known literal GitHub transports may agree on provenance, not trust."""
    source = normalize_source(value, local=True)
    if source.startswith("git@github.com:"):
        return "github.com/" + source.split(":", 1)[1]
    parsed = urlsplit(source)
    if parsed.hostname == "github.com" and parsed.port is None:
        return "github.com" + parsed.path
    return source


def literal_attribute(path: Path, name: str, *, strict=False):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                    return ast.literal_eval(node.value) if node.value is not None else None
    except (OSError, SyntaxError, ValueError, TypeError) as exc:
        if strict:
            raise PluginError("Cannot read literal %s declaration: %s" % (name, exc)) from exc
    return None


@dataclass(frozen=True)
class PluginMetadata:
    name: str
    module: str
    distribution: str
    version: str
    source: str
    repository: str
    requires_sparkrun: str
    requires_python: str
    dependencies: tuple[str, ...] = ()
    feature: str | None = None
    plugin_api: int | None = None
    application_profile_api: int | None = None
    declarations: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["dependencies"] = list(self.dependencies)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> PluginMetadata:
        try:
            if not isinstance(data, dict):
                raise PluginError("Metadata must be a mapping")
            metadata = cls(**{**data, "dependencies": tuple(data.get("dependencies", ()))})
            metadata.validate()
            return metadata
        except (TypeError, KeyError, ValueError) as exc:
            raise PluginError("Invalid plugin metadata: %s" % exc) from exc

    def validate(self) -> None:
        validate_declarations(self.declarations, self.module)
        safe_name(self.name)
        module_parts(self.module)
        safe_relative(self.source)
        if not isinstance(self.version, str) or not isinstance(self.repository, str):
            raise PluginError("Plugin version and repository must be strings")
        Version(self.version)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.distribution):
            raise PluginError("Invalid distribution name")
        for requirement_range in (self.requires_sparkrun, self.requires_python):
            if not isinstance(requirement_range, str) or not requirement_range.strip():
                raise PluginError("Managed plugins require explicit Sparkrun and Python compatibility ranges")
            SpecifierSet(requirement_range)
        if self.feature is not None and not re.fullmatch(r"[a-zA-Z0-9_.-]+", self.feature):
            raise PluginError("Invalid feature gate")
        if type(self.plugin_api) is not int:
            raise PluginError("Declare a literal SPARKRUN_PLUGIN_API_VERSION in the plugin entry module")
        if self.application_profile_api is not None and type(self.application_profile_api) is not int:
            raise PluginError("application_profile_api must be an integer")
        for requirement in self.dependencies:
            Requirement(requirement)
        if not any(canonicalize_name(Requirement(r).name) == "sparkrun" for r in self.dependencies):
            raise PluginError("project.dependencies must declare a compatible Sparkrun requirement")


def read_manifest(root: Path) -> PluginMetadata:
    try:
        manifest = tomllib.loads((root / "plugin.toml").read_text())
        project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
        if type(manifest.get("schema")) is not int or manifest["schema"] != 1:
            raise PluginError("plugin.toml must declare schema = 1")
        if set(project.get("dynamic", ())) & {"name", "version", "dependencies", "requires-python"}:
            raise PluginError("Source-copy installation needs static project metadata; use a wheel for dynamic builds")
        source = str(safe_relative(manifest["source"]))
        metadata = PluginMetadata(
            name=manifest["name"],
            module=manifest["module"],
            distribution=project["name"],
            version=project["version"],
            source=source,
            repository=manifest["repository"],
            requires_sparkrun=manifest["sparkrun"],
            requires_python=project["requires-python"],
            dependencies=tuple(project.get("dependencies", ())),
            feature=manifest.get("feature"),
            plugin_api=literal_attribute(root / source / "__init__.py", "SPARKRUN_PLUGIN_API_VERSION"),
            application_profile_api=manifest.get("application_profile_api"),
            declarations=read_declarations(root / source / "__init__.py", manifest["module"]),
        )
        metadata.validate()
        if not (root / source / "__init__.py").is_file():
            raise PluginError("The declared source must be a Python package with __init__.py")
        return metadata
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise PluginError("Cannot read plugin.toml/pyproject.toml: %s" % exc) from exc


LIFECYCLE_EVENTS = frozenset(
    {
        "plugin.install",
        "plugin.update",
        "plugin.rollback",
        "plugin.remove",
        "plugin.enable",
        "plugin.disable",
        "feature.enable",
        "feature.disable",
    }
)


def validate_declarations(data, module):
    if not isinstance(data, dict) or set(data) - {"api", "features", "hooks"}:
        raise PluginError("Invalid SPARKRUN_PLUGIN_MANIFEST declaration")
    if not data:
        return
    if type(data.get("api")) is not int or data["api"] != 1:
        raise PluginError("Unsupported plugin lifecycle declaration API")
    features, hooks = data.get("features", {}), data.get("hooks", {})
    if not isinstance(features, dict) or not isinstance(hooks, dict):
        raise PluginError("Plugin features and hooks must be mappings")
    for name, definition in features.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", name) or not isinstance(definition, dict):
            raise PluginError("Invalid plugin feature declaration")
        if set(definition) - {"description", "default", "channel_defaults"} or not isinstance(definition.get("description"), str):
            raise PluginError("Plugin features require descriptions and supported settings")
        if type(definition.get("default", False)) is not bool:
            raise PluginError("Plugin feature defaults must be boolean")
        channels = definition.get("channel_defaults", {})
        if not isinstance(channels, dict) or any(k not in {"stable", "beta", "alpha"} or type(v) is not bool for k, v in channels.items()):
            raise PluginError("Invalid feature channel defaults")
    for event, target in hooks.items():
        if event not in LIFECYCLE_EVENTS or not isinstance(target, str):
            raise PluginError("Invalid lifecycle event/handler")
        package, separator, function = target.partition(":")
        if not separator or not function.isidentifier() or not (package == module or package.startswith(module + ".")):
            raise PluginError("Lifecycle handlers must belong to their declaring plugin")
        if not all(part.isidentifier() for part in package.split(".")):
            raise PluginError("Invalid lifecycle handler module")


def read_declarations(origin, module):
    if origin is None:
        return {}
    data = literal_attribute(Path(origin), "SPARKRUN_PLUGIN_MANIFEST", strict=True)
    if data is None:
        return {}
    validate_declarations(data, module)
    return data
