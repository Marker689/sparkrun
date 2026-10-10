"""Shared, metadata-only plugin compatibility checks; never a startup hook."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import metadata as distributions
import platform

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from sparkrun.core.registration import PLUGIN_API_VERSION


@dataclass(frozen=True)
class HostSnapshot:
    core_version: str | None
    python_version: str = field(default_factory=platform.python_version)
    plugin_api: int = PLUGIN_API_VERSION
    profile_id: str = "sparkrun"
    profile_api: int = 1
    packages: dict[str, str] = field(default_factory=dict)
    requirements: dict[str, tuple[str, ...]] = field(default_factory=dict)
    core_commit: str | None = None
    markers: dict[str, str] = field(default_factory=lambda: {key: str(value) for key, value in default_environment().items()})

    @classmethod
    def current(cls) -> HostSnapshot:
        from sparkrun.core.application_profile import APPLICATION_PROFILE_API_VERSION, get_application_profile

        from sparkrun.core.version import installed_commit

        packages, requirements = {}, {}
        for dist in distributions.distributions():
            name = canonicalize_name(dist.metadata.get("Name", ""))
            if name:
                packages[name] = dist.version
                requirements[name] = tuple(dist.requires or ())
        return cls(
            packages.get("sparkrun"),
            profile_id=get_application_profile().id,
            profile_api=APPLICATION_PROFILE_API_VERSION,
            core_commit=installed_commit("sparkrun"),
            packages=packages,
            requirements=requirements,
        )

    def identity(self) -> dict:
        return {
            "core_version": self.core_version,
            "core_commit": self.core_commit,
            "python_version": self.python_version,
            "plugin_api": self.plugin_api,
            "profile_id": self.profile_id,
            "profile_api": self.profile_api,
        }


@dataclass(frozen=True)
class CompatibilityResult:
    name: str
    status: str
    reasons: tuple[str, ...]
    host: dict

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_plugin_compatibility(data: dict, host: HostSnapshot) -> CompatibilityResult:
    """One evaluator for inventory, install, update, rollback, and core update."""
    unknown, incompatible = [], []

    def check_range(label, requirement, installed):
        if not requirement or not installed:
            unknown.append("%s compatibility unknown (missing range or installed version)" % label)
            return
        try:
            if Version(installed) not in SpecifierSet(requirement):
                incompatible.append("%s %s does not satisfy %s" % (label, installed, requirement))
        except (TypeError, ValueError) as exc:
            unknown.append("Invalid %s compatibility metadata: %s" % (label, exc))

    check_range("Sparkrun", data.get("requires_sparkrun"), host.core_version)
    check_range("Python", data.get("requires_python"), host.python_version)
    api = data.get("plugin_api")
    if api is None:
        unknown.append("Plugin API declaration unavailable")
    elif type(api) is not int or api != host.plugin_api:
        incompatible.append("Plugin API %r is unsupported (host requires %s)" % (api, host.plugin_api))
    profile_api = data.get("application_profile_api")
    if host.profile_id != "sparkrun" and data.get("module") == "sparkrun.plugins.sparkroute" and profile_api is None:
        unknown.append("SparkRoute does not declare application profile API support")
    if profile_api is not None and (type(profile_api) is not int or profile_api != host.profile_api):
        incompatible.append("Application profile API %r is unsupported" % profile_api)

    visited: set[tuple[str, tuple[str, ...]]] = set()

    def dependencies(requirements, extras=()):
        for raw in requirements:
            try:
                req = Requirement(raw)
                if req.marker and not any(req.marker.evaluate({**host.markers, "extra": e}) for e in ("", *extras)):
                    continue
                name = canonicalize_name(req.name)
                installed = host.core_version if name == "sparkrun" else host.packages.get(name)
                if req.url:
                    unknown.append("Cannot verify direct dependency provenance: %s" % name)
                if name == "sparkrun" and not installed:
                    unknown.append("Installed Sparkrun dependency version unavailable")
                elif not installed:
                    incompatible.append("Missing dependency: %s" % raw)
                elif req.specifier and Version(installed) not in req.specifier:
                    incompatible.append("Dependency %s %s does not satisfy %s" % (name, installed, req.specifier))
                key = name, tuple(sorted(req.extras))
                if key not in visited:
                    visited.add(key)
                    dependencies(host.requirements.get(name, ()), tuple(req.extras))
            except (ValueError, TypeError) as exc:
                unknown.append("Invalid dependency metadata: %s" % exc)

    dependencies(data.get("dependencies", ()))
    reasons = tuple(dict.fromkeys((*incompatible, *unknown)))
    return CompatibilityResult(
        data.get("name", "unknown"), "incompatible" if incompatible else "unknown" if unknown else "compatible", reasons, host.identity()
    )


def audit_plugin_compatibility(inventory: list[dict], host: HostSnapshot) -> list[CompatibilityResult]:
    return [evaluate_plugin_compatibility(item, host) for item in inventory]


def metadata_for_plugin(plugin) -> dict:
    """Collect whatever a provider declares, without importing it or guessing."""
    import tomllib

    from sparkrun.core.plugin_metadata import bundled_distribution, plugin_origin
    from sparkrun.core.plugin_manifest import literal_attribute

    if plugin.managed:
        return dict(plugin.management["metadata"])
    data = {"name": plugin.name, "module": plugin.module}
    origin = plugin_origin(plugin.module, plugin.path)
    bundled = bundled_distribution(plugin.module, origin) if plugin.source == "in-tree" else None
    if bundled is not None:
        # Release-owned plugins ship with this exact core. Their own version
        # (e.g. OCI Relay's) is independent of this compatibility constraint.
        data["requires_sparkrun"] = "==%s" % bundled.version
    if origin:
        data["plugin_api"] = literal_attribute(origin, "SPARKRUN_PLUGIN_API_VERSION")
        for filename in ("plugin.toml", "VENDORED.toml"):
            try:
                declared = tomllib.loads((origin.parent / filename).read_text())
                if declared.get("sparkrun") is not None:
                    data["requires_sparkrun"] = declared["sparkrun"]
                data["application_profile_api"] = declared.get("application_profile_api")
                break
            except (OSError, ValueError):
                continue
    if bundled is not None or plugin.package:
        try:
            dist = bundled if bundled is not None else distributions.distribution(plugin.package)
            data["requires_python"] = dist.metadata.get("Requires-Python")
            data["dependencies"] = list(dist.requires or ())
            requirements = [Requirement(r) for r in data["dependencies"]]
            ranges = [
                str(r.specifier) for r in requirements if canonicalize_name(r.name) == "sparkrun" and (not r.marker or r.marker.evaluate())
            ]
            if ranges and isinstance(data.get("requires_sparkrun", ""), str):
                data["requires_sparkrun"] = ",".join(filter(None, [data.get("requires_sparkrun"), *ranges]))
        except (distributions.PackageNotFoundError, ValueError):
            pass
    return data
