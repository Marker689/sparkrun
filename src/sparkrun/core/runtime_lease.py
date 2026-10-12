"""Pin managed generations while a daemon/worker uses their files."""

from __future__ import annotations

from contextlib import contextmanager
from importlib.machinery import ModuleSpec, PathFinder
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from sparkrun.core.process_identity import alive, process_identity
from sparkrun.utils.fs import atomic_private_write, file_lock


def _path_spec(name, search):
    """Like PathFinder, but keep namespace locations concrete during discovery.

    PathFinder's lazy namespace paths require parents in sys.modules, even to
    construct a nested namespace spec. Path-entry finders expose the same
    locations without that requirement. Preserve import path/hook precedence.
    """
    locations = []
    for entry in sys.path if search is None else search:
        if not isinstance(entry, str):
            continue
        entry = entry or os.getcwd()
        if entry not in sys.path_importer_cache:
            finder = None
            for hook in sys.path_hooks:
                try:
                    finder = hook(entry)
                    break
                except ImportError:
                    continue
            sys.path_importer_cache[entry] = finder
        finder = sys.path_importer_cache[entry]
        spec = finder.find_spec(name) if finder is not None else None
        if spec is not None:
            if spec.loader is not None:
                return spec
            if spec.submodule_search_locations is not None:
                locations.extend(spec.submodule_search_locations)
    if locations:
        spec = ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = locations
        return spec
    return None


def _module_sources(module, recorded):
    """Resolve source ownership through import finders without importing modules.

    Calling PathFinder directly misses PEP 660 meta-path hooks. Walking specs
    ourselves also avoids importlib.util.find_spec importing parent packages.
    Namespace directories do not imply ownership of every package below them.
    """
    search = None
    package_root = None
    parts = module.split(".")
    for index in range(len(parts)):
        name = ".".join(parts[: index + 1])
        spec = None
        for finder in sys.meta_path:
            spec = _path_spec(name, search) if finder is PathFinder else finder.find_spec(name, search)
            if spec is not None:
                break
        if spec is None:
            raise RuntimeError("Cannot locate source for runtime component %r" % module)
        origin = Path(spec.origin).resolve() if spec.has_location and spec.origin else None
        search = spec.submodule_search_locations
        if search is not None and origin is not None and package_root is None:
            package_root = origin.parent
        if index < len(parts) - 1 and search is None:
            raise RuntimeError("Runtime component %r has a non-package parent" % module)

    # Ordinary wheels already identify every owned source file in RECORD.
    # Editable RECORDs contain the finder/.pth instead of the actual sources.
    if origin is None or not origin.is_file():
        raise RuntimeError("Runtime component %r has no trackable source location" % module)
    if origin in recorded:
        return {origin}
    if package_root is not None:
        return set(package_root.rglob("*.py")) | {origin}
    # A single-file module owns that file, not its site-packages directory.
    return {origin}


def _component_identity(package, module):
    """Metadata and source stamps without importing integration code."""
    from importlib.metadata import distribution, PackageNotFoundError

    paths = set()
    version = None
    if package:
        try:
            dist = distribution(package)
            version = dist.version
            paths.update(
                Path(str(dist.locate_file(path))).resolve()
                for path in dist.files or ()
                if path.suffix in {".py", ".pth"} or ".dist-info/" in str(path)
            )
        except PackageNotFoundError:
            pass
    if module:
        paths.update(_module_sources(module.split(":")[0], paths))
    stamps = []
    for path in sorted(paths):
        try:
            st = path.stat()
            stamps.append((str(path), st.st_ino, st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            stamps.append((str(path), None))
    return {"package": package, "module": module, "version": version, "stamp": fingerprint(stamps)}


def runtime_identity(config):
    from sparkrun import __version__
    from sparkrun.core.plugin_store import PluginStore

    store = PluginStore(config)
    records = store._records()
    import sparkrun

    package = Path(sparkrun.__file__).resolve()
    info = package.stat()
    # Include editable source changes and same-version reinstalls, not only __init__.
    stamps = []
    for path in sorted(package.parent.rglob("*.py")):
        st = path.stat()
        stamps.append((str(path.relative_to(package.parent)), st.st_ino, st.st_mtime_ns, st.st_size))
    source_stamp = hashlib.sha256(json.dumps(stamps).encode()).hexdigest()
    from sparkrun.core.installed_plugins import discover_installed_plugins

    profile = config.profile
    integrations = {row.name: _component_identity(row.package, row.module) for row in discover_installed_plugins(config) if row.selected}
    return {
        "version": __version__,
        "package": str(package),
        "package_stamp": [info.st_ino, info.st_mtime_ns, source_stamp],
        "plugins": {name: record["active_generation"] for name, record in sorted(records.items())},
        "application": {"id": profile.id, **_component_identity(profile.package, profile.profile_ref)},
        "integrations": integrations,
    }


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def live_leases(root):
    directory = Path(root) / "runtime-leases"
    if not directory.exists():
        return []
    result = []
    for path in directory.glob("*.json"):
        data = json.loads(path.read_text())
        if alive(data["process"]):
            result.append(data)
        else:
            path.unlink(missing_ok=True)
    return result


def assert_generations_available(root, records):
    """Called under the plugin transaction lock, before activation/deletion."""
    from sparkrun.core.plugin_manifest import PluginError

    for lease in live_leases(root):
        for name, generation in lease["runtime"]["plugins"].items():
            record = records.get(name, {})
            keep = {record.get("active_generation"), *record.get("previous_generations", [])}
            if generation not in keep:
                raise PluginError(
                    f"Plugin {name} generation is in use by PID {lease['process']['pid']}; drain/stop the daemon or worker and retry"
                )


def environment_state():
    """One protected coordination directory for this interpreter installation."""
    import sys
    import tempfile
    from sparkrun.daemon.state import private_directory

    key = hashlib.sha256(str(Path(sys.prefix).resolve()).encode()).hexdigest()[:20]
    from sparkrun.core.process_identity import user_identity

    uid = user_identity()
    return private_directory(Path(tempfile.gettempdir()).resolve() / f"sparkrun-runtime-{uid}-{key}")


@contextmanager
def runtime_lease(config, *, kind, expected=None):
    root = config.config_path.parent
    root.mkdir(parents=True, exist_ok=True)
    environment = environment_state()
    with file_lock(environment / "runtime.lock"), file_lock(root / "plugins.lock"):
        identity = runtime_identity(config)
        if expected is not None and identity != expected:
            raise RuntimeError("Runtime changed; restart the daemon before submitting work")
        process = process_identity()
        if process is None:
            raise RuntimeError("Strong process identity is unavailable on this platform")
        identifier = uuid.uuid4().hex + ".json"
        paths = []
        for parent in (root, environment):
            directory = parent / "runtime-leases"
            directory.mkdir(mode=0o700, exist_ok=True)
            path = directory / identifier
            atomic_private_write(
                path, json.dumps({"process": process, "runtime": identity, "kind": kind, "config": str(config.config_path)})
            )
            paths.append(path)
    try:
        yield identity
    finally:
        with file_lock(environment / "runtime.lock"), file_lock(root / "plugins.lock"):
            for path in paths:
                path.unlink(missing_ok=True)


def assert_runtime_idle(config):
    """Protect the entire interpreter installation, including other bindings."""
    leases = [v for v in live_leases(environment_state()) if v["process"]["pid"] != os.getpid()]
    if leases:
        raise RuntimeError(
            "Core update requires stopping daemons and draining workers first (active PIDs: %s)"
            % ", ".join(str(v["process"]["pid"]) for v in leases)
        )


@contextmanager
def update_guard(config):
    """Serialize replacement against process lease creation across all bindings."""
    with file_lock(environment_state() / "runtime.lock"):
        assert_runtime_idle(config)
        yield


def guarded_update(config, runner, *args, **kwargs):
    with update_guard(config):
        return runner(*args, **kwargs)


@contextmanager
def consistent_runtime(config, expected):
    """Serialize acceptance with plugin activation and environment replacement."""
    with file_lock(environment_state() / "runtime.lock"), file_lock(config.config_path.parent / "plugins.lock"):
        if runtime_identity(config) != expected:
            raise RuntimeError("Runtime changed; restart the daemon before submitting work")
        yield
