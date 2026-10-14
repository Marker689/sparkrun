"""Pin managed generations while a daemon/worker uses their files.

Coordination state lives at ``<OS account home>/.sparkrun-runtime/<environment>``.
The account home is read from the OS user database, never from ``HOME`` /
``USERPROFILE`` / ``XDG_*`` / ``TMPDIR``: a daemon started by a service manager
and an updater run from a shell routinely disagree on those, and a lease the
core-update guard cannot find is a lease that does not exist.

Test isolation is the one sanctioned relocation. The suite must not write into
the developer's real root, and a live lease there (another suite, an orphaned
worker, a real daemon) must not fail it -- but its subprocesses must still meet
the test process in *one* directory, so an in-process monkeypatch cannot do it.
``SPARKRUN_TEST_RUNTIME_STATE_ROOT`` replaces the account home **and** drops the
legacy temp roots, and is honored only while ``PYTEST_VERSION`` is also set
(pytest exports it for the whole session, so children inherit it). Neither is an
ordinary setting, and a stray export of the first in a shell is inert outside
pytest, so the guarantee above is unchanged for anything that is not a test run.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
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
        data = json.loads(path.read_text(encoding="utf-8"))
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


def _account_home():
    """OS account directory, independent of HOME/USERPROFILE/XDG/TEMP overrides."""
    if os.name == "nt":
        from sparkrun.utils.windows import account_home

        return account_home()
    import pwd

    return Path(pwd.getpwuid(os.getuid()).pw_dir)


#: Test-only relocation (see module docstring). Both must be present.
TEST_STATE_ROOT_ENV = "SPARKRUN_TEST_RUNTIME_STATE_ROOT"
TEST_SESSION_MARKER_ENV = "PYTEST_VERSION"

#: Fixed POSIX temp roots older releases used; see _legacy_environment_states.
_LEGACY_TEMP_ROOTS = ("/tmp", "/var/tmp", "/usr/tmp")


def _test_state_root():
    """The test suite's redirected root, or None outside a pytest session."""
    root = os.environ.get(TEST_STATE_ROOT_ENV)
    if not root or not os.environ.get(TEST_SESSION_MARKER_ENV):
        return None
    root = Path(root)
    if not root.is_absolute():
        raise RuntimeError("%s must be an absolute path" % TEST_STATE_ROOT_ENV)
    return root


def _environment_name(prefix=None):
    from sparkrun.core.process_identity import user_identity

    key = hashlib.sha256(str(Path(sys.prefix if prefix is None else prefix).resolve()).encode()).hexdigest()[:20]
    return f"sparkrun-runtime-{user_identity()}-{key}"


def environment_state(prefix=None):
    """One protected coordination directory per OS user and installation.

    Keep it outside the replaceable environment and independent of shell,
    service, application profile, config, and temporary-directory settings.
    Resolve the OS home first (macOS may expose it through a system alias).
    """
    from sparkrun.daemon.state import private_directory

    home = _test_state_root() or _account_home()
    if not home.is_absolute():
        raise RuntimeError("OS account home must be an absolute path for runtime coordination")
    return private_directory(home.resolve() / ".sparkrun-runtime" / _environment_name(prefix))


def _legacy_environment_states(prefix=None):
    """Known pre-release temp roots; old daemons must be restarted on upgrade.

    Also inspect the config's duplicate records in assert_runtime_idle. Arbitrary
    temp roots belonging to old processes on other bindings cannot be discovered
    reliably; these compatibility checks do not authorize keeping them running.
    """
    import tempfile
    from sparkrun.daemon.state import private_directory

    if _test_state_root() is not None:
        return []
    roots = {Path(tempfile.gettempdir()).resolve()}
    if os.name != "nt":
        roots.update(Path(path).resolve() for path in _LEGACY_TEMP_ROOTS)
    else:
        roots.add(_account_home().resolve() / "AppData" / "Local" / "Temp")
    return [private_directory(path) for root in sorted(roots) if (path := root / _environment_name(prefix)).exists()]


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


def assert_runtime_idle(config, *, target_prefix=None):
    """Protect the entire interpreter installation, including other bindings."""
    environment = environment_state() if target_prefix is None else environment_state(target_prefix)
    roots = [environment, *_legacy_environment_states(target_prefix), config.config_path.parent]
    leases = {v["process"]["pid"] for root in roots for v in live_leases(root) if v["process"]["pid"] != os.getpid()}
    if leases:
        raise RuntimeError(
            "Core update requires stopping daemons and draining workers first (active PIDs: %s)"
            % ", ".join(str(pid) for pid in sorted(leases))
        )


@contextmanager
def update_guard(config, *, target_prefix=None):
    """Serialize replacement against process lease creation across all bindings."""
    environment = environment_state() if target_prefix is None else environment_state(target_prefix)
    with ExitStack() as locks:
        locks.enter_context(file_lock(environment / "runtime.lock"))
        for legacy in _legacy_environment_states(target_prefix):
            locks.enter_context(file_lock(legacy / "runtime.lock"))
        assert_runtime_idle(config, target_prefix=target_prefix)
        yield


def guarded_update(config, runner, *args, target_prefix=None, **kwargs):
    with update_guard(config, target_prefix=target_prefix):
        return runner(*args, **kwargs)


@contextmanager
def consistent_runtime(config, expected):
    """Serialize acceptance with plugin activation and environment replacement."""
    with file_lock(environment_state() / "runtime.lock"), file_lock(config.config_path.parent / "plugins.lock"):
        if runtime_identity(config) != expected:
            raise RuntimeError("Runtime changed; restart the daemon before submitting work")
        yield
