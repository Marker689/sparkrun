"""Pin managed generations while a daemon/worker uses their files."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid

from sparkrun.core.process_identity import alive, process_identity
from sparkrun.utils.fs import atomic_private_write, file_lock


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
    return {
        "version": __version__,
        "package": str(package),
        "package_stamp": [info.st_ino, info.st_mtime_ns, source_stamp],
        "plugins": {name: record["active_generation"] for name, record in sorted(records.items())},
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
