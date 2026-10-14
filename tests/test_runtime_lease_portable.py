"""Real process leases across shell/service environments and config bindings."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from sparkrun.core import runtime_lease as leases
from sparkrun.core.config import SparkrunConfig

# These exercise the production lease-root resolution, sandboxing its inputs
# (_account_home, tempfile.gettempdir) themselves rather than via the suite's
# test-only redirect.
pytestmark = pytest.mark.usefixtures("real_runtime_state")


@contextmanager
def worker(tmp_path, config, *, legacy=None):
    marker = tmp_path / "ready.json"
    temp = tmp_path / "worker-temp"
    temp.mkdir()
    code = """
import json, sys
from pathlib import Path
from sparkrun.core import runtime_lease as leases
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.process_identity import process_identity
from sparkrun.daemon.state import private_directory
from sparkrun.utils.fs import atomic_private_write
root, config, legacy = Path(sys.argv[1]), SparkrunConfig(sys.argv[2]), sys.argv[3]
leases._account_home = lambda: root / 'account'
def ready():
    (root / 'ready.json').write_text(json.dumps({'environment':str(leases.environment_state())}), encoding='utf-8')
    sys.stdin.read()
if legacy:
    directory = config.config_path.parent if legacy == 'config' else root / 'worker-temp' / leases._environment_name()
    directory = private_directory(directory) / 'runtime-leases'
    directory.mkdir(exist_ok=True)
    record = directory / 'legacy.json'
    atomic_private_write(record, json.dumps({'process':process_identity(), 'kind':'worker'}))
    try:
        ready()
    finally:
        record.unlink()
else:
    with leases.runtime_lease(config, kind='test-worker'):
        ready()
"""
    env = {**os.environ, **dict.fromkeys(("TMPDIR", "TEMP", "TMP", "HOME", "USERPROFILE", "XDG_CACHE_HOME"), str(temp))}
    with subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path), str(config.config_path), legacy or ""],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as child:
        try:
            deadline = time.monotonic() + 15
            while not marker.exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    raise AssertionError("Worker did not register its lease")
                time.sleep(0.02)
            yield json.loads(marker.read_text(encoding="utf-8"))
        finally:
            try:
                _, stderr = child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate(timeout=5)
                raise
            assert child.returncode == 0, stderr.decode("utf-8", errors="replace")


def test_os_account_home_ignores_shell_environment(tmp_path, monkeypatch):
    before = leases._account_home()
    for name in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "XDG_CACHE_HOME", "TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(tmp_path / name))
    assert leases._account_home() == before
    assert before.is_absolute()


def test_environment_directory_is_private_and_keyed_by_canonical_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(leases, "_account_home", lambda: tmp_path / "account")
    directory = leases.environment_state(tmp_path / "installation")
    assert directory == leases.environment_state(tmp_path / "other" / ".." / "installation")
    assert directory != leases.environment_state(tmp_path / "different-installation")
    if os.name == "nt":
        from sparkrun.utils.windows import check_private

        check_private(directory)
    else:
        assert directory.stat().st_mode & 0o777 == 0o700


def test_live_worker_blocks_update_across_temp_roots_and_config_bindings(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(leases, "_account_home", lambda: tmp_path / "account")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "updater-temp"))
    config = SparkrunConfig(tmp_path / "updater" / "config.yaml")
    other = SparkrunConfig(tmp_path / "worker" / "config.yaml")
    calls = []
    with worker(tmp_path, other) as record:
        assert Path(record["environment"]) == leases.environment_state()
        with pytest.raises(RuntimeError, match="draining workers"):
            leases.guarded_update(config, lambda: calls.append("replace"))
        assert not calls
    leases.guarded_update(config, lambda: calls.append("replace"))
    assert calls == ["replace"]


@pytest.mark.parametrize("legacy", ["temp", "config"])
def test_transition_honors_legacy_records(tmp_path, monkeypatch, legacy):
    import tempfile

    monkeypatch.setattr(leases, "_account_home", lambda: tmp_path / "account")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / ("worker-temp" if legacy == "temp" else "different-temp")))
    config = SparkrunConfig(tmp_path / "config" / "config.yaml")
    with worker(tmp_path, config, legacy=legacy):
        with pytest.raises(RuntimeError, match="draining workers"):
            leases.guarded_update(config, lambda: pytest.fail("live legacy lease ignored"))
    leases.guarded_update(config, lambda: None)


def test_dead_environment_lease_does_not_block_update(tmp_path, monkeypatch):
    from sparkrun.core.process_identity import process_identity

    monkeypatch.setattr(leases, "_account_home", lambda: tmp_path / "account")
    directory = leases.environment_state() / "runtime-leases"
    directory.mkdir()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        identity = process_identity(child.pid)
        assert identity
        path = directory / "dead.json"
        path.write_text(json.dumps({"process": identity}), encoding="utf-8")
    finally:
        child.terminate()
        child.wait(timeout=10)
    leases.guarded_update(SparkrunConfig(tmp_path / "config.yaml"), lambda: None)
    assert not path.exists()
