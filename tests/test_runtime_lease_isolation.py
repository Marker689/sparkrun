"""The suite's runtime-lease redirect: shared with children, inert outside pytest.

``conftest.isolate_runtime_state`` points lease coordination at a per-test
directory. These pin the three properties it exists for: a spawned child meets
the test process in that directory, a live lease in the account root (another
suite, an orphaned worker, a real daemon) cannot fail an update-guard test, and
the redirect needs the pytest session marker, so a stray export is inert.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sparkrun.core import runtime_lease as leases
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.process_identity import process_identity


@contextmanager
def _live_process():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        identity = process_identity(child.pid)
        assert identity
        yield identity
    finally:
        child.kill()
        child.wait(timeout=10)


def _plant_lease(environment, identity):
    directory = environment / "runtime-leases"
    directory.mkdir(mode=0o700, exist_ok=True)
    (directory / "foreign.json").write_text(json.dumps({"process": identity, "kind": "worker"}), encoding="utf-8")


def test_redirected_root_is_shared_with_spawned_children(isolate_runtime_state, tmp_path):
    environment = leases.environment_state()
    assert environment.parent == isolate_runtime_state.resolve() / ".sparkrun-runtime"
    assert leases._legacy_environment_states() == []
    code = """
import sys
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.runtime_lease import environment_state, runtime_lease
with runtime_lease(SparkrunConfig(sys.argv[1]), kind='test-worker'):
    print(environment_state(), flush=True)
    sys.stdin.read()
"""
    worker_config = tmp_path / "worker" / "config.yaml"
    with subprocess.Popen(
        [sys.executable, "-c", code, str(worker_config)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    ) as child:
        try:
            assert Path(child.stdout.readline().strip()) == environment
            # A different config binding sees the child only through the shared root.
            with pytest.raises(RuntimeError, match="draining workers"):
                leases.assert_runtime_idle(SparkrunConfig(tmp_path / "updater" / "config.yaml"))
        finally:
            child.stdin.close()
            child.wait(timeout=10)
    assert child.returncode == 0
    leases.assert_runtime_idle(SparkrunConfig(tmp_path / "updater" / "config.yaml"))


def test_live_lease_in_account_and_legacy_roots_does_not_reach_the_suite(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(leases, "_account_home", lambda: tmp_path / "account")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "temp"))
    monkeypatch.setattr(leases, "_LEGACY_TEMP_ROOTS", ())
    config = SparkrunConfig(tmp_path / "config" / "config.yaml")
    with _live_process() as identity:
        # Where production would put them: the account root and a legacy temp root.
        monkeypatch.delenv(leases.TEST_STATE_ROOT_ENV)
        _plant_lease(leases.environment_state(), identity)
        (tmp_path / "temp").mkdir()
        legacy = tmp_path / "temp" / leases._environment_name()
        legacy.mkdir(mode=0o700)
        _plant_lease(legacy, identity)
        with pytest.raises(RuntimeError, match="draining workers"):
            leases.guarded_update(config, lambda: pytest.fail("planted lease ignored"))

        monkeypatch.setenv(leases.TEST_STATE_ROOT_ENV, str(tmp_path / "suite"))
        calls = []
        leases.guarded_update(config, lambda: calls.append("replace"))
        assert calls == ["replace"]


def test_redirect_is_inert_without_the_test_session_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(leases, "_account_home", lambda: tmp_path / "account")
    monkeypatch.delenv(leases.TEST_SESSION_MARKER_ENV)
    assert leases.environment_state().parent == (tmp_path / "account").resolve() / ".sparkrun-runtime"

    monkeypatch.setenv(leases.TEST_SESSION_MARKER_ENV, "1")
    monkeypatch.setenv(leases.TEST_STATE_ROOT_ENV, "relative/root")
    with pytest.raises(RuntimeError, match="absolute"):
        leases.environment_state()


def test_ordinary_environment_does_not_relocate_the_production_root(tmp_path, monkeypatch, real_runtime_state):
    monkeypatch.setattr(leases, "_account_home", lambda: tmp_path / "account")
    before = leases.environment_state()
    for name in ("HOME", "USERPROFILE", "XDG_CACHE_HOME", "XDG_RUNTIME_DIR", "TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(tmp_path / name))
    assert leases.environment_state() == before
    assert os.environ.get(leases.TEST_STATE_ROOT_ENV) is None
