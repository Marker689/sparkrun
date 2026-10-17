"""Regressions for collection-time state and swallowed external I/O."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _python(script, *, env=None):
    tests = str(Path(__file__).parent)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, **(env or {}), "PYTHONPATH": tests + os.pathsep + os.environ.get("PYTHONPATH", "")},
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_guard_records_caught_process_and_network_escapes():
    # A fresh interpreter tests the real audit hook without adding intentional
    # violations to the outer pytest session's escape list.
    _python("""
import socket, subprocess, sys
from _test_isolation import IOGuard

guard = IOGuard()
sys.addaudithook(guard)
for command in (["ssh", "fixture.invalid", "true"], ["docker", "image", "inspect", "fixture"], ["uvx", "llama-benchy", "--version"]):
    try:
        subprocess.run(command, check=True)
    except OSError:
        pass  # Application best-effort paths swallow this too.
assert len(guard.attempts) == 3, guard.attempts

# Emit CPython's socket events without risking a packet if the guard regresses.
with socket.socket() as sock:
    for event, args in (
        ("socket.connect", (sock, ("192.0.2.1", 22))),
        ("socket.getaddrinfo", ("fixture.invalid", 22, 0, 0, 0)),
        ("socket.sendto", (sock, b"payload", ("192.0.2.1", 22))),
    ):
        try:
            sys.audit(event, *args)
        except OSError:
            pass
assert len(guard.attempts) == 6, guard.attempts

# Real local integrations still work.
with socket.socket() as server:
    server.bind(("127.0.0.1", 0))
    server.listen()
    with socket.create_connection(server.getsockname()) as client:
        connection, _ = server.accept()
        with connection:
            client.sendall(b"local")
            assert connection.recv(5) == b"local"
subprocess.run([sys.executable, "-c", "pass"], check=True)
assert len(guard.attempts) == 6
""")


def test_collection_and_children_ignore_ambient_profile_config(tmp_path):
    poison = tmp_path / "ambient"
    poison.mkdir()
    (poison / "config.yaml").write_text("self_update:\n  channel: unsupported\n")
    _python(
        """
from pathlib import Path
import os, subprocess, sys
import conftest  # Must isolate home BEFORE its first application import.
try:
    from sparkrun.cli import main
    from sparkrun.core.application_profile import ApplicationProfile, select_application_profile, _reset_application_profile_for_tests
    from sparkrun.core.config import SparkrunConfig
    assert SparkrunConfig().self_update_channel == "stable"
    _reset_application_profile_for_tests()
    select_application_profile(ApplicationProfile(id="alternate-test", display_name="Test", command="alternate-test", package="alternate-test"))
    config = SparkrunConfig()
    assert config.config_path == Path.home() / ".config/alternate-test/config.yaml"
    assert config.self_update_channel == "stable"
    child_home = subprocess.check_output([sys.executable, "-c", "from pathlib import Path; print(Path.home())"], text=True).strip()
    assert child_home == str(Path.home())
    assert "ALTERNATE_TEST_CONFIG_DIR" not in os.environ
finally:
    conftest.pytest_unconfigure(None)
""",
        env={"SPARKRUN_CONFIG_DIR": str(poison), "ALTERNATE_TEST_CONFIG_DIR": str(poison)},
    )


def test_guard_accepts_windows_audit_payload_and_local_ssh_parser():
    """CPython on Windows sends a command line with executable=None."""
    import pytest
    from _test_isolation import IOGuard

    guard = IOGuard()
    command = subprocess.list2cmdline([r"C:\Program Files\Python\python.exe", "-c", "print('a', \"b\")"])
    guard("subprocess.Popen", (None, command, None, None))
    guard("subprocess.Popen", (None, r'"C:\Program Files\Python\python.exe" -c "pass"', None, None))
    guard("subprocess.Popen", (None, r'"C:\Program Files\OpenSSH\ssh.exe" -G -F "C:\Temp\test config" fixture', None, None))
    assert guard.attempts == []
    with pytest.raises(OSError, match="Unmocked external I/O"):
        guard("subprocess.Popen", (None, r'"C:\Program Files\Docker\docker.exe" image inspect fixture', None, None))
    assert len(guard.attempts) == 1
