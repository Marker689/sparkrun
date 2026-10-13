"""Real reader processes exercise cancellation, completion and resource cleanup."""

import os
import subprocess
import sys
import threading
from unittest.mock import Mock

import psutil
import pytest

from sparkrun.core.log_source import LogSource
from sparkrun.orchestration import logs


SOURCE = LogSource(host="localhost", container="reader-contract")


@pytest.fixture
def spawned(monkeypatch):
    processes = []
    original = logs._spawn

    def spawn(command):
        proc = original(command)
        processes.append(proc)
        return proc

    monkeypatch.setattr(logs, "_spawn", spawn)
    yield processes
    for proc in processes:
        if proc.poll() is None:
            logs._terminate(proc)


def assert_reaped(processes):
    assert processes and all(proc.poll() is not None for proc in processes)
    assert all(proc.stdout.closed for proc in processes)
    assert not any(t.name == "sparkrun-log-watch-" + SOURCE.label for t in threading.enumerate())


@pytest.mark.parametrize(
    "script",
    [
        "import time; print('ready', flush=True); time.sleep(60)",
        "import time\nwhile True:\n print('ready', flush=True)\n time.sleep(0.001)",
        "import os,time; print('ready', flush=True); os.close(1); os.close(2); time.sleep(60)",
        pytest.param(
            "import subprocess,sys\nsubprocess.run([sys.executable, '-c', "
            '"import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
            "print('child:' + str(os.getpid()), flush=True); time.sleep(60)\"])",
            marks=pytest.mark.skipif(os.name == "nt", reason="POSIX process-group and SIGTERM contract"),
        ),
    ],
    ids=["silent", "busy", "closed-output", "stubborn-child"],
)
def test_finite_cancel_reaps_readers_and_children(monkeypatch, spawned, script):
    monkeypatch.setattr(logs, "build_read_command", lambda *a, **kw: [sys.executable, "-c", script])
    cancel = threading.Event()
    ready = threading.Event()
    errors, children = [], []

    def consume():
        try:
            for line in logs.read_log_sources(Mock(), [SOURCE], cancel=cancel, check=True):
                if line.text.startswith("child:"):
                    children.append(psutil.Process(int(line.text.split(":")[1])))
                ready.set()
        except Exception as exc:
            errors.append(exc)

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    try:
        assert ready.wait(5)
    finally:
        cancel.set()
        reader.join(5)
    assert not reader.is_alive()
    assert errors == []  # Termination requested by the caller is not a read failure.
    assert_reaped(spawned)
    for child in children:
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE


@pytest.mark.parametrize("follow", [False, True])
@pytest.mark.parametrize("check", [False, True])
def test_exit_status_is_checked_in_both_modes(monkeypatch, spawned, follow, check):
    command = [sys.executable, "-c", "import sys; print('ssh failed', flush=True); sys.exit(7)"]
    monkeypatch.setattr(logs, "build_read_command", lambda *a, **kw: command)
    reader = logs.read_log_sources(Mock(), [SOURCE], follow=follow, check=check)
    if check:
        with pytest.raises(subprocess.CalledProcessError) as failure:
            list(reader)
        assert failure.value.returncode == 7
    else:
        assert [line.text for line in reader] == ["ssh failed"]
    assert_reaped(spawned)


@pytest.mark.parametrize("close_output", [False, True])
def test_timeout_reaps_silent_reader_even_after_stdout_closes(monkeypatch, spawned, close_output):
    script = "import os,time; " + ("os.close(1); os.close(2); " if close_output else "") + "time.sleep(60)"
    monkeypatch.setattr(logs, "build_read_command", lambda *a, **kw: [sys.executable, "-c", script])
    with pytest.raises(subprocess.TimeoutExpired):
        list(logs.read_log_sources(Mock(), [SOURCE], check=True, timeout=0.15))
    assert_reaped(spawned)


def test_close_partial_iterator_reaps_reader_and_watcher(spawned):
    reader = logs.read_log_command(
        [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(60)"],
        SOURCE,
        check=True,
        cancel=threading.Event(),
    )
    assert next(reader).text == "ready"
    reader.close()
    assert_reaped(spawned)


@pytest.mark.parametrize("follow", [False, True])
def test_already_cancelled_does_not_spawn(spawned, follow):
    cancel = threading.Event()
    cancel.set()
    assert list(logs.read_log_sources(Mock(), [SOURCE], follow=follow, cancel=cancel, check=True)) == []
    assert spawned == []


def test_finite_cancel_does_not_open_later_sources(monkeypatch, spawned):
    monkeypatch.setattr(logs, "build_read_command", lambda *a, **kw: [sys.executable, "-c", "print('ready')"])
    cancel = threading.Event()
    reader = logs.read_log_sources(Mock(), [SOURCE, LogSource("other", "other")], cancel=cancel, check=True)
    next(reader)
    cancel.set()
    assert list(reader) == []
    assert len(spawned) == 1
    assert_reaped(spawned)


def test_reader_has_no_terminal_stdin(spawned):
    command = [sys.executable, "-c", "import sys; print('eof:' + str(sys.stdin.read() == ''))"]
    assert [line.text for line in logs.read_log_command(command, SOURCE, check=True, timeout=5)] == ["eof:True"]
    assert_reaped(spawned)
