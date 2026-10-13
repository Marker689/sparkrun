"""Real measurement subprocess contracts, also run on native macOS/Windows CI."""

import json
import os
import subprocess
import sys
import time

import psutil
import pytest

from sparkrun.benchmarking._credentials import BenchmarkCredentials
from sparkrun.benchmarking._process import run_benchmark_process
from sparkrun.core.operation import OperationState, operation_scope
from sparkrun.core.subprocesses import _WINDOWS_BOOTSTRAP


def execute(code, *, timeout=10, stdout=None, stderr=None):
    return run_benchmark_process(
        [sys.executable, "-c", code],
        timeout=timeout,
        credentials=BenchmarkCredentials("secret é"),
        stdout=stdout or (lambda line: None),
        stderr=stderr,
    )


def assert_stopped(pids):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        running = []
        for pid in pids:
            try:
                process = psutil.Process(pid)
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    running.append(process)
            except psutil.NoSuchProcess:
                pass
        if not running:
            return
        time.sleep(0.02)
    # Avoid leaking processes even when containment regresses.
    for process in running:
        process.kill()
    pytest.fail(f"Measurement processes survived cleanup: {pids}")


def tree_code(*, exit_parent=False):
    return (
        "import subprocess, sys, os, json, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "print(json.dumps([os.getpid(), child.pid]), flush=True); " + ("sys.exit(0)" if exit_parent else "time.sleep(60)")
    )


@pytest.mark.parametrize("exit_code", [0, 17])
def test_large_dual_stream_output_is_drained_and_redacted(exit_code):
    payload = ("hello secret é 世界\n" * 10000).encode()
    # Keep argv below Windows' command-line limit; the child constructs the data.
    code = (
        "import os, sys; data = ('hello secret é 世界\\n' * 10000).encode(); "
        "[(os.write(1, data[i:i+257]), os.write(2, data[i:i+257])) for i in range(0, len(data), 257)]; "
        f"sys.exit({exit_code})"
    )
    stdout, stderr = [], []
    assert execute(code, stdout=stdout.append, stderr=stderr.append) == exit_code
    expected = payload.decode().replace("secret é", "[REDACTED]")
    assert "".join(stdout) == "".join(stderr) == expected


def test_quiet_timeout_is_bounded_and_does_not_expose_command():
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as error:
        execute("import time; secret = 'secret é'; time.sleep(60)", timeout=0.5)
    assert error.value.cmd == "benchmark"
    assert "secret" not in str(error.value)
    assert time.monotonic() - started < 5


def test_python_output_uses_utf8_independently_of_controller_locale(monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    output = []
    assert execute("print('secret é 世界')", stdout=output.append) == 0
    assert "".join(output).strip() == "[REDACTED] 世界"


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_callback_failure_cleans_up_descendants(error):
    pids = []

    def output(line):
        pids.extend(json.loads(line))
        raise error("subscriber stopped")

    with pytest.raises(error, match="subscriber stopped"):
        execute(tree_code(), stdout=output)
    assert len(pids) == 2
    assert_stopped(pids)


def test_pipe_inheriting_descendant_cannot_escape_deadline_after_parent_exits():
    pids = []
    with pytest.raises(subprocess.TimeoutExpired):
        execute(tree_code(exit_parent=True), timeout=2, stdout=lambda line: pids.extend(json.loads(line)))
    assert len(pids) == 2
    assert_stopped(pids)


def test_success_cleans_up_descendant_that_closed_its_pipes():
    code = (
        "import subprocess, sys, json, os; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "print(json.dumps([os.getpid(), child.pid]), flush=True)"
    )
    pids = []
    assert execute(code, stdout=lambda line: pids.extend(json.loads(line))) == 0
    assert len(pids) == 2
    assert_stopped(pids)


def test_operation_cancellation_interrupts_quiet_measurement():
    pids = []
    state = OperationState()

    def output(line):
        pids.extend(json.loads(line))
        state.cancel.set()

    with operation_scope(state=state), pytest.raises(InterruptedError, match="cancelled"):
        execute(tree_code(), timeout=None, stdout=output)
    assert_stopped(pids)


@pytest.mark.parametrize("release", [False, True])
def test_bootstrap_does_not_execute_until_parent_releases_gate(tmp_path, release):
    marker = tmp_path / "started"
    command = [
        sys.executable,
        "-I",
        "-S",
        "-c",
        _WINDOWS_BOOTSTRAP,
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).touch()",
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as child:
        time.sleep(0.1)
        assert not marker.exists()
        child.communicate(b"1" if release else b"", timeout=10)
    assert marker.exists() is release


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job Objects")
def test_failed_job_assignment_cannot_start_measurement(monkeypatch, tmp_path):
    from sparkrun.utils import windows

    marker = tmp_path / "started"

    def fail(pid):
        raise OSError("assignment denied")

    monkeypatch.setattr(windows, "ChildJob", fail)
    with pytest.raises(OSError, match="assignment denied"):
        execute(f"from pathlib import Path; Path({str(marker)!r}).touch()")
    assert not marker.exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows nested daemon/measurement containment")
def test_measurement_job_can_nest_inside_daemon_worker_job():
    source = """
import subprocess, sys
from sparkrun.utils.windows import WorkerJob
from sparkrun.core.process_identity import process_identity
from sparkrun.benchmarking._process import run_benchmark_process
from sparkrun.benchmarking._credentials import BenchmarkCredentials
job = WorkerJob(process_identity())
lines = []
assert run_benchmark_process([sys.executable, '-c', 'print(42)'], timeout=10,
    credentials=BenchmarkCredentials(), stdout=lines.append) == 0
assert ''.join(lines).strip() == '42'
try:
    run_benchmark_process([sys.executable, '-c', 'import time; time.sleep(60)'], timeout=0.5,
        credentials=BenchmarkCredentials(), stdout=lines.append)
except subprocess.TimeoutExpired:
    pass
else:
    raise AssertionError('timeout was not enforced')
job.release_success()
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
