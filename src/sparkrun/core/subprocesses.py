"""Owned short-lived process trees with portable, fail-closed containment."""

from contextlib import contextmanager
import logging
import os
import signal
import subprocess
import sys

from sparkrun.core.process_identity import detached_process_options

logger = logging.getLogger(__name__)

# The base interpreter avoids Windows venv redirectors spawning outside the job
# before assignment. Isolated/no-site startup cannot run environment hooks. The
# command is argv data, and cannot start until its parent installs containment.
_WINDOWS_BOOTSTRAP = """import sys
if sys.stdin.buffer.read(1) != b'1':
    sys.exit(1)
import subprocess
sys.exit(subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL))
"""


@contextmanager
def owned_process(command, *, env=None, separate_stderr=False):
    """Own a process group on POSIX or a nested Job Object on Windows.

    Keep Windows containment in the parent so grandchildren remain owned after
    the command/launcher exits. A parent crash also closes the job. POSIX uses
    the existing process-group boundary; deliberate session escape is excluded.
    """
    windows = os.name == "nt"
    argv = [getattr(sys, "_base_executable", sys.executable), "-I", "-S", "-c", _WINDOWS_BOOTSTRAP, *command] if windows else command
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if windows else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if separate_stderr else subprocess.STDOUT,
        env=env,
        bufsize=0,
        **detached_process_options(),
    )
    job = None
    try:
        if windows:
            from sparkrun.utils.windows import ChildJob

            job = ChildJob(proc.pid)
            assert proc.stdin is not None
            proc.stdin.write(b"1")
            proc.stdin.close()
        yield proc
    finally:
        try:
            if job is not None:
                job.close()
            elif windows:
                proc.kill()  # Assignment failed; the command never passed its gate.
            else:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.warning("Owned process did not exit after termination")
            finally:
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
