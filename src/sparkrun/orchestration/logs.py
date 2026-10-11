"""Reading :class:`~sparkrun.core.log_source.LogSource` streams into lines.

The transport half of the log path.  A runtime says *what* to read
(:meth:`~sparkrun.runtimes.base.RuntimePlugin.log_sources`), an executor
says *how* to read it on its substrate
(:meth:`~sparkrun.orchestration.executors._base.Executor.read_logs_cmd`),
and this module actually runs those commands — locally or over SSH — and
turns their output into :class:`~sparkrun.core.log_source.LogLine` records.

Ordering contract (deliberate, and not the same in both modes):

* ``follow=True`` — one reader thread per source feeding a shared queue, so
  lines are yielded **as they arrive**.  That is true time order for live
  streaming, and each line is stamped with its arrival time.
* ``follow=False`` — sources are read **rank-grouped**: each source in full,
  head first, then workers by rank.  Cross-host time-interleaving is not
  possible here: the serve log written by ``scripts/exec_serve_detached.sh``
  carries no per-line capture timestamps, and a dump delivers every line at
  once, so arrival time carries no information either.  Inferring order from
  the runtime's own log prefixes would silently mis-order every unstamped
  line (tracebacks, progress bars, warnings), which is worse than honest
  grouping.  Getting true dump ordering requires stamping at capture time in
  the launch redirect — a deliberate, separate change.

``sparkrun.api.logs`` composes this into the public iterator; the CLI
renders it.  Nothing here imports ``api`` (layering).
"""

from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager, ExitStack
import math
import os
import queue
import signal
import subprocess
import threading
import time
from typing import Callable, Generator, Iterator, Sequence

from sparkrun.core.log_source import LogLine, LogSource, LogSourceChanged, source_sort_key

logger = logging.getLogger(__name__)

_QUEUE_SENTINEL = object()


def build_read_command(
    executor,
    source: LogSource,
    *,
    follow: bool = False,
    tail: int | None = None,
    ssh_kwargs: dict | None = None,
) -> list[str]:
    """Build the argv that reads *source*, dispatching local vs SSH.

    Mirrors the dispatch every other orchestration primitive uses
    (:func:`~sparkrun.orchestration.ssh.should_run_locally`), so reading a
    workload on the control machine itself doesn't pointlessly round-trip
    through sshd — and doesn't break when SSH-to-self isn't configured.

    The two branches quote differently, and must. Locally the argv reaches
    ``execve`` untouched, so *command* is already a single argument. Over SSH
    it does not: ``ssh`` joins its trailing argv into one string and the
    remote login shell re-splits it on whitespace. An unquoted
    ``["bash", "-c", command]`` therefore arrives as
    ``bash -c docker exec … tail …``, where ``bash -c`` takes only ``docker``
    as the command and the rest become ``$0, $1, …`` — so the remote runs a
    bare ``docker`` and prints its help instead of the logs. Quoting collapses
    *command* back into the single word ``bash -c`` expects.
    """
    command = executor.read_logs_cmd(source, follow=follow, tail=tail)
    return _dispatch_argv(source, command, ssh_kwargs)


def _dispatch_argv(source: LogSource, command: str | list[str], ssh_kwargs: dict | None = None) -> list[str]:
    from sparkrun.orchestration.ssh import build_ssh_cmd, should_run_locally
    from sparkrun.utils.shell import quote

    if isinstance(command, list):
        return list(command)
    kwargs = dict(ssh_kwargs or {})
    if should_run_locally(source.host, kwargs.get("ssh_user")):
        return ["bash", "-c", command]
    return build_ssh_cmd(source.host, **kwargs) + ["bash", "-c", quote(command)]


def build_read_window_command(
    executor,
    source: LogSource,
    *,
    skip: int,
    count: int,
    ssh_kwargs: dict | None = None,
) -> list[str] | None:
    """Build the argv that reads a log window for *source*, dispatching local vs SSH.

    Returns ``None`` when the executor does not support windowed reads.
    """
    command = executor.read_logs_window_cmd(source, skip=skip, count=count)
    if command is None:
        return None
    return _dispatch_argv(source, command, ssh_kwargs)


def _spawn(cmd: list[str], *, merge_stderr: bool = True) -> subprocess.Popen:
    # Background SSH readers must not consume the CLI/TUI's keystrokes.
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )


@contextmanager
def _reader_process(cmd: list[str], source: LogSource, *, merge_stderr: bool):
    """Own the process and drain diagnostics without buffering unbounded output."""
    proc = _spawn(cmd, merge_stderr=merge_stderr)
    diagnostics: deque[str] = deque(maxlen=16)
    stderr_thread = None

    def drain() -> None:
        assert proc.stderr is not None
        # Bound both the retained tail and individual chunks, even for a tool
        # emitting a very long line. Drain concurrently to avoid pipe deadlock.
        while chunk := proc.stderr.readline(4096):
            diagnostics.append(chunk)
            logger.warning("Log reader %s: %s", source.label, chunk.rstrip("\n"))

    try:
        if proc.stderr is not None:
            thread = threading.Thread(target=drain, name="sparkrun-log-stderr-%s" % source.label, daemon=True)
            thread.start()
            stderr_thread = thread
        yield proc, diagnostics
    finally:
        # Reap children before joining: a child can hold stderr open after its
        # parent exits. Consumers inspect diagnostics only after this cleanup.
        _terminate(proc)
        if stderr_thread is not None:
            stderr_thread.join()
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def _line(source: LogSource, text: str, *, timestamp: float | None = None) -> LogLine:
    return LogLine(
        host=source.host,
        container=source.container,
        text=text.rstrip("\n"),
        role=source.role,
        rank=source.rank,
        timestamp=timestamp,
        source_id=source.source_id,
        resource=source.resource,
    )


def _read_one(
    executor,
    source: LogSource,
    *,
    follow: bool,
    tail: int | None,
    ssh_kwargs: dict | None,
    cancel: threading.Event | None,
    check: bool,
    timeout: float | None,
) -> Iterator[LogLine]:
    """Yield every line from a single source, then return."""
    with _validated_source(executor, source, cancel=cancel, timeout=timeout) as remaining:
        if cancel is not None and cancel.is_set():
            return
        cmd = build_read_command(executor, source, follow=follow, tail=tail, ssh_kwargs=ssh_kwargs)
        logger.debug("Reading logs from %s: %s", source.label, " ".join(cmd))
        yield from read_log_command(cmd, source, cancel=cancel, check=check, timeout=remaining(), merge_stderr=False)


@contextmanager
def _validated_source(executor, source, *, cancel, timeout: float | None = None):
    """Keep finite reads/pages tied to one native instance, within one deadline."""
    deadline = None if timeout is None else time.monotonic() + timeout

    def remaining() -> float | None:
        if deadline is None:
            return None
        left = deadline - time.monotonic()
        if left <= 0:
            assert timeout is not None
            raise subprocess.TimeoutExpired(source.label, timeout)
        return left

    def validate():
        if source.resource is not None and not (cancel is not None and cancel.is_set()):
            executor.validate_log_source(source, cancel=cancel, timeout=remaining())

    validate()
    yield remaining
    # Do not mask reader errors or run extra requests on generator.close().
    validate()


def read_log_command(
    cmd: list[str],
    source: LogSource,
    *,
    check: bool = False,
    cancel: threading.Event | None = None,
    timeout: float | None = None,
    merge_stderr: bool = True,
) -> Generator[LogLine, None, None]:
    """Read a local argv into structured lines; closing the iterator stops it.

    Direct callers retain merged stderr by default. Executor-backed readers
    set merge_stderr=False: stderr is drained/logged separately and its bounded
    tail is attached to read failures. Native control-plane plugins reuse this
    reader without routing a local client command through SSH.
    With check=True, a nonzero reader exit raises CalledProcessError.
    Cancellation ends iteration normally, including while waiting on a silent
    command. A positive finite timeout bounds the command's wall-clock lifetime
    and raises TimeoutExpired. Neither cancellation nor timeout stops the workload.
    """
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Log read timeout must be positive and finite")
    if cancel is not None and cancel.is_set():
        return
    with _reader_process(cmd, source, merge_stderr=merge_stderr) as (proc, diagnostics):
        finished = threading.Event()
        expired = threading.Event()
        deadline = None if timeout is None else time.monotonic() + timeout

        def watch() -> None:
            while not finished.wait(0.05):
                if cancel is not None and cancel.is_set():
                    _terminate(proc)
                    return
                if deadline is not None and time.monotonic() >= deadline:
                    expired.set()
                    _terminate(proc)
                    return

        watcher = None
        try:
            if cancel is not None or timeout is not None:
                thread = threading.Thread(target=watch, name="sparkrun-log-watch-%s" % source.label, daemon=True)
                thread.start()
                watcher = thread
            assert proc.stdout is not None
            for text in proc.stdout:
                if expired.is_set() or (cancel is not None and cancel.is_set()):
                    break
                yield _line(source, text)
            rc = proc.wait()
        finally:
            finished.set()
            if watcher is not None:
                watcher.join()
    if cancel is not None and cancel.is_set():
        return
    if expired.is_set():
        assert timeout is not None
        raise subprocess.TimeoutExpired(cmd, timeout, stderr="".join(diagnostics))
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, stderr="".join(diagnostics))


def read_log_window(
    executor,
    source: LogSource,
    *,
    skip: int,
    count: int,
    ssh_kwargs: dict | None = None,
    timeout: float = 30.0,
) -> list[LogLine]:
    """Read *count* lines preceding the last *skip* lines for *source*.

    Returns an empty list when the executor does not support windowed reads.
    Raises :class:`subprocess.CalledProcessError` if the reader command exits non-zero,
    or :class:`subprocess.TimeoutExpired` if the reader does not complete within *timeout*.
    """
    skip = max(0, int(skip))
    count = max(0, int(count))
    cmd = build_read_window_command(executor, source, skip=skip, count=count, ssh_kwargs=ssh_kwargs)
    if cmd is None:
        return []

    proc = _spawn(cmd)
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    finally:
        _terminate(proc)
        if proc.stdout is not None:
            proc.stdout.close()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout)

    return [_line(source, text) for text in stdout.splitlines()]


def _terminate(proc: subprocess.Popen) -> None:
    """Reap the reader session, including children that outlive their shell."""
    try:
        if os.name == "nt":
            from sparkrun.core.process_identity import process_identity, terminate_tree

            if proc.poll() is None:
                terminate_tree(process_identity(proc.pid))
            proc.wait(timeout=1)
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        # A shell can exit before a child that ignores TERM. Always finish
        # the owned process group, even if poll()/wait() says the shell exited.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=1)
    except Exception:
        logger.debug("Log reader cleanup failed", exc_info=True)


def read_log_sources(
    executor,
    sources: Sequence[LogSource],
    *,
    follow: bool = False,
    tail: int | None = None,
    ssh_kwargs: dict | None = None,
    cancel: threading.Event | None = None,
    check: bool = False,
    timeout: float | None = None,
) -> Generator[LogLine, None, None]:
    """Read sources in arrival order when following, otherwise rank-grouped.

    In both modes, ``cancel`` ends even a silent reader and ``check`` propagates
    reader failures. ``timeout`` bounds each non-follow command; following has
    no deadline and must use cancellation. No option changes workload lifetime.
    """
    if follow and timeout is not None:
        raise ValueError("Log read timeout is only supported for non-follow reads")
    if not sources or (cancel is not None and cancel.is_set()):
        return
    if not follow:
        for source in sorted(sources, key=source_sort_key):
            if cancel is not None and cancel.is_set():
                return
            yield from _read_one(
                executor, source, follow=False, tail=tail, ssh_kwargs=ssh_kwargs, cancel=cancel, check=check, timeout=timeout
            )
        return
    yield from _interleave(executor, sources, tail=tail, ssh_kwargs=ssh_kwargs, cancel=cancel, check=check)


def _interleave(
    executor,
    sources: Sequence[LogSource],
    *,
    tail: int | None,
    ssh_kwargs: dict | None,
    cancel: threading.Event | None = None,
    check: bool = False,
) -> Iterator[LogLine]:
    """Follow sources concurrently, with bounded, cancellation-aware queues."""
    lines: queue.Queue = queue.Queue(maxsize=1000)
    procs: list[subprocess.Popen] = []
    stop = threading.Event()
    lock = threading.Lock()

    def put(item) -> None:
        # A full queue must not strand a producer after its consumer exits.
        while not stop.is_set():
            try:
                lines.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def pump(source: LogSource) -> None:
        try:
            with ExitStack() as readers:
                readers.enter_context(_validated_source(executor, source, cancel=stop))
                cmd = build_read_command(executor, source, follow=True, tail=tail, ssh_kwargs=ssh_kwargs)
                logger.debug("Following logs from %s: %s", source.label, " ".join(cmd))
                with lock:
                    if stop.is_set():
                        return
                    proc, diagnostics = readers.enter_context(_reader_process(cmd, source, merge_stderr=False))
                    procs.append(proc)
                assert proc.stdout is not None
                for text in proc.stdout:
                    if stop.is_set():
                        break
                    put(_line(source, text, timestamp=time.time()))
                rc = proc.wait() if check and not stop.is_set() else 0
            if check and not stop.is_set() and rc != 0:
                raise subprocess.CalledProcessError(rc, cmd, stderr="".join(diagnostics))
        except Exception as error:
            logger.debug("Log reader for %s failed: %s", source.label, error)
            if check or isinstance(error, LogSourceChanged):
                put(error)
        finally:
            put(_QUEUE_SENTINEL)

    threads = [threading.Thread(target=pump, args=(source,), daemon=True, name="sparkrun-logs-%s" % source.label) for source in sources]
    for thread in threads:
        thread.start()
    remaining = len(threads)
    try:
        while remaining:
            if cancel is not None and cancel.is_set():
                break
            try:
                item = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _QUEUE_SENTINEL:
                remaining -= 1
            elif isinstance(item, Exception):
                raise item
            else:
                yield item
    finally:
        # Serialize spawn/registration with shutdown so no late SSH child leaks.
        with lock:
            stop.set()
            active = list(procs)
        for proc in active:
            _terminate(proc)
        for thread in threads:
            thread.join(timeout=1)


class LogFollower:
    """Console-free attachment with explicit completion, error and cancellation.

    EOF says only that logs ended; callers decide independently whether the
    workload is ready. Stopping reaps reader processes, never the workload.
    """

    def __init__(self, executor, sources, *, on_line, ssh_kwargs=None, tail=100):
        self.done = threading.Event()
        self.error: Exception | None = None
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sparkrun-log-follow", daemon=True)
        self._executor = executor
        self._sources = sources
        self._on_line = on_line
        self._ssh_kwargs = ssh_kwargs
        self._tail = tail

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        from contextlib import closing

        try:
            with closing(
                read_log_sources(
                    self._executor,
                    self._sources,
                    follow=True,
                    tail=self._tail,
                    ssh_kwargs=self._ssh_kwargs,
                    cancel=self._cancel,
                    check=True,
                )
            ) as lines:
                for line in lines:
                    self._on_line(line)
        except Exception as error:
            self.error = error
        finally:
            self.done.set()

    def stop(self):
        self._cancel.set()
        self._thread.join(timeout=12)


def print_log_sources(
    executor,
    sources: Sequence[LogSource],
    *,
    follow: bool = False,
    tail: int | None = None,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
) -> None:
    """Print *sources* to stdout — the printing shim over :func:`read_log_sources`.

    Used by :meth:`~sparkrun.runtimes.base.RuntimePlugin.follow_logs` for the
    post-launch attach.  The ``sparkrun logs`` command renders
    :func:`sparkrun.api.logs` instead; both read through the same machinery.
    """
    if not sources:
        logger.warning("No log sources to read.")
        return

    if dry_run:
        for source in sources:
            cmd = build_read_command(executor, source, follow=follow, tail=tail, ssh_kwargs=ssh_kwargs)
            logger.info("[dry-run] Would stream logs from %s: %s", source.label, " ".join(cmd))
        return

    if follow:
        logger.info(
            "Following logs from %d source(s): %s (Ctrl-C to stop)...",
            len(sources),
            ", ".join(source.label for source in sources),
        )

    prefixed = len(sources) > 1
    try:
        for line in read_log_sources(executor, sources, follow=follow, tail=tail, ssh_kwargs=ssh_kwargs):
            print("[%s] %s" % (line.host, line.text) if prefixed else line.text)
    except KeyboardInterrupt:
        logger.info("\nLog following stopped.")


def dump_log_sources(
    executor,
    sources: Sequence[LogSource],
    fh,
    *,
    ssh_kwargs: dict | None = None,
    cancel: threading.Event | None = None,
    prefix: bool | str | Callable[[LogLine], str] = False,
) -> int:
    """Stream non-follow reads of *sources* to file handle *fh*.

    Sources are read in rank-grouped order (sorted by rank, host, container)
    without buffering whole logs into memory.
    If *cancel* is set, reading terminates promptly, reaps child processes, and
    returns the partial count of lines written so far (callers should check
    *cancel* if partial dumps must be discarded/cleaned up).
    Returns the total number of lines written.
    """
    if not sources:
        return 0

    sorted_sources = sorted(
        sources,
        key=lambda s: (s.rank is None, s.rank if s.rank is not None else 0, s.host, s.container),
    )
    multi_host = len({s.host for s in sorted_sources}) > 1
    count = 0

    for source in sorted_sources:
        if cancel is not None and cancel.is_set():
            break
        cmd = build_read_command(executor, source, follow=False, tail=None, ssh_kwargs=ssh_kwargs)
        proc = _spawn(cmd)
        stop_watcher = threading.Event()
        if cancel is not None:

            def _watch_cancel(active_proc=proc, active_stop=stop_watcher):
                while not active_stop.is_set():
                    if cancel.wait(timeout=0.05):
                        if not active_stop.is_set():
                            _terminate(active_proc)
                        break

            threading.Thread(target=_watch_cancel, daemon=True, name=f"sparkrun-dump-cancel-{source.label}").start()

        try:
            assert proc.stdout is not None
            for text in proc.stdout:
                if cancel is not None and cancel.is_set():
                    break
                line = _line(source, text)
                if callable(prefix):
                    p = prefix(line)
                    if p:
                        fh.write(f"{p} " if not p.endswith(" ") else p)
                    fh.write(f"{line.text}\n")
                elif prefix is True:
                    tag = f"[{line.host}/{line.role}]" if multi_host else f"[{line.role}]"
                    fh.write(f"{tag} {line.text}\n")
                elif isinstance(prefix, str) and prefix:
                    fh.write(f"{prefix} " if not prefix.endswith(" ") else prefix)
                    fh.write(f"{line.text}\n")
                else:
                    fh.write(f"{line.text}\n")
                count += 1
            if cancel is not None and cancel.is_set():
                break
            if (rc := proc.wait()) != 0:
                raise subprocess.CalledProcessError(rc, cmd)
        finally:
            stop_watcher.set()
            _terminate(proc)
            if proc.stdout is not None:
                proc.stdout.close()

    return count


__all__ = [
    "build_read_command",
    "build_read_window_command",
    "dump_log_sources",
    "print_log_sources",
    "read_log_sources",
    "read_log_window",
]
