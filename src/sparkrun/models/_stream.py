"""Bounded, cancellable model-transfer processes on borrowed host sessions."""

from __future__ import annotations

import threading
import time

from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.transports.session import HostCommandResult, StreamingHostSession


class ModelTransferStalled(ModelArtifactError):
    """A transfer's byte count stopped advancing for its idle interval."""

    def __init__(self, idle: float, transferred: int):
        super().__init__("model transfer stalled: no progress for %ds (%d bytes transferred)" % (idle, transferred))
        self.idle = idle
        self.transferred = transferred


class ProgressWatch:
    """Idle clock reset only when a transferred-byte count actually grows.

    Output alone proves nothing: rsync repeats an unchanged progress line, and
    a heartbeat would keep a dead transfer alive. One watch per attempt, since
    a retried rsync counts from zero again.

    Two windows move no bytes while healthy, so the clock does not run there:
    once *expected* bytes have moved (the receiver may spend minutes flushing
    and closing on a slow shared cache), and, with ``armed=False``, until the
    first byte (a resumed rsync first checksums the whole partial file). The
    process's own bounds (rsync ``--timeout``, an explicit ceiling) cover both.
    """

    def __init__(self, *, expected: int | None = None, armed: bool = True, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self.expected = expected
        self.armed = armed
        self.transferred = 0
        self._last = clock()

    def advance(self, count: int) -> None:
        with self._lock:
            if count > self.transferred:
                self.transferred = count
                self._last = self._clock()
                self.armed = True

    def touch(self) -> None:
        """Restart the idle clock at a phase boundary the caller recognizes."""
        with self._lock:
            self._last = self._clock()

    def idle(self) -> float:
        with self._lock:
            if not self.armed or (self.expected is not None and self.transferred >= self.expected):
                return 0.0
            return self._clock() - self._last


def execute_stream(
    session,
    host,
    script,
    *,
    timeout,
    cancelled,
    stdout=None,
    stdout_limit: int | None = 16 * 1024**2,
    idle_timeout: float | None = None,
    watch: ProgressWatch | None = None,
):
    """Drain both pipes while supervising the process, including stdin delivery.

    Only this handle is cancelled; the session may also own publication leases
    and other callers' work. Output retention is bounded independently of live
    progress. Remote open_process supplies the SSH disconnect guard.

    *timeout* ``None`` means no total ceiling. With *idle_timeout*, *watch*
    (fed by the caller's *stdout* parser) must keep advancing or the process
    is cancelled with :class:`ModelTransferStalled`.
    """
    if not isinstance(session, StreamingHostSession):
        raise ModelArtifactError("model copies require a streaming host session")
    if idle_timeout is not None and watch is None:
        raise ValueError("an idle timeout needs a progress watch")
    deadline = None if timeout is None else time.monotonic() + timeout
    if cancelled():
        raise ModelArtifactError("model copy cancelled")
    process = session.open_process(host, ["bash", "-s"])
    tails = [bytearray(), bytearray()]
    errors = []

    def drain(stream, tail, callback):
        try:
            while data := stream.read(16384):
                tail.extend(data)
                limit = 65536 if callback is not None or stream is process.stderr else stdout_limit
                if limit is not None:
                    del tail[:-limit]
                if callback is not None:
                    callback(data)
        except Exception as exc:
            errors.append(exc)

    def write():
        try:
            data = memoryview(script.encode())
            while data:
                count = process.stdin.write(data)
                if not count:
                    raise OSError("host process stopped accepting its script")
                data = data[count:]
        except (BrokenPipeError, OSError):
            # A failed command can exit before reading all stdin. Its status
            # and stderr are authoritative; pipe closure is not a second error.
            pass
        finally:
            process.stdin.close()

    threads = [
        threading.Thread(target=drain, args=(process.stdout, tails[0], stdout), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, tails[1], None), daemon=True),
        threading.Thread(target=write, daemon=True),
    ]
    try:
        for thread in threads:
            thread.start()
        while process.poll() is None or any(thread.is_alive() for thread in threads):
            if cancelled():
                raise ModelArtifactError("model copy cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                raise ModelArtifactError("model transfer deadline exceeded")
            if idle_timeout is not None and watch is not None and (idle := watch.idle()) >= idle_timeout:
                raise ModelTransferStalled(idle, watch.transferred)
            if errors:
                raise ModelArtifactError("could not read model transfer output") from errors[0]
            time.sleep(0.05)
        if errors:
            raise ModelArtifactError("could not read model transfer output") from errors[0]
        return HostCommandResult(host, process.wait(), bytes(tails[0]), bytes(tails[1]))
    finally:
        process.cancel()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=5)
        process.close()
