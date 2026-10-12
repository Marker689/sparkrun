"""Bounded, cancellable model-transfer processes on borrowed host sessions."""

from __future__ import annotations

import threading
import time

from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.transports.session import HostCommandResult, StreamingHostSession


def execute_stream(session, host, script, *, timeout, cancelled, stdout=None):
    """Drain both pipes while supervising the process, including stdin delivery.

    Only this handle is cancelled; the session may also own publication leases
    and other callers' work. Output retention is bounded independently of live
    progress. Remote open_process supplies the SSH disconnect guard.
    """
    if not isinstance(session, StreamingHostSession):
        raise ModelArtifactError("model copies require a streaming host session")
    deadline = time.monotonic() + timeout
    if cancelled():
        raise ModelArtifactError("model copy cancelled")
    process = session.open_process(host, ["bash", "-s"])
    tails = [bytearray(), bytearray()]
    errors = []

    def drain(stream, tail, callback):
        try:
            while data := stream.read(16384):
                tail.extend(data)
                limit = 65536 if callback is not None or stream is process.stderr else 16 * 1024**2
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
            if time.monotonic() >= deadline:
                raise ModelArtifactError("model transfer deadline exceeded")
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
