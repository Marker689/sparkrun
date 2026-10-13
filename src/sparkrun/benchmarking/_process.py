"""Private benchmark process lifetime and redacted output handling (portable controllers)."""

from __future__ import annotations

import codecs
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence

from sparkrun.benchmarking._credentials import BenchmarkCredentials

from sparkrun.core.operation import CURRENT
from sparkrun.core.subprocesses import owned_process


class _Output:
    """Decode and redact across reads before emitting complete text lines."""

    def __init__(self, sink: Callable[[str], None], credentials: BenchmarkCredentials):
        self.sink = sink
        self.key = credentials.api_key
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.pending = ""
        self.line = ""

    def feed(self, data: bytes, *, final: bool = False) -> None:
        text = self.pending + self.decoder.decode(data, final=final)
        self.pending = ""
        if self.key:
            parts = text.split(self.key)
            ready = "[REDACTED]".join(parts[:-1]) + ("[REDACTED]" if len(parts) > 1 else "")
            text = parts[-1]
            if not final:
                for size in range(min(len(self.key) - 1, len(text)), 0, -1):
                    if text.endswith(self.key[:size]):
                        text, self.pending = text[:-size], text[-size:]
                        break
            text = ready + text
        # Only CR/LF frame transport lines. Unicode separators are payload.
        # Hold a trailing CR until the next read so split CRLF stays together.
        text = self.line + text
        start = 0
        for match in re.finditer(r"\r\n|\n|\r(?!$)", text):
            self.sink(text[start : match.end()])
            start = match.end()
        self.line = text[start:]
        if final and self.line:
            self.sink(self.line)
            self.line = ""


def run_benchmark_process(
    command: Sequence[str],
    *,
    timeout: float | None,
    credentials: BenchmarkCredentials,
    stdout: Callable[[str], None],
    stderr: Callable[[str], None] | None = None,
) -> int:
    """Drain both output pipes on the caller's thread under one deadline.

    Python 3.12 supports nonblocking pipes on Windows as well as POSIX. Polling
    two pipes avoids Windows' socket-only selector and reader-thread leaks.
    Callbacks remain synchronous; errors/cancellation unwind owned containment.
    """
    deadline = time.monotonic() + timeout if timeout is not None else None
    operation = CURRENT.get()
    # The transport decoder is UTF-8; Python tools must not inherit Windows'
    # pipe code page (or a controller's explicit legacy IO encoding).
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with owned_process(command, env=env, separate_stderr=stderr is not None) as proc:
        streams = {}
        for stream, sink in ((proc.stdout, stdout), (proc.stderr, stderr)):
            if stream is not None and sink is not None:
                os.set_blocking(stream.fileno(), False)
                streams[stream.fileno()] = _Output(sink, credentials)
        while proc.poll() is None or streams:
            if operation is not None:
                operation.check()
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0 and timeout is not None:
                # Never attach credential-bearing command arguments.
                raise subprocess.TimeoutExpired("benchmark", timeout)
            received = False
            for fd, output in list(streams.items()):
                try:
                    data = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                received = True
                output.feed(data, final=not data)
                if not data:
                    del streams[fd]
            if not received:
                time.sleep(min(remaining, 0.01) if remaining is not None else 0.01)
        return proc.returncode
