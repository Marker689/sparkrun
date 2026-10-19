"""A supervised host process that owns both the cache lock and publication."""

from __future__ import annotations

import base64
import logging
import os
import socket
import threading
import time

from sparkrun.core.progress import PROGRESS
from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)
_WAIT_INTERVAL = 10


class PublicationLease:
    def __init__(self, session, host, root, path, nonce, *, check=None):
        self.root = root
        self.host = host
        self.error = None
        self._acquired = False
        self._holder = "unknown"
        self._mutex = threading.Lock()
        # FD 9 belongs to the holder shell and is inherited by each publication
        # child. Killing either process cannot release the flock while the other
        # still writes. No Python/uv installation is needed on offline hosts.
        from pathlib import PurePosixPath

        identity = f"controller {socket.gethostname()} pid {os.getpid()} operation {nonce.split(':', 1)[0]}"
        program = (
            "set -euo pipefail\n"
            + f"mkdir -p -- {quote(str(PurePosixPath(path).parent))}\n"
            + f"exec 9<> {quote(path)}\n"
            # Exit 75 means contention; other flock errors must not loop forever.
            + "while true; do\n"
            + f"  if flock -E 75 -w {_WAIT_INTERVAL} 9; then break; else status=$?; fi\n"
            + '  [ "$status" = 75 ] || exit "$status"\n'
            + f"  printf 'WAITING '; head -c 4096 -- {quote(path)} | base64 -w 0; printf '\\n'\n"
            + "done\n"
            + f": > {quote(path)}\nprintf '%s\\n%s\\n' {quote(nonce)} {quote(identity)} >&9\nprintf 'LOCKED\\n'\n"
            + "while IFS= read -r request; do\n"
            + "  if printf %s \"$request\" | base64 -d | bash; then printf 'OK\\n'; else printf 'ERROR\\n'; fi\n"
            + "done\n"
        )
        self.process = session.open_process(host or "localhost", ["bash", "-c", program])
        try:
            while True:
                # This bounds an unresponsive channel, not time spent waiting
                # for another preparation. Every WAITING renews supervision.
                response = self._exchange(None, timeout=_WAIT_INTERVAL + 30, check=check)
                if response == b"LOCKED\n":
                    self._acquired = True
                    break
                if not response.startswith(b"WAITING "):
                    raise ModelArtifactError(self._failure_message())
                owner = base64.b64decode(response.removeprefix(b"WAITING "), validate=False).decode(errors="replace").splitlines()
                self._holder = " ".join(owner[1:]) if len(owner) > 1 else ("operation " + owner[0] if owner else "unknown")
                logger.log(
                    PROGRESS, "Waiting for model cache publication lease on %s (%s); held by %s", host or "control", root, self._holder
                )
        except BaseException:
            self.process.close()
            raise

    def _failure_message(self):
        if self._acquired:
            return "model cache publication lease lost on " + (self.host or "control")
        return "could not acquire model cache publication lease on %s (%s); last reported holder: %s" % (
            self.host or "control",
            self.root,
            self._holder,
        )

    def check(self):
        if self.error is None and self.process.poll() is not None:
            self.error = self._failure_message()
        if self.error is not None:
            raise ModelArtifactError(self.error)

    def _exchange(self, payload, *, timeout=30, check=None):
        # Pipes are not selectable on Windows controllers. A bounded helper
        # handles duplex IO; the caller supervises all leases while it waits.
        done = threading.Event()
        result, errors = [], []

        def communicate():
            try:
                if payload is not None:
                    remaining = memoryview(payload)
                    while remaining:
                        count = self.process.stdin.write(remaining)
                        if not count:
                            raise OSError("publication pipe closed")
                        remaining = remaining[count:]
                    self.process.stdin.flush()
                result.append(self.process.stdout.readline(16384))
            except Exception as error:
                errors.append(error)
            finally:
                done.set()

        worker = threading.Thread(target=communicate, name="model-publication", daemon=True)
        deadline = time.monotonic() + timeout
        worker.start()
        try:
            while not done.wait(0.05):
                self.check()
                if check is not None:
                    check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("model publication response timed out")
            self.check()
            if check is not None:
                check()
            if errors or not result or not result[0]:
                raise OSError("model publication pipe closed")
            return result[0]
        except BaseException as error:
            self.error = self._failure_message()
            self.process.cancel()
            if not isinstance(error, Exception) or isinstance(error, ModelArtifactError):
                raise
            raise ModelArtifactError(self.error) from error
        finally:
            worker.join(timeout=5)

    def publish(self, path, contents, *, check):
        with self._mutex:
            check()
            script = (
                "set -eu\n"
                + f"p={quote(path)}\n"
                + 'mkdir -p -- "$(dirname -- "$p")"\n'
                + 'tmp=$(mktemp "$(dirname -- "$p")/.manifest-XXXXXXXX")\n'
                + "trap 'rm -f -- \"$tmp\"' EXIT\n"
                + f'printf %s {quote(contents)} > "$tmp"\n'
                + 'mv -f -- "$tmp" "$p"\n'
            )
            payload = base64.b64encode(script.encode()) + b"\n"
            if self._exchange(payload, check=check) != b"OK\n":
                raise ModelArtifactError("model publication failed on " + (self.host or "control"))
