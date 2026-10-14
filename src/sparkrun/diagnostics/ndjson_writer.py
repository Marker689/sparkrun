"""Append-only, immediate-flush NDJSON writer.

Each call to :meth:`NDJSONWriter.emit` writes one JSON line with envelope
fields (``_type``, ``_seq``, ``_ts``) and flushes immediately so records
survive crashes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class NDJSONWriter:
    """Append-only NDJSON writer with immediate flush.

    Every emitted record is wrapped in an envelope with:
    - ``_type``: record type string
    - ``_seq``: monotonically increasing sequence number
    - ``_ts``: ISO-8601 UTC timestamp

    Usage::

        with NDJSONWriter("output.ndjson") as w:
            w.emit("host_hardware", {"host": "10.0.0.1", "cpu_cores": 16})
    """

    def __init__(self, path: str | Path, *, exclusive=False, redact=None, max_bytes=None):
        self._path = Path(path)
        self._seq = 0
        self._fh = None
        self._exclusive = exclusive
        self._redact = redact
        self._max_bytes = max_bytes
        self._bytes = 0
        self._lock = threading.RLock()
        self.truncated = False
        self.error = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def seq(self) -> int:
        return self._seq

    def open(self) -> NDJSONWriter:
        flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if self._exclusive else os.O_APPEND)
        fd = os.open(self._path, flags, 0o600)
        self._fh = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        return self

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def emit(self, record_type: str, data: dict | None = None) -> dict:
        """Write one NDJSON record and flush immediately.

        Args:
            record_type: Value for the ``_type`` envelope field.
            data: Payload fields merged into the record.

        Returns:
            The full record dict that was written.
        """
        with self._lock:
            return self._emit(record_type, data)[0]

    @property
    def available_bytes(self) -> int:
        """Ordinary capacity, excluding the completion reserve; no formatting work."""
        with self._lock:
            if self.error or self._fh is None:
                return 0
            return max(0, (self._max_bytes or 64 * 1024 * 1024) - 8192 - self._bytes)

    def emit_bounded(self, record_type: str, data: dict, *, max_bytes: int) -> tuple[int, str | None]:
        """Emit prevalidated data once; return bytes flushed and a bounded rejection reason."""
        with self._lock:
            if self.error or self._fh is None:
                return 0, "storage"
            _, size, reason = self._emit(record_type, data, record_limit=max_bytes)
            return size, reason

    def _emit(self, record_type, data, *, record_limit=None):
        self._seq += 1
        record = {
            **(data or {}),
            "_type": record_type,
            "_seq": self._seq,
            "_ts": datetime.now(timezone.utc).isoformat(),
        }
        if self._redact is not None:
            record = self._redact(record)
        line = json.dumps(record, default=str)
        size = len((line + "\n").encode("utf-8"))
        if record_limit is not None and size > record_limit:
            self.truncated = True
            return record, 0, "record_bytes"
        if self._max_bytes is not None:
            # Reserve space for the completion marker even after capture fills up.
            limit = self._max_bytes if record_type == "bundle_complete" else self._max_bytes - 8192
            if size > 65536 or self._bytes + size > limit:
                self.truncated = True
                return record, 0, "bundle_bytes"
        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
                self._bytes += size
                return record, size, None
            except OSError as exc:
                if self._max_bytes is None:
                    raise
                self.error = type(exc).__name__
        else:
            logger.warning("NDJSONWriter not open; record %d dropped", self._seq)

        return record, 0, "storage"

    def __enter__(self) -> NDJSONWriter:
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
