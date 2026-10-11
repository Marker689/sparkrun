"""Public, frontend-independent plugin feedback API (version 1).

Create a reporter inside an operation, then pass it to background work. Events
are advisory: they never replace the operation's result or determine success.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from contextlib import contextmanager
import threading
import logging
import math
import time
from typing import Literal, Mapping, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from sparkrun.core.operation import OperationState

FEEDBACK_API_VERSION = 1
FeedbackLevel = Literal["debug", "info", "progress", "warning", "error"]
FeedbackStatus = Literal["running", "succeeded", "failed", "skipped"]
MetricValue = str | int | float | bool | None
_LEVELS = {"debug": 10, "info": 20, "progress": 25, "warning": 30, "error": 40}
LOG_EXTRA = {"sparkrun_feedback": True}


@dataclass(frozen=True)
class FeedbackEvent:
    source: str
    message: str
    activity_id: str
    level: FeedbackLevel = "progress"
    parent_id: str | None = None
    stage: str | None = None
    status: FeedbackStatus | None = None
    host: str | None = None
    current: float | None = None
    total: float | None = None
    total_kind: Literal["exact", "upper_bound", "unknown"] = "unknown"
    unit: str | None = None
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a bounded JSON payload, without accepting arbitrary objects."""
        if self.level not in _LEVELS or self.status not in {None, "running", "succeeded", "failed", "skipped"}:
            raise ValueError("Invalid feedback level or status")
        if self.total_kind not in {"exact", "upper_bound", "unknown"}:
            raise ValueError("Invalid feedback total kind")
        for name in ("source", "message", "activity_id", "parent_id", "stage", "host", "unit"):
            value = getattr(self, name)
            limit = 16384 if name == "message" else 512
            if value is not None and (not isinstance(value, str) or not value or len(value) > limit):
                raise ValueError("Invalid feedback " + name)
            if name in {"source", "message", "activity_id"} and value is None:
                raise ValueError("Missing feedback " + name)
        for value in (self.current, self.total):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("Feedback counters must be finite and nonnegative")
        if (self.total is None) != (self.total_kind == "unknown"):
            raise ValueError("A feedback total requires exact or upper_bound semantics")
        metrics = dict(self.metrics)
        if len(metrics) > 32:
            raise ValueError("Too many feedback metrics")
        for key, value in metrics.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise ValueError("Invalid feedback metric name")
            if value is not None and type(value) not in (str, int, float, bool):
                raise ValueError("Feedback metrics must be JSON scalars")
            if isinstance(value, str) and len(value) > 512:
                raise ValueError("Feedback metric string exceeds limit")
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("Feedback metrics must be finite")
        result = {
            name: getattr(self, name)
            for name in (
                "source",
                "message",
                "activity_id",
                "level",
                "parent_id",
                "stage",
                "status",
                "host",
                "current",
                "total",
                "total_kind",
                "unit",
            )
            if getattr(self, name) is not None
        }
        result.update(type="feedback", feedback_version=FEEDBACK_API_VERSION, timestamp=time.time(), metrics=metrics)
        return result


@dataclass(frozen=True)
class FeedbackReporter:
    """An operation-bound reporter. Use :func:`get_feedback` to construct one.

    Calls may come from any thread. Subscribers run synchronously and should
    enqueue work quickly; subscriber exceptions do not fail the operation.
    Close/join producer threads before the operation ends.
    """

    source: str
    activity_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    host: str | None = None
    _state: OperationState | None = field(default=None, repr=False, compare=False)
    _logger: logging.Logger | None = field(default=None, repr=False, compare=False)

    def child(self, *, source=None, activity_id=None, host=None, logger=None) -> FeedbackReporter:
        return replace(
            self,
            source=source or self.source,
            activity_id=activity_id or uuid.uuid4().hex,
            parent_id=self.activity_id,
            host=host or self.host,
            _logger=logger or self._logger,
        )

    def report(self, message: str, *, level: FeedbackLevel = "progress", **fields) -> None:
        event = FeedbackEvent(
            source=self.source,
            message=message,
            activity_id=self.activity_id,
            parent_id=self.parent_id,
            host=fields.pop("host", self.host),
            level=level,
            **fields,
        )
        payload = event.to_dict()
        if self._state is not None:
            self._state.feedback(payload)
        (self._logger or logging.getLogger(self.source)).log(_LEVELS[level], "%s", message, extra=LOG_EXTRA)

    @contextmanager
    def heartbeat(self, label, *, interval: float = 30):
        """Report silent work from a bound background thread until scope exit."""
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Heartbeat interval must be positive and finite")
        stopped = threading.Event()
        started = time.monotonic()

        def report():
            while not stopped.wait(interval):
                message = label() if callable(label) else label
                elapsed = time.monotonic() - started
                self.progress(
                    "%s — still running (%.0fs)" % (message, elapsed),
                    stage="heartbeat",
                    status="running",
                    metrics={"elapsed_seconds": elapsed},
                )

        thread = threading.Thread(target=report, name="sparkrun-feedback-heartbeat", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join()

    def progress(self, message: str, **fields) -> None:
        self.report(message, **fields)

    def warning(self, message: str, **fields) -> None:
        self.report(message, level="warning", **fields)

    def error(self, message: str, **fields) -> None:
        self.report(message, level="error", **fields)


def get_feedback(source: str, *, sctx=None, logger: logging.Logger | None = None) -> FeedbackReporter:
    """Capture this operation once, including for custom plugin pipelines.

    With no operation/subscriber, messages still use ordinary Python logging.
    ``sctx`` supports library/plugin entry points outside a core API boundary.
    """
    from sparkrun.core.operation import CURRENT, OperationState

    state = getattr(sctx, "operation", None) or CURRENT.get()
    if sctx is not None:
        if state is None:
            state = OperationState()
        sctx.operation = state
        if getattr(sctx, "events", None) is not None:
            state.events = sctx.events
    return FeedbackReporter(source, _state=state, _logger=logger)


__all__ = ["FEEDBACK_API_VERSION", "FeedbackEvent", "FeedbackReporter", "get_feedback"]
