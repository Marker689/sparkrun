"""Operation-local state, propagated explicitly into nested work."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
import threading
import time
import uuid
from typing import Any, Callable


@dataclass
class OperationState:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    events: Callable[[dict], None] | None = None
    observers: list[Callable[[dict], None]] = field(default_factory=list, repr=False, compare=False, kw_only=True)
    secrets: set[str] = field(default_factory=set, repr=False, compare=False, kw_only=True)
    diagnostics_sink: Any = field(default=None, repr=False, compare=False, kw_only=True)
    cancel: threading.Event = field(default_factory=threading.Event)
    deadline: float | None = None
    hub: Any = None
    via_daemon: bool = False
    # Set at daemon admission, never from operation arguments or client labels.
    manager: str = "daemon"
    _feedback_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    _feedback_local: threading.local = field(default_factory=threading.local, repr=False, compare=False)

    def feedback(self, payload):
        """Deliver advisory feedback without letting a UI callback fail work."""
        if getattr(self._feedback_local, "busy", False):
            return
        with self._feedback_lock:
            for subscriber in ([self.events] if self.events is not None else []) + list(self.observers):
                self._feedback_local.busy = True
                try:
                    subscriber({**payload, "operation_id": self.id})
                except Exception:
                    import logging

                    logging.getLogger(__name__).warning("Operation feedback subscriber failed", extra={"sparkrun_feedback": True})
                finally:
                    self._feedback_local.busy = False

    def check(self):
        if self.cancel.is_set():
            raise InterruptedError("Operation cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Operation deadline exceeded")

    def emit(self, kind, **data):
        self.check()
        if kind in {"progress", "feedback", "warning", "log"}:
            self.feedback({"type": kind, **data})
        else:
            # Observers are advisory; the existing primary event sink retains
            # its delivery/error contract for durable operation events.
            for observer in list(self.observers):
                try:
                    observer({"type": kind, "operation_id": self.id, **data})
                except Exception:
                    pass
            if self.events is not None:
                self.events({"type": kind, "operation_id": self.id, **data})


CURRENT: ContextVar[OperationState | None] = ContextVar("sparkrun_operation", default=None)


@contextmanager
def operation_scope(sctx=None, *, state=None):
    """Reuse one state across chained API calls, never across requests."""
    state = state or getattr(sctx, "operation", None) or CURRENT.get() or OperationState()
    if sctx is not None:
        sctx.operation = state
        if getattr(sctx, "events", None) is not None:
            state.events = sctx.events
    token = CURRENT.set(state)
    try:
        state.check()
        from contextlib import nullcontext
        from sparkrun.core.feedback_logging import forward_operation_logs

        if sctx is not None and getattr(sctx, "progress", None) is None and (state.events is not None or state.observers):
            from sparkrun.core.progress import LaunchProgress

            sctx.progress = LaunchProgress(timeline=getattr(sctx, "timing", None))
        with forward_operation_logs() if state.events is not None else nullcontext():
            yield state
    finally:
        CURRENT.reset(token)


def emit(kind, **data):
    state = CURRENT.get()
    if state is not None:
        state.emit(kind, **data)


def submit(executor, function, *args, **kwargs):
    """Submit with a distinct context copy; the operation state remains shared."""
    return executor.submit(copy_context().run, function, *args, **kwargs)
