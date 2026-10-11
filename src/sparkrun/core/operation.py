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
    cancel: threading.Event = field(default_factory=threading.Event)
    deadline: float | None = None
    hub: Any = None
    via_daemon: bool = False

    def check(self):
        if self.cancel.is_set():
            raise InterruptedError("Operation cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Operation deadline exceeded")

    def emit(self, kind, **data):
        self.check()
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
