"""Compatibility bridge for legacy operation-scoped progress/warning logs."""

from contextlib import contextmanager
import logging
import threading
import time

from sparkrun.core.operation import CURRENT

_lock = threading.RLock()
_users = 0
_worker = None


class _FeedbackHandler(logging.Handler):
    def __init__(self):
        super().__init__(25)
        self.local = threading.local()

    def createLock(self):
        # Like NullHandler, this sink has no shared I/O lock to register at fork.
        # Delivery is serialized per operation to avoid lock inversion between
        # concurrent operations and their callbacks.
        self.lock = None

    def handle(self, record):
        # Python 3.13's Handler.handle requires a context-manager lock. Perform
        # the same filtering here, including replacement records, without one.
        accepted = self.filter(record)
        if isinstance(accepted, logging.LogRecord):
            record = accepted
        if accepted:
            self.emit(record)
        return accepted

    def emit(self, record):
        if getattr(record, "sparkrun_feedback", False) or getattr(self.local, "busy", False):
            return
        state = CURRENT.get() or _worker
        if state is None or state.events is None:
            return
        self.local.busy = True
        try:
            state.feedback(
                {
                    "type": "log",
                    "source": record.name,
                    "message": record.getMessage()[:16384],
                    "level": "error" if record.levelno >= 40 else "warning" if record.levelno >= 30 else "progress",
                    "timestamp": time.time(),
                }
            )
        except Exception:
            logging.getLogger(__name__).warning("Could not forward operation log", extra={"sparkrun_feedback": True})
        finally:
            self.local.busy = False


_handler = _FeedbackHandler()


@contextmanager
def forward_operation_logs(*, worker=None):
    """Library scopes preserve logging configuration; isolated workers enable progress.

    Only a single-operation worker may attribute context-free plugin threads to
    its operation. Concurrent in-process callers require propagated context.
    """
    global _users, _worker
    root = logging.getLogger()
    with _lock:
        old_level = root.level
        old_worker = _worker
        if worker is not None:
            _worker = worker
            root.setLevel(min(root.level, 25))
        if _users == 0:
            root.addHandler(_handler)
        _users += 1
    try:
        yield
    finally:
        with _lock:
            _users -= 1
            if _users == 0:
                root.removeHandler(_handler)
            if worker is not None:
                _worker = old_worker
                root.setLevel(old_level)
