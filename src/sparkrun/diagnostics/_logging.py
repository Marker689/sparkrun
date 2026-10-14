"""Temporarily admit DEBUG records while preserving ordinary handler output."""

import logging
import threading

_lock = threading.RLock()
_captures: set[logging.Handler] = set()
_protected: set[logging.Handler] = set()
_original_level: int | None = None
_capture_level = logging.DEBUG


class _OriginalThreshold(logging.Filter):
    def filter(self, record):
        # Explicit levels on named loggers retain their normal semantics. Only
        # inheritance from the temporarily lowered root needs compensation.
        logger = logging.getLogger(record.name)
        root = logging.getLogger()
        while logger is not root:
            if logger.level:
                return record.levelno >= logger.level
            logger = logger.parent or root
        level = _original_level if root.level == _capture_level else root.level
        return level is None or record.levelno >= level


_original_threshold = _OriginalThreshold()


def attach(handler):
    """Share the DEBUG admission scope across nested/concurrent captures.

    Keep handler levels/formatters intact, including NOTSET handlers in library
    applications. Console output still follows the original logger threshold.
    Explicit per-logger suppression and logging.disable() remain authoritative.
    """
    global _original_level, _capture_level
    root = logging.getLogger()
    with _lock:
        if not _captures:
            _original_level = root.level
            _capture_level = min(root.level, logging.DEBUG)
        _captures.add(handler)
        # Include handlers attached directly to named loggers, and handlers
        # added since an overlapping capture began. Never filter another bundle.
        loggers = [root, *logging.Logger.manager.loggerDict.copy().values()]
        for logger in loggers:
            if not isinstance(logger, logging.Logger):
                continue
            for existing in logger.handlers:
                if existing not in _captures and existing not in _protected:
                    existing.filters.insert(0, _original_threshold)
                    _protected.add(existing)
        root.addHandler(handler)
        root.setLevel(_capture_level)


def detach(handler):
    global _original_level, _capture_level
    root = logging.getLogger()
    with _lock:
        root.removeHandler(handler)
        _captures.discard(handler)
        if _captures:
            return
        # Do not overwrite a deliberate application reconfiguration.
        if root.level == _capture_level and _original_level is not None:
            root.setLevel(_original_level)
        for existing in _protected:
            existing.removeFilter(_original_threshold)
        _protected.clear()
        _original_level = None
