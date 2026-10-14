"""Process/runtime environment helpers (SSH user, env merge, logging config)."""

from __future__ import annotations

import logging

# Loggers that produce excessive output at DEBUG/INFO level.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore.http11",
    "httpcore.connection",
    "httpcore2.http11",
    "httpcore2.connection",
    "urllib3.connectionpool",
)


def suppress_noisy_loggers() -> None:
    """Suppress verbose HTTP/transport loggers."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def local_username() -> str:
    """Return the local login name, cross-platform, or ``""`` when unknown.

    ``os.environ["USER"]`` is POSIX-only (Windows sets ``USERNAME``), so a bare
    ``USER`` lookup with a ``"root"`` default made every Windows control
    machine connect as ``root``.  ``getpass.getuser`` consults ``LOGNAME`` /
    ``USER`` / ``LNAME`` / ``USERNAME`` and then the password database.
    """
    import getpass

    try:
        return getpass.getuser() or ""
    except Exception:  # no env var and no pwd entry (e.g. a container uid)
        return ""


def resolve_ssh_user(cluster_user: str | None, config, fallback: str = "root") -> str:
    """Resolve SSH user from cluster definition, config, or the local login.

    *fallback* applies only when the local login cannot be determined; pass
    ``""`` to let the caller refuse rather than guess.
    """
    return cluster_user or getattr(config, "ssh_user", None) or local_username() or fallback


def merge_env(*env_dicts: dict[str, str] | None) -> dict[str, str]:
    """Merge multiple environment dicts (later values win)."""
    merged: dict[str, str] = {}
    for d in env_dicts:
        if d:
            merged.update(d)
    return merged
