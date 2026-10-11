"""Signal cleanup shared by CLI and isolated workers."""


def install_termination_handlers() -> list[int]:
    """Make ``SIGTERM`` / ``SIGHUP`` unwind like Ctrl-C.

    Two things follow from raising :class:`KeyboardInterrupt` instead of dying
    where the default handler would:

    - **Remote work is torn down.** ``subprocess.run`` kills its child from the
      bare ``except`` in its cleanup path, so the ``ssh`` client dies, sshd
      exits, and the remote session guard
      (:func:`~sparkrun.orchestration.ssh.wrap_with_session_guard`) fires.
      Without this, ``kill <sparkrun-pid>`` leaves the ``ssh`` client running
      as a local orphan, the session stays healthy, and the guard correctly
      never fires — the remote download keeps going.
    - **Existing cleanup runs.** ``KeyboardInterrupt`` is already the signal
      every state-preserving path in the codebase handles (``api._run``,
      ``api._benchmark``, the benchmark scheduler), so ``SIGTERM`` now gets the
      same treatment Ctrl-C has always had.

    ``SIGKILL`` is unreachable by definition; a launch killed with ``-9`` can
    still orphan the *streaming* distribution path, whose ``ssh`` stdout is an
    inherited terminal rather than a pipe back to us.

    Best-effort: signal handlers can only be installed from the main thread, so
    a non-main-thread caller (embedded use, a test runner) is a silent no-op.
    An already-customised handler is left alone rather than clobbered.

    Returns:
        The signal numbers for which a handler was installed.
    """
    import signal

    installed: list[int] = []
    candidates = [getattr(signal, name, None) for name in ("SIGTERM", "SIGHUP")]
    for sig in candidates:
        if sig is None:  # e.g. SIGHUP on Windows
            continue
        try:
            if signal.getsignal(sig) is not signal.SIG_DFL:
                continue  # someone else owns it — don't clobber
            signal.signal(sig, _terminate_as_interrupt)
        except (ValueError, OSError, RuntimeError):
            # Not the main thread, or the platform refuses this signal.
            continue
        installed.append(int(sig))
    return installed


def _terminate_as_interrupt(signum, _frame):
    """Unwind existing cleanup paths on SIGTERM/SIGHUP."""
    raise KeyboardInterrupt
