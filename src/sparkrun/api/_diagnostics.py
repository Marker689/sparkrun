"""Opt-in diagnostic bundles for Python, CLI, and daemon callers."""

from sparkrun.diagnostics.session import DiagnosticsCapture, DiagnosticsResult, capture
from contextlib import AbstractContextManager


def capture_diagnostics(
    output_path,
    *,
    sctx=None,
    max_bytes=16 * 1024 * 1024,
    log_tail=200,
    dry_run=False,
) -> AbstractContextManager[DiagnosticsCapture]:
    """Capture chained run/benchmark calls in one bounded, redacted NDJSON bundle.

    Returns a context manager yielding :class:`DiagnosticsCapture`. Nested
    captures reuse the outer bundle (including its destination and limits).
    Opening a new output file can fail before work begins; collection failures
    after opening are recorded without changing the operation's outcome.
    Existing files are never overwritten. Logging preserves application levels.
    """
    return capture(output_path, sctx=sctx, max_bytes=max_bytes, log_tail=log_tail, dry_run=dry_run)


def collect_diagnostics(
    output_path,
    *,
    hosts=None,
    cluster=None,
    workload_id=None,
    sudo_password=None,
    sctx=None,
    max_bytes=16 * 1024 * 1024,
    log_tail=200,
    dry_run=False,
) -> DiagnosticsResult:
    """Collect config, host probes and optional workload log tails independently.

    Hosts default to the named/default cluster, or saved workload placement.
    Native Kubernetes destinations skip SSH host probes explicitly. Each log
    source is read independently through the shared log API, including native
    sources. ``sudo_password`` opts into dmidecode probes; it is never stored.
    Partial sections are reported in the result and bundle, not silently lost.
    Inside a capture scope, collection adds to the existing bundle and returns
    its current status; the outer scope writes the completion marker on exit.
    """
    with capture(output_path, sctx=sctx, max_bytes=max_bytes, log_tail=log_tail, dry_run=dry_run, secrets=(sudo_password,)) as session:
        session.collect_target(hosts=hosts, cluster=cluster, workload_id=workload_id, sudo_password=sudo_password, dry_run=dry_run)
        if workload_id:
            session.collect_logs(workload_id, dry_run=dry_run)
    # An explicit standalone collection owns its result. Nested callers should
    # use the outer capture's result after it exits.
    if session.result is None:
        return session.snapshot()
    return session.result


__all__ = ["capture_diagnostics", "collect_diagnostics", "DiagnosticsCapture", "DiagnosticsResult"]
