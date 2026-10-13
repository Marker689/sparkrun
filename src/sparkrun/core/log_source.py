"""Log source / log line dataclasses — the shape of "where does this workload's output live?"

This module is **data only**, mirroring :mod:`sparkrun.core.cluster_status`.
Production of :class:`LogSource` lives on the runtime
(:meth:`~sparkrun.runtimes.base.RuntimePlugin.log_sources`), because *what*
to read is a property of how the runtime launched the workload; turning a
source into a concrete command lives on the executor
(:meth:`~sparkrun.orchestration.executors._base.Executor.read_logs_cmd`),
because *how* to read it is a property of the substrate.  Reading and
merging live in :mod:`sparkrun.orchestration.logs`; composition lives in
:func:`sparkrun.api.logs`.

The split exists because the two halves genuinely differ per runtime *and*
per executor.  Most sparkrun runtimes use the sleep-infinity + exec pattern:
container PID 1 is ``sleep infinity`` and the serve process writes to a file
*inside* the container, so ``docker logs`` is structurally blind to the
output (see ``scripts/exec_serve_detached.sh``).  A few — TRT-LLM's cluster
mode, Ray *worker* containers — really do put their output on the
container's stdout.  Meanwhile the ``local`` executor has no container at
all and redirects to a host file, and k8s reads through ``kubectl logs``.
Encoding that as data rather than as a print action is what lets the CLI be
a renderer instead of a caller of a printer.

These dataclasses live in ``core`` (not ``api``) so that ``orchestration``
can produce them without importing ``api`` — the layering rule is
``cli → api → {core, orchestration}``.
"""

from __future__ import annotations

from dataclasses import dataclass

SERVE_LOG_PATH = "/tmp/sparkrun_serve.log"
"""In-container path the sleep-infinity + exec launch redirects serve output to.

Written by ``scripts/exec_serve_detached.sh``; the default ``path`` for any
:class:`LogSource` with ``mode == "file"``.
"""

MODE_FILE = "file"
"""Output lives in a file *inside* the container (read via an exec + tail)."""

MODE_STDOUT = "stdout"
"""Output is the container/pod's own stdout (read via the substrate's log stream)."""

SCOPE_HEAD = "head"
"""Only the head/solo source — the workload's primary log."""

SCOPE_ALL = "all"
"""Every source the runtime can name: head plus each worker/rank."""


@dataclass(frozen=True)
class LogSource:
    """One readable stream of a workload's output, on one host.

    Args:
        host: Host the container/process lives on.
        container: Container (or pod / process) name on that host.
        role: ``solo`` / ``head`` / ``worker`` / ``node_<rank>`` — the same
            vocabulary :class:`~sparkrun.core.cluster_status.ContainerDetail`
            uses, so log output and status output name things identically.
        rank: Distributed rank when known; ``None`` for sources with no rank
            (e.g. a Ray worker container, which hosts ranks but isn't one).
            Also the ordering key for rank-grouped (non-follow) reads.
        mode: :data:`MODE_FILE` or :data:`MODE_STDOUT`.
        path: In-container path when ``mode == MODE_FILE``.  Ignored by
            executors whose substrate has no in-container filesystem
            indirection (``local``, ``k8s``).
    """

    host: str
    container: str
    role: str = "solo"
    rank: int | None = None
    mode: str = MODE_FILE
    path: str | None = SERVE_LOG_PATH

    @property
    def label(self) -> str:
        """Short ``host/role`` label for prefixing interleaved output."""
        return "%s/%s" % (self.host, self.role)


@dataclass(frozen=True)
class LogLine:
    """A single line yielded by :func:`sparkrun.api.logs`.

    Re-exported as ``sparkrun.api.LogLine``; that is the stable public path.
    """

    host: str
    container: str
    text: str
    stream: str = "stdout"
    """``"stdout"`` or ``"stderr"`` — best-effort, may be ``"stdout"`` if
    the executor doesn't preserve stream identity."""
    timestamp: float | None = None
    """Epoch seconds for the line, when available.

    Populated with the *arrival* time when following (which is what makes
    live interleaving time-ordered).  ``None`` for non-follow reads of
    :data:`MODE_FILE` sources: the serve log carries no capture timestamps,
    so there is nothing to sort on and those reads are rank-grouped instead.
    """
    role: str = "solo"
    """Role of the source that produced this line (see :attr:`LogSource.role`)."""
    rank: int | None = None
    """Rank of the source that produced this line, when known."""


def source_sort_key(source: LogSource) -> tuple[float, str, str]:
    """Deterministic read/display order: primary source, ranks, unranked workers."""
    rank = source.rank if source.rank is not None else (-1 if source.role in {"head", "solo"} else float("inf"))
    return rank, source.host, source.container


__all__ = [
    "LogLine",
    "LogSource",
    "MODE_FILE",
    "MODE_STDOUT",
    "SCOPE_ALL",
    "SCOPE_HEAD",
    "SERVE_LOG_PATH",
    "source_sort_key",
]
