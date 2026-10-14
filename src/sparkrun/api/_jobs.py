"""``sparkrun.api.list_jobs`` — enumerate persisted job metadata.

Walks ``~/.cache/sparkrun/jobs/*.yaml`` (or a caller-supplied
``cache_dir``) and surfaces each as a :class:`~sparkrun.api.JobInfo`.
Stale entries whose YAML fails to parse are logged at debug and
skipped — the function never raises on a single bad file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sparkrun.api._models import JobInfo

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)


def list_jobs(
    *,
    cache_dir: str | Path | None = None,
    limit: int | None = None,
    sctx: "SparkrunContext | None" = None,
) -> list[JobInfo]:
    """Return a list of :class:`JobInfo` for every persisted job metadata file.

    Args:
        cache_dir: Override for the sparkrun cache root.  Takes
            precedence when set.  Otherwise falls back to
            the supplied or current application's configured cache root.
            Reading configuration does not initialize plugins.
        limit: Return at most this many of the most recent jobs, and —
            crucially — only parse that many files.  See below.
        sctx: Optional shared :class:`SparkrunContext`.

    Returns:
        :class:`JobInfo` entries sorted by ``started_at`` descending
        (most recent first); entries without a timestamp come last,
        ordered by ``cluster_id``.

    **On ``limit`` and why it exists.** Each metadata file embeds the full
    serialized recipe state (command template, env, metadata, hooks), so the
    cost here is dominated by YAML parsing, not by the directory walk: an
    unpruned cache of ~740 jobs is ~1.7 MB and takes ~1.5 s to load. That is
    fine for a report and far too slow for shell completion, which runs on
    every TAB.

    So ``limit`` pre-ranks by file **mtime** — a stat per file, no parsing —
    and parses only the top N, then re-sorts that subset by the resolved
    ``started_at``. mtime is a proxy: a job whose file was rewritten later
    than it was launched could in principle rank above a newer job and
    displace it from the window. That is the deliberate trade, and it is the
    same proxy :func:`_resolve_started_at` already falls back to, so a cache
    written before ``started_at`` existed ranks identically either way.
    Callers that need exactness omit ``limit``.
    """
    from sparkrun.core.config import resolve_configured_cache_dir

    cache_dir = resolve_configured_cache_dir(cache_dir, config=sctx.config if sctx is not None else None)

    jobs_dir = Path(cache_dir) / "jobs"
    if not jobs_dir.is_dir():
        return []

    paths = list(jobs_dir.glob("*.yaml"))
    if limit is not None:
        if limit <= 0:
            return []
        paths = _newest_by_mtime(paths, limit)

    entries: list[JobInfo] = []
    for meta_path in paths:
        info = _job_info_from_file(meta_path)
        if info is not None:
            entries.append(info)

    # Stable order: most-recent first, untimed jobs last.
    def _sort_key(j: JobInfo):
        return (j.started_at is None, -(j.started_at or 0.0), j.workload_id)

    entries.sort(key=_sort_key)
    return entries


def _newest_by_mtime(paths: list[Path], limit: int) -> list[Path]:
    """The *limit* most recently modified paths, by stat alone.

    Files that vanish between the glob and the stat (a concurrent prune, a
    ``stop`` racing us) are dropped rather than raising — this is a
    best-effort ranking, and completion must never fail on a race.
    """
    stamped: list[tuple[float, Path]] = []
    for path in paths:
        try:
            stamped.append((path.stat().st_mtime, path))
        except OSError:
            continue
    stamped.sort(key=lambda item: -item[0])
    return [path for _, path in stamped[:limit]]


def _resolve_started_at(raw, meta_path: Path) -> float | None:
    """Resolve a job's launch time, falling back to the file's mtime.

    ``started_at`` was added to the metadata after the fact, so every job
    written by an earlier sparkrun lacks it — on a long-lived cache that is
    most of them.  Without a fallback those all sort as "no timestamp", which
    is exactly the ordering bug the field was added to fix: the sort key sends
    untimed entries to the back and orders them alphabetically by cluster_id.

    The file's mtime is a good proxy — it is when the launch wrote the
    metadata — and it is only ever consulted when the recorded value is absent
    or unparseable, so a real ``started_at`` always wins over a mtime that a
    later rewrite (backup restore, cache rsync) may have moved.
    """
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            logger.debug("list_jobs: unparseable started_at %r in %s", raw, meta_path)
    try:
        return meta_path.stat().st_mtime
    except OSError:
        return None


def _job_info_from_file(meta_path: Path) -> JobInfo | None:
    """Load one job metadata YAML and return a :class:`JobInfo`, or ``None`` on failure."""
    try:
        from sparkrun.utils import load_yaml

        data = load_yaml(meta_path) or {}
    except Exception:
        logger.debug("list_jobs: failed to load %s", meta_path, exc_info=True)
        return None
    if not isinstance(data, dict):
        return None

    from sparkrun.core.ownership import owns_metadata

    cluster_id = data.get("cluster_id") or _cluster_id_from_filename(meta_path)
    # Legacy metadata omitted cluster_id; its filename in the active metadata
    # directory supplies the historical Sparkrun identity. An explicit owner wins.
    if not owns_metadata({**data, "cluster_id": cluster_id}):
        return None
    if not cluster_id:
        return None

    hosts_raw = data.get("hosts") or ()
    hosts = tuple(str(h) for h in hosts_raw) if isinstance(hosts_raw, (list, tuple)) else ()

    started_at = _resolve_started_at(data.get("started_at"), meta_path)

    # Decompose the cluster_id when the metadata didn't already record
    # intent_id / placement_token (e.g. a job metadata file written by
    # an older tool that doesn't persist these keys).  Non-canonical
    # cluster_ids surface as ``None`` on both fields — a data-quality
    # signal that callers can detect via :class:`JobInfo`.
    intent_id = data.get("intent_id")
    placement_token = data.get("placement_token")
    if intent_id is None or placement_token is None:
        try:
            from sparkrun.orchestration.job_metadata import parse_cluster_id

            parsed_intent, parsed_token = parse_cluster_id(str(cluster_id))
            if intent_id is None:
                intent_id = parsed_intent
            if placement_token is None:
                placement_token = parsed_token
        except ValueError:
            pass

    return JobInfo(
        workload_id=str(cluster_id),
        intent_id=intent_id,
        placement_token=placement_token,
        recipe=data.get("recipe"),
        runtime=data.get("runtime"),
        hosts=hosts,
        started_at=started_at,
        metadata=dict(data),
    )


def _cluster_id_from_filename(meta_path: Path) -> str:
    """Recover ``sparkrun_<digest>`` from a metadata filename (back-compat)."""
    stem = meta_path.stem
    return ("sparkrun_%s" % stem) if not stem.startswith("sparkrun_") else stem


__all__ = ["list_jobs"]
