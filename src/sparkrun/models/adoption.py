"""Adopt existing Hugging Face cache snapshots so sparkrun tracks them.

A revision downloaded outside sparkrun (``hf download``, a notebook, an earlier
sparkrun without manifest preparation) has no usage record, so the cache
operations never consider it: ``list`` hides it, ``prune`` never frees it. Adopting
writes that record and nothing else. No manifest is saved: a snapshot is often
a deliberate *subset* of its revision (one GGUF quant), and a saved manifest is
what a pinned launch reads as the complete inventory.

What is recorded must be the snapshot's real content, so each snapshot is judged
before it is adopted:

* **online** (default): every file present must exist in the model origin's
  inventory for that commit, at the same size. Files the snapshot lacks are
  fine; a file the origin does not know, or knows at another size, refuses.
* **offline**: no origin to ask, so every entry must be a link into the blob
  store, the layout the Hub client writes (and names by content). A plain
  file proves nothing offline and is refused.

``last_used`` starts at *now* by default, so nothing becomes prune-eligible the
moment it is adopted; ``mtime`` uses the newest entry's modification time.
"""

from __future__ import annotations

import logging
import time

from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactRequest, ModelInventoryUnavailable
from sparkrun.models.cache import host_cache, resolve_cache_locations
from sparkrun.models.cache_reports import CacheReport
from sparkrun.models.host_io import ModelHostIO

logger = logging.getLogger(__name__)

LAST_USED_CHOICES = ("now", "mtime")


def _reason(code: str, detail: str = "") -> dict:
    from sparkrun.models._cache_contract import CACHE_REASONS

    text = CACHE_REASONS[code] + (": " + detail if detail else "")
    return {"reason": text, "reason_code": code}


def judge(snapshot: dict, *, offline: bool, origin) -> dict:
    """``{"reason": …, "reason_code": …}`` for one scanned snapshot; ``adoptable`` or why not.

    *origin(repo, revision)* returns ``{path: size}`` from the model origin, or
    raises :class:`ModelInventoryUnavailable`.
    """
    entries = snapshot["entries"]
    if snapshot.get("problem"):
        return _reason("adopt_unverifiable", snapshot["problem"])
    if not entries:
        return _reason("not_adoptable", "empty snapshot")
    for entry in entries:
        if entry["kind"] in {"dangling", "other"}:
            return _reason("not_adoptable", "%s is %s" % (entry["path"], "a broken link" if entry["kind"] == "dangling" else "not a file"))
    if offline:
        plain = [e["path"] for e in entries if e["kind"] != "blob"]
        if plain:
            return _reason("adopt_unverifiable", "offline adoption needs Hub-layout entries; %s is not one" % plain[0])
        return _reason("adoptable")
    try:
        sizes = origin(snapshot["repo"], snapshot["revision"])
    except (ModelInventoryUnavailable, ModelArtifactError) as exc:
        return _reason("adopt_unverifiable", "%s; retry, or use --offline for Hub-layout snapshots" % exc)
    for entry in entries:
        if entry["path"] not in sizes:
            return _reason("not_adoptable", "%s is not part of revision %s" % (entry["path"], snapshot["revision"][:12]))
        if sizes[entry["path"]] != entry["size"]:
            return _reason("not_adoptable", "%s differs in size from revision %s" % (entry["path"], snapshot["revision"][:12]))
    return _reason("adoptable")


def _origin(endpoint: str, token: str | None):
    known: dict = {}

    def sizes(repo: str, revision: str) -> dict:
        from sparkrun.models.manifest import fetch_inventory

        if (repo, revision) not in known:
            try:
                inventory = fetch_inventory(ModelArtifactRequest(repo, revision, endpoint=endpoint, file_selection="all"), token=token)
                if inventory["revision"] != revision:
                    raise ModelArtifactError("the origin returned another revision")
                known[(repo, revision)] = {f["path"]: f["size"] for f in inventory["files"]}
            except (ModelInventoryUnavailable, ModelArtifactError) as exc:
                known[(repo, revision)] = exc
        result = known[(repo, revision)]
        if isinstance(result, Exception):
            raise result
        return result

    return sizes


def adopt_snapshots(
    *,
    cache_dir: str,
    hosts=("localhost",),
    repo: str | None = None,
    revision: str | None = None,
    endpoint: str = "https://huggingface.co",
    offline: bool = False,
    dry_run: bool = True,
    last_used: str = "now",
    pinned: bool = False,
    token: str | None = None,
    ssh_kwargs=None,
    session=None,
) -> tuple[CacheReport, ...]:
    """Judge, and unless *dry_run* adopt, untracked snapshots; one report per cache location.

    *repo* None means every model in the cache (the caller decides whether
    that was asked for). Each report lists only the snapshots considered, with
    ``action`` ``would_adopt`` / ``adopted`` for those that qualify.
    """
    if last_used not in LAST_USED_CHOICES:
        raise ValueError("last_used must be one of: " + ", ".join(LAST_USED_CHOICES))
    origin = None if offline else _origin(endpoint, token)
    reports = []
    locations = resolve_cache_locations(cache_dir=cache_dir, hosts=hosts, ssh_kwargs=ssh_kwargs, session=session, verify=not dry_run)
    with ModelHostIO(ssh_kwargs, session=session) as io:
        for location in locations:
            host = location.hosts[0]
            root = location.paths[host]
            base = host_cache(io, host, root, action="inventory")
            scanned = host_cache(io, host, root, action="scan", repo=repo, revision=revision)["snapshots"]
            entries = []
            for snapshot in scanned:
                verdict = judge(snapshot, offline=offline, origin=origin)
                row = {
                    "repo": snapshot["repo"],
                    "revision": snapshot["revision"],
                    "snapshot": snapshot["snapshot"],
                    "tracked": False,
                    "present": True,
                    "last_used": None,
                    "logical_bytes": sum(e["size"] or 0 for e in snapshot["entries"]),
                    "reclaimable_bytes": 0,
                    "pinned": False,
                    "claims": {},
                    "eligible": False,
                    **verdict,
                }
                if verdict["reason_code"] == "adoptable":
                    if dry_run:
                        row["action"] = "would_adopt"
                    else:
                        newest = max((e["mtime"] for e in snapshot["entries"]), default=time.time())
                        try:
                            row = host_cache(
                                io,
                                host,
                                root,
                                action="adopt",
                                repo=snapshot["repo"],
                                revision=snapshot["revision"],
                                files=[[e["path"], e["kind"], e["size"]] for e in snapshot["entries"]],
                                endpoint=endpoint,
                                adopt_last_used=time.time() if last_used == "now" else newest,
                                pinned=pinned,
                            )
                            row.pop("host", None)  # host_cache stamps every result; a row has none
                            row["action"] = "adopted"
                        except ModelArtifactError as exc:
                            row.update(_reason("adopt_failed", str(exc).rsplit(": ", 1)[-1].strip()))
                entries.append(row)
            report = {**base, "dry_run": dry_run, "entries": entries, "planned_bytes": 0, "reclaimed_bytes": 0}
            reports.append(CacheReport.from_dict({**report, "location": location}))
    return tuple(reports)


__all__ = ["LAST_USED_CHOICES", "adopt_snapshots", "judge"]
