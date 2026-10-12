"""Standard-library host helper for explicit inventory and conservative pruning.

Shipped over the existing host session; no Sparkrun or Hub install is required.
Never delete blobs, lock files, unknown snapshots, or paths outside HF_HOME.
"""

import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import time

try:
    from ._cache_contract import STATE_DIRECTORY as STATE, CACHE_REASONS, validate_usage
except ImportError:  # Shipped by the controller without installing Sparkrun.
    from importlib import import_module

    _contract = import_module("_sparkrun_cache_contract")
    STATE, CACHE_REASONS, validate_usage = _contract.STATE_DIRECTORY, _contract.CACHE_REASONS, _contract.validate_usage


def traversal_error(error):
    raise error


def safe_directory(path):
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("symlinked cache directory: " + str(parent))
        if parent.exists() and not parent.is_dir():
            raise ValueError("not a cache directory: " + str(parent))


def read_record(path, slug, revision):
    if path.is_symlink():
        raise ValueError("symlinked usage record")
    record = json.loads(path.read_text())
    return validate_usage(record, slug, revision)


def observe(root, slug, revision, *, now, min_age_seconds, exclude_repos):
    snapshot = root / "hub" / slug / "snapshots" / revision
    path = root / STATE / "usage" / slug.removeprefix("models--") / (revision + ".json")
    row = {
        "repo": slug.removeprefix("models--").replace("--", "/", 1),
        "revision": revision,
        "snapshot": str(snapshot),
        "tracked": False,
        "present": snapshot.exists(),
        "last_used": None,
        "logical_bytes": 0,
        "reclaimable_bytes": 0,
        "pinned": False,
        "claims": {},
        "eligible": False,
        "reason": CACHE_REASONS["untracked"],
        "reason_code": "untracked",
    }
    try:
        safe_directory(snapshot)
        safe_directory(path.parent)
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            return row
        if not path.exists():
            return row
        record = read_record(path, slug.removeprefix("models--"), revision)
        row.update({k: record[k] for k in ("repo", "last_used", "pinned", "claims")})
        row.update(tracked=True, usage=record)
        unknown = False
        files = set(record["files"])
        if snapshot.exists():
            for directory, dirs, names in os.walk(snapshot, followlinks=False, onerror=traversal_error):
                for child in dirs + names:
                    item = Path(directory) / child
                    info = item.lstat()
                    if stat.S_ISDIR(info.st_mode):
                        continue
                    relative = str(item.relative_to(snapshot))
                    if relative not in files or not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                        unknown = True
                    if stat.S_ISREG(info.st_mode):
                        row["logical_bytes"] += info.st_size
                        if info.st_nlink == 1:
                            row["reclaimable_bytes"] += info.st_blocks * 512
                    elif stat.S_ISLNK(info.st_mode):
                        # Canonical/shared blobs belong to HF, never this sweep.
                        try:
                            row["logical_bytes"] += item.stat().st_size
                        except OSError:
                            pass
        reason_code = (
            "absent"
            if not row["present"]
            else "pinned"
            if record["pinned"]
            else "workload_claim"
            if record["claims"]
            else "requested_model"
            if record["repo"] in exclude_repos
            else "untracked_files"
            if unknown
            else "retention_age"
            if now - record["last_used"] < min_age_seconds
            else "eligible"
        )
        row.update(reason=CACHE_REASONS[reason_code], reason_code=reason_code, eligible=reason_code == "eligible")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row.update(reason=CACHE_REASONS["unverifiable"] + ": " + str(exc), reason_code="unverifiable", eligible=False)
    return row


def inventory(root, **policy):
    safe_directory(root)
    safe_directory(root / "hub")
    safe_directory(root / STATE / "usage")
    identities = set()
    for path in (root / "hub").glob("models--*/snapshots/*"):
        identities.add((path.parent.parent.name, path.name))
    for path in (root / STATE / "usage").glob("*/*.json"):
        identities.add(("models--" + path.parent.name, path.stem))
    return [observe(root, slug, revision, **policy) for slug, revision in sorted(identities)]


def write_record(path, record):
    import tempfile

    fd, temp = tempfile.mkstemp(prefix=".usage-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, sort_keys=True)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def run(
    root,
    *,
    action="inventory",
    dry_run=True,
    min_age_seconds=7 * 86400,
    max_bytes=None,
    min_free_bytes=None,
    exclude_repos=(),
    repo=None,
    revision=None,
    pinned=True,
    claim=None,
    confirm_inactive=False,
    endpoint=None,
    now=None,
):
    root = Path(root)
    if not root.is_absolute() or root == Path("/"):
        raise ValueError("an absolute HF cache home is required")
    for value in (min_age_seconds, max_bytes, min_free_bytes):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("cache limits must be finite and nonnegative")
    if action not in {"inventory", "prune", "pin", "reconcile"}:
        raise ValueError("unknown cache action")
    if type(dry_run) is not bool or type(pinned) is not bool or type(confirm_inactive) is not bool:
        raise ValueError("cache mutation flags must be boolean")
    now = time.time() if now is None else now
    policy = dict(now=now, min_age_seconds=min_age_seconds, exclude_repos=exclude_repos)
    rows = inventory(root, **policy)
    probe = root
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    result = {
        "cache_root": str(root),
        "dry_run": dry_run,
        "entries": rows,
        "free_bytes": free,
        "reclaimed_bytes": 0,
        "planned_bytes": 0,
        "target_met": True,
        "budget_scope": "private_managed_snapshots",
        "reclaimed_bytes_kind": "allocated_bytes_estimate",
        "free_bytes_after_kind": "projected" if dry_run else "measured",
    }
    if action == "inventory":
        return result
    if action in {"pin", "reconcile"}:
        rows = [r for r in rows if r["repo"] == repo and r["revision"] == revision and r["tracked"]]
        if len(rows) != 1:
            raise ValueError("tracked model revision not found")
        if action == "reconcile" and (not confirm_inactive or not claim):
            raise ValueError("reconciliation requires a claim and explicit confirmation that its workloads stopped")
    total = sum(r["reclaimable_bytes"] for r in result["entries"])

    def satisfied():
        return (max_bytes is None or total <= max_bytes) and (min_free_bytes is None or free >= min_free_bytes)

    for row in sorted(rows, key=lambda r: (r["last_used"] or 0, r["repo"], r["revision"])):
        if action == "prune" and (not row["eligible"] or ((max_bytes is not None or min_free_bytes is not None) and satisfied())):
            continue
        slug = Path(row["snapshot"]).parent.parent.name
        lock = root / STATE / (slug.removeprefix("models--") + ".lock")
        safe_directory(lock.parent)
        # Preparation creates the lock before recording usage. Never create it
        # in a dry run, nor unlink it (which would split the locking domain).
        try:
            fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            row.update(eligible=False, reason=CACHE_REASONS["publication_lock_unavailable"], reason_code="publication_lock_unavailable")
            continue
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                row.update(eligible=False, reason=CACHE_REASONS["invalid_publication_lock"], reason_code="invalid_publication_lock")
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                row.update(eligible=False, reason=CACHE_REASONS["cache_busy"], reason_code="cache_busy")
                continue
            observed_last_used = row["last_used"]
            fresh = observe(root, slug, row["revision"], **policy)
            row.update(fresh)
            if not fresh["tracked"]:
                continue
            path = root / STATE / "usage" / slug.removeprefix("models--") / (row["revision"] + ".json")
            if action != "prune":
                record = fresh["usage"]
                if endpoint is not None and record["endpoint"] != endpoint:
                    raise ValueError("cache usage endpoint mismatch")
                if action == "pin":
                    record["pinned"] = pinned
                    row["pinned"] = pinned
                else:
                    if claim not in record["claims"]:
                        raise ValueError("outstanding claim not found")
                    del record["claims"][claim]
                    record["last_used"] = record["last_released"] = now
                if not dry_run:
                    write_record(path, record)
                row["action"] = action
                continue
            if not fresh["eligible"]:
                continue
            if row["last_used"] != observed_last_used:
                row.update(eligible=False, reason=CACHE_REASONS["usage_changed_during_cleanup"], reason_code="usage_changed_during_cleanup")
                continue
            amount = row["reclaimable_bytes"]
            result["planned_bytes"] += amount
            row["action"] = "would_remove" if dry_run else "removed"
            if not dry_run:
                # shutil.rmtree uses fd-based symlink-safe traversal on Linux.
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise ValueError("safe snapshot deletion is unavailable on this host")
                shutil.rmtree(row["snapshot"])
                record = fresh["usage"]
                record["last_evicted"] = now
                write_record(path, record)
                result["reclaimed_bytes"] += amount
            total -= amount
            free = free + amount if dry_run else shutil.disk_usage(root).free
        finally:
            os.close(fd)
    result["target_met"] = satisfied()
    result["free_bytes_after"] = free
    return result
