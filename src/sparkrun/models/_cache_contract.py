"""Shared dependency-free cache receipt contract, also shipped to cache hosts."""

import math
import re

STATE_DIRECTORY = ".sparkrun-model-artifacts"
USAGE_SCHEMA_VERSION = 1

# Codes are the machine contract; display text may change independently.
CACHE_REASONS = {
    "untracked": "untracked",
    "absent": "absent",
    "pinned": "pinned",
    "workload_claim": "workload claim",
    "requested_model": "requested model",
    "untracked_files": "untracked files",
    "retention_age": "retention age",
    "eligible": "eligible",
    "unverifiable": "cannot verify cache",
    "publication_lock_unavailable": "publication lock unavailable",
    "invalid_publication_lock": "invalid publication lock",
    "cache_busy": "cache busy",
    "usage_changed_during_cleanup": "usage changed during cleanup",
}


def validate_usage(record, slug, revision):
    repo = record["repo"]
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)?", repo):
        raise ValueError("invalid repository")
    if record["schema"] != USAGE_SCHEMA_VERSION or repo.replace("/", "--") != slug or record["revision"] != revision:
        raise ValueError("usage identity mismatch")
    if not isinstance(record["claims"], dict) or type(record["pinned"]) is not bool:
        raise ValueError("invalid usage protection")
    if not isinstance(record["files"], list) or not record["files"]:
        raise ValueError("invalid file inventory")
    for name in record["files"]:
        if not isinstance(name, str) or not name or name.startswith("/") or any(p in {"", ".", ".."} for p in name.split("/")):
            raise ValueError("invalid inventory path")
    for name in ("last_used", "first_seen"):
        value = record[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid usage time")
    return record
