"""Public retention DTOs. Mapping access is available to report renderers."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ReportMapping(Mapping):
    def __getitem__(self, key):
        if key not in {field.name for field in fields(self)}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return (field.name for field in fields(self))

    def __len__(self):
        return len(fields(self))


@dataclass(frozen=True)
class CacheEntry(ReportMapping):
    repo: str
    revision: str
    snapshot: str
    tracked: bool
    present: bool
    last_used: float | None
    logical_bytes: int
    reclaimable_bytes: int
    pinned: bool
    claims: dict[str, Any]
    eligible: bool
    reason: str
    reason_code: str
    usage: dict[str, Any] | None = None
    action: str | None = None


@dataclass(frozen=True)
class CacheReport(ReportMapping):
    host: str
    cache_root: str
    dry_run: bool
    entries: tuple[CacheEntry, ...]
    free_bytes: int
    reclaimed_bytes: int
    planned_bytes: int
    target_met: bool
    budget_scope: str
    reclaimed_bytes_kind: str
    free_bytes_after_kind: str
    free_bytes_after: int | None = None

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "entries": tuple(CacheEntry(**entry) for entry in value["entries"])})
