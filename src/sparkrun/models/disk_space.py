"""Capacity checks for the files still needed by a manifest preparation."""

from __future__ import annotations

from dataclasses import dataclass
import errno

from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.observation import path_assignment


def disk_failure(error) -> str | None:
    """Recognize native errors and the wrapped errors emitted by Hub/Xet/rsync."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        text = str(error).lower()
        if getattr(error, "errno", None) == errno.ENOSPC or any(
            part in text for part in ("no space left on device", "out of disk space", "os error 28")
        ):
            return "out of disk space"
        if getattr(error, "errno", None) == getattr(errno, "EDQUOT", 122) or "disk quota exceeded" in text:
            return "disk quota exceeded"
        error = getattr(error, "__cause__", None) or getattr(error, "__context__", None)
    return None


def _bytes(value: int) -> str:
    for unit, scale in (("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if value >= scale:
            return "%.2f %s" % (value / scale, unit)
    return "%d B" % value


class ModelDiskSpaceError(ModelArtifactError):
    def __init__(self, host, cache_root, *, required_bytes=None, free_bytes=None, reason="out of disk space", action="model preparation"):
        self.host = host or "control"
        self.cache_root = cache_root
        self.required_bytes = required_bytes
        self.free_bytes = free_bytes
        self.reason = "insufficient disk space" if required_bytes is not None else reason
        detail = "%s on %s for %s at %s" % (self.reason, self.host, action, cache_root)
        if required_bytes is not None and free_bytes is not None:
            detail += ": %s required for remaining files, %s available (%s short)" % (
                _bytes(required_bytes),
                _bytes(free_bytes),
                _bytes(max(0, required_bytes - free_bytes)),
            )
        super().__init__(detail + ". Free space on this filesystem and retry; completed model files will be reused.")


@dataclass(frozen=True)
class DiskSpace:
    free_bytes: int
    filesystem: tuple[str, str]


@dataclass
class _Budget:
    host: str | None
    root: str
    available: int
    required: int = 0


def probe_disk_space(io, host, cache_root: str) -> DiskSpace:
    """Measure available bytes on the cache filesystem without creating it."""
    script = (
        "set -euo pipefail\nexport LC_ALL=C\n"
        + path_assignment("probe", cache_root)
        + r"""
while [ ! -e "$probe" ]; do
    probe=${probe%/*}
    [ -n "$probe" ] || probe=/
done
stat -f -c '%a %S' -- "$probe"
stat -c '%d' -- "$probe"
cat /proc/sys/kernel/random/boot_id
"""
    )
    try:
        values = io.checked(host, script).split()
        if len(values) != 4 or not all(v.isdigit() for v in values[:3]) or int(values[1]) <= 0:
            raise ValueError("invalid filesystem capacity report")
        return DiskSpace(int(values[0]) * int(values[1]), (values[3], values[2]))
    except (ModelArtifactError, ValueError) as exc:
        raise ModelArtifactError("could not check model disk space on %s at %s: %s" % (host or "control", cache_root, exc)) from exc


def require_model_space(io, manifest, requirements):
    """Check all caches before any payload work, with aliases counted once.

    Requirements are (host, cache root, required file names). Complete files
    and reused blobs cost nothing. Replacement files need their full size:
    atomic publication keeps the old entry until its replacement is ready.
    Existing partials are not credited: a provider can stage a fresh copy.
    Separate caches on the same host filesystem share one capacity budget.
    """
    domains: dict[str, tuple[str | None, str, set[str]]] = {}
    for host, root, files in requirements:
        if not files:
            continue
        domain = io.lock(host, root, manifest.repo)
        if domain not in domains:
            domains[domain] = (host, root, set())
        domains[domain][2].update(files)
    sizes = {f.path: f.size for f in manifest.files}
    budgets: dict[tuple[str, str], _Budget] = {}
    for host, root, files in domains.values():
        disk = probe_disk_space(io, host, root)
        budget = budgets.setdefault(disk.filesystem, _Budget(host, root, disk.free_bytes))
        budget.required += sum(sizes[name] for name in files)
        budget.available = min(budget.available, disk.free_bytes)
    for budget in budgets.values():
        if budget.required > budget.available:
            raise ModelDiskSpaceError(budget.host, budget.root, required_bytes=budget.required, free_bytes=budget.available)
