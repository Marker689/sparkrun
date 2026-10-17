"""Which target hosts share one model cache, so cache operations run once per cache.

A cluster whose hosts mount one NFS export for the HF cache is one cache, not
N. Run per host, ``model-cache list`` printed the same table N times and
``delete`` previewed (and asked to confirm) N times the bytes it would free,
then did the work on the first host and found nothing on the rest. So the
cache operations work per *location*: a set of hosts that see the same cache
directory.

Identity is probed read-only, one round-trip per host:

* a **network** filesystem (NFS, SMB, Ceph, ...) is keyed by mount source, the
  cache root's path within the export, and its inode. The inode is the
  server's file id, so it agrees across clients, and requiring it beside the
  source guards against a stale or confusing mount;
* a **local** filesystem is keyed by ``boot_id``, device and inode. Two disks
  are never merged, but one machine reached under two names (the control
  machine that is also a cluster member, an IP and a hostname) is;
* a host whose probe fails is a location of its own.

The key can only err toward *more* locations: a share mounted under two
spellings of the server's name is shown twice; two caches are never merged.
That is the safe direction for operations that delete.

Mutating operations go further (:func:`verify_locations`): a nonce written
through one host's path must be read back through every other host's path.
A host that cannot confirm it is split off and handled on its own, which is
exactly what every operation did before locations existed.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
import secrets
import time

from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.cache_reports import CacheLocation
from sparkrun.models.manifest import STATE_DIRECTORY
from sparkrun.models.observation import path_assignment
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)

#: Filesystem types several machines can mount as one. Only these merge hosts.
NETWORK_FILESYSTEMS = frozenset(
    {"nfs", "nfs4", "cifs", "smb3", "smbfs", "ceph", "fuse.ceph", "glusterfs", "fuse.glusterfs", "lustre", "beegfs", "gpfs"}
)

#: Persistent file the nonce is written into. Rewritten in place rather than
#: replaced: a client that has looked the name up keeps resolving it, whereas a
#: new name can sit behind NFS negative-lookup caching for tens of seconds.
PROBE_FILE = "location-probe"

#: How long a host may take to see another host's write (NFS attribute caching).
VERIFY_TIMEOUT_S = 10.0


def probe_script(cache_dir: str) -> str:
    """Report the identity facts of *cache_dir* on one host, as ``KEY=value`` lines."""
    return (
        "set -uo pipefail\nexport LC_ALL=C\n"
        + path_assignment("root", cache_dir)
        + r"""
real=$(realpath -m -- "$root") || exit 3
# The cache may not exist yet: identify the nearest existing ancestor, and
# carry the rest of the path as a suffix.
probe=$real
while [ ! -e "$probe" ]; do probe=$(dirname -- "$probe"); done
printf 'REAL=%s\n' "$real"
printf 'PROBE=%s\n' "$probe"
stat -c 'INODE=%i' -- "$probe"
stat -c 'DEVICE=%d' -- "$probe"
printf 'BOOT=%s\n' "$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
findmnt -n -P -o FSTYPE,SOURCE,TARGET,FSROOT -T "$probe" 2>/dev/null | tail -n 1
"""
    )


def _pairs(line: str) -> dict[str, str]:
    """Parse ``findmnt -P`` output: ``KEY="value"`` pairs, with its ``\\xNN`` escapes decoded."""
    import re
    import shlex

    try:
        pairs = dict(part.split("=", 1) for part in shlex.split(line) if "=" in part)
    except ValueError:
        return {}
    # findmnt escapes spaces and other special characters as \x20 etc.
    return {k: re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), v) for k, v in pairs.items()}


def identity(host: str, stdout: str) -> tuple[str, str, str, str] | None:
    """``(key, kind, source, path)`` for one host's probe output, or None if unusable."""
    facts: dict[str, str] = {}
    for line in stdout.splitlines():
        if line.startswith("FSTYPE="):
            facts.update(_pairs(line))
        elif "=" in line:
            name, value = line.split("=", 1)
            facts[name] = value
    real, probe = facts.get("REAL", ""), facts.get("PROBE", "")
    if not real.startswith("/") or not probe.startswith("/") or not facts.get("INODE", "").isdigit():
        return None
    suffix = str(PurePosixPath(real).relative_to(probe)) if real != probe else ""
    fstype = facts.get("FSTYPE", "")
    if fstype in NETWORK_FILESYSTEMS and facts.get("SOURCE") and facts.get("TARGET", "").startswith("/"):
        try:
            within = PurePosixPath(probe).relative_to(facts["TARGET"])
        except ValueError:
            return None
        export_path = str(PurePosixPath(facts.get("FSROOT") or "/") / within)
        source = facts["SOURCE"]
        key = "|".join(("network", fstype.removesuffix("4"), source, export_path, facts["INODE"], suffix))
        return key, "network", source, real
    boot = facts.get("BOOT", "")
    if not boot or not facts.get("DEVICE", "").isdigit():
        return None
    key = "|".join(("local", boot, facts["DEVICE"], facts["INODE"], suffix))
    return key, "local", host, real


def resolve_locations(io, hosts, cache_dir: str) -> tuple[CacheLocation, ...]:
    """Group *hosts* by the cache they see, in host order. Read-only."""
    groups: dict[str, dict] = {}
    for host in dict.fromkeys(hosts):
        found = None
        try:
            result = io.execute(host, probe_script(cache_dir), timeout=30)
            if result.returncode == 0:
                found = identity(host, result.stdout.decode(errors="replace"))
        except ModelArtifactError as exc:
            logger.debug("Cache location probe failed on %s: %s", host, exc)
        if found is None:
            logger.debug("Cache location of %s is unknown; it is treated as its own cache", host)
            try:
                path = io.root(host, cache_dir)
            except ModelArtifactError:
                path = cache_dir  # the operation itself will report the host
            found = ("host|" + host, "unknown", host, path)
        key, kind, source, path = found
        group = groups.setdefault(key, {"kind": kind, "source": source, "hosts": [], "paths": {}})
        group["hosts"].append(host)
        group["paths"][host] = path
    return tuple(
        CacheLocation(key, g["kind"], g["source"], tuple(g["hosts"]), dict(g["paths"]), verified=len(g["hosts"]) == 1)
        for key, g in groups.items()
    )


def _probe_path(root: str) -> str:
    return str(PurePosixPath(root) / STATE_DIRECTORY / PROBE_FILE)


def _write_nonce(io, host: str, root: str, nonce: str) -> bool:
    path = _probe_path(root)
    script = (
        "set -euo pipefail\n"
        + path_assignment("p", path)
        + 'mkdir -p -- "$(dirname -- "$p")"\n'
        # In place (not a rename), so other clients keep resolving the same file.
        + 'printf %%s %s > "$p"\n' % quote(nonce)
    )
    try:
        return io.execute(host, script, timeout=30).returncode == 0
    except ModelArtifactError as exc:
        logger.debug("Could not write the cache location probe on %s: %s", host, exc)
        return False


def _sees_nonce(io, host: str, root: str, nonce: str, *, timeout: float) -> bool:
    script = path_assignment("p", _probe_path(root)) + 'cat -- "$p" 2>/dev/null\n'
    deadline = time.monotonic() + timeout
    delay = 0.2
    while True:
        try:
            result = io.execute(host, script, timeout=30)
            if result.returncode == 0 and result.stdout.decode(errors="replace") == nonce:
                return True
        except ModelArtifactError as exc:
            logger.debug("Could not read the cache location probe on %s: %s", host, exc)
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(delay)
        delay = min(delay * 2, 2.0)


def verify_locations(io, locations, *, timeout: float = VERIFY_TIMEOUT_S) -> tuple[CacheLocation, ...]:
    """Confirm each multi-host location by a nonce round-trip; split what does not confirm.

    For mutating operations. Writes one small file in the cache's state
    directory. A host that does not see the nonce becomes its own location.
    """
    confirmed: list[CacheLocation] = []
    for location in locations:
        if len(location.hosts) == 1:
            confirmed.append(location)
            continue
        nonce = secrets.token_hex(16)
        # The first host that can write is the reference; one that is down
        # must not cost the rest of the group its membership.
        first = next((h for h in location.hosts if _write_nonce(io, h, location.paths[h], nonce)), location.hosts[0])
        members = [first] + [h for h in location.hosts if h != first and _sees_nonce(io, h, location.paths[h], nonce, timeout=timeout)]
        members.sort(key=location.hosts.index)
        apart = [h for h in location.hosts if h not in members]
        if apart:
            logger.info(
                "Model cache on %s could not be confirmed as the cache %s uses; treating %s separately",
                ", ".join(apart),
                first,
                "it" if len(apart) == 1 else "them",
            )
        confirmed.append(
            CacheLocation(
                location.key, location.kind, location.source, tuple(members), {h: location.paths[h] for h in members}, verified=True
            )
        )
        confirmed += [
            CacheLocation(
                "host|" + h, location.kind, h if location.kind != "network" else location.source, (h,), {h: location.paths[h]}, True
            )
            for h in apart
        ]
    return tuple(confirmed)


__all__ = ["CacheLocation", "NETWORK_FILESYSTEMS", "identity", "probe_script", "resolve_locations", "verify_locations"]
