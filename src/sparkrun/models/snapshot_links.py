"""Reuse content a host already holds for a snapshot, as hard links.

Sparkrun keeps its own snapshots self-contained: entries are regular files,
which is what lets the cache agent budget and prune them (``st_nlink == 1``
bytes are reclaimable; HF blobs are never touched). Neither the repair
download nor the copy leg looked at what the host already had, though, so a
new revision of a model whose files sit unchanged in the repo's HF blob store
(``blobs/<etag>``, written by an earlier ``hf download``) was downloaded and
copied again in full, as a second copy of every shared file. On a revision
bump of a ~450 GiB model that differs by a few files, that filled the disks.

The Hub names a per-repo blob by its etag, which is the file's checksum
(sha256 for LFS files, the git blob id otherwise) — exactly what the manifest
records. So for each manifest file whose ``blobs/<checksum>`` has the
expected size, this pass makes the snapshot entry a **hard link** to it:

* a missing (or dangling) entry is linked, and needs no download or copy;
* a regular-file entry that is a separate copy is replaced by the link, which
  reclaims a duplicate left by an earlier run.

A hard link keeps the entry a regular file, so the cache agent's accounting
stays right without change: the bytes count as shared (not reclaimable) while
the HF revision still holds them, and become the snapshot's own once that
revision is deleted. An HF-style symlink entry that resolves is left alone.

Nothing is ever downloaded or written into ``blobs/``. Under the checksum
validation policy the blob's digest is verified before it is linked; otherwise
the Hub's own name for it is trusted, as ``hf_hub_download`` itself does.
Callers hold the repository's publication lease. The pass is best effort: a
failure costs the reuse, never the launch, since observation decides
readiness afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import PurePosixPath
import shutil

from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest
from sparkrun.models.observation import path_assignment
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)

#: Prefix of the temporary link created beside an entry and renamed over it.
#: An interrupted pass can leave one; the next pass (holding the lease) removes it.
TEMPORARY_PREFIX = ".sparkrun-link."


@dataclass(frozen=True)
class LinkResult:
    linked: int = 0
    deduplicated: int = 0
    reclaimed_bytes: int = 0
    failed: int = 0
    unlinkable: int = 0
    reason: str = ""


def link_script(manifest: ModelArtifactManifest, cache_root: str, *, checksums: bool = False) -> str:
    """Bash for one host; targets need no Python.

    Prints ``LINKS <linked> <deduplicated> <bytes> <failed> <unlinkable>``,
    preceded by ``REASON <first ln error>`` when a blob could not be linked.
    """
    candidates = [f for f in manifest.files if f.checksum is not None and f.checksum_algorithm in {"sha256", "git-sha1"}]
    snapshot = PurePosixPath(manifest.relative_snapshot)
    arrays = {
        "names": [f.path for f in candidates],
        "sizes": [str(f.size) for f in candidates],
        "algorithms": [f.checksum_algorithm for f in candidates],
        "sums": [f.checksum for f in candidates],
    }
    script = "set -uo pipefail\nexport LC_ALL=C\n" + path_assignment("root", cache_root)
    script += 'snap="$root"/%s\nblobs="$root"/%s\n' % (quote(str(snapshot)), quote(str(snapshot.parent.parent / "blobs")))
    script += "verify=%d\ntemporary=%s\n" % (int(checksums), quote(TEMPORARY_PREFIX))
    script += "".join(name + "=(" + " ".join(quote(value) for value in values) + ")\n" for name, values in arrays.items())
    script += r"""
linked=0 deduplicated=0 reclaimed=0 failed=0 unlinkable=0 reason=
if [ -d "$snap" ]; then
    find "$snap" -name "$temporary*" -type f -delete 2>/dev/null
fi
for i in "${!names[@]}"; do
    blob="$blobs/${sums[$i]}"
    [ -f "$blob" ] || continue
    real=$(readlink -f -- "$blob") || continue
    [ "$(stat -c %s -- "$real" 2>/dev/null)" = "${sizes[$i]}" ] || continue
    entry="$snap/${names[$i]}"
    duplicate=0
    if [ -L "$entry" ]; then
        # An HF-style entry that resolves is already shared.
        [ -e "$entry" ] && continue
    elif [ -e "$entry" ]; then
        [ -f "$entry" ] || continue
        [ "$entry" -ef "$real" ] && continue
        duplicate=1
        read -r links blocks unit < <(stat -c '%h %b %B' -- "$entry") || continue
    fi
    parent=$(dirname -- "$entry")
    staged="$parent/$temporary$$.$i"
    mkdir -p -- "$parent" || { failed=$((failed + 1)); continue; }
    # Link first: a blob that cannot be hard-linked (owned by another user
    # under fs.protected_hardlinks, or on another filesystem) is never hashed.
    if ! error=$(ln -- "$real" "$staged" 2>&1); then
        unlinkable=$((unlinkable + 1))
        [ -n "$reason" ] || reason=$error
        continue
    fi
    if [ "$verify" = 1 ]; then
        case "${algorithms[$i]}" in
            sha256) digest=$(sha256sum < "$real") || digest= ;;
            git-sha1) digest=$( (printf 'blob %s\0' "${sizes[$i]}"; cat -- "$real") | sha1sum) || digest= ;;
        esac
        if [ "${digest%% *}" != "${sums[$i]}" ]; then
            rm -f -- "$staged"
            continue
        fi
    elif [ "$duplicate" = 1 ] && ! cmp -s -- "$entry" "$real"; then
        # Replacing a copy we hold trusts the blob with good data, so the
        # name and size are not enough: a bit-rotted blob must not win.
        rm -f -- "$staged"
        continue
    fi
    # The rename swaps the entry atomically; a reader never sees it missing.
    if mv -fT -- "$staged" "$entry"; then
        if [ "$duplicate" = 1 ]; then
            deduplicated=$((deduplicated + 1))
            [ "$links" = 1 ] && reclaimed=$((reclaimed + blocks * unit))
        else
            linked=$((linked + 1))
        fi
    else
        rm -f -- "$staged"
        failed=$((failed + 1))
    fi
done
[ -z "$reason" ] || printf 'REASON %s\n' "$(printf '%s' "$reason" | head -n 1)"
printf 'LINKS %d %d %d %d %d\n' "$linked" "$deduplicated" "$reclaimed" "$failed" "$unlinkable"
"""
    return script


def parse_links(stdout: str) -> LinkResult | None:
    reason = ""
    for line in stdout.splitlines():
        if line.startswith("REASON "):
            reason = line[len("REASON ") :].strip()
    for line in reversed(stdout.splitlines()):
        parts = line.split()
        if len(parts) == 6 and parts[0] == "LINKS" and all(p.isdigit() for p in parts[1:]):
            return LinkResult(*(int(p) for p in parts[1:]), reason=reason)
    return None


def link_existing_blobs(io, host: str | None, cache_root: str, manifest: ModelArtifactManifest, *, checksums: bool = False) -> LinkResult:
    """Run the pass on one host; never raises for a failed pass (see module docstring)."""
    from sparkrun.core.progress import PROGRESS

    label = host or "control"
    if host is None and shutil.which("bash") is None:
        # A Windows control machine: nothing to link there, and no shell to try.
        logger.debug("No bash on the control machine; skipping cached-blob reuse")
        return LinkResult()
    try:
        result = io.execute(host, link_script(manifest, cache_root, checksums=checksums), timeout=7200 if checksums else 300)
    except ModelArtifactError as exc:
        logger.warning("Could not reuse cached blobs for %s on %s: %s", manifest.repo, label, exc)
        return LinkResult()
    links = parse_links(result.stdout.decode(errors="replace")) if result.returncode == 0 else None
    if links is None:
        logger.warning(
            "Could not reuse cached blobs for %s on %s (rc=%d): %s",
            manifest.repo,
            label,
            result.returncode,
            result.stderr.decode(errors="replace")[-500:],
        )
        return LinkResult()
    if links.linked or links.deduplicated:
        logger.log(
            PROGRESS,
            "  Model cache on %s: %d file(s) linked from blobs already on the host%s",
            label,
            links.linked + links.deduplicated,
            ", %.2f GiB of duplicate copies reclaimed" % (links.reclaimed_bytes / 2**30) if links.reclaimed_bytes else "",
        )
    if links.unlinkable:
        # A property of the host (blob owner under fs.protected_hardlinks, or a
        # blob store on another filesystem), not a fault: it recurs every launch.
        logger.info(
            "Model cache on %s: %d cached blob(s) cannot be hard-linked (%s); those files are transferred instead",
            label,
            links.unlinkable,
            links.reason or "unknown reason",
        )
    if links.failed:
        logger.warning("Model cache on %s: %d file(s) could not be linked to cached blobs; they will be transferred", label, links.failed)
    return links


__all__ = ["LinkResult", "TEMPORARY_PREFIX", "link_existing_blobs", "link_script", "parse_links"]
