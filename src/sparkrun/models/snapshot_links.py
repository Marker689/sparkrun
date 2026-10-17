"""Reuse content a host already holds for a snapshot, as hard links.

Sparkrun keeps its own snapshots self-contained: entries are regular files,
which is what lets the cache agent budget and prune them (``st_nlink == 1``
bytes are reclaimable; HF blobs are never touched). Neither the repair
download nor the copy leg looked at what the host already had, though, so a
new revision of a model whose files sit unchanged on the host was downloaded
and copied again in full, as a second copy of every shared file. On a revision
bump of a ~450 GiB model that differs by a few files, that filled the disks.

Two local sources name content by the checksum the manifest records, so
neither needs hashing to find:

* the repo's HF blob store: the Hub names ``blobs/<etag>`` by the file's
  checksum (sha256 for LFS files, the git blob id otherwise), for revisions
  fetched by ``hf download``;
* other revisions sparkrun prepared on this host: each left its manifest under
  the state directory, so ``snapshots/<rev>/<path>`` with the same checksum
  and size is the same content. Without this, reuse would end the moment the
  HF revision was deleted to make room — every later bump would be a full
  download again.

For each manifest file, the first candidate of the expected size becomes the
snapshot entry as a **hard link**:

* a missing (or dangling) entry is linked, and needs no download or copy;
* a regular-file entry that is a separate copy is replaced by the link, which
  reclaims a duplicate left by an earlier run.

A hard link keeps the entry a regular file, so the cache agent's accounting
stays right without change: the bytes count as shared (not reclaimable) while
another revision still holds them, and become this snapshot's own once that
revision is deleted. An HF-style symlink entry that resolves is left alone.

When no candidate can be hard-linked (a blob owned by another user under
``fs.protected_hardlinks``, or on another filesystem), a *missing* entry falls
back to the relative ``../../blobs/<checksum>`` symlink the Hub itself writes.
Prune frees such a blob only once no other revision links it (the bytes are
the blob store's while one does), so it is a weaker form of reuse, not an unsafe one: ``hf cache rm`` and the cache agent's delete both keep a blob any
snapshot links to. The fallback never targets another revision's snapshot
file (deleting that revision would leave the entry dangling) and never
replaces a copy we already hold.

Nothing is downloaded or written into ``blobs/``. A missing entry trusts a
candidate's name and size, as ``hf_hub_download`` trusts a blob's name;
replacing a copy we hold needs ``cmp``; under the checksum policy every
candidate's digest is verified first. Callers hold the repository's
publication lease. The pass is best effort: a failure costs the reuse, never
the launch, since observation decides readiness afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import PurePosixPath
import shutil

from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest
from sparkrun.models.manifest import STATE_DIRECTORY
from sparkrun.models.observation import path_assignment
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)

#: Prefix of the temporary link created beside an entry and renamed over it.
#: An interrupted pass can leave one; the next pass (holding the lease) removes it.
TEMPORARY_PREFIX = ".sparkrun-link."

#: Saved manifests are tens of KiB; anything far larger is not one of ours.
_MANIFEST_SIZE_LIMIT = "4M"


@dataclass(frozen=True)
class LinkResult:
    linked: int = 0
    deduplicated: int = 0
    reclaimed_bytes: int = 0
    failed: int = 0
    unlinkable: int = 0
    symlinked: int = 0
    reason: str = ""


def _reusable(manifest: ModelArtifactManifest):
    return [f for f in manifest.files if f.checksum is not None and f.checksum_algorithm in {"sha256", "git-sha1"}]


def saved_manifests_script(cache_root: str, repo: str) -> str:
    """Print this host's saved manifests for *repo*, NUL-framed, for :func:`sibling_sources`."""
    needle = json.dumps({"repo": repo}, separators=(",", ":"))[1:-1]
    return (
        "set -uo pipefail\nexport LC_ALL=C\n"
        + path_assignment("root", cache_root)
        + 'state="$root"/%s\n' % quote(STATE_DIRECTORY)
        + "needle=%s\n" % quote(needle)
        + r"""
[ -d "$state" ] || exit 0
find "$state" -maxdepth 1 -type f -name '*.json' -size -"""
        + _MANIFEST_SIZE_LIMIT
        + r""" -print0 2>/dev/null |
while IFS= read -r -d '' path; do
    grep -qF -- "$needle" "$path" 2>/dev/null || continue
    cat -- "$path" 2>/dev/null || continue
    printf '\0'
done
"""
    )


def sibling_sources(manifests: list[ModelArtifactManifest], manifest: ModelArtifactManifest) -> dict[str, list[str]]:
    """Map each reusable file of *manifest* to same-content files of the other revisions.

    Paths are relative to the cache root. Identity is (algorithm, checksum,
    size) as each revision's own manifest recorded it.
    """
    wanted = {(f.checksum_algorithm, f.checksum, f.size): f.path for f in _reusable(manifest)}
    sources: dict[str, list[str]] = {}
    for other in manifests:
        if other.repo != manifest.repo or other.revision == manifest.revision:
            continue
        for entry in other.files:
            path = wanted.get((entry.checksum_algorithm, entry.checksum, entry.size))
            if path is None:
                continue
            candidate = str(PurePosixPath(other.relative_snapshot) / entry.path)
            if candidate not in sources.setdefault(path, []):
                sources[path].append(candidate)
    return sources


def parse_saved_manifests(stdout: bytes) -> list[ModelArtifactManifest]:
    manifests = []
    for chunk in stdout.split(b"\0"):
        if not chunk.strip():
            continue
        try:
            manifests.append(ModelArtifactManifest.from_json(chunk.decode()))
        except (ModelArtifactError, ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
            logger.debug("Ignoring an unreadable saved model manifest: %s", exc)
    return manifests


def link_script(
    manifest: ModelArtifactManifest,
    cache_root: str,
    *,
    checksums: bool = False,
    siblings: dict[str, list[str]] | None = None,
) -> str:
    """Bash for one host; targets need no Python.

    *siblings* maps a manifest path to same-content files (relative to the
    cache root) from other revisions, tried after the blob. Prints
    ``LINKS <linked> <deduplicated> <bytes> <failed> <unlinkable> <symlinked>``,
    preceded by ``REASON <cause>`` when a hard link was refused for a reason the
    host explains (another filesystem, another user's file, a blob resolving
    outside the cache).
    """
    files = _reusable(manifest)
    snapshot = PurePosixPath(manifest.relative_snapshot)
    siblings = siblings or {}
    arrays = {
        "names": [f.path for f in files],
        "sizes": [str(f.size) for f in files],
        "algorithms": [f.checksum_algorithm for f in files],
        "sums": [f.checksum for f in files],
        # The Hub's own entry form: up from the entry's directory to the repo.
        "relblobs": ["../" * (1 + len(PurePosixPath(f.path).parts)) + "blobs/" + f.checksum for f in files],
        # Candidates grouped per file; each file slices its own run of them.
        "cand_start": [str(sum(len(siblings.get(g.path, ())) for g in files[:i])) for i in range(len(files))],
        "cand_count": [str(len(siblings.get(f.path, ()))) for f in files],
        "cand_path": [path for f in files for path in siblings.get(f.path, ())],
    }
    script = "set -uo pipefail\nexport LC_ALL=C\n" + path_assignment("root", cache_root)
    script += 'snap="$root"/%s\nblobs="$root"/%s\n' % (quote(str(snapshot)), quote(str(snapshot.parent.parent / "blobs")))
    script += "verify=%d\ntemporary=%s\n" % (int(checksums), quote(TEMPORARY_PREFIX))
    script += "".join(name + "=(" + " ".join(quote(value) for value in values) + ")\n" for name, values in arrays.items())
    script += r"""
linked=0 deduplicated=0 reclaimed=0 failed=0 unlinkable=0 symlinked=0 reason=
# A hard-link source must resolve inside the hub: any component of a candidate
# path can be a symlink, and a planted one must not hard-link a host file into a
# cache that containers mount.
hubreal=$(readlink -f -- "$root/hub") || hubreal=

digest_ok() {
    # $1 file, $2 manifest index. Only consulted under the checksum policy.
    local digest=
    case "${algorithms[$2]}" in
        sha256) digest=$(sha256sum < "$1") || return 1 ;;
        git-sha1) digest=$( (printf 'blob %s\0' "${sizes[$2]}"; cat -- "$1") | sha1sum) || return 1 ;;
    esac
    [ "${digest%% *}" = "${sums[$2]}" ]
}

if [ -d "$snap" ]; then
    find "$snap" -name "$temporary*" \( -type f -o -type l \) -delete 2>/dev/null
fi
for i in "${!names[@]}"; do
    entry="$snap/${names[$i]}"
    duplicate=0
    if [ -L "$entry" ]; then
        # An HF-style entry that resolves is already shared.
        [ -e "$entry" ] && continue
    elif [ -e "$entry" ]; then
        [ -f "$entry" ] || continue
        duplicate=1
        read -r links blocks unit < <(stat -c '%h %b %B' -- "$entry") || continue
    fi
    blob="$blobs/${sums[$i]}"
    candidates=("$blob")
    for path in "${cand_path[@]:${cand_start[$i]}:${cand_count[$i]}}"; do
        candidates+=("$root/$path")
    done
    parent=$(dirname -- "$entry")
    staged="$parent/$temporary$$.$i"
    shared=0 staged_from= blob_ok=0 refused= broken=0
    for source in "${candidates[@]}"; do
        [ -n "$hubreal" ] && [ -f "$source" ] || continue
        real=$(readlink -f -- "$source") || continue
        [ "$(stat -c %s -- "$real" 2>/dev/null)" = "${sizes[$i]}" ] || continue
        [ "$source" = "$blob" ] && blob_ok=1
        case "$real" in
            "$hubreal"/*) ;;
            *)
                # Never hard-linked: that would pull a host file into the mount.
                # A blob may still get the Hub's own relative symlink, which
                # exposes nothing a container could not already resolve.
                [ "$source" = "$blob" ] && [ -z "$refused" ] && refused="outside the cache"
                continue
                ;;
        esac
        if [ -e "$entry" ] && [ "$entry" -ef "$real" ]; then
            shared=1
            break
        fi
        mkdir -p -- "$parent" || break
        # Link first: a candidate that cannot be hard-linked (owned by another
        # user under fs.protected_hardlinks, or on another filesystem) is
        # never hashed.
        if ln -- "$real" "$staged" 2>/dev/null; then
            if [ "$verify" = 1 ] && ! digest_ok "$real" "$i"; then
                rm -f -- "$staged"
                # Already hashed and wrong: the fallback must not offer it again.
                [ "$source" = "$blob" ] && blob_ok=2
                continue
            fi
            if [ "$duplicate" = 1 ] && [ "$verify" != 1 ] && ! cmp -s -- "$entry" "$real"; then
                # Replacing a copy we hold trusts the candidate with good data,
                # so a name and size are not enough: bit rot must not win.
                rm -f -- "$staged"
                continue
            fi
            staged_from=$real
            break
        fi
        # Only these two are properties of the host that a symlink can work
        # around; anything else (read-only, full, permissions) is a failure.
        if [ "$(stat -c %d -- "$real")" != "$(stat -c %d -- "$parent")" ]; then
            [ -n "$refused" ] || refused="on another filesystem"
        elif [ ! -O "$real" ]; then
            [ -n "$refused" ] || refused="owned by another user"
        else
            broken=1
        fi
    done
    [ "$shared" = 1 ] && continue
    if [ -n "$staged_from" ]; then
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
        continue
    fi
    # A copy we hold that could not be swapped simply stays; nothing moves for it.
    [ "$duplicate" = 0 ] || continue
    if [ -z "$refused" ]; then
        # No refusal: an ordinary transfer, unless a link failed for no reason
        # the host explains.
        [ "$broken" = 1 ] && failed=$((failed + 1))
        continue
    fi
    # Nothing could be hard-linked. A missing entry may still point at the blob
    # store the way the Hub's own entries do; a copy we hold stays as it is.
    if [ "$blob_ok" = 1 ] && { [ "$verify" != 1 ] || digest_ok "$blob" "$i"; }; then
        if mkdir -p -- "$parent" && ln -s -- "${relblobs[$i]}" "$staged" 2>/dev/null && mv -fT -- "$staged" "$entry"; then
            symlinked=$((symlinked + 1))
            [ -n "$reason" ] || reason=$refused
            continue
        fi
        rm -f -- "$staged"
    fi
    unlinkable=$((unlinkable + 1))
    [ -n "$reason" ] || reason=$refused
done
[ -z "$reason" ] || printf 'REASON %s\n' "$(printf '%s' "$reason" | head -n 1)"
printf 'LINKS %d %d %d %d %d %d\n' "$linked" "$deduplicated" "$reclaimed" "$failed" "$unlinkable" "$symlinked"
"""
    return script


def parse_links(stdout: str) -> LinkResult | None:
    reason = ""
    for line in stdout.splitlines():
        if line.startswith("REASON "):
            reason = line[len("REASON ") :].strip()
    for line in reversed(stdout.splitlines()):
        parts = line.split()
        if len(parts) == 7 and parts[0] == "LINKS" and all(p.isdigit() for p in parts[1:]):
            return LinkResult(*(int(p) for p in parts[1:]), reason=reason)
    return None


def _host_siblings(io, host: str | None, cache_root: str, manifest: ModelArtifactManifest, label: str) -> dict[str, list[str]]:
    try:
        result = io.execute(host, saved_manifests_script(cache_root, manifest.repo), timeout=60)
    except ModelArtifactError as exc:
        logger.debug("Could not read saved model manifests on %s: %s", label, exc)
        return {}
    if result.returncode:
        logger.debug("Could not read saved model manifests on %s (rc=%d)", label, result.returncode)
        return {}
    return sibling_sources(parse_saved_manifests(result.stdout), manifest)


def link_existing_blobs(io, host: str | None, cache_root: str, manifest: ModelArtifactManifest, *, checksums: bool = False) -> LinkResult:
    """Run the pass on one host; never raises for a failed pass (see module docstring)."""
    from sparkrun.core.progress import PROGRESS

    label = host or "control"
    if host is None and shutil.which("bash") is None:
        # A Windows control machine: nothing to link there, and no shell to try.
        logger.debug("No bash on the control machine; skipping cached-blob reuse")
        return LinkResult()
    siblings = _host_siblings(io, host, cache_root, manifest, label)
    try:
        result = io.execute(
            host,
            link_script(manifest, cache_root, checksums=checksums, siblings=siblings),
            timeout=7200 if checksums else 300,
        )
    except ModelArtifactError as exc:
        logger.warning("Could not reuse cached files for %s on %s: %s", manifest.repo, label, exc)
        return LinkResult()
    links = parse_links(result.stdout.decode(errors="replace")) if result.returncode == 0 else None
    if links is None:
        logger.warning(
            "Could not reuse cached files for %s on %s (rc=%d): %s",
            manifest.repo,
            label,
            result.returncode,
            result.stderr.decode(errors="replace")[-500:],
        )
        return LinkResult()
    reused = links.linked + links.deduplicated + links.symlinked
    if reused:
        logger.log(
            PROGRESS,
            "  Model cache on %s: %d file(s) reused from files already on the host%s",
            label,
            reused,
            ", %.2f GiB of duplicate copies reclaimed" % (links.reclaimed_bytes / 2**30) if links.reclaimed_bytes else "",
        )
    if links.symlinked or links.unlinkable:
        # A property of the host (owner under fs.protected_hardlinks, or another
        # filesystem), not a fault: it recurs every launch, so INFO, once.
        logger.info(
            "Model cache on %s: cached files could not be hard-linked (%s); %d file(s) linked to the blob store instead,"
            " %d transferred. Bytes linked this way are freed only when no other revision uses them%s",
            label,
            links.reason or "unknown reason",
            links.symlinked,
            links.unlinkable,
            "; `sparkrun setup permissions` restores hard-link reuse" if "another user" in links.reason else "",
        )
    if links.failed:
        logger.warning("Model cache on %s: %d file(s) could not be linked; they will be transferred", label, links.failed)
    return links


__all__ = [
    "LinkResult",
    "TEMPORARY_PREFIX",
    "link_existing_blobs",
    "link_script",
    "parse_links",
    "parse_saved_manifests",
    "saved_manifests_script",
    "sibling_sources",
]
