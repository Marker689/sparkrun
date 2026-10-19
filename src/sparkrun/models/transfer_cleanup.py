"""Discard resumable copies only after their selected snapshot is verified."""

from __future__ import annotations

import logging
from pathlib import PurePosixPath

from sparkrun.models.artifacts import CacheValidationReport, ModelArtifactError, ModelArtifactManifest
from sparkrun.models.observation import path_assignment
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)
PARTIAL_DIRECTORY = ".sparkrun-partial"


def cleanup_script(manifest: ModelArtifactManifest, cache_root: str) -> str:
    snapshot = str(PurePosixPath(cache_root) / manifest.relative_snapshot)
    names = {f.path for f in manifest.files}
    partials = []
    for name in sorted(names):
        path = PurePosixPath(name)
        partial = str(path.parent / PARTIAL_DIRECTORY / path.name)
        # A model can itself contain a file named like our staging area.
        if PARTIAL_DIRECTORY not in path.parts and partial not in names:
            partials.append(partial)
    script = "set -euo pipefail\nexport LC_ALL=C\n" + path_assignment("snap", snapshot)
    script += "partials=(" + " ".join(quote(p) for p in partials) + ")\n"
    script += "names=(" + " ".join(quote(p) for p in sorted(names)) + ")\n"
    return (
        script
        + r"""
pending=()
for name in "${partials[@]}"; do
    p="$snap/$name"
    [ -e "$p" ] || [ -L "$p" ] || continue
    # Never follow a staging-directory or ancestor symlink when deleting.
    parent=${p%/*}
    while [ -n "$parent" ]; do
        if [ -L "$parent" ]; then
            printf 'symlinked transfer staging directory: %s\n' "$parent" >&2
            exit 1
        fi
        parent=${parent%/*}
    done
    if [ ! -f "$p" ] || [ -L "$p" ]; then
        printf 'unexpected transfer partial file type: %s\n' "$p" >&2
        exit 1
    fi
    pending+=("$p")
done
[ "${#pending[@]}" -gt 0 ] || { printf '0\n'; exit 0; }
# A snapshot entry resolving into staging is live data, not a redundant copy.
# Resolve all entries before deleting anything, including cross-file links.
declare -A live=()
for name in "${names[@]}"; do
    IFS= read -r -d '' resolved < <(realpath -e -z -- "$snap/$name")
    live["$resolved"]=1
done
for p in "${pending[@]}"; do
    if [ -n "${live[$p]:-}" ]; then
        printf 'model entry still references transfer partial: %s\n' "$p" >&2
        exit 1
    fi
done
for p in "${pending[@]}"; do
    rm -f -- "$p"
    rmdir -- "${p%/*}" 2>/dev/null || true  # Other file selections may still need theirs.
done
printf '%d\n' "${#pending[@]}"
"""
    )


def cleanup_transfer_partials(io, host: str | None, cache_root: str, manifest: ModelArtifactManifest, report: CacheValidationReport):
    """Clean this selection under its held lease; failed transfers keep resume data."""
    if not report.complete or report.manifest_id != manifest.identity:
        raise ModelArtifactError("transfer cleanup requires a complete validation of this model manifest")
    io.lock(host, cache_root, manifest.repo)
    removed = int(io.checked(host, cleanup_script(manifest, cache_root), timeout=60).strip())
    if removed:
        logger.info("Removed %d completed model transfer partial(s) on %s", removed, host or "control")
