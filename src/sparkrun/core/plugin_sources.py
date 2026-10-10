"""Acquire immutable plugin snapshots without executing repository code."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib

from sparkrun.core.plugin_manifest import PluginError, PluginMetadata, normalize_source, read_manifest, safe_relative

MAX_PAYLOAD_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class PluginCandidate:
    metadata: PluginMetadata
    root: Path
    source: dict
    revision: str

    @property
    def package_path(self) -> Path:
        return self.root / self.metadata.source


def run_git(*args: str, cwd: Path | None = None) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.ext.allow=never", *args],
            cwd=cwd,
            capture_output=True,
            timeout=120,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PluginError("Git acquisition failed: %s" % exc) from exc
    if result.returncode:
        # Credentials belong to helpers, never source URLs or rendered commands.
        raise PluginError("Git acquisition failed: %s" % result.stderr.decode(errors="replace")[-2000:].strip())
    return result.stdout


def resolve_ref(repository: Path, ref: str) -> tuple[str, str]:
    if not ref or ref.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_./-]+", ref) or ".." in ref:
        raise PluginError("Invalid Git ref %r" % ref)
    refs = run_git("for-each-ref", "--format=%(refname)", cwd=repository).decode().splitlines()
    if re.fullmatch(r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", ref):
        # Fetch also permits reachable commits not included by a shallow upstream.
        run_git("fetch", "--no-tags", "origin", ref, cwd=repository)
        selected, kind = ref, "commit"
    elif ref.startswith(("refs/heads/", "refs/tags/")):
        if ref not in refs:
            raise PluginError("Git ref %r does not exist; select an existing --ref" % ref)
        selected, kind = ref, "branch" if ref.startswith("refs/heads/") else "tag"
    else:
        matches = [r for r in ("refs/heads/" + ref, "refs/tags/" + ref) if r in refs]
        if len(matches) != 1:
            raise PluginError("Git ref %r is missing or ambiguous; use --ref refs/heads/NAME or refs/tags/NAME" % ref)
        selected = matches[0]
        kind = "branch" if selected.startswith("refs/heads/") else "tag"
    commit = run_git("rev-parse", "--verify", selected + "^{commit}", cwd=repository).decode().strip()
    if not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", commit):
        raise PluginError("Git did not resolve an immutable commit")
    return commit, kind


def _extract_snapshot(repository: Path, commit: str, destination: Path) -> None:
    for name in ("plugin.toml", "pyproject.toml"):
        raw = run_git("show", commit + ":" + name, cwd=repository)
        if len(raw) > 1024 * 1024:
            raise PluginError("Plugin metadata is too large")
        (destination / name).write_bytes(raw)
    try:
        manifest = tomllib.loads((destination / "plugin.toml").read_text())
        project = tomllib.loads((destination / "pyproject.toml").read_text())["project"]
        source = str(safe_relative(manifest["source"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise PluginError("Invalid source manifest: %s" % exc) from exc
    tracked = run_git("ls-tree", "-r", "-z", commit, cwd=repository).decode().split("\0")
    paths = []
    patterns = project.get("license-files", ["LICENSE*", "NOTICE*", "COPYING*", "LICENSES/*"])
    if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
        raise PluginError("project.license-files must be a list of repository paths/globs")
    for line in filter(None, tracked):
        mode_type, path = line.split("\t", 1)
        mode = mode_type.split()[0]
        if path.startswith(source + "/") or any(Path(path).match(pattern) for pattern in patterns):
            safe_relative(path)
            if mode not in {"100644", "100755"}:
                raise PluginError("Plugin export contains a symlink or submodule: %s" % path)
            if Path(path).suffix in {".so", ".pyd", ".dll", ".pyc"}:
                raise PluginError("Source-copy installation does not support compiled/generated payloads: %s" % path)
            paths.append(path)
    if source + "/__init__.py" not in paths:
        raise PluginError("plugin.toml source must contain a tracked __init__.py")
    archive = run_git("archive", "--format=tar", commit, "--", *paths, cwd=repository)
    if len(archive) > MAX_PAYLOAD_BYTES:
        raise PluginError("Plugin payload exceeds the 128 MiB limit")
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            path = destination / str(safe_relative(member.name.rstrip("/")))
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = tar.extractfile(member)
                if stream is None:
                    raise PluginError("Unreadable plugin archive member")
                with stream, path.open("wb") as output:
                    shutil.copyfileobj(stream, output)
                path.chmod(0o755 if member.mode & 0o111 else 0o644)
            else:
                raise PluginError("Plugin archive contains a link or special file")


@contextmanager
def acquire_git(source: str, ref: str = "main"):
    url = normalize_source(source, local=True)
    # Identity normalization must not rewrite transport paths on arbitrary Git servers.
    fetch_url = url if url.startswith("file://") else source
    with tempfile.TemporaryDirectory(prefix="sparkrun-plugin-source-") as temporary:
        root = Path(temporary)
        repository, export = root / "git", root / "export"
        export.mkdir()
        run_git("clone", "--bare", "--no-local", "--", fetch_url, str(repository))
        commit, kind = resolve_ref(repository, ref)
        _extract_snapshot(repository, commit, export)
        metadata = read_manifest(export)
        policy = {"kind": "git", "url": url, "fetch_url": fetch_url, "ref": ref, "ref_kind": kind}
        if kind in {"tag", "commit"}:
            policy["pinned_revision"] = commit
        yield PluginCandidate(metadata, export, policy, commit)
