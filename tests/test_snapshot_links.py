"""Reusing blobs a host already holds as hard-linked snapshot entries.

The pass is bash run on every host, so these run the real script under bash
against fixture caches; the end-to-end tests drive ``prepare_model`` through
the same local-host harness ``test_model_artifact_modes`` uses.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
from unittest import mock

import pytest

from sparkrun.models import preparation, snapshot_links, transport
from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.snapshot_links import LinkResult, link_existing_blobs, link_script, parse_links
from sparkrun.transports.session import HostCommandResult
from test_model_artifacts import DATA, artifact, cache, observe


def _blobs(root: Path, data=DATA, *, shared_store=False) -> Path:
    """An older revision fetched by ``hf download``: per-repo blobs named by checksum."""
    repo_blobs = root / "hub/models--org--model/blobs"
    repo_blobs.mkdir(parents=True, exist_ok=True)
    for contents in data.values():
        digest = hashlib.sha256(contents).hexdigest()
        if shared_store:
            backing = root / "hub/blobs" / digest[:2] / digest
            backing.parent.mkdir(parents=True, exist_ok=True)
            backing.write_bytes(contents)
            (repo_blobs / digest).symlink_to(os.path.relpath(backing, repo_blobs))
        else:
            (repo_blobs / digest).write_bytes(contents)
    return repo_blobs


def _run(manifest, root: Path, *, checksums=False) -> LinkResult:
    proc = subprocess.run(["bash"], input=link_script(manifest, str(root), checksums=checksums), text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    result = parse_links(proc.stdout)
    assert result is not None, proc.stdout
    return result


def _blob_of(root: Path, name: str) -> Path:
    return root / "hub/models--org--model/blobs" / hashlib.sha256(DATA[name]).hexdigest()


@pytest.mark.parametrize("shared_store", [False, True], ids=["classic", "shared-store"])
def test_missing_entries_become_hard_links_to_cached_blobs(tmp_path, shared_store):
    manifest = artifact()
    _blobs(tmp_path, shared_store=shared_store)

    result = _run(manifest, tmp_path)

    assert result == LinkResult(linked=len(DATA))
    snapshot = tmp_path / manifest.relative_snapshot
    for name, contents in DATA.items():
        entry = snapshot / name
        # A regular file sharing the blob's inode: the cache agent's model of a
        # sparkrun snapshot, with the bytes counted as shared, not reclaimable.
        assert entry.is_file() and not entry.is_symlink()
        assert os.path.samefile(entry, _blob_of(tmp_path, name))
        assert entry.stat().st_nlink == 2
        assert entry.read_bytes() == contents
    assert observe(manifest, snapshot).complete
    assert not list(snapshot.glob(snapshot_links.TEMPORARY_PREFIX + "*"))


def test_duplicate_copies_are_replaced_and_their_space_reported(tmp_path):
    manifest = artifact()
    _blobs(tmp_path)
    snapshot = cache(tmp_path, manifest)  # regular-file copies, as earlier runs left them

    result = _run(manifest, tmp_path)

    assert result.deduplicated == len(DATA) and result.linked == 0 and result.failed == 0
    assert result.reclaimed_bytes > 0
    for name in DATA:
        assert os.path.samefile(snapshot / name, _blob_of(tmp_path, name))
    # Idempotent: a second pass finds nothing to do.
    assert _run(manifest, tmp_path) == LinkResult()


def test_hf_style_symlink_entries_are_left_alone(tmp_path):
    manifest = artifact()
    snapshot = cache(tmp_path, manifest, shared=True)
    before = {name: os.readlink(snapshot / name) for name in DATA}

    assert _run(manifest, tmp_path) == LinkResult()
    assert {name: os.readlink(snapshot / name) for name in DATA} == before


def test_dangling_entry_is_relinked(tmp_path):
    manifest = artifact()
    _blobs(tmp_path)
    snapshot = tmp_path / manifest.relative_snapshot
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").symlink_to("../../blobs/" + "0" * 64)

    assert _run(manifest, tmp_path).linked == len(DATA)
    assert (snapshot / "config.json").read_bytes() == DATA["config.json"]


def test_a_wrong_size_blob_is_never_used(tmp_path):
    manifest = artifact()
    blob = _blob_of(tmp_path, "model.safetensors")
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"short")

    assert _run(manifest, tmp_path) == LinkResult()
    assert not (tmp_path / manifest.relative_snapshot / "model.safetensors").exists()


def test_checksum_policy_refuses_a_corrupt_blob_of_the_right_size(tmp_path):
    manifest = artifact()
    _blobs(tmp_path)
    corrupt = _blob_of(tmp_path, "model.safetensors")
    corrupt.write_bytes(b"x" * len(DATA["model.safetensors"]))

    result = _run(manifest, tmp_path, checksums=True)

    assert result.linked == len(DATA) - 1
    assert not (tmp_path / manifest.relative_snapshot / "model.safetensors").exists()
    # Without the policy the Hub's name for the blob is trusted, as hf_hub_download does.
    assert _run(manifest, tmp_path).linked == 1


def test_git_sha1_named_blobs_are_reused(tmp_path):
    data = {"config.json": b'{"a":1}'}
    git = hashlib.sha1(b"blob %d\0" % len(data["config.json"]) + data["config.json"]).hexdigest()
    base = artifact(data)
    manifest = replace(base, files=(replace(base.files[0], checksum_algorithm="git-sha1", checksum=git),))
    blobs = tmp_path / "hub/models--org--model/blobs"
    blobs.mkdir(parents=True)
    (blobs / git).write_bytes(data["config.json"])

    assert _run(manifest, tmp_path, checksums=True).linked == 1


def test_stale_temporaries_from_an_interrupted_pass_are_removed(tmp_path):
    manifest = artifact()
    snapshot = tmp_path / manifest.relative_snapshot
    snapshot.mkdir(parents=True)
    stale = snapshot / (snapshot_links.TEMPORARY_PREFIX + "123.0")
    stale.write_bytes(b"debris")

    _run(manifest, tmp_path)
    assert not stale.exists()


def test_no_repo_directory_is_a_clean_no_op(tmp_path):
    assert _run(artifact(), tmp_path) == LinkResult()


def test_a_failed_pass_never_fails_preparation(caplog):
    io = mock.Mock()
    io.execute.return_value = HostCommandResult("h1", 1, b"", b"boom")
    assert link_existing_blobs(io, "h1", "/hf", artifact()) == LinkResult()
    io.execute.side_effect = ModelArtifactError("model host session unavailable on h1")
    assert link_existing_blobs(io, "h1", "/hf", artifact()) == LinkResult()
    assert "Could not reuse cached blobs" in caplog.text


# ---------------------------------------------------------------------------
# prepare_model, end to end on local fixture "hosts"
# ---------------------------------------------------------------------------


def _local_hosts(monkeypatch, locations):
    copies, repairs = [], []

    class LocalHosts(ModelHostIO):
        def root(self, host, path):
            return str(locations[host])

        def execute(self, host, script, **kwargs):
            return super().execute(None, script, **kwargs)

        def lock(self, host, root, repo):
            return super().lock(None, root, repo)

    original = transport.copy_script

    def local_copy(request, target, ssh_kwargs, **kwargs):
        copies.append((target.host, target.required_files))
        return original(replace(request, source_host=None), replace(target, host="localhost"), {}, **kwargs)

    def acquire(io, host, request, root, **kwargs):
        repairs.append((host, kwargs["files"]))
        for name in kwargs["files"]:
            (Path(root) / kwargs["manifest"].relative_snapshot / name).write_bytes(DATA[name])

    monkeypatch.setattr(preparation, "ModelHostIO", LocalHosts)
    monkeypatch.setattr(transport, "copy_script", local_copy)
    monkeypatch.setattr(preparation, "run_adapter", acquire)
    return copies, repairs


def test_delegated_new_revision_reuses_every_hosts_blobs(tmp_path, monkeypatch):
    """The reported failure: a revision bump re-downloaded and re-copied ~450 GiB.

    Every host held the files unchanged as HF blobs from the previous
    revision; the head stored a second copy of each, the fan-out a third, and
    the disks filled. Now nothing is downloaded or copied at all.
    """
    manifest = artifact()
    locations = {None: tmp_path / "control", "head": tmp_path / "head", "worker": tmp_path / "worker"}
    for host in ("head", "worker"):
        _blobs(locations[host])
    copies, repairs = _local_hosts(monkeypatch, locations)

    prepared = preparation.prepare_model("org/model", ["head", "worker"], cache_dir="/hf", manifest=manifest, transfer_mode="delegated")

    assert all(binding.report.complete for binding in prepared.bindings)
    assert repairs == [] and copies == []
    for host in ("head", "worker"):
        for name in DATA:
            assert os.path.samefile(locations[host] / manifest.relative_snapshot / name, _blob_of(locations[host], name))


def test_delegated_transfers_only_what_no_host_holds(tmp_path, monkeypatch):
    manifest = artifact()
    locations = {None: tmp_path / "control", "head": tmp_path / "head", "worker": tmp_path / "worker"}
    _blobs(locations["head"], {k: v for k, v in DATA.items() if k != "tokenizer.json"})
    _blobs(locations["worker"], {k: v for k, v in DATA.items() if k != "config.json"})
    copies, repairs = _local_hosts(monkeypatch, locations)

    prepared = preparation.prepare_model("org/model", ["head", "worker"], cache_dir="/hf", manifest=manifest, transfer_mode="delegated")

    assert all(binding.report.complete for binding in prepared.bindings)
    assert repairs == [("head", ("tokenizer.json",))]
    assert copies == [("worker", ("config.json",))]


def test_an_unlinkable_blob_is_reported_with_its_cause(tmp_path):
    """EXDEV here; EPERM (fs.protected_hardlinks, a root-owned blob) takes the same path."""
    elsewhere = Path("/dev/shm")
    if not elsewhere.is_dir() or elsewhere.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("needs a second filesystem")
    manifest = artifact()
    blobs = tmp_path / "hub/models--org--model/blobs"
    blobs.mkdir(parents=True)
    with __import__("tempfile").TemporaryDirectory(dir=elsewhere) as other:
        for contents in DATA.values():
            digest = hashlib.sha256(contents).hexdigest()
            (Path(other) / digest).write_bytes(contents)
            (blobs / digest).symlink_to(Path(other) / digest)

        result = _run(manifest, tmp_path, checksums=True)

    assert result.unlinkable == len(DATA) and result.linked == 0 and result.failed == 0
    assert "cross-device" in result.reason.lower()
    assert not list((tmp_path / manifest.relative_snapshot).glob("*"))


def test_a_good_copy_is_never_replaced_by_a_corrupt_blob(tmp_path):
    """Without the checksum policy, replacing data we hold needs proof, not a name and a size."""
    manifest = artifact()
    _blobs(tmp_path)
    snapshot = cache(tmp_path, manifest)
    _blob_of(tmp_path, "model.safetensors").write_bytes(b"x" * len(DATA["model.safetensors"]))

    result = _run(manifest, tmp_path)

    assert result.deduplicated == len(DATA) - 1
    entry = snapshot / "model.safetensors"
    assert entry.read_bytes() == DATA["model.safetensors"] and entry.stat().st_nlink == 1


def test_whole_model_delete_credits_inodes_shared_by_its_snapshots(tmp_path):
    """The HF revision and the linked sparkrun revision are one inode: deleting both frees it."""
    from test_model_cache_delete import REV_A, REV_B, delete, hf_repo

    payload = b"w" * 20000
    hf_repo(tmp_path, "org/model", REV_A, {"w.safetensors": payload})
    newer = replace(artifact({"w.safetensors": payload}), revision=REV_B)
    assert _run(newer, tmp_path).linked == 1

    whole = delete(tmp_path, "org/model")
    assert whole["planned_bytes"] >= len(payload)
    # One revision alone frees nothing: the other still holds the inode.
    assert delete(tmp_path, "org/model", REV_A)["planned_bytes"] == 0
    assert delete(tmp_path, "org/model", REV_B)["planned_bytes"] == 0

    applied = delete(tmp_path, "org/model", dry_run=False)
    assert applied["reclaimed_bytes"] == whole["planned_bytes"]


def test_budget_prune_skips_a_snapshot_that_frees_nothing(tmp_path):
    """The least recently used snapshot is fully linked; evicting it would free nothing."""
    from sparkrun.api import model_cache
    from test_model_cache_retention import tracked

    _, linked, _ = tracked(tmp_path, last_used=1)
    _blobs(tmp_path)
    assert _run(artifact(), tmp_path).deduplicated == len(DATA)
    _, private, _ = tracked(tmp_path, revision="b" * 40, last_used=2)

    report = model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=0, max_bytes=0, dry_run=False)[0]

    assert linked.exists() and not private.exists()
    assert report["reclaimed_bytes"] > 0
