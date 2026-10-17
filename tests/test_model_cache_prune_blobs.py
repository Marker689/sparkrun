"""Prune frees a Hub-layout revision's space: the blobs only it links to go with it.

Prune used to remove only the snapshot directory. For a revision whose entries
are symlinks into ``blobs/`` that reclaimed nothing and orphaned the blobs.
Real host-agent runs against fixture caches; no downloads.
"""

from __future__ import annotations

from dataclasses import replace
import fcntl
import json
from pathlib import Path

import pytest

from sparkrun.api import model_cache
from sparkrun.models import _cache_agent as agent
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.usage import record_use, usage_path
from test_model_artifacts import artifact
from test_model_cache_delete import REV_A, REV_B, hf_repo, shared_blob

SHARED = b"s" * 30000
ONLY_A = b"a" * 30000
ONLY_B = b"b" * 30000


def track(root, revision, files, *, shared=False, last_used=1):
    """A Hub-layout revision that sparkrun tracks (it has a usage record)."""
    repo = hf_repo(root, "org/model", revision, files, shared=shared)
    manifest = replace(artifact(files), revision=revision)
    with ModelHostIO() as io:
        record_use(io, "localhost", str(root), manifest)
    path = Path(usage_path(root, manifest.repo, revision))
    record = json.loads(path.read_text())
    record["last_used"] = last_used
    path.write_text(json.dumps(record))
    return repo


def prune(root, **kwargs):
    return model_cache.prune_cache(cache_dir=str(root), min_age_days=0, dry_run=False, **kwargs)[0]


def blob(repo, data):
    import hashlib

    return repo / "blobs" / hashlib.sha256(data).hexdigest()


def test_evicting_a_hub_layout_revision_frees_its_blobs(tmp_path):
    repo = track(tmp_path, REV_A, {"w.safetensors": ONLY_A})

    report = prune(tmp_path)

    assert not (repo / "snapshots" / REV_A).exists()
    assert not blob(repo, ONLY_A).exists()
    assert report["reclaimed_bytes"] >= len(ONLY_A)


def test_a_blob_another_revision_links_to_is_kept(tmp_path):
    from sparkrun.models.artifacts import ModelArtifactRequest

    track(tmp_path, REV_A, {"w.safetensors": ONLY_A, "shared.bin": SHARED}, last_used=1)
    repo = track(tmp_path, REV_B, {"w.safetensors": ONLY_B, "shared.bin": SHARED}, last_used=2)
    model_cache.pin_revision(ModelArtifactRequest("org/model", REV_B), cache_dir=str(tmp_path))
    before = model_cache.inventory_cache(cache_dir=str(tmp_path))[0]
    reclaimable = {e.revision: e.reclaimable_bytes for e in before.entries}

    prune(tmp_path)

    assert not (repo / "snapshots" / REV_A).exists() and (repo / "snapshots" / REV_B).exists()
    assert not blob(repo, ONLY_A).exists()
    assert blob(repo, SHARED).exists() and blob(repo, ONLY_B).exists()
    assert (repo / "snapshots" / REV_B / "shared.bin").read_bytes() == SHARED
    # The inventory counted exclusive blobs only: the shared one is in neither row.
    assert len(ONLY_A) <= reclaimable[REV_A] < len(ONLY_A) + len(SHARED)


def test_shared_store_payloads_follow_huggingface_hubs_protocol(tmp_path):
    repo = track(tmp_path, REV_A, {"w.safetensors": ONLY_A}, shared=True)
    payload = shared_blob(tmp_path, ONLY_A)
    assert payload.exists()

    report = prune(tmp_path)

    assert not blob(repo, ONLY_A).is_symlink() and not payload.exists()
    assert report["reclaimed_bytes"] >= len(ONLY_A)


def test_a_payload_another_repository_references_is_kept(tmp_path):
    repo = track(tmp_path, REV_A, {"w.safetensors": ONLY_A}, shared=True)
    hf_repo(tmp_path, "org/other", REV_B, {"w.safetensors": ONLY_A}, shared=True)  # same content, untracked

    prune(tmp_path)

    assert not (repo / "snapshots" / REV_A).exists() and not blob(repo, ONLY_A).is_symlink()
    assert shared_blob(tmp_path, ONLY_A).exists()


@pytest.mark.parametrize("action", ["prune", "delete"])
def test_unwritable_shared_lock_keeps_payload_and_finishes_bookkeeping(tmp_path, monkeypatch, action):
    repo = track(tmp_path, REV_A, {"w.safetensors": ONLY_A}, shared=True)
    payload = shared_blob(tmp_path, ONLY_A)
    lock = payload.with_name(payload.name + ".lock")
    original = agent.os.open

    def denied(path, *args, **kwargs):
        if Path(path) == lock:
            raise PermissionError("shared store is owned by another user")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(agent.os, "open", denied)
    report = agent.run(tmp_path, action=action, repo="org/model", dry_run=False, min_age_seconds=0)
    assert not (repo / "snapshots" / REV_A).exists()
    assert payload.read_bytes() == ONLY_A
    assert report["reclaimed_bytes"] == 0
    record = Path(usage_path(tmp_path, "org/model", REV_A))
    if action == "prune":
        assert json.loads(record.read_text())["last_evicted"] > 0
    else:
        assert not record.exists()


def test_prune_refuses_while_a_download_of_the_model_is_active(tmp_path):
    repo = track(tmp_path, REV_A, {"w.safetensors": ONLY_A})
    locks = tmp_path / "hub/.locks/models--org--model"
    locks.mkdir(parents=True)
    with open(locks / "x.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        report = prune(tmp_path)

    (row,) = report["entries"]
    assert row["reason_code"] == "download_in_progress"
    assert blob(repo, ONLY_A).exists() and (repo / "snapshots" / REV_A).exists()


def test_an_unreadable_sibling_makes_nothing_exclusive(tmp_path):
    repo = track(tmp_path, REV_A, {"w.safetensors": ONLY_A})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (repo / "snapshots" / ("c" * 40)).symlink_to(elsewhere)

    plan = agent.plan_removal(agent.scan_removal(tmp_path, "models--org--model", [REV_A]))

    assert plan.scan.blobs == () and plan.scan.targets == {} and plan.planned_bytes == 0


def test_a_hard_linked_sibling_protects_the_blob(tmp_path):
    """A sparkrun revision reusing the blob by hard link is a reference like any symlink."""
    import os

    repo = track(tmp_path, REV_A, {"w.safetensors": ONLY_A})
    newer = repo / "snapshots" / REV_B
    newer.mkdir()
    os.link(blob(repo, ONLY_A), newer / "w.safetensors")

    plan = agent.plan_removal(agent.scan_removal(tmp_path, "models--org--model", [REV_A]))
    assert plan.scan.blobs == () and plan.scan.targets == {} and plan.planned_bytes == 0
    prune(tmp_path)
    assert (newer / "w.safetensors").read_bytes() == ONLY_A and blob(repo, ONLY_A).exists()


def test_a_sibling_link_spelled_differently_still_protects_the_blob(tmp_path):
    """Absolute, or through a symlinked cache path: identity decides, not spelling."""
    import os

    repo = track(tmp_path, REV_A, {"w.safetensors": ONLY_A})
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path)
    newer = repo / "snapshots" / REV_B
    newer.mkdir()
    (newer / "abs.safetensors").symlink_to(alias / "hub/models--org--model/blobs" / blob(repo, ONLY_A).name)

    prune(tmp_path)

    assert not (repo / "snapshots" / REV_A).exists()
    assert (newer / "abs.safetensors").read_bytes() == ONLY_A
    assert os.path.exists(blob(repo, ONLY_A))


def test_a_blob_directory_prune_cannot_change_is_neither_credited_nor_fatal(tmp_path):
    """A container's root-owned blobs/: evict the snapshot, keep the blob, finish the run."""
    import os

    repo = track(tmp_path, REV_A, {"w.safetensors": ONLY_A})
    os.chmod(repo / "blobs", 0o555)
    try:
        assert agent.scan_removal(tmp_path, "models--org--model", [REV_A]).blobs == ()
        report = prune(tmp_path)
    finally:
        os.chmod(repo / "blobs", 0o755)
    assert not (repo / "snapshots" / REV_A).exists() and blob(repo, ONLY_A).exists()
    assert report["entries"][0]["action"] == "removed"


def test_a_blob_hugging_face_is_busy_with_is_kept_and_not_reported_freed(tmp_path):
    repo = track(tmp_path, REV_A, {"w.safetensors": ONLY_A})
    locks = tmp_path / "hub/.locks/models--org--model"
    locks.mkdir(parents=True)
    plan = agent.plan_removal(agent.scan_removal(tmp_path, "models--org--model", [REV_A]))
    with open(locks / (blob(repo, ONLY_A).name + ".lock"), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        kept = agent.remove_exclusive_blobs(tmp_path, "models--org--model", plan)
    assert blob(repo, ONLY_A).exists() and blob(repo, ONLY_A) in kept
    assert plan.planned_bytes > 0 and plan.reclaimed_bytes(kept) == 0


def test_a_link_made_while_pruning_is_seen_before_the_unlink(tmp_path):
    """The re-read under the per-blob lock catches a revision that linked the blob meanwhile."""
    import os

    repo = track(tmp_path, REV_A, {"w.safetensors": ONLY_A})
    plan = agent.plan_removal(agent.scan_removal(tmp_path, "models--org--model", [REV_A]))
    newer = repo / "snapshots" / REV_B
    newer.mkdir()
    (newer / "w.safetensors").symlink_to(os.path.relpath(blob(repo, ONLY_A), newer))

    agent.remove_exclusive_blobs(tmp_path, "models--org--model", plan)

    assert (newer / "w.safetensors").read_bytes() == ONLY_A


def test_the_budget_is_recounted_as_blobs_change_hands(tmp_path):
    """Shared by A and B, the blob is private to neither; once A goes it is B's, and the budget sees it.

    Subtracting A's figure from a total taken at the start would declare this
    budget met with B, and the shared blob, still over it.
    """
    track(tmp_path, REV_A, {"w.safetensors": ONLY_A, "shared.bin": SHARED}, last_used=1)
    repo = track(tmp_path, REV_B, {"w.safetensors": ONLY_B, "shared.bin": SHARED}, last_used=2)
    start = sum(e.reclaimable_bytes for e in model_cache.inventory_cache(cache_dir=str(tmp_path))[0].entries)

    report = prune(tmp_path, max_bytes=start - 1)

    # After A, B privately holds w + shared, which is more than the budget: B goes too.
    assert not (repo / "snapshots" / REV_B).exists() and not blob(repo, SHARED).exists()
    assert report["target_met"]


@pytest.mark.parametrize("dry_run", [True, False])
def test_the_preview_predicts_the_apply(tmp_path, dry_run):
    track(tmp_path, REV_A, {"w.safetensors": ONLY_A})
    preview = model_cache.prune_cache(cache_dir=str(tmp_path), min_age_days=0)[0]
    assert preview["planned_bytes"] >= len(ONLY_A)
    if not dry_run:
        assert prune(tmp_path)["reclaimed_bytes"] == preview["planned_bytes"]
