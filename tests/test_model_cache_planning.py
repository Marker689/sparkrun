"""Removal plans account for real allocations and survive changes at lock boundaries."""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fcntl")

from sparkrun.api import model_cache
from sparkrun.models import _cache_agent as agent
from sparkrun.models.artifacts import ModelArtifactRequest
from sparkrun.models.usage import usage_path
from test_model_cache_delete import REV_A, REV_B, hf_repo, shared_blob
from test_model_cache_prune_blobs import ONLY_A, blob, track


SLUG = "models--org--model"


def hardlinked_snapshot(root, *, shared):
    repo = track(root, REV_A, {"weights.bin": ONLY_A}, shared=shared)
    payload = shared_blob(root, ONLY_A) if shared else blob(repo, ONLY_A)
    entry = repo / "snapshots" / REV_A / "weights.bin"
    entry.unlink()
    os.link(payload, entry)
    return repo, payload, entry


def operation(root, name, *, dry_run):
    if name == "prune":
        return model_cache.prune_cache(cache_dir=str(root), min_age_days=0, dry_run=dry_run)[0]
    request = ModelArtifactRequest("org/model", REV_A if name == "revision" else None)
    return model_cache.delete_model(request, cache_dir=str(root), dry_run=dry_run)[0]


@pytest.mark.parametrize("shared", [False, True], ids=["classic", "shared-store"])
@pytest.mark.parametrize("name", ["prune", "revision", "model"])
@pytest.mark.parametrize("external", [False, True], ids=["last-names", "external-hardlink"])
def test_shared_accounting_counts_all_removed_names_once(tmp_path, shared, name, external):
    repo, payload, entry = hardlinked_snapshot(tmp_path, shared=shared)
    allocated = payload.stat().st_blocks * 512
    outside = tmp_path / "external-copy"
    if external:
        os.link(payload, outside)
    expected = 0 if external else allocated

    preview = operation(tmp_path, name, dry_run=True)
    assert preview.planned_bytes == sum(row.reclaimable_bytes for row in preview.entries) == expected
    assert entry.read_bytes() == ONLY_A and payload.exists()

    applied = operation(tmp_path, name, dry_run=False)
    assert applied.planned_bytes == applied.reclaimed_bytes == expected
    assert not entry.exists() and not blob(repo, ONLY_A).exists()
    if external:
        assert outside.read_bytes() == ONLY_A


def test_whole_model_attributes_a_shared_inode_to_only_one_revision(tmp_path):
    repo, payload, first = hardlinked_snapshot(tmp_path, shared=True)
    second = repo / "snapshots" / REV_B
    second.mkdir()
    os.link(payload, second / "weights.bin")
    allocated = payload.stat().st_blocks * 512
    report = operation(tmp_path, "model", dry_run=True)
    assert len(report.entries) == 2
    assert sum(row.logical_bytes for row in report.entries) == len(ONLY_A) * 2
    assert sum(row.reclaimable_bytes for row in report.entries) == report.planned_bytes == allocated
    assert operation(tmp_path, "model", dry_run=False).reclaimed_bytes == allocated
    assert not first.exists() and not payload.exists()


@pytest.mark.parametrize("action", ["prune", "delete"])
def test_retained_shared_payload_cancels_the_entire_hardlink_credit(tmp_path, monkeypatch, action):
    repo, payload, entry = hardlinked_snapshot(tmp_path, shared=True)
    preview = agent.run(tmp_path, action=action, repo="org/model", min_age_seconds=0)
    assert preview["planned_bytes"] == payload.stat().st_blocks * 512 > 0
    lock = payload.with_name(payload.name + ".lock")
    original = agent.os.open

    def denied(path, *args, **kwargs):
        if Path(path) == lock:
            raise PermissionError("payload lock is owned by another user")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(agent.os, "open", denied)
    report = agent.run(tmp_path, action=action, repo="org/model", min_age_seconds=0, dry_run=False)
    assert report["planned_bytes"] == preview["planned_bytes"] and report["reclaimed_bytes"] == 0
    assert not entry.exists() and not blob(repo, ONLY_A).exists() and payload.read_bytes() == ONLY_A
    record = Path(usage_path(tmp_path, "org/model", REV_A))
    if action == "prune":
        assert json.loads(record.read_text())["last_evicted"] > 0
    else:
        assert not record.exists()


def test_planning_and_report_attribution_require_no_filesystem_access(tmp_path, monkeypatch):
    _, payload, _ = hardlinked_snapshot(tmp_path, shared=True)
    allocated = payload.stat().st_blocks * 512
    evidence = agent.scan_removal(tmp_path, SLUG, [REV_A])

    def forbidden(*args, **kwargs):
        raise AssertionError("planning must use the supplied observations")

    with monkeypatch.context() as m:
        m.setattr(Path, "stat", forbidden)
        m.setattr(Path, "lstat", forbidden)
        m.setattr(os, "open", forbidden)
        m.setattr(Path, "unlink", forbidden)
        plan = agent.plan_removal(evidence)
        assert plan.planned_bytes == allocated
        assert plan.row_bytes(REV_A) == (len(ONLY_A), allocated)
        assert plan.reclaimed_bytes({payload}) == 0
        result = agent.cache_report(tmp_path, [], dry_run=True, free=123)
        agent.describe_delete_plan(result, [{"revision": REV_A}], plan)
        assert result["entries"][0]["reclaimable_bytes"] == allocated


def test_unknown_payload_references_do_not_become_an_empty_reference_set(tmp_path):
    _, payload, entry = hardlinked_snapshot(tmp_path, shared=True)
    payload.with_name(payload.name + ".refs").unlink()
    preview = operation(tmp_path, "revision", dry_run=True)
    assert preview.planned_bytes == 0
    applied = operation(tmp_path, "revision", dry_run=False)
    assert applied.reclaimed_bytes == 0 and not entry.exists()
    assert payload.read_bytes() == ONLY_A


def test_reference_observations_belong_to_one_operation(tmp_path):
    repo = track(tmp_path, REV_A, {"weights.bin": ONLY_A})
    first = agent.ObservationContext()
    assert agent.scan_removal(tmp_path, SLUG, [REV_A], context=first).blobs
    hf_repo(tmp_path, "org/model", REV_B, {"weights.bin": ONLY_A})
    second = agent.ObservationContext()
    assert agent.scan_removal(tmp_path, SLUG, [REV_A], context=second).blobs == ()
    first.refresh()
    assert agent.scan_removal(tmp_path, SLUG, [REV_A], context=first).blobs == ()
    assert (repo / "snapshots" / REV_B / "weights.bin").read_bytes() == ONLY_A


def test_delete_reselects_after_acquiring_publication_lock(tmp_path, monkeypatch):
    repo = track(tmp_path, REV_A, {"weights.bin": ONLY_A})
    lock = tmp_path / agent.STATE / "org--model.lock"
    lock_identity = lock.stat().st_ino
    original = agent.fcntl.flock
    added = False

    def add_revision(fd, operation):
        nonlocal added
        original(fd, operation)
        if not added and os.fstat(fd).st_ino == lock_identity:
            added = True
            hf_repo(tmp_path, "org/model", REV_B, {"weights.bin": ONLY_A})

    monkeypatch.setattr(agent.fcntl, "flock", add_revision)
    report = agent.run(tmp_path, action="delete", repo="org/model", allowed=[REV_A], dry_run=False)
    assert added and not report["target_met"]
    assert {row["reason_code"] for row in report["entries"]} == {"selection_changed"}
    assert (repo / "snapshots" / REV_A / "weights.bin").read_bytes() == ONLY_A
    assert (repo / "snapshots" / REV_B / "weights.bin").read_bytes() == ONLY_A


def test_snapshot_disappearing_before_publication_lock_is_a_noop(tmp_path, monkeypatch):
    import shutil

    repo = track(tmp_path, REV_A, {"weights.bin": ONLY_A})
    lock_identity = (tmp_path / agent.STATE / "org--model.lock").stat().st_ino
    original = agent.fcntl.flock

    def remove_selection(fd, operation):
        original(fd, operation)
        if os.fstat(fd).st_ino == lock_identity:
            shutil.rmtree(repo)
            Path(usage_path(tmp_path, "org/model", REV_A)).unlink()

    monkeypatch.setattr(agent.fcntl, "flock", remove_selection)
    report = agent.run(tmp_path, action="delete", repo="org/model", dry_run=False)
    assert report["target_met"] and report["entries"] == []
    assert report["planned_bytes"] == report["reclaimed_bytes"] == 0


def test_empty_confirmation_refuses_every_present_revision(tmp_path):
    repo = track(tmp_path, REV_A, {"weights.bin": ONLY_A})
    report = model_cache.delete_model(ModelArtifactRequest("org/model"), cache_dir=str(tmp_path), dry_run=False, allowed_revisions=[])[0]
    assert not report.target_met and report.entries[0].reason_code == "selection_changed"
    assert (repo / "snapshots" / REV_A / "weights.bin").read_bytes() == ONLY_A
