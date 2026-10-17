"""Explicit model-cache delete: what is removed, what is kept, and what refuses."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from sparkrun.api import model_cache
from sparkrun.cli import _model_cache as cli
from sparkrun.models.artifacts import ModelArtifactRequest
from sparkrun.models.cache_reports import CacheReport
from test_model_cache_retention import tracked
from test_model_artifacts import SHA

REV_A = "a" * 40
REV_B = "b" * 40
SHARED = b"s" * 20000
ONLY_A = b"a" * 20000
ONLY_B = b"b" * 20000


def hf_repo(root, repo, revision, files, *, shared=False, ref=None):
    """Lay out one snapshot the way huggingface_hub does (optionally with the shared blob store)."""
    repo_dir = root / "hub" / ("models--" + repo.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    (repo_dir / "blobs").mkdir(exist_ok=True)
    for name, data in files.items():
        digest = hashlib.sha256(data).hexdigest()
        blob = repo_dir / "blobs" / digest
        if not blob.exists():
            if shared:
                # huggingface_hub's layout-1 store: marker, payload, and a .refs manifest.
                store = root / "hub" / "blobs"
                store.mkdir(parents=True, exist_ok=True)
                (store / ".huggingface-shared-blobs").write_text("1\n")
                backing = store / digest[:2] / digest
                backing.parent.mkdir(parents=True, exist_ok=True)
                if not backing.exists():
                    backing.write_bytes(data)
                blob.symlink_to(os.path.relpath(backing, blob.parent))
                with open(backing.with_name(digest + ".refs"), "a") as manifest:
                    manifest.write(blob.relative_to(root / "hub").as_posix() + "\n")
            else:
                blob.write_bytes(data)
        (snapshot / name).symlink_to(os.path.relpath(blob, snapshot))
    if ref:
        (repo_dir / "refs").mkdir(exist_ok=True)
        (repo_dir / "refs" / ref).write_text(revision)
    return repo_dir


def shared_blob(root, data):
    digest = hashlib.sha256(data).hexdigest()
    return root / "hub" / "blobs" / digest[:2] / digest


def delete(root, model, revision=None, **kwargs):
    return model_cache.delete_model(ModelArtifactRequest(model, revision), cache_dir=str(root), **kwargs)[0]


def test_preview_is_default_and_changes_nothing(tmp_path):
    repo = hf_repo(tmp_path, "org/model", REV_A, {"w.safetensors": ONLY_A})
    report = delete(tmp_path, "org/model")
    assert report.dry_run and report.target_met
    assert [e.action for e in report.entries] == ["would_delete"]
    assert report.entries[0].reason_code == "explicit_delete"
    assert report.planned_bytes > 0 and report.reclaimed_bytes == 0
    assert repo.exists()


def test_revision_delete_keeps_blobs_other_revisions_use(tmp_path):
    repo = hf_repo(tmp_path, "org/model", REV_A, {"shared.bin": SHARED, "a.bin": ONLY_A}, ref="main")
    hf_repo(tmp_path, "org/model", REV_B, {"shared.bin": SHARED, "b.bin": ONLY_B})
    report = delete(tmp_path, "org/model", REV_A, dry_run=False)
    assert [e.action for e in report.entries] == ["deleted"]
    assert report.reclaimed_bytes > 0
    blobs = {p.name for p in (repo / "blobs").iterdir()}
    assert blobs == {hashlib.sha256(SHARED).hexdigest(), hashlib.sha256(ONLY_B).hexdigest()}
    assert not (repo / "snapshots" / REV_A).exists()
    assert (repo / "snapshots" / REV_B / "shared.bin").read_bytes() == SHARED
    assert not (repo / "refs" / "main").exists()  # it pointed at the deleted revision


@pytest.mark.parametrize("reference", ["alias", "hardlink", "snapshot_symlink"])
def test_revision_delete_protects_sibling_references_by_identity(tmp_path, reference):
    repo = hf_repo(tmp_path, "org/model", REV_A, {"shared.bin": SHARED, "a.bin": ONLY_A})
    hf_repo(tmp_path, "org/model", REV_B, {"shared.bin": SHARED})
    sibling = repo / "snapshots" / REV_B
    entry = sibling / "shared.bin"
    blob = repo / "blobs" / hashlib.sha256(SHARED).hexdigest()
    if reference == "snapshot_symlink":
        moved = tmp_path / "moved-snapshot"
        sibling.rename(moved)
        (moved / "shared.bin").unlink()
        (moved / "shared.bin").symlink_to(blob)
        sibling.symlink_to(moved)
    else:
        entry.unlink()
        if reference == "alias":
            alias = tmp_path / "alias"
            alias.symlink_to(tmp_path)
            entry.symlink_to(alias / blob.relative_to(tmp_path))
        else:
            entry.hardlink_to(blob)
    preview = delete(tmp_path, "org/model", REV_A)
    report = delete(tmp_path, "org/model", REV_A, dry_run=False)
    assert preview.target_met and report.target_met
    assert not (repo / "snapshots" / REV_A).exists()
    assert blob.exists()
    assert entry.read_bytes() == SHARED


def test_revision_delete_keeps_blobs_when_a_sibling_cannot_be_statted(tmp_path, monkeypatch):
    from sparkrun.models import _cache_agent as agent

    repo = hf_repo(tmp_path, "org/model", REV_A, {"shared.bin": SHARED})
    hf_repo(tmp_path, "org/model", REV_B, {"shared.bin": SHARED})
    entry = repo / "snapshots" / REV_B / "shared.bin"
    original = Path.stat

    def denied(path, *args, **kwargs):
        if path == entry and kwargs.get("follow_symlinks", True):
            raise PermissionError("cannot resolve sibling entry")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    report = agent.run(tmp_path, action="delete", repo="org/model", revision=REV_A, dry_run=False)
    assert report["target_met"]
    assert entry.read_bytes() == SHARED


def test_revision_sharing_weights_reports_what_it_frees(tmp_path):
    from sparkrun.utils.cli_formatters import format_model_cache_reports

    hf_repo(tmp_path, "org/model", REV_A, {"weights.bin": SHARED * 50, "a.bin": ONLY_A})
    hf_repo(tmp_path, "org/model", REV_B, {"weights.bin": SHARED * 50, "b.bin": ONLY_B})
    report = delete(tmp_path, "org/model", REV_A)
    (row,) = report.entries
    assert row.logical_bytes == len(SHARED) * 50 + len(ONLY_A)
    assert 0 < row.reclaimable_bytes == report.planned_bytes < len(SHARED) * 50
    text = format_model_cache_reports([report], action="delete")
    assert "stays: those files are shared with other cached revisions" in text

    whole = delete(tmp_path, "org/model")
    assert whole.planned_bytes >= len(SHARED) * 50
    assert sum(e.reclaimable_bytes for e in whole.entries) == whole.planned_bytes
    assert "stays" not in format_model_cache_reports([whole], action="delete")
    untracked = delete(tmp_path, "org/model", REV_B)
    assert not untracked.entries[0].tracked
    from sparkrun.utils.cli_formatters import format_bytes

    row_line = format_model_cache_reports([untracked], action="delete").splitlines()[2]
    assert format_bytes(untracked.entries[0].logical_bytes) in row_line


def test_revision_may_be_named_by_ref(tmp_path):
    repo = hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A}, ref="main")
    hf_repo(tmp_path, "org/model", REV_B, {"b.bin": ONLY_B})
    report = delete(tmp_path, "org/model", "main", dry_run=False)
    assert [e.revision for e in report.entries] == [REV_A]
    assert (repo / "snapshots" / REV_B).exists()


def test_whole_model_delete_removes_repo_and_usage_records(tmp_path):
    _, snapshot, usage = tracked(tmp_path)
    report = delete(tmp_path, "org/model", dry_run=False)
    assert report.target_met and [e.action for e in report.entries] == ["deleted"]
    assert not (tmp_path / "hub" / "models--org--model").exists()
    assert not usage.exists() and not usage.parent.exists()
    assert model_cache.inventory_cache(cache_dir=str(tmp_path))[0].entries == ()


def test_pinned_and_untracked_revisions_are_deleted_explicitly(tmp_path):
    tracked(tmp_path)
    model_cache.pin_revision(ModelArtifactRequest("org/model", SHA), cache_dir=str(tmp_path))
    hf_repo(tmp_path, "org/loose", REV_A, {"a.bin": ONLY_A})
    assert delete(tmp_path, "org/model", dry_run=False).target_met
    assert delete(tmp_path, "org/loose", dry_run=False).target_met
    assert not list((tmp_path / "hub").glob("models--*"))


def test_shared_store_blob_survives_while_another_repo_links_it(tmp_path):
    hf_repo(tmp_path, "org/one", REV_A, {"common.bin": SHARED, "mine.bin": ONLY_A}, shared=True)
    hf_repo(tmp_path, "org/two", REV_B, {"common.bin": SHARED}, shared=True)
    delete(tmp_path, "org/one", dry_run=False)
    assert shared_blob(tmp_path, SHARED).exists()
    assert not shared_blob(tmp_path, ONLY_A).exists()
    assert (tmp_path / "hub" / "models--org--two" / "snapshots" / REV_B / "common.bin").read_bytes() == SHARED
    delete(tmp_path, "org/two", dry_run=False)
    assert not shared_blob(tmp_path, SHARED).exists()


def test_targets_outside_the_cache_are_never_removed(tmp_path):
    outside = tmp_path / "elsewhere.bin"
    outside.write_bytes(ONLY_A)
    snapshot = tmp_path / "hf" / "hub" / "models--org--model" / "snapshots" / REV_A
    snapshot.mkdir(parents=True)
    (snapshot / "w.bin").symlink_to(outside)
    delete(tmp_path / "hf", "org/model", dry_run=False)
    assert outside.read_bytes() == ONLY_A


def test_claims_block_every_revision_on_the_host_unless_forced(tmp_path):
    from sparkrun.models.usage import usage_path

    tracked(tmp_path, revision=REV_A)
    tracked(tmp_path, revision=REV_B)
    path = Path(usage_path(tmp_path, "org/model", REV_A))
    record = json.loads(path.read_text())
    record["claims"] = {"claim-1": {"workload": "wl", "hosts": ["n1"], "created": 1}}
    path.write_text(json.dumps(record))

    report = delete(tmp_path, "org/model", dry_run=False)
    assert not report.target_met
    assert {e.revision: e.reason_code for e in report.entries} == {REV_A: "workload_claim", REV_B: "delete_blocked"}
    assert all(e.action is None for e in report.entries)
    assert len(list((tmp_path / "hub" / "models--org--model" / "snapshots").iterdir())) == 2

    assert delete(tmp_path, "org/model", dry_run=False, force=True).target_met
    assert not (tmp_path / "hub" / "models--org--model").exists()


def test_active_hf_download_refuses(tmp_path):
    repo = hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A})
    lock = tmp_path / "hub" / ".locks" / "models--org--model" / "abc.lock"
    lock.parent.mkdir(parents=True)
    lock.touch()
    with open(lock) as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        report = delete(tmp_path, "org/model", dry_run=False)
    assert not report.target_met and report.entries[0].reason_code == "download_in_progress"
    assert repo.exists()
    assert delete(tmp_path, "org/model", dry_run=False).target_met  # lock released


def test_uncached_model_reports_no_entries(tmp_path):
    report = delete(tmp_path, "org/absent")
    assert report.entries == () and report.target_met
    assert not (tmp_path / "hub").exists()


def test_apply_deletes_nowhere_when_any_host_refuses(monkeypatch):
    calls = []

    def operation(*, dry_run, **kwargs):
        calls.append(dry_run)
        refused = SimpleNamespace(target_met=False)
        return (SimpleNamespace(target_met=True), refused)

    monkeypatch.setattr("sparkrun.models.cache.cache_operation", operation)
    # The apply verifies its grouping once, before the gating preview.
    monkeypatch.setattr("sparkrun.models.cache.resolve_cache_locations", lambda **kwargs: ())
    reports = model_cache.delete_model(ModelArtifactRequest("org/model"), cache_dir="/c", hosts=("a", "b"), dry_run=False)
    assert calls == [True]
    assert not reports[1].target_met


def test_delete_report_matches_the_wire_schema(tmp_path):
    from sparkrun.api._wire import to_wire, validate
    from sparkrun.api._wire_schemas import RESULT_SCHEMAS

    tracked(tmp_path)
    validate(
        to_wire(model_cache.delete_model(ModelArtifactRequest("org/model"), cache_dir=str(tmp_path))), RESULT_SCHEMAS["model_cache.delete"]
    )
    validate(
        to_wire(model_cache.delete_model(ModelArtifactRequest("org/model"), cache_dir=str(tmp_path), dry_run=False)),
        RESULT_SCHEMAS["model_cache.delete"],
    )


@pytest.fixture
def invoke(monkeypatch, tmp_path):
    config = SimpleNamespace(ssh_user=None, ssh_key=None, ssh_options=None, default_hosts=None, hf_cache_dir=str(tmp_path))
    clusters = Mock(get_default=Mock(return_value=None))
    monkeypatch.setattr(cli, "_get_context", lambda ctx: SimpleNamespace(config=config, cluster_manager=clusters))

    def run(*args, tty=False, input=None):
        monkeypatch.setattr(cli, "_interactive", lambda: tty)
        return CliRunner().invoke(cli.model_cache, ["delete", *args], input=input)

    return run


def test_cli_dry_run_previews_without_deleting(invoke, tmp_path):
    repo = hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A})
    result = invoke("org/model", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "would delete" in result.output and "Would delete 1 revision(s)" in result.output
    assert repo.exists()


def test_cli_requires_confirmation_off_a_terminal(invoke, tmp_path):
    repo = hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A})
    result = invoke("org/model")
    assert result.exit_code != 0 and "--yes" in result.output
    assert repo.exists()


def test_cli_prompt_and_yes(invoke, tmp_path):
    repo = hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A})
    declined = invoke("org/model", tty=True, input="n\n")
    assert declined.exit_code != 0 and repo.exists()
    accepted = invoke("org/model", tty=True, input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert "Deleted 1 revision(s)" in accepted.output and not repo.exists()

    repo = hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A})
    result = invoke("org/model", "--yes", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["entries"][0]["action"] == "deleted"
    assert not repo.exists()


def test_cli_uncached_model_exits_nonzero(invoke):
    result = invoke("org/absent", "--yes")
    assert result.exit_code == 1
    assert "not cached on the target hosts" in result.output


def test_cli_refusal_exits_nonzero_without_prompting(invoke, monkeypatch):
    refused = Mock(return_value=(CacheReport.from_dict(_refused_report()),))
    monkeypatch.setattr(cli.api, "delete_model", refused)
    result = invoke("org/model", tty=True)
    assert result.exit_code == 1
    assert "Nothing deleted on this host" in result.output
    assert refused.call_count == 1  # preview only


def _refused_report():
    entry = dict(
        repo="org/model",
        revision=REV_A,
        snapshot="/c/hub/models--org--model/snapshots/" + REV_A,
        tracked=True,
        present=True,
        last_used=1.0,
        logical_bytes=0,
        reclaimable_bytes=0,
        pinned=False,
        claims={"c": {}},
        eligible=False,
        reason="workload claim",
        reason_code="workload_claim",
    )
    return dict(
        host="localhost",
        cache_root="/c",
        dry_run=True,
        entries=[entry],
        free_bytes=0,
        reclaimed_bytes=0,
        planned_bytes=0,
        target_met=False,
        budget_scope="private_managed_snapshots",
        reclaimed_bytes_kind="allocated_bytes_estimate",
        free_bytes_after_kind="projected",
    )


def test_preview_refuses_a_held_publication_lock_without_creating_one(tmp_path):
    hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A})
    state = tmp_path / ".sparkrun-model-artifacts"
    report = delete(tmp_path, "org/model")
    assert report.target_met and not state.exists()  # preview wrote nothing
    state.mkdir()
    lock = state / "org--model.lock"
    lock.touch()
    with open(lock) as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        preview = delete(tmp_path, "org/model")
    assert not preview.target_met and preview.entries[0].reason_code == "cache_busy"


@pytest.mark.parametrize("name", [".", "..", ".hidden"])
def test_dot_revisions_are_rejected(tmp_path, name):
    from sparkrun.models.artifacts import ModelArtifactError

    repo = hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A})
    with pytest.raises(ModelArtifactError, match="invalid revision"):
        delete(tmp_path, "org/model", name, dry_run=False)
    assert (repo / "snapshots" / REV_A).exists()


def test_dot_named_usage_records_are_never_selected(tmp_path):
    repo = hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A})
    usage = tmp_path / ".sparkrun-model-artifacts" / "usage" / "org--model"
    usage.mkdir(parents=True)
    (usage / "..json").write_text("{}")
    (usage / "...json").write_text("{}")
    report = delete(tmp_path, "org/model")
    assert [e.revision for e in report.entries] == [REV_A]
    assert repo.exists()


def test_shared_blob_linked_through_a_symlinked_repository_is_kept(tmp_path):
    hf_repo(tmp_path, "org/one", REV_A, {"common.bin": SHARED}, shared=True)
    hf_repo(tmp_path, "org/two", REV_B, {"common.bin": SHARED}, shared=True)
    moved = tmp_path / "other-disk" / "models--org--two"
    moved.parent.mkdir()
    (tmp_path / "hub" / "models--org--two").rename(moved)
    (tmp_path / "hub" / "models--org--two").symlink_to(moved)
    # The store links are relative, so re-point the moved repo's link at the store.
    blob = moved / "blobs" / hashlib.sha256(SHARED).hexdigest()
    blob.unlink()
    blob.symlink_to(shared_blob(tmp_path, SHARED))
    assert delete(tmp_path, "org/one", dry_run=False).target_met
    assert shared_blob(tmp_path, SHARED).exists()


def test_shared_blob_without_a_readable_manifest_is_kept(tmp_path):
    hf_repo(tmp_path, "org/one", REV_A, {"mine.bin": ONLY_A}, shared=True)
    target = shared_blob(tmp_path, ONLY_A)
    target.with_name(target.name + ".refs").unlink()
    report = delete(tmp_path, "org/one", dry_run=False)
    assert report.target_met and report.planned_bytes < len(ONLY_A)
    assert target.exists()  # leaked, never guessed


def test_shared_store_with_an_unknown_layout_is_not_touched(tmp_path):
    hf_repo(tmp_path, "org/one", REV_A, {"mine.bin": ONLY_A}, shared=True)
    (tmp_path / "hub" / "blobs" / ".huggingface-shared-blobs").write_text("2\n")
    delete(tmp_path, "org/one", dry_run=False)
    assert shared_blob(tmp_path, ONLY_A).exists()


def test_whole_model_delete_tolerates_a_missing_shared_payload(tmp_path):
    repo = hf_repo(tmp_path, "org/one", REV_A, {"mine.bin": ONLY_A}, shared=True)
    shared_blob(tmp_path, ONLY_A).unlink()
    report = delete(tmp_path, "org/one", dry_run=False)
    assert report.target_met and report.reclaimed_bytes == 0
    assert not repo.exists()


def test_shared_blob_held_by_huggingface_hub_is_kept(tmp_path, monkeypatch):
    from sparkrun.models import _cache_agent

    hf_repo(tmp_path, "org/one", REV_A, {"mine.bin": ONLY_A}, shared=True)
    target = shared_blob(tmp_path, ONLY_A)
    monkeypatch.setattr(_cache_agent, "SHARED_LOCK_TIMEOUT", 0.2)
    with open(target.with_name(target.name + ".lock"), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        _cache_agent.sweep_shared(target, tmp_path / "hub", tmp_path / "hub" / "blobs")
    assert target.exists()


def test_stale_manifest_lines_do_not_protect_a_blob(tmp_path):
    hf_repo(tmp_path, "org/one", REV_A, {"mine.bin": ONLY_A}, shared=True)
    target = shared_blob(tmp_path, ONLY_A)
    with open(target.with_name(target.name + ".refs"), "a") as manifest:
        manifest.write("models--org--gone/blobs/" + target.name + "\n../escape\n/abs/path\n")
    delete(tmp_path, "org/one", dry_run=False)
    assert not target.exists() and not target.with_name(target.name + ".refs").exists()


def test_symlinked_snapshot_is_unlinked_not_followed(tmp_path):
    repo = hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.bin").write_bytes(ONLY_B)
    (repo / "snapshots" / REV_B).symlink_to(elsewhere)
    report = delete(tmp_path, "org/model", REV_B, dry_run=False)
    assert report.target_met
    assert not (repo / "snapshots" / REV_B).is_symlink()
    assert (elsewhere / "keep.bin").read_bytes() == ONLY_B
    assert (repo / "snapshots" / REV_A / "a.bin").exists()


def test_revisions_added_after_confirmation_are_refused(tmp_path):
    repo = hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A})
    hf_repo(tmp_path, "org/model", REV_B, {"b.bin": ONLY_B})
    reports = model_cache.delete_model(ModelArtifactRequest("org/model"), cache_dir=str(tmp_path), dry_run=False, allowed_revisions={REV_A})
    assert not reports[0].target_met
    assert {e.reason_code for e in reports[0].entries} == {"selection_changed"}
    assert (repo / "snapshots" / REV_A).exists()


def test_daemon_delete_preserves_confirmed_revisions(tmp_path):
    from sparkrun import api
    from sparkrun.core.daemon_registry import get_daemon_op
    from sparkrun.daemon.ops import check_request, register_core_ops

    repo = hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A})
    hf_repo(tmp_path, "org/model", REV_B, {"b.bin": ONLY_B})
    register_core_ops()
    op = get_daemon_op("model_cache.delete")
    arguments = {
        "model": "org/model",
        "cache_dir": str(tmp_path),
        "dry_run": False,
        "allowed_revisions": [REV_A],
    }
    result = op.handler_loader()(check_request(op, arguments), api.default_sctx())
    assert not result[0]["target_met"]
    assert {e["reason_code"] for e in result[0]["entries"]} == {"selection_changed"}
    assert (repo / "snapshots" / REV_A).exists() and (repo / "snapshots" / REV_B).exists()
    for invalid in (None, REV_A, [None], [1]):
        with pytest.raises(ValueError):
            check_request(op, {**arguments, "allowed_revisions": invalid})


def test_fifo_ref_does_not_hang(tmp_path):
    repo = hf_repo(tmp_path, "org/model", REV_A, {"a.bin": ONLY_A})
    (repo / "refs").mkdir()
    os.mkfifo(repo / "refs" / "main")
    assert delete(tmp_path, "org/model", "main").entries == ()
    assert delete(tmp_path, "org/model", REV_A, dry_run=False).target_met


def test_blob_only_repository_can_be_deleted(tmp_path):
    repo = tmp_path / "hub" / "models--org--model"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "partial.incomplete").write_bytes(ONLY_A)
    preview = delete(tmp_path, "org/model")
    assert [e.revision for e in preview.entries] == ["-"]
    assert preview.planned_bytes > 0 and preview.entries[0].reclaimable_bytes == preview.planned_bytes
    assert delete(tmp_path, "org/model", dry_run=False).target_met
    assert not repo.exists()
