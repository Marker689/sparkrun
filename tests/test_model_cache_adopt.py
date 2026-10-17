"""model-cache adopt: track snapshots downloaded outside sparkrun.

The host agent runs for real against fixture caches (Hub layout from
``test_model_cache_delete.hf_repo``); the model origin is a stub, so nothing
here touches the network.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from sparkrun.api import model_cache
from sparkrun.models import _cache_agent as agent
from sparkrun.models.adoption import judge
from sparkrun.models.artifacts import ModelArtifactRequest, ModelInventoryUnavailable
from test_model_cache_delete import REV_A, REV_B, hf_repo

W = b"w" * 20000
CONFIG = b'{"model_type": "x"}'
FILES = {"model.safetensors": W, "config.json": CONFIG}


def sizes(files=FILES):
    return {name: len(data) for name, data in files.items()}


@pytest.fixture
def origin(monkeypatch):
    """The model origin's inventory per (repo, revision); unknown revisions are unavailable."""
    known = {}

    def fetch_inventory(request, token=None, cache=None):
        if (request.repo, request.revision) not in known:
            raise ModelInventoryUnavailable("Hugging Face inventory is unavailable for " + request.repo)
        files = known[(request.repo, request.revision)]
        return {"revision": request.revision, "files": [{"path": p, "size": s} for p, s in files.items()]}

    monkeypatch.setattr("sparkrun.models.manifest.fetch_inventory", fetch_inventory)
    return known


def adopt(root, model="org/model", revision=None, **kwargs):
    request = ModelArtifactRequest(model, revision) if model else None
    return model_cache.adopt_cache(request, cache_dir=str(root), **kwargs)[0]


def rows(report):
    return {(e.revision): e for e in report.entries}


# ---------------------------------------------------------------------------
# judge(): what may be adopted
# ---------------------------------------------------------------------------


def snapshot(entries):
    return {"repo": "org/model", "revision": REV_A, "snapshot": "/s", "entries": entries}


def entry(path, size, kind="blob"):
    return {"path": path, "size": size, "kind": kind, "mtime": 1.0}


def test_a_partial_download_is_adoptable():
    """One GGUF quant of many: the snapshot is a subset of its revision, which is normal."""
    verdict = judge(snapshot([entry("config.json", 3)]), offline=False, origin=lambda r, v: {"config.json": 3, "big.gguf": 10})
    assert verdict["reason_code"] == "adoptable"


@pytest.mark.parametrize(
    "entries, expected",
    [
        ([entry("stray.bin", 3)], "not part of revision"),
        ([entry("config.json", 4)], "differs in size"),
        ([entry("config.json", None, "dangling")], "broken link"),
        ([], "empty snapshot"),
    ],
)
def test_content_the_origin_does_not_vouch_for_is_refused(entries, expected):
    verdict = judge(snapshot(entries), offline=False, origin=lambda r, v: {"config.json": 3})
    assert verdict["reason_code"] == "not_adoptable" and expected in verdict["reason"]


def test_an_unreachable_origin_is_unverifiable_and_points_at_offline():
    def down(repo, revision):
        raise ModelInventoryUnavailable("Hugging Face inventory is unavailable for org/model")

    verdict = judge(snapshot([entry("config.json", 3)]), offline=False, origin=down)
    assert verdict["reason_code"] == "adopt_unverifiable" and "--offline" in verdict["reason"]


def test_offline_accepts_only_the_hub_layout():
    assert judge(snapshot([entry("a", 1), entry("b", 2)]), offline=True, origin=None)["reason_code"] == "adoptable"
    refused = judge(snapshot([entry("a", 1), entry("b", 2, "regular")]), offline=True, origin=None)
    assert refused["reason_code"] == "adopt_unverifiable" and "b is not one" in refused["reason"]


# ---------------------------------------------------------------------------
# the host agent's scan and adopt
# ---------------------------------------------------------------------------


def test_scan_lists_untracked_snapshots_with_their_entry_kinds(tmp_path):
    hf_repo(tmp_path, "org/model", REV_A, FILES, ref="main")
    plain = tmp_path / "hub/models--org--model/snapshots" / REV_B
    plain.mkdir(parents=True)
    (plain / "config.json").write_bytes(CONFIG)

    found = {s["revision"]: s for s in agent.scan(tmp_path)}

    assert {e["kind"] for e in found[REV_A]["entries"]} == {"blob"}
    assert {e["path"]: e["size"] for e in found[REV_A]["entries"]} == sizes()
    assert [e["kind"] for e in found[REV_B]["entries"]] == ["regular"]
    assert [s["revision"] for s in agent.scan(tmp_path, repo="org/model", revision="main")] == [REV_A]


def test_adopt_writes_a_record_that_tracks_every_file(tmp_path):
    hf_repo(tmp_path, "org/model", REV_A, FILES)

    row = agent.adopt(
        tmp_path,
        repo="org/model",
        revision=REV_A,
        files=scanned(tmp_path),
        endpoint="https://huggingface.co",
        last_used=5.0,
        pinned=False,
        now=10.0,
    )

    # Tracked, with no "untracked files": the record names exactly what is there.
    assert row["tracked"] and row["reason_code"] in {"eligible", "retention_age"}
    record = json.loads((tmp_path / ".sparkrun-model-artifacts/usage/org--model" / (REV_A + ".json")).read_text())
    assert record["last_used"] == 5.0 and record["last_adopted"] == 10.0 and record["files"] == sorted(FILES)
    assert agent.scan(tmp_path) == []


def scanned(root):
    (found,) = agent.scan(root)
    return [[e["path"], e["kind"], e["size"]] for e in found["entries"]]


@pytest.mark.parametrize("change", ["tracked", "file added", "same name, other content"])
def test_adopt_refuses_what_changed_since_the_scan(tmp_path, change):
    hf_repo(tmp_path, "org/model", REV_A, FILES)
    kwargs = dict(repo="org/model", revision=REV_A, files=scanned(tmp_path), endpoint="https://huggingface.co", last_used=1.0, pinned=False)
    snapshot = tmp_path / "hub/models--org--model/snapshots" / REV_A
    if change == "tracked":
        agent.adopt(tmp_path, now=2.0, **kwargs)
        with pytest.raises(ValueError, match="already tracked"):
            agent.adopt(tmp_path, now=3.0, **kwargs)
        return
    if change == "file added":
        (snapshot / "extra.txt").write_text("new")
    else:
        # Replaced under its name by something the origin never vouched for.
        (snapshot / "config.json").unlink()
        (snapshot / "config.json").write_bytes(b"x" * 99)
    with pytest.raises(ValueError, match="changed since it was scanned"):
        agent.adopt(tmp_path, now=2.0, **kwargs)


# ---------------------------------------------------------------------------
# api.model_cache.adopt_cache, end to end on the local host
# ---------------------------------------------------------------------------


def test_the_preview_changes_nothing(tmp_path, origin):
    hf_repo(tmp_path, "org/model", REV_A, FILES)
    origin[("org/model", REV_A)] = sizes()
    before = sorted(p for p in tmp_path.rglob("*"))

    report = adopt(tmp_path)

    assert rows(report)[REV_A].action == "would_adopt"
    assert sorted(p for p in tmp_path.rglob("*")) == before


def test_adopting_makes_the_revision_listed_and_manageable(tmp_path, origin):
    repo = hf_repo(tmp_path, "org/model", REV_A, FILES)
    origin[("org/model", REV_A)] = sizes()

    adopted = rows(adopt(tmp_path, dry_run=False))[REV_A]
    assert adopted.action == "adopted" and adopted.tracked

    (listed,) = model_cache.inventory_cache(cache_dir=str(tmp_path))[0].entries
    assert listed.tracked and listed.reclaimable_bytes >= len(W)  # its blobs are its own
    # last_used defaults to now, so the default retention age protects it ...
    assert model_cache.prune_cache(cache_dir=str(tmp_path), dry_run=False)[0]["reclaimed_bytes"] == 0
    # ... and once eligible, prune frees the blobs too.
    model_cache.prune_cache(cache_dir=str(tmp_path), dry_run=False, min_age_days=0)
    assert not (repo / "snapshots" / REV_A).exists() and not any((repo / "blobs").iterdir())


def test_last_used_mtime_starts_the_clock_at_the_newest_file(tmp_path, origin):
    repo = hf_repo(tmp_path, "org/model", REV_A, FILES)
    origin[("org/model", REV_A)] = sizes()
    old = 1_000_000_000
    for path in (repo / "snapshots" / REV_A).iterdir():
        os.utime(path, (old, old), follow_symlinks=False)

    adopt(tmp_path, dry_run=False, last_used="mtime")

    (listed,) = model_cache.inventory_cache(cache_dir=str(tmp_path))[0].entries
    assert listed.last_used == old
    assert listed.eligible  # long past the default seven days


def test_pin_adopts_and_protects(tmp_path, origin):
    hf_repo(tmp_path, "org/model", REV_A, FILES)
    origin[("org/model", REV_A)] = sizes()

    adopt(tmp_path, dry_run=False, pinned=True)

    (listed,) = model_cache.inventory_cache(cache_dir=str(tmp_path))[0].entries
    assert listed.pinned and listed.reason_code == "pinned"


def test_a_snapshot_the_origin_disagrees_with_is_left_alone(tmp_path, origin):
    hf_repo(tmp_path, "org/model", REV_A, FILES)
    origin[("org/model", REV_A)] = {"model.safetensors": len(W) + 1, "config.json": len(CONFIG)}

    row = rows(adopt(tmp_path, dry_run=False))[REV_A]

    assert row.action is None and row.reason_code == "not_adoptable"
    assert not (tmp_path / ".sparkrun-model-artifacts/usage").exists()


def test_offline_needs_no_origin(tmp_path, monkeypatch):
    def unreachable(*args, **kwargs):
        raise AssertionError("offline adoption must not ask the origin")

    monkeypatch.setattr("sparkrun.models.manifest.fetch_inventory", unreachable)
    hf_repo(tmp_path, "org/model", REV_A, FILES)

    assert rows(adopt(tmp_path, dry_run=False, offline=True))[REV_A].action == "adopted"


def test_all_models_scope_and_its_guard(tmp_path, origin):
    hf_repo(tmp_path, "org/model", REV_A, FILES)
    hf_repo(tmp_path, "org/other", REV_B, FILES)
    origin[("org/model", REV_A)] = origin[("org/other", REV_B)] = sizes()

    assert {e.repo for e in adopt(tmp_path, model=None, all_models=True).entries} == {"org/model", "org/other"}
    with pytest.raises(ValueError, match="name a model"):
        model_cache.adopt_cache(None, cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="not both"):
        model_cache.adopt_cache(ModelArtifactRequest("org/model"), cache_dir=str(tmp_path), all_models=True)


def test_reports_match_the_wire_schema(tmp_path, origin):
    from sparkrun.api._wire import to_wire, validate
    from sparkrun.api._wire_schemas import RESULT_SCHEMAS

    hf_repo(tmp_path, "org/model", REV_A, FILES)
    origin[("org/model", REV_A)] = sizes()
    for dry_run in (True, False):
        validate(to_wire((adopt(tmp_path, dry_run=dry_run),)), RESULT_SCHEMAS["model_cache.adopt"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args, message", [([], "Name a MODEL"), (["org/model", "--all"], "Name a MODEL"), (["--all", "--revision", "x"], "needs a MODEL")]
)
def test_cli_requires_exactly_one_scope(args, message):
    from sparkrun.cli._model_cache import model_cache as group

    result = CliRunner().invoke(group, ["adopt", *args, "--hosts", "localhost"])
    assert result.exit_code == 2 and message in result.output


def test_cli_preview_renders_sizes_and_the_apply_hint(tmp_path, origin):
    from sparkrun.cli._model_cache import model_cache as group

    hf_repo(tmp_path, "org/model", REV_A, FILES)
    origin[("org/model", REV_A)] = sizes()
    result = CliRunner().invoke(group, ["adopt", "org/model", "--hosts", "localhost", "--cache-dir", str(tmp_path)], obj=None)

    assert result.exit_code == 0, result.output
    assert "would adopt" in result.output and "Re-run with --apply" in result.output


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def test_shared_store_snapshots_are_hub_layout(tmp_path):
    hf_repo(tmp_path, "org/model", REV_A, FILES, shared=True)
    (found,) = agent.scan(tmp_path)
    assert {e["kind"] for e in found["entries"]} == {"blob"}
    assert Path(os.path.realpath(tmp_path / "hub/models--org--model/blobs" / _digest(W))).parent.parent.name == "blobs"


# ---------------------------------------------------------------------------
# Review regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, same",
    [
        ("https://hf-mirror.com/", "https://HF-mirror.com", True),
        ("HTTPS://huggingface.co", "https://huggingface.co", True),
        ("https://huggingface.co", "https://hf-mirror.com", False),
        ("https://hf.example/a", "https://hf.example/b", False),
    ],
)
def test_origins_compare_normalized(a, b, same):
    from sparkrun.models._cache_contract import same_origin

    assert same_origin(a, b) is same


def test_a_launch_takes_over_an_adopted_records_origin(tmp_path, origin):
    """Adoption names whatever origin the adopter used; it must not fail the first real launch."""
    from dataclasses import replace

    from sparkrun.models.host_io import ModelHostIO
    from sparkrun.models.usage import record_use, usage_path
    from test_model_artifacts import artifact

    hf_repo(tmp_path, "org/model", REV_A, FILES)
    model_cache.adopt_cache(None, all_models=True, cache_dir=str(tmp_path), dry_run=False, offline=True, endpoint="https://adopted.example")
    manifest = replace(artifact(FILES), revision=REV_A)  # endpoint https://huggingface.co

    with ModelHostIO() as io:
        record_use(io, "localhost", str(tmp_path), manifest)
    record = json.loads(Path(usage_path(tmp_path, "org/model", REV_A)).read_text())
    assert record["endpoint"] == "https://huggingface.co"

    # Once prepared, the origin is authoritative again: a different one still refuses.
    from sparkrun.models.artifacts import ModelArtifactError

    with ModelHostIO() as io, pytest.raises(ModelArtifactError, match="endpoint mismatch"):
        record_use(io, "localhost", str(tmp_path), replace(manifest, endpoint="https://other.example"))


def test_an_unreadable_record_is_reported_not_silently_skipped(tmp_path, origin):
    hf_repo(tmp_path, "org/model", REV_A, FILES)
    record = tmp_path / ".sparkrun-model-artifacts/usage/org--model" / (REV_A + ".json")
    record.parent.mkdir(parents=True)
    record.write_text("{not json")

    row = rows(adopt(tmp_path))[REV_A]

    assert row.reason_code == "adopt_unverifiable" and "usage record is unreadable" in row.reason and str(record) in row.reason


def test_a_symlinked_snapshots_directory_is_never_scanned(tmp_path):
    real = hf_repo(tmp_path / "elsewhere", "org/model", REV_A, FILES)
    repo = tmp_path / "hub/models--org--model"
    repo.mkdir(parents=True)
    (repo / "snapshots").symlink_to(real / "snapshots")
    assert agent.scan(tmp_path) == []


def test_the_daemon_op_handles_all_models_and_rejects_a_bare_revision(monkeypatch):
    from sparkrun.daemon import special_ops

    seen = {}
    monkeypatch.setattr(model_cache, "adopt_cache", lambda request, **kwargs: seen.update(request=request, **kwargs) or ())
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda config: {})

    def call(data):
        return special_ops.execute("model_cache.adopt", dict(data), sctx=type("Ctx", (), {"config": None})())

    # Previously revision/endpoint stayed in the arguments without a model: TypeError.
    call({"cache_dir": "/c", "all_models": True, "endpoint": "https://hf-mirror.com"})
    assert seen["request"] is None and seen["endpoint"] == "https://hf-mirror.com"
    with pytest.raises(ValueError, match="revision needs a model"):
        call({"cache_dir": "/c", "all_models": True, "revision": "main"})
