"""Completed snapshots discard resume data; incomplete selections retain it."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from sparkrun.core.model_distribution import ModelTransferResult
from sparkrun.models import _cache_agent, preparation
from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactRequest
from sparkrun.models.manifest import select_manifest
from sparkrun.models.transfer_cleanup import PARTIAL_DIRECTORY
from test_model_artifacts import DATA, SHA, artifact, cache, observe


def partial(snapshot, name="model.safetensors"):
    path = Path(name)
    staged = snapshot / path.parent / PARTIAL_DIRECTORY / path.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"unfinished")
    return staged


@pytest.mark.parametrize("shared", [False, True])
def test_warm_snapshot_cleans_nested_partials_and_preserves_other_files(tmp_path, shared):
    data = {**DATA, "nested/quote'\nfile.bin": b"nested"}
    manifest = artifact(data)
    snapshot = cache(tmp_path, manifest, data, shared=shared)
    leftovers = [partial(snapshot, name) for name in data]
    notes = snapshot / "operator-notes"
    notes.write_text("keep")
    other_revision = cache(tmp_path, replace(manifest, revision="b" * 40), data)
    other_partial = partial(other_revision)

    for _ in range(2):
        result = preparation.prepare_model(
            "org/model", ["localhost", "127.0.0.1"], cache_dir=str(tmp_path), manifest=manifest, offline=True, checksums=True
        )
        assert all(binding.report.complete for binding in result.bindings)
        assert not any(path.exists() or path.parent.exists() for path in leftovers)
        assert observe(manifest, snapshot, checksums=True).complete
        assert notes.read_text() == "keep"
        assert other_partial.read_bytes() == b"unfinished"


def test_successful_copy_cleans_source_and_target_even_for_files_not_copied(tmp_path):
    manifest = artifact()
    source, target = tmp_path / "source", tmp_path / "target"
    source_snapshot = cache(source, manifest, shared=True)
    target_snapshot = cache(target, manifest)
    source_partial, target_partial = partial(source_snapshot), partial(target_snapshot)
    (target_snapshot / "tokenizer.json").unlink()

    prepared = preparation.prepare_model(
        "org/model", ["localhost"], cache_dir=str(target), local_cache_dir=str(source), manifest=manifest, offline=True, checksums=True
    )

    assert prepared.bindings[0].report.complete
    assert not source_partial.exists() and not target_partial.exists()
    assert observe(manifest, target_snapshot, checksums=True).complete
    row = _cache_agent.run(str(target))["entries"][0]
    assert row["logical_bytes"] == sum(len(value) for value in DATA.values())
    assert row["reason_code"] == "retention_age"


def test_blob_reuse_removes_old_transfer_partials_without_copying(tmp_path, monkeypatch):
    manifest = artifact()
    snapshot = cache(tmp_path, manifest, shared=True)
    staged = partial(snapshot)
    (snapshot / "model.safetensors").unlink()
    monkeypatch.setattr(preparation, "try_model_transfer", lambda *a, **k: pytest.fail("already cached"))
    monkeypatch.setattr(preparation, "builtin_copy", lambda *a, **k: pytest.fail("already cached"))

    preparation.prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest, offline=True, checksums=True)

    assert not staged.exists()
    assert observe(manifest, snapshot, checksums=True).complete


@pytest.mark.parametrize("failure", ["missing", "wrong_size", "checksum"])
def test_incomplete_snapshot_preserves_all_resume_data(tmp_path, failure):
    manifest = artifact()
    snapshot = cache(tmp_path, manifest)
    staged = partial(snapshot)
    broken = snapshot / "tokenizer.json"
    if failure == "missing":
        broken.unlink()
    else:
        broken.write_bytes(b"wrong" if failure == "wrong_size" else b"x" * len(DATA["tokenizer.json"]))

    with pytest.raises(ModelArtifactError, match="offline: required model files"):
        preparation.prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest, offline=True, checksums=True)

    assert staged.read_bytes() == b"unfinished"


@pytest.mark.parametrize("succeeds", [False, True])
def test_pull_cleanup_requires_observed_completeness(tmp_path, monkeypatch, succeeds):
    manifest = artifact()
    snapshot = cache(tmp_path, manifest)
    staged = partial(snapshot)
    missing = snapshot / "tokenizer.json"
    missing.unlink()

    def transfer(request, **kwargs):
        assert staged.read_bytes() == b"unfinished"
        if succeeds:
            missing.write_bytes(DATA["tokenizer.json"])
        return ModelTransferResult(manifest.identity, {"localhost": "complete"})

    monkeypatch.setattr(preparation, "try_model_transfer", transfer)
    kwargs = dict(cache_dir=str(tmp_path), manifest=manifest, transfer_mode="pull", checksums=True)
    if succeeds:
        assert preparation.prepare_model("org/model", ["localhost"], **kwargs).bindings[0].report.complete
        assert not staged.exists()
    else:
        with pytest.raises(ModelArtifactError, match="source still invalid"):
            preparation.prepare_model("org/model", ["localhost"], **kwargs)
        assert staged.read_bytes() == b"unfinished"


def test_filtered_selection_preserves_resume_data_for_unselected_files(tmp_path):
    request = ModelArtifactRequest("org/model", allow_patterns=("*.json",))
    manifest = select_manifest(request, {"revision": SHA, "files": [asdict(f) for f in artifact().files]})
    snapshot = cache(tmp_path, manifest, {name: value for name, value in DATA.items() if name.endswith(".json")})
    complete_partial = partial(snapshot, "tokenizer.json")
    unfinished_partial = partial(snapshot)

    preparation.prepare_model(
        "org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest, offline=True, allow_patterns=request.allow_patterns
    )

    assert not complete_partial.exists()
    assert unfinished_partial.read_bytes() == b"unfinished"


@pytest.mark.parametrize("link", ["directory", "file"])
def test_cleanup_refuses_symlinked_staging_without_touching_referent(tmp_path, link):
    root, outside = tmp_path / "cache", tmp_path / "outside"
    manifest = artifact()
    snapshot = cache(root, manifest)
    outside.mkdir()
    protected = outside / "model.safetensors"
    protected.write_bytes(b"keep")
    if link == "directory":
        (snapshot / PARTIAL_DIRECTORY).symlink_to(outside, target_is_directory=True)
    else:
        (snapshot / PARTIAL_DIRECTORY).mkdir()
        (snapshot / PARTIAL_DIRECTORY / protected.name).symlink_to(protected)

    with pytest.raises(ModelArtifactError, match="symlinked transfer|unexpected transfer partial"):
        preparation.prepare_model("org/model", ["localhost"], cache_dir=str(root), manifest=manifest, offline=True)

    assert protected.read_bytes() == b"keep"
    assert observe(manifest, snapshot, checksums=True).complete


def test_snapshot_entry_referencing_staging_is_not_deleted(tmp_path):
    manifest = artifact()
    snapshot = cache(tmp_path, manifest)
    staged = partial(snapshot)
    staged.write_bytes(DATA["model.safetensors"])
    entry = snapshot / "model.safetensors"
    entry.unlink()
    entry.symlink_to(staged.relative_to(snapshot))

    with pytest.raises(ModelArtifactError, match="still references transfer partial"):
        preparation.prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest, offline=True, checksums=True)

    assert staged.read_bytes() == DATA["model.safetensors"]
    assert observe(manifest, snapshot, checksums=True).complete
