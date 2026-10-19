"""File subsets retain their scope through selection, validation and cache reuse."""

from dataclasses import asdict, replace
import hashlib
import json

import pytest

from sparkrun.api._wire import to_wire
from sparkrun.api.model_cache import repair_model
from sparkrun.core import model_distribution as md
from sparkrun.models import preparation
from sparkrun.models.artifacts import (
    CacheValidationReport,
    ModelArtifactError,
    ModelArtifactRequest,
    ModelInventoryUnavailable,
    SELECTION_VERSION,
    ValidatedModelBinding,
)
from sparkrun.models.manifest import check_request, manifest_path, select_manifest
from test_model_artifacts import artifact, cache, SHA

DATA = {
    "config.json": b"{}",
    "nested/config.json": b"{}",
    "nested/modeling.py": b"# model code",
    "tokenizer.json": b"{}",
    "model.safetensors": b"weights",
    "model.safetensors.index.json": b"not a valid weight map",
    "CONFIG.JSON": b"case sensitive",
}


def select(**kwargs):
    request = ModelArtifactRequest("org/model", **kwargs)
    inventory = {"revision": SHA, "files": [asdict(f) for f in artifact(DATA).files]}
    return request, select_manifest(request, inventory)


@pytest.mark.parametrize("file_selection", ["auto", "all", "safetensors"])
def test_patterns_override_weight_selection_match_relative_paths_and_ignore_wins(file_selection):
    request, manifest = select(allow_patterns=("*.json", "*.py"), ignore_patterns=("*.index.json",), file_selection=file_selection)
    assert [f.path for f in manifest.files] == ["config.json", "nested/config.json", "nested/modeling.py", "tokenizer.json"]
    assert manifest.has_file_patterns
    assert not manifest.weight_files
    check_request(manifest, request)


def test_ignore_only_starts_from_entire_inventory():
    _, manifest = select(ignore_patterns=("*.safetensors", "*.index.json"))
    assert {f.path for f in manifest.files} == set(DATA) - {"model.safetensors", "model.safetensors.index.json"}


@pytest.mark.parametrize("kwargs", [{"allow_patterns": ()}, {"allow_patterns": ("*.missing",)}, {"ignore_patterns": ("*",)}])
def test_empty_selection_is_an_error(kwargs):
    with pytest.raises(ModelArtifactError, match="select no files"):
        select(**kwargs)


def test_request_identity_is_canonical_distinct_and_preserves_unfiltered_keys():
    full = ModelArtifactRequest("org/model")
    before = dict(
        model="org/model",
        revision="main",
        endpoint=full.endpoint,
        projector=None,
        file_selection="auto",
        selection_version=SELECTION_VERSION,
    )
    expected = hashlib.sha256(json.dumps(before, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    assert full.key == expected
    assert replace(full, ignore_patterns=()).key == full.key
    first = replace(full, allow_patterns=("*.py", "*.json", "*.json"))
    second = replace(full, allow_patterns=("*.json", "*.py"))
    assert first.key == second.key
    assert first.key != full.key
    assert replace(first, ignore_patterns=("config.json",)).key != first.key
    assert ModelArtifactRequest(**to_wire(first)) == first


def test_partial_inventory_cannot_validate_a_full_or_different_request():
    request, partial = select(allow_patterns=("*.json",), ignore_patterns=("*.index.json",))
    for other in (ModelArtifactRequest("org/model"), replace(request, allow_patterns=("*.py",))):
        with pytest.raises(ModelArtifactError, match="file patterns"):
            check_request(partial, other)
    with pytest.raises(ModelArtifactError, match="file patterns"):
        check_request(artifact(DATA), request)
    with pytest.raises(ModelArtifactError, match="outside"):
        check_request(replace(partial, files=artifact(DATA).files), request)


@pytest.mark.parametrize("field", ["allow_patterns", "ignore_patterns"])
@pytest.mark.parametrize("value", ["*.json", True, {"glob": "*.json"}, [None], [1], [""], [" "], ["bad\x00glob"]])
def test_invalid_patterns_are_rejected_before_host_access(field, value, tmp_path):
    with pytest.raises(ModelArtifactError, match=field):
        preparation.prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), **{field: value})


def test_filtered_cache_reuses_only_its_own_manifest_offline(tmp_path):
    request, manifest = select(allow_patterns=("*.py",))
    snapshot = cache(tmp_path, manifest, {"nested/modeling.py": DATA["nested/modeling.py"]})
    prepared = repair_model(request, cache_dir=str(tmp_path), hosts=["localhost"], manifest=manifest, offline=True)
    assert prepared.manifest == manifest
    assert manifest_path(str(tmp_path), request).is_file()
    assert not manifest_path(str(tmp_path), ModelArtifactRequest("org/model")).exists()
    assert not (snapshot.parent.parent / "refs/main").exists()
    reused = repair_model(request, cache_dir=str(tmp_path), hosts=["localhost"], offline=True)
    assert reused.manifest == manifest
    pinned = repair_model(replace(request, revision=SHA), cache_dir=str(tmp_path), hosts=["localhost"], offline=True)
    assert pinned.manifest == manifest
    with pytest.raises(ModelInventoryUnavailable, match="no saved authoritative"):
        repair_model(ModelArtifactRequest("org/model"), cache_dir=str(tmp_path), hosts=["localhost"], offline=True)
    (snapshot / "nested/modeling.py").unlink()
    with pytest.raises(ModelArtifactError, match="offline: required model files"):
        repair_model(request, cache_dir=str(tmp_path), hosts=["localhost"], offline=True)


@pytest.fixture
def binding_scope():
    token = md._BINDINGS.set({})
    yield
    md._BINDINGS.reset(token)


def binding(manifest, root="/cache", host="head", level="metadata"):
    return ValidatedModelBinding(
        "org/model",
        host,
        root,
        manifest,
        CacheValidationReport(host, manifest.identity, level, "complete", checked_files=len(manifest.files)),
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_partial_bindings_union_files_without_losing_validation(binding_scope, reverse):
    _, left = select(allow_patterns=("config.json", "nested/*.py"))
    _, right = select(allow_patterns=("*.json",), ignore_patterns=("*.index.json",))
    entries = [binding(left, level="checksum"), binding(right)]
    for entry in reversed(entries) if reverse else entries:
        md.record_binding(entry)
    merged = md.prepared_model("org/model", "head")
    assert {f.path for f in merged.manifest.files} == {f.path for f in left.files + right.files}
    assert merged.manifest.has_file_patterns
    assert merged.report.complete and merged.report.level == "metadata"
    assert merged.report.manifest_id == merged.manifest.identity
    assert merged.report.checked_files == len(merged.manifest.files)
    assert md.prepared_model("org/model", "worker") is None


@pytest.mark.parametrize("reverse", [False, True])
def test_full_binding_survives_overlapping_subset(binding_scope, reverse):
    _, partial = select(allow_patterns=("config.json",))
    full = artifact(DATA)
    entries = [binding(full), binding(partial)]
    for entry in reversed(entries) if reverse else entries:
        md.record_binding(entry)
    assert md.prepared_model("org/model", "head").manifest == full


def test_repeated_partial_merges_preserve_gguf_runtime_paths(binding_scope):
    data = {"model.gguf": b"weights", "config.json": b"{}", "tokenizer.json": b"{}"}
    full = replace(
        artifact(data),
        selection="gguf-v1",
        files=artifact({"model.gguf": data["model.gguf"]}).files,
        weight_path="model.gguf",
        weight_files=("model.gguf",),
    )
    md.record_binding(binding(full))
    for name in ("config.json", "tokenizer.json"):
        _, partial = select(allow_patterns=(name,))
        md.record_binding(binding(partial))
    merged = md.prepared_model("org/model", "head")
    assert merged.manifest.weight_path == "model.gguf"
    assert merged.manifest.weight_files == ("model.gguf",)
    assert {f.path for f in merged.manifest.files} == set(data)


@pytest.mark.parametrize("mismatch", ["revision", "root", "file"])
def test_overlapping_scopes_reject_conflicting_artifacts(binding_scope, mismatch):
    _, partial = select(allow_patterns=("config.json",))
    md.record_binding(binding(partial))
    other = replace(partial, selection="file-patterns-v1:other")
    root = "/cache"
    if mismatch == "revision":
        other = replace(other, revision="b" * 40)
    elif mismatch == "root":
        root = "/other-cache"
    else:
        other = replace(other, files=(replace(other.files[0], size=999),))
    with pytest.raises(ModelArtifactError):
        md.record_binding(binding(other, root=root))


def test_merged_scope_cannot_change_an_existing_runtime_weight(binding_scope):
    files = artifact({"a.gguf": b"a", "b.gguf": b"b"})
    partial_union = replace(files, selection="file-patterns-v1:union", weight_path="a.gguf")
    md.record_binding(binding(partial_union))
    with pytest.raises(ModelArtifactError, match="weight_path"):
        md.record_binding(binding(replace(files, selection="gguf-v1", weight_path="b.gguf")))
