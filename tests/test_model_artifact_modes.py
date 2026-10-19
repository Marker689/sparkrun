"""All acquisition topologies share immutable selection and real file copies."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from sparkrun.models import preparation, transport
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.artifacts import ModelArtifactRequest
from sparkrun.models.manifest import select_manifest
from test_model_artifacts import artifact, cache, DATA, SHA


@pytest.mark.parametrize("mode", ["local", "push", "delegated", "pull"])
def test_all_modes_repair_only_missing_files_and_preserve_source(tmp_path, monkeypatch, mode):
    manifest = artifact()
    locations = {None: tmp_path / "control", "head": tmp_path / "head", "worker": tmp_path / "worker"}
    source = cache(locations[None], manifest, shared=True)
    head = cache(locations["head"], manifest, shared=True)
    worker = cache(locations["worker"], manifest)
    (worker / "tokenizer.json").unlink()
    repairs = []

    class LocalHosts(ModelHostIO):
        def root(self, host, path):
            return str(locations[host])

        def execute(self, host, script, **kwargs):
            return super().execute(None, script, **kwargs)

        def lock(self, host, root, repo):
            return super().lock(None, root, repo)

    original = transport.copy_script

    def local_copy(request, target, ssh_kwargs, **kwargs):
        return original(replace(request, source_host=None), replace(target, host="localhost"), {}, **kwargs)

    def acquire(io, host, request, root, **kwargs):
        repairs.append((host, kwargs["files"]))
        assert kwargs["manifest"].identity == manifest.identity
        for name in kwargs["files"]:
            (Path(root) / manifest.relative_snapshot / name).write_bytes(DATA[name])

    monkeypatch.setattr(preparation, "ModelHostIO", LocalHosts)
    monkeypatch.setattr(transport, "copy_script", local_copy)
    monkeypatch.setattr(preparation, "run_adapter", acquire)
    prepared = preparation.prepare_model("org/model", ["head", "worker"], cache_dir="/hf", manifest=manifest, transfer_mode=mode)
    assert len(prepared.bindings) == 2
    assert all(b.report.complete for b in prepared.bindings)
    assert (source / "model.safetensors").is_symlink()
    assert (head / "model.safetensors").is_symlink()
    assert (worker / "tokenizer.json").read_bytes() == DATA["tokenizer.json"]
    assert repairs == ([("worker", ("tokenizer.json",))] if mode == "pull" else [])


@pytest.mark.parametrize("mode", ["local", "push", "delegated", "pull"])
def test_all_modes_copy_only_filtered_files_from_full_source(tmp_path, monkeypatch, mode):
    data = {**DATA, "nested/modeling.py": b"# code"}
    request = ModelArtifactRequest("org/model", allow_patterns=("*.json", "*.py"))
    manifest = select_manifest(request, {"revision": SHA, "files": [asdict(f) for f in artifact(data).files]})
    locations = {None: tmp_path / "control", "head": tmp_path / "head", "worker": tmp_path / "worker"}
    source_host = None if mode in {"local", "push"} else "head"
    source = cache(locations[source_host], manifest, data, shared=True)
    repairs = []

    class LocalHosts(ModelHostIO):
        def root(self, host, path):
            return str(locations[host])

        def execute(self, host, script, **kwargs):
            return super().execute(None, script, **kwargs)

        def lock(self, host, root, repo):
            return super().lock(None, root, repo)

    original = transport.copy_script

    def local_copy(request, target, ssh_kwargs, **kwargs):
        return original(replace(request, source_host=None), replace(target, host="localhost"), {}, **kwargs)

    def acquire(io, host, request, root, **kwargs):
        repairs.append((host, kwargs["files"]))
        assert request.allow_patterns == ("*.json", "*.py")
        assert "model.safetensors" not in kwargs["files"]
        cache(Path(root), manifest, {name: data[name] for name in kwargs["files"]})

    monkeypatch.setattr(preparation, "ModelHostIO", LocalHosts)
    monkeypatch.setattr(transport, "copy_script", local_copy)
    monkeypatch.setattr(preparation, "run_adapter", acquire)
    prepared = preparation.prepare_model(
        "org/model", ["head", "worker"], cache_dir="/hf", manifest=manifest, transfer_mode=mode, allow_patterns=request.allow_patterns
    )
    assert all(b.report.complete for b in prepared.bindings)
    for host in ("head", "worker"):
        snapshot = locations[host] / manifest.relative_snapshot
        assert (snapshot / "nested/modeling.py").read_bytes() == data["nested/modeling.py"]
        assert (snapshot / "model.safetensors").exists() is (host == source_host)
    assert (source / "model.safetensors").is_symlink()
    assert len(repairs) == (1 if mode == "pull" else 0)
