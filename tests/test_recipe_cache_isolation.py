"""A model variant must preserve its base recipe's weights and runtime artifacts."""

import hashlib
from pathlib import Path
import subprocess

import pytest
import yaml

from sparkrun.core.env_templates import render_env_template
from sparkrun.core.recipe import Recipe
from sparkrun.core.runtime_cache import build_runtime_cache_mounts, resolve_runtime_cache_settings
from sparkrun.models.artifacts import ModelArtifactFile, ModelArtifactManifest
from sparkrun.models.preparation import prepare_model
from sparkrun.orchestration.runtime_cache import generate_runtime_cache_script
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.utils.text import render_template


@pytest.mark.parametrize("base_layout", ["regular", "hf-symlink"])
def test_inherited_model_has_separate_weight_and_runtime_caches(tmp_path, monkeypatch, base_layout):
    base_id = "deepseek-ai/DeepSeek-V4.1-Flash"
    child_id = base_id + "-1017"
    base_revision, child_revision = "a" * 40, "b" * 40
    base_data = {
        "model": base_id,
        "model_revision": base_revision,
        "runtime": "sglang",
        "container": "example/sglang:test",
        "defaults": {"revision": "{model_revision}"},
        "metadata": {"model_url": "https://huggingface.co/" + base_id, "model_dtype": "mxfp4"},
        "env": {"DSV41_PACKED_DIR": "{launch.runtime_cache_dir}/engram/v1/{launch.model_revision}"},
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base_data))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({"include": "base", "model": child_id, "model_revision": child_revision}))
    base, child = (Recipe.load(tmp_path / name) for name in ("base.yaml", "child.yaml"))
    child.distribution_config.resolve(child)
    (entry,) = child.distribution_config.models.entries
    assert (entry.name, entry.revision) == (child_id, child_revision)
    assert render_template(child.defaults["revision"], child.build_config_chain()) == child_revision
    # Inherited descriptive metadata must not redirect acquisition to the base.
    assert child.metadata["model_url"] == base.metadata["model_url"]

    filename = "model.safetensors"
    original, replacement = b"original weights", b"new 1017 weights"
    manifest = ModelArtifactManifest(
        child_id,
        child_revision,
        "https://huggingface.co",
        "full-snapshot-v1",
        (ModelArtifactFile(filename, len(replacement), "sha256", hashlib.sha256(replacement).hexdigest()),),
    )
    source, target = tmp_path / "control", tmp_path / "worker"
    base_files = []
    for root in (source, target):
        repo = root / "hub" / ("models--" + base_id.replace("/", "--"))
        weight = repo / "snapshots" / base_revision / filename
        weight.parent.mkdir(parents=True)
        if base_layout == "hf-symlink":
            blob = repo / "blobs" / hashlib.sha256(original).hexdigest()
            blob.parent.mkdir()
            blob.write_bytes(original)
            weight.symlink_to("../../blobs/" + blob.name)
        else:
            weight.write_bytes(original)
        base_files.append((weight, weight.stat().st_ino))

    downloads = []

    def download(repo, name, *, revision, local_dir, **kwargs):
        downloads.append((repo, revision, name))
        assert (repo, revision, name) == (child_id, child_revision, filename)
        path = Path(local_dir) / name
        path.write_bytes(replacement)
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    prepared = prepare_model(
        entry.name,
        ["localhost"],
        revision=entry.revision,
        cache_dir=str(target),
        local_cache_dir=str(source),
        manifest=manifest,
        checksums=True,
    )
    assert prepared is not None and prepared.bindings[0].report.complete
    assert downloads == [(child_id, child_revision, filename)]
    for root in (source, target):
        assert (root / manifest.relative_snapshot / filename).read_bytes() == replacement
    for weight, inode in base_files:
        assert weight.read_bytes() == original and weight.stat().st_ino == inode
        assert weight.is_symlink() == (base_layout == "hf-symlink")

    runtime = SglangRuntime()
    mounts, packed_files = [], []
    for recipe, contents in ((base, original), (child, replacement)):
        mount = build_runtime_cache_mounts(
            runtime=runtime,
            recipe=recipe,
            settings=resolve_runtime_cache_settings(runtime=runtime, recipe=recipe),
            root=str(tmp_path / "runtime-cache"),
            image=recipe.container,
        )
        assert mount is not None
        subprocess.run(["bash"], input=generate_runtime_cache_script(mount), text=True, check=True, capture_output=True)
        packed = render_env_template(
            recipe.env["DSV41_PACKED_DIR"], {}, {"runtime_cache_dir": mount.leaf, "model_revision": recipe.model_revision}
        )
        path = Path(packed) / "engram.bin"
        path.parent.mkdir(parents=True)
        path.write_bytes(contents)
        mounts.append(mount)
        packed_files.append(path)
    assert mounts[0].leaf != mounts[1].leaf
    assert list(mounts[0].volumes.values()) == list(mounts[1].volumes.values()) == ["/cache/runtime"]
    assert [path.read_bytes() for path in packed_files] == [original, replacement]
