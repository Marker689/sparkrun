"""One bounded index contract for controller and generated acquisition adapters."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.models import acquisition, manifest
from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactRequest

SHA = "a" * 40
LIMIT = 16 * 1024 * 1024


@pytest.fixture(params=[False, True], ids=["controller", "delegated"])
def resolver(request, monkeypatch):
    raw = json.dumps({"weight_map": {"tensor": "model.safetensors"}}).encode()
    index = SimpleNamespace(rfilename="model.safetensors.index.json", size=len(raw), blob_id="", lfs=None)
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha=SHA, siblings=[index])
    monkeypatch.setattr("huggingface_hub.HfApi", Mock(return_value=api))

    def download(*args, **kwargs):
        path = Path(kwargs["local_dir"]) / index.rfilename
        path.write_bytes(raw)
        return str(path)

    fetch = Mock(side_effect=download)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)

    def set_payload(size):
        nonlocal raw
        raw = raw.ljust(size, b" ")
        index.size = size
        index.blob_id = hashlib.sha1(b"blob " + str(size).encode() + b"\0" + raw).hexdigest()

    def execute(host, shell, **kwargs):
        program = shell.split("<<'SPARKRUN_HUB_ADAPTER'\n", 1)[1].rsplit("\nSPARKRUN_HUB_ADAPTER", 1)[0]
        stdout, stderr = StringIO(), StringIO()
        rc = 0
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print("SPARKRUN_MODEL_ADAPTER_READY")
            try:
                exec(compile(program, "remote-adapter", "exec"), {})
            except SystemExit as error:
                rc = error.code
        return SimpleNamespace(returncode=rc, stdout=stdout.getvalue().encode(), stderr=stderr.getvalue().encode())

    def resolve(selection="auto"):
        options = ModelArtifactRequest("org/model", file_selection=selection)
        if request.param:
            return acquisition.run_adapter(SimpleNamespace(execute=execute), "head", options, "/unused-cache", token=None)
        return manifest.fetch_inventory(options)

    set_payload(len(raw))
    return SimpleNamespace(index=index, fetch=fetch, resolve=resolve, set_payload=set_payload)


@pytest.mark.parametrize("size", [None, LIMIT + 1, -1, True, "unknown"])
def test_invalid_index_size_reports_actual_reason_without_download(resolver, size):
    resolver.index.size = size
    with pytest.raises(ModelArtifactError) as caught:
        resolver.resolve()
    message = str(caught.value)
    assert f"org/model@{SHA}/model.safetensors.index.json" in message
    assert "https://huggingface.co" in message
    if size is None:
        assert "size metadata is missing" in message and "exceed" not in message
        assert "file selection all" not in message  # Full inventory also requires known sizes.
    elif size == LIMIT + 1:
        assert str(size) + " bytes" in message and "exceeding the 16 MiB" in message
    else:
        assert "size metadata is invalid" in message
    resolver.fetch.assert_not_called()


@pytest.mark.parametrize("size", [251269, LIMIT])
def test_valid_index_including_limit_is_authenticated_and_selected(resolver, size):
    # The smaller case matches the reported Ornith base index's actual byte size.
    resolver.set_payload(size)
    assert resolver.resolve()["weight_map"] == {"tensor": "model.safetensors"}
    resolver.fetch.assert_called_once()


def test_all_selection_skips_index_fetch_even_when_index_exceeds_limit(resolver):
    resolver.index.size = LIMIT + 1
    result = resolver.resolve("all")
    assert "weight_map" not in result
    assert result["files"][0]["size"] == LIMIT + 1
    resolver.fetch.assert_not_called()


def test_actual_index_larger_than_inventory_is_rejected(resolver):
    resolver.index.size -= 1
    with pytest.raises(ModelArtifactError, match="size does not match origin inventory"):
        resolver.resolve()


def test_index_checksum_validation_is_preserved(resolver):
    resolver.index.blob_id = "0" * 40
    with pytest.raises(ModelArtifactError, match="checksum does not match origin inventory"):
        resolver.resolve()
