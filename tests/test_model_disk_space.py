"""Capacity preflight precedes all payload work; disk-full errors retain their host."""

import errno
import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.core.model_distribution import ModelTransferResult
from sparkrun.core import model_distribution
from sparkrun.models import acquisition, disk_space, preparation
from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactRequest
from sparkrun.models.disk_space import DiskSpace, ModelDiskSpaceError, probe_disk_space, require_model_space
from sparkrun.models.host_io import ModelHostIO
from sparkrun.orchestration.distribution import _prepare_manifest_model
from sparkrun.transports.session import HostCommandResult
from test_model_artifacts import DATA, SHA, artifact, cache


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    roots = {None: tmp_path / "control", "head": tmp_path / "head", "worker": tmp_path / "worker"}

    class LocalHosts(ModelHostIO):
        def root(self, host, path):
            return str(roots[host])

        def execute(self, host, script, **kwargs):
            return super().execute(None, script, **kwargs)

        def lock(self, host, root, repo):
            return super().lock(None, root, repo)

    monkeypatch.setattr(preparation, "ModelHostIO", LocalHosts)
    free = {host: 2**40 for host in roots}
    monkeypatch.setattr(disk_space, "probe_disk_space", lambda io, host, root: DiskSpace(free[host], (host or "control", root)))
    return roots, free


@pytest.mark.parametrize("mode", ["local", "push", "delegated", "pull"])
def test_full_worker_aborts_before_any_source_download_or_transfer(cluster, monkeypatch, mode):
    roots, free = cluster
    free["worker"] = 1
    download = Mock(side_effect=AssertionError("must not download"))
    provider = Mock(side_effect=AssertionError("must not invoke provider"))
    monkeypatch.setattr(acquisition, "repair_snapshot_files", download)
    monkeypatch.setattr(preparation, "run_adapter", download)
    monkeypatch.setattr(preparation, "try_model_transfer", provider)

    with pytest.raises(ModelDiskSpaceError) as raised:
        preparation.prepare_model("org/model", ["head", "worker"], cache_dir="/hf", manifest=artifact(), transfer_mode=mode)

    assert raised.value.host == "worker"
    assert raised.value.required_bytes == sum(map(len, DATA.values()))
    assert raised.value.free_bytes == 1
    download.assert_not_called()
    provider.assert_not_called()
    assert not any((root / artifact().relative_snapshot).exists() for root in roots.values())


@pytest.mark.parametrize("mode,host", [("local", None), ("push", None), ("delegated", "head"), ("pull", "head")])
def test_full_source_aborts_without_downloading(cluster, monkeypatch, mode, host):
    _, free = cluster
    free[host] = 0
    monkeypatch.setattr(preparation, "_acquire", lambda *a, **kw: pytest.fail("download began"))
    with pytest.raises(ModelDiskSpaceError) as raised:
        preparation.prepare_model("org/model", ["head", "worker"], cache_dir="/hf", manifest=artifact(), transfer_mode=mode)
    assert raised.value.host == (host or "control")


def test_only_remaining_files_count_and_retry_reuses_completed_files(cluster, monkeypatch):
    roots, free = cluster
    manifest = artifact()
    snapshot = cache(roots["head"], manifest)
    (snapshot / "tokenizer.json").unlink()
    free["head"] = len(DATA["tokenizer.json"]) - 1
    download = Mock()
    monkeypatch.setattr(preparation, "run_adapter", download)
    with pytest.raises(ModelDiskSpaceError) as raised:
        preparation.prepare_model("org/model", ["head"], cache_dir="/hf", manifest=manifest, transfer_mode="pull")
    assert raised.value.required_bytes == len(DATA["tokenizer.json"])
    download.assert_not_called()

    def repair(io, host, request, root, **kwargs):
        assert kwargs["files"] == ("tokenizer.json",)
        (snapshot / "tokenizer.json").write_bytes(DATA["tokenizer.json"])

    download.side_effect = repair
    free["head"] += 1
    result = preparation.prepare_model("org/model", ["head"], cache_dir="/hf", manifest=manifest, transfer_mode="pull")
    assert result.bindings[0].report.complete
    download.assert_called_once()


@pytest.mark.parametrize("damage", ["short", "same_size_corrupt"])
def test_replacement_budgets_full_file_and_preserves_old_entry(cluster, damage):
    roots, free = cluster
    snapshot = cache(roots["head"], artifact())
    old = b"bad" if damage == "short" else b"x" * len(DATA["model.safetensors"])
    (snapshot / "model.safetensors").write_bytes(old)
    free["head"] = len(DATA["model.safetensors"]) - 1
    with pytest.raises(ModelDiskSpaceError) as raised:
        preparation.prepare_model("org/model", ["head"], cache_dir="/hf", manifest=artifact(), transfer_mode="pull", checksums=True)
    assert raised.value.required_bytes == len(DATA["model.safetensors"])
    assert (snapshot / "model.safetensors").read_bytes() == old


@pytest.mark.parametrize("reuse_blob", [False, True])
def test_complete_or_reused_cache_needs_no_payload_capacity(cluster, monkeypatch, reuse_blob):
    roots, _ = cluster
    snapshot = cache(roots["head"], artifact(), shared=reuse_blob)
    if reuse_blob:
        (snapshot / "model.safetensors").unlink()
    monkeypatch.setattr(disk_space, "probe_disk_space", lambda *a: pytest.fail("nothing to download"))
    result = preparation.prepare_model("org/model", ["head"], cache_dir="/hf", manifest=artifact(), transfer_mode="pull", offline=True)
    assert result.bindings[0].report.complete


def test_aliases_count_once_but_separate_caches_share_filesystem_budget(monkeypatch):
    io = SimpleNamespace(lock=lambda host, root, repo: root)
    monkeypatch.setattr(disk_space, "probe_disk_space", lambda *a: DiskSpace(10, ("machine", "disk")))
    need = ("model.safetensors",)
    require_model_space(io, artifact(), [("head", "/cache", need), ("alias", "/cache", need)])
    with pytest.raises(ModelDiskSpaceError) as raised:
        require_model_space(io, artifact(), [("head", "/source", need), ("alias", "/target", need)])
    assert raised.value.required_bytes == 2 * len(DATA["model.safetensors"])


def test_probe_uses_cache_filesystem_and_does_not_create_missing_directory(tmp_path):
    missing = tmp_path / "cache 'quote'\npath" / "missing"
    with ModelHostIO() as io:
        report = probe_disk_space(io, None, str(missing))
    assert report.free_bytes >= 0
    assert report.filesystem[1] == str(tmp_path.stat().st_dev)
    assert report.filesystem[0]
    assert not missing.parent.exists()
    expected = os.statvfs(tmp_path)
    assert abs(report.free_bytes - expected.f_bavail * expected.f_frsize) < 2**20


@pytest.mark.parametrize("output", ["", "garbage", "-1 4096 1 boot", "2 0 1 boot", "1 4096 1 boot extra"])
def test_unreadable_capacity_is_not_treated_as_room(output):
    io = Mock(checked=Mock(return_value=output))
    with pytest.raises(ModelArtifactError, match="could not check model disk space on worker"):
        probe_disk_space(io, "worker", "/cache")


@pytest.mark.parametrize(
    "error", [OSError(errno.ENOSPC, "full"), RuntimeError("File reconstruction error: IO Error: No space left on device (os error 28)")]
)
def test_local_download_disk_full_is_classified_and_staging_is_removed(cluster, monkeypatch, error):
    roots, _ = cluster
    snapshot = cache(roots[None], artifact())
    (snapshot / "tokenizer.json").unlink()

    def download(*a, **kw):
        from pathlib import Path

        (Path(kw["local_dir"]) / "partial").write_bytes(b"partial")
        raise error

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    with pytest.raises(ModelDiskSpaceError, match="out of disk space on control"):
        preparation.prepare_model("org/model", ["worker"], cache_dir="/hf", manifest=artifact())
    assert (snapshot / "model.safetensors").read_bytes() == DATA["model.safetensors"]
    assert not list(snapshot.parent.glob(".sparkrun-repair-*"))


@pytest.mark.parametrize(
    "detail,reason", [("No space left on device (os error 28)", "out of disk space"), ("Disk quota exceeded", "disk quota exceeded")]
)
def test_remote_download_error_is_concise_and_has_the_source_host(detail, reason):
    from sparkrun.models.artifacts import ModelArtifactRequest

    io = Mock()
    io.execute.return_value = HostCommandResult(
        "head", 1, b"SPARKRUN_MODEL_ADAPTER_READY\nSPARKRUN_MODEL_ERROR=invalid\n", ("warning secret-token\n" + detail).encode()
    )
    with pytest.raises(ModelDiskSpaceError) as raised:
        acquisition.run_adapter(
            io, "head", ModelArtifactRequest("org/model"), "/hf", token="secret-token", manifest=artifact(), files=tuple(DATA)
        )
    assert raised.value.host == "head"
    assert raised.value.reason == reason
    assert "secret-token" not in str(raised.value) and "warning" not in str(raised.value)
    assert "completed model files will be reused" in str(raised.value)


def test_copy_disk_full_after_preflight_reports_only_affected_worker(cluster, monkeypatch):
    roots, _ = cluster
    cache(roots[None], artifact())
    cache(roots["head"], artifact())
    monkeypatch.setattr(
        preparation,
        "builtin_copy",
        lambda *a: ModelTransferResult(artifact().identity, {"worker": "failed"}, {"worker": "out of disk space: write failed"}),
    )
    with pytest.raises(ModelDiskSpaceError) as raised:
        preparation.prepare_model("org/model", ["head", "worker"], cache_dir="/hf", manifest=artifact())
    assert raised.value.host == "worker"
    assert raised.value.required_bytes is None


def test_distribution_preserves_disk_failure_host_and_does_not_use_legacy_fallback(monkeypatch):
    error = ModelDiskSpaceError("head", "/hf", required_bytes=95 * 2**30, free_bytes=3 * 2**30)
    monkeypatch.setattr(preparation, "prepare_model", Mock(side_effect=error))
    failures = _prepare_manifest_model("org/model", ["head", "worker"], "/hf", "/local", "delegated", {}, SHA, None, False)
    assert len(failures) == 1
    assert failures[0].host == "head"
    assert failures[0].reason == "insufficient disk space"
    assert "95.00 GiB required" in failures[0].detail
    assert "3.00 GiB available" in failures[0].detail
    assert "92.00 GiB short" in failures[0].detail


@pytest.mark.parametrize("mode", ["local", "pull"])
@pytest.mark.parametrize("raised_error", [False, True])
def test_provider_disk_full_preserves_failed_host_without_fallback(cluster, monkeypatch, mode, raised_error):
    roots, _ = cluster
    if mode == "local":
        cache(roots[None], artifact())
        cache(roots["head"], artifact())

    def fail(request):
        if raised_error:
            raise OSError(errno.ENOSPC, "disk full")
        states = {target.host: "failed" for target in request.targets}
        return ModelTransferResult(request.manifest.identity, states, {host: "No space left on device" for host in states})

    monkeypatch.setattr(model_distribution, "_PROVIDERS", {"fixture": SimpleNamespace(copy=fail, pull=fail)})
    monkeypatch.setattr(preparation, "builtin_copy", lambda *a: pytest.fail("must not retry provider failure"))
    with pytest.raises(ModelDiskSpaceError) as raised:
        preparation.prepare_model("org/model", ["head", "worker"], cache_dir="/hf", manifest=artifact(), transfer_mode=mode)
    assert raised.value.host == ("head" if mode == "pull" else "worker")


def test_filtered_manifest_only_budgets_selected_files(cluster):
    _, free = cluster
    request = ModelArtifactRequest("org/model", allow_patterns=("*.json",))
    manifest = replace(
        artifact(), selection=request.pattern_selection, files=tuple(f for f in artifact().files if f.path.endswith(".json"))
    )
    free["head"] = 0
    with pytest.raises(ModelDiskSpaceError) as raised:
        preparation.prepare_model(
            "org/model", ["head"], cache_dir="/hf", manifest=manifest, transfer_mode="pull", allow_patterns=request.allow_patterns
        )
    assert raised.value.required_bytes == len(DATA["config.json"]) + len(DATA["tokenizer.json"])


def test_capacity_is_checked_again_after_source_download(cluster, monkeypatch):
    roots, free = cluster

    def acquire(io, host, request, root, **kwargs):
        cache(roots[host], artifact())
        # A concurrent workload consumes the worker's space during acquisition.
        free["worker"] = 0

    monkeypatch.setattr(preparation, "run_adapter", acquire)
    monkeypatch.setattr(preparation, "builtin_copy", lambda *a: pytest.fail("worker copy must not start"))
    with pytest.raises(ModelDiskSpaceError) as raised:
        preparation.prepare_model("org/model", ["head", "worker"], cache_dir="/hf", manifest=artifact(), transfer_mode="delegated")
    assert raised.value.host == "worker"
    assert not (roots["worker"] / artifact().relative_snapshot).exists()
