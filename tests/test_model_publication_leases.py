"""Real lease loss, lock-owned publication and cancellation; temporary data only."""

from concurrent.futures import ThreadPoolExecutor
import base64
import os
import signal
import threading
import time
from unittest.mock import Mock

import pytest

pytest.importorskip("fcntl")

from sparkrun.models import _cache_agent, usage
from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.preparation import prepare_model
from sparkrun.transports.session import SshHostSession
from sparkrun.utils.shell import quote
from test_model_cache_retention import tracked


def kill_holder(io):
    holders = tuple(io._owners.values())
    assert holders
    for owner in holders:
        os.killpg(owner.process.process.pid, signal.SIGKILL)
        owner.process.wait(timeout=5)


@pytest.mark.parametrize("alias", ["localhost", "127.0.0.1"])
def test_lost_lease_is_terminal_including_shared_aliases(tmp_path, alias):
    tracked(tmp_path)
    with pytest.raises(ModelArtifactError, match="lease lost"), ModelHostIO() as io:
        io.lock("localhost", str(tmp_path), "org/model")
        kill_holder(io)
        assert io.cancelled()
        with pytest.raises(ModelArtifactError, match="lease lost"):
            io.lock(alias, str(tmp_path), "org/model")
        with pytest.raises(ModelArtifactError, match="lease lost"):
            io.publish(alias, str(tmp_path), "org/model", str(tmp_path / "invalid"), "must not publish")
        assert not (tmp_path / "invalid").exists()
        assert _cache_agent.run(str(tmp_path), action="prune", dry_run=False, min_age_seconds=0)["entries"][0]["action"] == "removed"
        # Even catching an intermediate error cannot turn this into success.


def test_preparation_never_publishes_binding_after_lease_loss(tmp_path, monkeypatch):
    manifest, snapshot, _ = tracked(tmp_path)
    original = usage.record_use
    published = Mock()
    monkeypatch.setattr("sparkrun.models.preparation.record_binding", published)

    def lose_before_claim(io, host, root, artifact, **kwargs):
        kill_holder(io)
        report = _cache_agent.run(str(tmp_path), action="prune", dry_run=False, min_age_seconds=0)
        assert report["entries"][0]["action"] == "removed"
        return original(io, host, root, artifact, **kwargs)

    monkeypatch.setattr(usage, "record_use", lose_before_claim)
    with usage.usage_operation():
        usage.begin_workload("fixture", {})
        with pytest.raises(ModelArtifactError, match="lease lost"):
            prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest, offline=True)
        assert not usage.workload_claims()
    assert not snapshot.exists()
    published.assert_not_called()


def test_loss_cancels_inflight_copy_without_closing_borrowed_session(tmp_path):
    ready = threading.Event()
    session = SshHostSession()
    unrelated = session.open_process("localhost", ["sleep", "60"])
    try:
        with pytest.raises(ModelArtifactError, match="lease lost"), ModelHostIO(session=session) as io:
            io.lock("localhost", str(tmp_path), "org/model")
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(io.execute, "localhost", "printf ready; sleep 60", timeout=10, stdout=lambda _: ready.set())
                assert ready.wait(5)
                kill_holder(io)
                with pytest.raises(ModelArtifactError, match="lease lost"):
                    future.result(timeout=5)
            assert unrelated.poll() is None
            assert not session._closed
        assert unrelated.poll() is None
        assert session.execute("localhost", ["true"]).returncode == 0
    finally:
        session.close()


def test_publication_child_keeps_flock_if_holder_dies(tmp_path):
    _, snapshot, _ = tracked(tmp_path)
    ready, release, written = (tmp_path / name for name in ("ready", "release", "written"))
    with pytest.raises(ModelArtifactError, match="lease lost"), ModelHostIO() as io:
        nonce = io.lock("localhost", str(tmp_path), "org/model")
        holder = io._owners[nonce].process
        # Pause a real command inside the publication channel. FD ownership must
        # continue in this child even if only the supervising holder exits.
        script = (
            f"touch {quote(str(ready))}; while [ ! -e {quote(str(release))} ]; do sleep .01; done; printf done > {quote(str(written))}\n"
        )
        holder.stdin.write(base64.b64encode(script.encode()) + b"\n")
        holder.stdin.flush()
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        os.kill(holder.process.pid, signal.SIGKILL)
        holder.wait(timeout=5)
        report = _cache_agent.run(str(tmp_path), action="prune", dry_run=False, min_age_seconds=0)
        assert report["entries"][0]["reason_code"] == "cache_busy"
        assert snapshot.exists()
        release.touch()
        while not written.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert written.read_text() == "done"


def test_publication_translates_shared_mount_alias_and_preserves_text(tmp_path):
    cache = tmp_path / "actual"
    cache.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(cache, target_is_directory=True)
    name = "metadata 'quoted'\n.json"
    contents = 'é\n{"value": "$HOME `false`"}\n' * 10000
    with ModelHostIO() as io:
        first = io.lock("localhost", str(cache), "org/model")
        assert io.lock("127.0.0.1", str(alias), "org/model") == first
        io.publish("127.0.0.1", str(alias), "org/model", str(alias / name), contents)
        assert (cache / name).read_text(encoding="utf-8") == contents
        assert len(io._owners) == 1


def test_provider_gets_lease_loss_cancellation(tmp_path, monkeypatch):
    from sparkrun.models import preparation
    from sparkrun.core.model_distribution import ModelTransferResult

    manifest, snapshot, _ = tracked(tmp_path)
    (snapshot / "model.safetensors").unlink()
    session = SshHostSession()
    observed = []

    def transfer(request, **kwargs):
        assert not request.cancelled()
        for holder in tuple(session._processes):
            os.killpg(holder.pid, signal.SIGKILL)
            holder.wait(timeout=5)
        observed.append(request.cancelled())
        return ModelTransferResult(manifest.identity, {"localhost": "complete"})

    monkeypatch.setattr(preparation, "try_model_transfer", transfer)
    try:
        with pytest.raises(ModelArtifactError, match="lease lost"):
            prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest, transfer_mode="pull", session=session)
        assert observed == [True]
        assert not session._closed
    finally:
        session.close()


def test_local_acquisition_discards_staged_file_after_cancellation(tmp_path, monkeypatch):
    from pathlib import Path
    from sparkrun.models.acquisition import repair_snapshot_files

    manifest, snapshot, _ = tracked(tmp_path)
    original = (snapshot / "model.safetensors").read_bytes()
    cancelled = threading.Event()

    def download(repo, name, **kwargs):
        target = Path(kwargs["local_dir"]) / name
        target.write_bytes(b"replacement")
        cancelled.set()
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    with pytest.raises(RuntimeError, match="acquisition cancelled"):
        repair_snapshot_files(
            manifest.repo, manifest.revision, str(tmp_path), manifest.endpoint, None, ("model.safetensors",), cancelled=cancelled.is_set
        )
    assert (snapshot / "model.safetensors").read_bytes() == original
    assert not list(snapshot.parent.glob(".sparkrun-repair-*"))


def test_lease_supervision_preserves_large_metadata_reads(tmp_path):
    # Using the transfer supervisor for finite host IO must not apply its tail
    # retention cap to full manifests or usage metadata.
    contents = "x" * (17 * 1024**2)
    path = tmp_path / "large.json"
    path.write_text(contents)
    with ModelHostIO() as io:
        io.lock("localhost", str(tmp_path), "org/model")
        assert io.read("localhost", str(path)) == contents
