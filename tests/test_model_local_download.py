"""Controller repairs stop the actual downloader on timeout or lease loss."""

import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import pytest

from _hub_download_stub import stub_hub_download
from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.preparation import prepare_model
from test_model_artifacts import artifact, cache

pytestmark = pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"), reason="requires POSIX bash")


def _config(**settings):
    return SimpleNamespace(get=settings.get)


def _assert_stopped(pidfile):
    pid = int(pidfile.read_text())
    # An orphan may briefly be a zombie until init reaps it.
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        stat = Path("/proc/%d/stat" % pid)
        if stat.exists() and stat.read_text().split(")", 1)[1].split()[0] == "Z":
            return
        time.sleep(0.05)
    pytest.fail("download process survived cancellation")


@pytest.mark.parametrize("mode", ["local", "push"])
@pytest.mark.parametrize("limit,match", [("idle", "no download progress"), ("total", "deadline exceeded"), ("cancel", "lease lost")])
def test_local_repair_cancels_process_and_cleans_staging(tmp_path, monkeypatch, mode, limit, match):
    pidfile = tmp_path / "downloader.pid"
    source = tmp_path / "source"
    manifest = artifact({"config.json": b"{}", "model.safetensors": b"weights"})
    snapshot = cache(source, manifest, {"config.json": b"{}"})
    stub_hub_download(
        monkeypatch,
        f"""
        import os, time
        from pathlib import Path
        Path({str(pidfile)!r}).write_text(str(os.getpid()))
        (Path(kwargs['local_dir']) / 'partial').write_bytes(b'partial')
        time.sleep(30)
    """,
    )
    if limit == "cancel":
        check = ModelHostIO.check_leases

        def lose_lease(self):
            check(self)
            if pidfile.exists():
                raise ModelArtifactError("lease lost")

        monkeypatch.setattr(ModelHostIO, "check_leases", lose_lease)
    config = _config(model_transfer_idle_timeout_s=1.5 if limit == "idle" else 60, model_transfer_timeout_s=1.5 if limit == "total" else 0)
    started = time.monotonic()
    with pytest.raises(ModelArtifactError, match=match):
        prepare_model(
            "org/model",
            ["localhost"],
            cache_dir=str(tmp_path / "target"),
            local_cache_dir=str(source),
            manifest=manifest,
            config=config,
            transfer_mode=mode,
        )
    assert time.monotonic() - started < 12
    _assert_stopped(pidfile)
    assert (snapshot / "config.json").read_bytes() == b"{}"
    assert not (snapshot / "model.safetensors").exists()
    assert not list(snapshot.parent.glob(".sparkrun-repair-*"))


def test_local_repair_with_progress_can_exceed_idle_limit(tmp_path, monkeypatch):
    data = b"x" * 12288
    manifest = artifact({"model.safetensors": data})
    stub_hub_download(
        monkeypatch,
        """
        import time
        from pathlib import Path
        target = Path(kwargs['local_dir']) / filename
        with target.open('wb') as stream:
            for _ in range(3):
                stream.write(b'x' * 4096)
                stream.flush()
                time.sleep(2.2)
        return str(target)
    """,
    )
    started = time.monotonic()
    result = prepare_model(
        "org/model",
        ["localhost"],
        cache_dir=str(tmp_path),
        manifest=manifest,
        config=_config(model_transfer_idle_timeout_s=3),
    )
    assert time.monotonic() - started > 6
    assert result.bindings[0].report.complete
    snapshot = tmp_path / manifest.relative_snapshot
    assert (snapshot / "model.safetensors").read_bytes() == data
    assert not list(snapshot.parent.glob(".sparkrun-repair-*"))
