"""Exercise the installed Hub HTTP client and generated adapter against a local origin."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import requires
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import huggingface_hub
import pytest
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.utils import close_session
from huggingface_hub.utils import _http
from packaging.requirements import Requirement

from sparkrun.models import acquisition, hub, manifest
from sparkrun.models.artifacts import ModelArtifactRequest, ModelInventoryUnavailable

REVISION = "a" * 40
FILES = {
    "config.json": b'{"model_type":"test"}',
    "model.safetensors": b"test weights",
    "model.safetensors.index.json": b'{"weight_map":{"layer":"model.safetensors"}}',
}


def git_hash(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


@pytest.fixture
def hub_origin(monkeypatch):
    # Opt out of the suite's mocked downloads, using only this loopback origin.
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_DISABLE_TELEMETRY", True)
    monkeypatch.setattr(hub._STATE, "client_configured", False)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    previous_factory = _http._GLOBAL_CLIENT_FACTORY
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, head=False):
            path = urlsplit(self.path).path
            requests.append((self.command, path))
            if path in ("/api/models/org/tiny", "/api/models/org/tiny/revision/" + REVISION):
                data = json.dumps(
                    {
                        "id": "org/tiny",
                        "sha": REVISION,
                        "siblings": [{"rfilename": name, "size": len(value), "blobId": git_hash(value)} for name, value in FILES.items()],
                    }
                ).encode()
            elif path.startswith("/org/tiny/resolve/" + REVISION + "/") and path.rsplit("/", 1)[-1] in FILES:
                data = FILES[path.rsplit("/", 1)[-1]]
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.send_header("X-Error-Code", "RepoNotFound")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Type", "application/json")
            self.send_header("ETag", '"' + git_hash(data) + '"')
            self.send_header("X-Repo-Commit", REVISION)
            self.end_headers()
            if not head:
                self.wfile.write(data)

        def do_HEAD(self):
            self.respond(head=True)

        def do_GET(self):
            self.respond()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        hub.configure_hub_client(SimpleNamespace(hub_timeout_s=2.0))
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        close_session()
        huggingface_hub.set_client_factory(previous_factory)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_remote_adapter_pin_matches_controller_dependency():
    dependency = next(Requirement(value) for value in requires("sparkrun") if Requirement(value).name == "huggingface-hub")
    assert str(dependency.specifier) == "==" + manifest.HUB_ADAPTER_VERSION
    assert huggingface_hub.__version__ == manifest.HUB_ADAPTER_VERSION


def test_real_inventory_repair_and_offline_reuse(hub_origin, tmp_path):
    endpoint, requests = hub_origin
    request = ModelArtifactRequest("org/tiny", endpoint=endpoint)
    inventory = manifest.fetch_inventory(request, token=False)
    assert inventory["revision"] == REVISION
    assert inventory["weight_map"] == {"layer": "model.safetensors"}
    snapshot = tmp_path / "hub/models--org--tiny/snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"damaged regular snapshot entry")
    acquisition.repair_snapshot_files("org/tiny", REVISION, str(tmp_path), endpoint, False, tuple(FILES))
    for filename, expected in FILES.items():
        assert (snapshot / filename).read_bytes() == expected
    request_count = len(requests)
    cached = hf_hub_download(
        "org/tiny",
        "model.safetensors",
        revision=REVISION,
        endpoint=endpoint,
        token=False,
        cache_dir=str(tmp_path / "hub"),
        local_files_only=True,
    )
    assert Path(cached).read_bytes() == FILES["model.safetensors"]
    assert len(requests) == request_count


def test_real_controller_http_error_is_inventory_unavailable(hub_origin):
    endpoint, _ = hub_origin
    with pytest.raises(ModelInventoryUnavailable):
        manifest.fetch_inventory(ModelArtifactRequest("org/missing", endpoint=endpoint), token=False)


@pytest.mark.parametrize("remote", [False, True])
def test_file_filters_never_fetch_excluded_weights_or_indexes(hub_origin, tmp_path, capsys, remote):
    endpoint, requests = hub_origin
    request = ModelArtifactRequest("org/tiny", endpoint=endpoint, allow_patterns=("*.json",), ignore_patterns=("*.index.json",))

    def adapter(**kwargs):
        script = acquisition.hub_script(request, str(tmp_path), token=None, **kwargs)
        program = script.split("<<'SPARKRUN_HUB_ADAPTER'\n", 1)[1].rsplit("\nSPARKRUN_HUB_ADAPTER", 1)[0]
        exec(compile(program, "remote-adapter", "exec"), {})
        return json.loads(capsys.readouterr().out.split("SPARKRUN_MODEL_RESULT=", 1)[1])

    inventory = adapter() if remote else manifest.fetch_inventory(request, token=False)
    assert "weight_map" not in inventory
    selected = manifest.select_manifest(request, inventory)
    files = tuple(f.path for f in selected.files)
    assert files == ("config.json",)
    if remote:
        assert adapter(manifest=selected, files=files)["repaired"] == 1
    else:
        acquisition.repair_snapshot_files(request.repo, selected.revision, str(tmp_path), endpoint, False, files)
    snapshot = tmp_path / selected.relative_snapshot
    assert (snapshot / "config.json").read_bytes() == FILES["config.json"]
    assert not (snapshot / "model.safetensors").exists()
    assert not (snapshot / "model.safetensors.index.json").exists()
    assert not any(path.endswith(("model.safetensors", "model.safetensors.index.json")) for _, path in requests)


@pytest.mark.parametrize("model", ["org/tiny", "org/missing"])
def test_generated_adapter_uses_the_hub_transport(hub_origin, tmp_path, capsys, model):
    endpoint, _ = hub_origin
    script = acquisition.hub_script(ModelArtifactRequest(model, endpoint=endpoint), str(tmp_path), token=None)
    program = script.split("<<'SPARKRUN_HUB_ADAPTER'\n", 1)[1].rsplit("\nSPARKRUN_HUB_ADAPTER", 1)[0]
    if model == "org/missing":
        with pytest.raises(SystemExit) as error:
            exec(compile(program, "remote-adapter", "exec"), {})
        assert error.value.code == 1
        assert "SPARKRUN_MODEL_ERROR=unavailable" in capsys.readouterr().out
    else:
        exec(compile(program, "remote-adapter", "exec"), {})
        result = json.loads(capsys.readouterr().out.split("SPARKRUN_MODEL_RESULT=", 1)[1])
        assert result["revision"] == REVISION
        assert result["weight_map"] == {"layer": "model.safetensors"}


def test_repair_reports_progress_by_bytes(hub_origin, tmp_path):
    endpoint, _ = hub_origin
    calls = []
    sizes = {name: len(data) for name, data in FILES.items()}

    acquisition.repair_snapshot_files(
        "org/tiny", REVISION, str(tmp_path), endpoint, False, tuple(FILES), sizes=sizes, progress=lambda *a: calls.append(a)
    )

    total = sum(sizes.values())
    assert calls[-1] == (total, total)
    assert [done for done, _ in calls] == sorted(done for done, _ in calls)
    # Downloads stay private regular files: the cache agent budgets and prunes those.
    snapshot = tmp_path / "hub/models--org--tiny/snapshots" / REVISION
    assert all((snapshot / name).is_file() and not (snapshot / name).is_symlink() for name in FILES)
    assert not (tmp_path / "hub/models--org--tiny/blobs").exists()


def test_generated_adapter_repair_streams_progress(hub_origin, tmp_path, capsys):
    endpoint, _ = hub_origin
    request = ModelArtifactRequest("org/tiny", endpoint=endpoint)
    selected = manifest.select_manifest(request, manifest.fetch_inventory(request, token=False))
    files = tuple(f.path for f in selected.files)

    script = acquisition.hub_script(request, str(tmp_path), token=None, manifest=selected, files=files)
    program = script.split("<<'SPARKRUN_HUB_ADAPTER'\n", 1)[1].rsplit("\nSPARKRUN_HUB_ADAPTER", 1)[0]
    exec(compile(program, "remote-adapter", "exec"), {})

    out = capsys.readouterr().out
    result = json.loads(out.split("SPARKRUN_MODEL_RESULT=", 1)[1])
    assert result["repaired"] == len(files)
    progress = acquisition.RepairProgress("h1")
    progress.feed(out.encode())
    total = sum(f.size for f in selected.files)
    assert progress.state == (total, total)


def test_repair_progress_throttles_and_reports_rate():
    from unittest import mock

    clock = SimpleNamespace(now=100.0)
    progress = acquisition.RepairProgress("h1", clock=lambda: clock.now)
    progress.reporter = mock.Mock()
    report = progress.reporter.report
    clock.now += 2
    progress.feed(b"SPARKRUN_MODEL_PROGRESS %d %d\nnoise\nSPARKRUN_MODEL_PRO" % (2**30, 20 * 2**30))
    clock.now += 0.1
    progress.feed(b"GRESS %d %d\n" % (2 * 2**30, 20 * 2**30))  # split marker, throttled
    clock.now += 1.9
    progress.feed(b"SPARKRUN_MODEL_PROGRESS garbage\n")
    progress.finish(True)
    messages = [call.args[0] for call in report.call_args_list]
    assert messages == [
        "Model download on h1: 1.00 / 20.00 GiB (512.0 MiB/s)",
        "Model download on h1: 2.00 / 20.00 GiB (512.0 MiB/s)",
    ]
    final = report.call_args_list[-1].kwargs
    assert final["status"] == "succeeded" and final["log_message"] is True
    assert final["current"] == 2 * 2**30 and final["total"] == 20 * 2**30 and final["unit"] == "bytes"


def _tiny_manifest():
    request = ModelArtifactRequest("org/tiny")
    return request, manifest.ModelArtifactManifest(
        "org/tiny", REVISION, request.endpoint, "auto", (manifest.ModelArtifactFile("config.json", 3, "git-sha1", "1" * 40),)
    )


def test_run_adapter_streams_repair_output_to_progress():
    from unittest import mock

    from sparkrun.transports.session import HostCommandResult

    request, selected = _tiny_manifest()
    io = mock.Mock()

    def execute(host, script, *, timeout, stdout=None, idle_timeout=None, watch=None):
        # No total ceiling: the download is bounded by its progress (#324).
        assert stdout is not None and timeout is None and idle_timeout == 600 and watch is not None
        stdout(b"SPARKRUN_MODEL_PROGRESS 3 3\n")
        return HostCommandResult(host, 0, b'SPARKRUN_MODEL_RESULT={"revision":"%s","repaired":1}\n' % REVISION.encode(), b"")

    io.execute.side_effect = execute
    with (
        mock.patch.object(acquisition.RepairProgress, "update") as update,
        mock.patch.object(acquisition.RepairProgress, "finish") as finish,
    ):
        result = acquisition.run_adapter(io, "h1", request, "/cache", token=None, manifest=selected, files=("config.json",))
    update.assert_called_once_with(3, 3)
    finish.assert_called_once_with(True)
    assert result["repaired"] == 1


def test_run_adapter_closes_progress_when_the_session_fails():
    """A frontend must not be left showing a download as running forever."""
    from unittest import mock

    from sparkrun.models.artifacts import ModelArtifactError

    request, selected = _tiny_manifest()
    io = mock.Mock()
    io.execute.side_effect = ModelArtifactError("model transfer deadline exceeded")
    with mock.patch.object(acquisition.RepairProgress, "finish") as finish, pytest.raises(ModelArtifactError):
        acquisition.run_adapter(io, "h1", request, "/cache", token=None, manifest=selected, files=("config.json",))
    finish.assert_called_once_with(False)
