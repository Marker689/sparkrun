"""Pinned, required-metadata and file-repair adapter for Hugging Face."""

from __future__ import annotations

from dataclasses import asdict
import json
import inspect
from textwrap import indent

from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest, ModelArtifactRequest, ModelInventoryUnavailable
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.manifest import HUB_ADAPTER_VERSION, read_weight_index
from sparkrun.utils.shell import quote


def repair_snapshot_files(repo, revision, cache, endpoint, token, files, *, cancelled=None, sizes=None, progress=None):
    """Publish only selected snapshot entries; never unlink shared HF blobs.

    Hub force_download updates a blob but leaves an existing regular snapshot
    entry untouched. A private local_dir followed by rename handles both that
    representation and classic/shared symlinks without a second weight copy.
    This function is also shipped verbatim to the pinned remote adapter.

    Content the host already holds is not this function's concern: the link
    pass (``models.snapshot_links``) runs first and turns such entries into
    hard links, so *files* is only what genuinely has to be fetched.

    *progress* is called as ``progress(done, total)`` in bytes, *total* from
    *sizes* (file -> expected size): after each file, and every two seconds
    while one downloads, measured from the staging directory, which holds the
    in-flight ``.incomplete`` file.
    """
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import os
    import threading
    from huggingface_hub import hf_hub_download

    snapshot = Path(cache) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    total = sum(int((sizes or {}).get(name, 0)) for name in files)
    state = {"done": 0, "staging": None}
    stopped = threading.Event()

    def staged_bytes(directory):
        found = 0
        for parent, _, names in os.walk(directory):
            for item in names:
                try:
                    found += os.lstat(os.path.join(parent, item)).st_size
                except OSError:
                    pass
        return found

    def sample():
        while not stopped.wait(2):
            directory = state["staging"]
            progress(state["done"] + (staged_bytes(directory) if directory else 0), total)

    sampler = threading.Thread(target=sample, daemon=True) if progress is not None else None
    if sampler is not None:
        sampler.start()
    try:
        for name in files:
            if cancelled is not None and cancelled():
                raise RuntimeError("model acquisition cancelled")
            with TemporaryDirectory(prefix=".sparkrun-repair-", dir=snapshot.parent) as staging:
                state["staging"] = staging
                try:
                    downloaded = hf_hub_download(
                        repo,
                        name,
                        revision=revision,
                        endpoint=endpoint,
                        token=token,
                        local_dir=staging,
                    )
                finally:
                    state["staging"] = None
                if cancelled is not None and cancelled():
                    raise RuntimeError("model acquisition cancelled")
                target = snapshot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                fetched = os.stat(downloaded).st_size
                os.replace(downloaded, target)
            state["done"] += fetched
            if progress is not None:
                progress(state["done"], total)
    finally:
        stopped.set()
        if sampler is not None:
            sampler.join()


def hub_script(
    request: ModelArtifactRequest,
    cache: str,
    *,
    token: str | None,
    manifest: ModelArtifactManifest | None = None,
    files: tuple[str, ...] = (),
) -> str:
    """One pinned adapter for controller/head acquisition; tokens travel on stdin."""
    from sparkrun.core.tooling import UV_INSTALL_BIN_DIR, UV_INSTALL_URL

    payload = {
        "request": asdict(request),
        "cache": cache,
        "token": token,
        "revision": manifest.revision if manifest else None,
        "files": files,
        "sizes": {f.path: f.size for f in (manifest.files if manifest else ()) if f.path in set(files)},
    }
    program = (
        inspect.getsource(read_weight_index)
        + "\n"
        + inspect.getsource(repair_snapshot_files)
        + "\n"
        + """import json, os
from huggingface_hub import HfApi, hf_hub_download
payload = json.loads(PAYLOAD)
request = payload['request']
repo = request['model'].split(':', 1)[0]
if payload['revision'] is None:
    info = HfApi(endpoint=request['endpoint'], token=payload['token']).model_info(repo, revision=request['revision'], files_metadata=True, timeout=30)
    files = []
    for f in info.siblings or ():
        algorithm = 'sha256' if f.lfs else 'git-sha1' if f.blob_id else None
        checksum = f.lfs.sha256 if f.lfs else f.blob_id
        files.append(dict(path=f.rfilename, size=f.size, checksum_algorithm=algorithm, checksum=checksum))
    result = dict(revision=info.sha, files=files)
    index = next((f for f in info.siblings or () if f.rfilename == 'model.safetensors.index.json'), None)
    if index is not None and request['file_selection'] != 'all':
        entry = next(f for f in files if f['path'] == index.rfilename)
        result['weight_map'] = read_weight_index(repo, info.sha, request['endpoint'], payload['token'], entry)
else:
    def report(done, total):
        print('SPARKRUN_MODEL_PROGRESS %d %d' % (done, total), flush=True)
    repair_snapshot_files(
        repo, payload['revision'], payload['cache'], request['endpoint'], payload['token'], payload['files'],
        sizes=payload['sizes'], progress=report,
    )
    result = dict(revision=payload['revision'], repaired=len(payload['files']))
print('SPARKRUN_MODEL_RESULT=' + json.dumps(result, separators=(',', ':')))
""".replace("PAYLOAD", repr(json.dumps(payload)))
    )
    # Keep a network/tool outage distinct from an invalid authoritative index.
    # Neither payload acquisition nor a malformed inventory may use the
    # compatibility fallback merely because the subprocess exited nonzero.
    program = (
        "import sys\nfrom huggingface_hub.utils import httpx\ntry:\n"
        + indent(program, "    ")
        + "\nexcept (httpx.HTTPError, OSError) as error:\n"
        "    print('SPARKRUN_MODEL_ERROR=unavailable')\n"
        "    print(str(error), file=sys.stderr)\n    sys.exit(1)\n"
        "except Exception as error:\n"
        "    print('SPARKRUN_MODEL_ERROR=invalid')\n"
        "    print(str(error), file=sys.stderr)\n    sys.exit(1)\n"
    )
    # Invoke the pinned installed client, otherwise a pinned isolated uv env.
    # This path is never reached for offline or already-complete observations.
    version_check = "import huggingface_hub; assert huggingface_hub.__version__ == %r" % HUB_ADAPTER_VERSION
    script = "set -euo pipefail\nexport HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30\n"
    script += f"if command -v python3 >/dev/null && python3 -c {quote(version_check)} 2>/dev/null; then\n  runner=(python3)\nelse\n"
    script += f'  export PATH="{UV_INSTALL_BIN_DIR}:$PATH"\n'
    script += f"  if ! command -v uv >/dev/null; then curl -fLsS {quote(UV_INSTALL_URL)} | sh >&2; fi\n"
    script += f"  runner=(uv run --no-project --with huggingface-hub=={HUB_ADAPTER_VERSION} python)\nfi\n"
    # Bootstrap failure is eligible for compatibility fallback; an unexpected
    # adapter failure after this point is not an unavailable-inventory result.
    script += "\"${runner[@]}\" -c 'from huggingface_hub.utils import httpx'\n"
    script += "printf '%s\\n' SPARKRUN_MODEL_ADAPTER_READY\n"
    script += "\"${runner[@]}\" - <<'SPARKRUN_HUB_ADAPTER'\n" + program + "\nSPARKRUN_HUB_ADAPTER\n"
    return script


class RepairProgress:
    """Feedback for a repair download, the peer of the copy leg's ``CopyProgress``.

    A delegated head runs the download with its output captured, so without
    this the launch printed nothing between "Distributing model" and the end
    of a download that can take hours. Feed it the adapter's stdout
    (:meth:`feed`) or call :meth:`update` directly for a control-machine
    repair. Events go out at most twice a second; a log line every ten.
    """

    _MARKER = b"SPARKRUN_MODEL_PROGRESS "

    def __init__(self, host: str | None, *, clock=None):
        import threading
        import time

        from sparkrun.feedback import get_feedback

        self.host = host or "control"
        self.reporter = get_feedback("model-distribution").child(source="model-download", host=self.host)
        self._clock = clock or time.monotonic
        self.started = self._clock()
        self.last_event = self.last_log = None
        self.state = (0, 0)
        self._pending = b""
        # A control-machine repair updates from its sampler thread and its loop.
        self._lock = threading.Lock()

    def feed(self, data: bytes) -> None:
        lines = (self._pending + data).split(b"\n")
        self._pending = lines.pop()[-4096:]
        for line in lines:
            if not line.startswith(self._MARKER):
                continue
            try:
                done, total = (int(value) for value in line[len(self._MARKER) :].split())
            except ValueError:
                continue
            self.update(done, total)

    def update(self, done: int, total: int) -> None:
        with self._lock:
            self.state = (done, total)
            self._emit("running")

    def finish(self, success: bool) -> None:
        with self._lock:
            self._emit("succeeded" if success else "failed", force=True)

    def _emit(self, status: str, *, force: bool = False) -> None:
        now = self._clock()
        if not force and self.last_event is not None and now - self.last_event < 0.5:
            return
        self.last_event = now
        log = force or self.last_log is None or now - self.last_log >= 10
        if log:
            self.last_log = now
        done, total = self.state
        rate = done / max(now - self.started, 0.001)
        self.reporter.report(
            "Model download on %s: %.2f / %.2f GiB (%.1f MiB/s)" % (self.host, done / 2**30, total / 2**30, rate / 2**20),
            stage="download",
            status=status,
            current=done,
            total=total if total else None,
            total_kind="exact" if total else "unknown",
            unit="bytes",
            metrics={"bytes_per_second": rate},
            log_message=log,
        )


def run_adapter(
    io: ModelHostIO,
    host: str | None,
    request: ModelArtifactRequest,
    cache: str,
    *,
    token: str | None,
    manifest: ModelArtifactManifest | None = None,
    files: tuple[str, ...] = (),
) -> dict:
    script = hub_script(request, cache, token=token, manifest=manifest, files=files)
    if not (manifest and files):
        result = io.execute(host, script, timeout=90)
    else:
        progress = RepairProgress(host)
        try:
            result = io.execute(host, script, timeout=7200, stdout=progress.feed)
        except BaseException:
            progress.finish(False)
            raise
        progress.finish(result.returncode == 0)
    if result.returncode:
        # Avoid printing a shell payload (which contains an optional HF token).
        detail = result.stderr.decode(errors="replace")[-1500:]
        if token:
            detail = detail.replace(token, "<redacted>")
        lines = result.stdout.splitlines()
        unavailable = b"SPARKRUN_MODEL_ERROR=unavailable" in lines or b"SPARKRUN_MODEL_ADAPTER_READY" not in lines
        error_type = (
            ModelInventoryUnavailable
            if manifest is None and unavailable and b"SPARKRUN_MODEL_ERROR=invalid" not in lines
            else ModelArtifactError
        )
        raise error_type(
            "Hugging Face %s on %s failed (rc=%d): %s"
            % ("repair" if manifest else "inventory", host or "control", result.returncode, detail)
        )
    records = [
        line.removeprefix("SPARKRUN_MODEL_RESULT=")
        for line in result.stdout.decode().splitlines()
        if line.startswith("SPARKRUN_MODEL_RESULT=")
    ]
    if len(records) != 1:
        raise ModelArtifactError("Hub adapter returned malformed metadata")
    try:
        parsed = json.loads(records[0])
    except json.JSONDecodeError as exc:
        raise ModelArtifactError("Hub adapter returned malformed metadata") from exc
    return parsed
