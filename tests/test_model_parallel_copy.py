"""Model copy concurrency, actual rsync semantics, progress and process lifetime."""

from collections import Counter
from dataclasses import replace
import json
import os
import shlex
import shutil
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.core.model_distribution import ModelCopyRequest, ModelTarget
from sparkrun.core.operation import OperationState, operation_scope
from sparkrun.models import preparation, transport
from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.host_io import ModelHostIO
from sparkrun.transports.session import HostCommandResult, SshHostSession
from test_model_artifacts import artifact, cache

POSIX = pytest.mark.skipif(sys.platform == "win32" or not shutil.which("rsync"), reason="requires POSIX rsync and bash")
DATA = {"nested/shard-%d.safetensors" % i: bytes([i]) * 40 for i in range(6)} | {"config.json": b"{}", "tokenizer.json": b"[]"}


def request(hosts=("a", "b", "c"), **kwargs):
    manifest = artifact(DATA)
    return ModelCopyRequest(
        manifest, None, "/source", tuple(ModelTarget(h, "/" + h, "ib-" + h, tuple(DATA)) for h in hosts), "op", **kwargs
    )


@pytest.fixture
def simulated(monkeypatch):
    monkeypatch.setattr(transport, "_BATCH_BYTES", 32)
    monkeypatch.setattr(transport, "copy_script", lambda req, target, ssh, **kw: json.dumps([target.host, target.required_files]))


@pytest.mark.parametrize("per_host,maximum", [(2, 4), (1, 1), (4, 2), (1, 3)])
def test_limits_fairness_exact_selection_and_feedback(simulated, per_host, maximum):
    active, peaks, calls = Counter(), Counter(), []
    lock = threading.Lock()
    ready = threading.Event()
    expected = min(maximum, per_host * 3)
    max_total = 0
    callback = Mock()
    events = []

    def execute(host, script, *, cancelled, stdout, **kwargs):
        nonlocal max_total
        target, files = json.loads(script)
        with lock:
            active[target] += 1
            calls.append((target, files))
            peaks[target] = max(peaks[target], active[target])
            max_total = max(max_total, sum(active.values()))
            if len(calls) == expected:
                ready.set()
        assert ready.wait(3), "scheduler did not fill its available stream slots"
        assert not cancelled()
        stdout(b"\r       2")
        stdout(b"0  50%    1MB/s    0:00:00\r")
        time.sleep(0.005)
        with lock:
            active[target] -= 1
        return HostCommandResult(target, 0)

    state = OperationState(events=events.append)
    with operation_scope(state=state):
        req = request(config={"model_transfer_streams_per_host": per_host, "model_transfer_max_streams": maximum}, progress=callback)
        result = transport.builtin_copy(req, SimpleNamespace(execute=execute), {})
    assert result.outcomes == {h: "complete" for h in ("a", "b", "c")}
    assert max_total == expected
    assert max(peaks.values()) <= per_host
    assert len({h for h, _ in calls[:expected]}) == min(3, expected)
    for host in result.outcomes:
        assert Counter(p for h, files in calls if h == host for p in files) == Counter(DATA.keys())
        callback.assert_any_call(host, sum(map(len, DATA.values())))
    assert callback.call_count == 3
    aggregate = [e for e in events if e["source"] == "model-rsync" and "host" not in e]
    assert aggregate[0]["status"] == "running"
    assert aggregate[-1]["status"] == "succeeded"
    assert aggregate[-1]["current"] == aggregate[-1]["total"] == 3 * sum(map(len, DATA.values()))
    assert aggregate[-1]["metrics"]["completed_files"] == 3 * len(DATA)
    assert all(e["operation_id"] == state.id for e in events)
    assert all(e["metrics"]["active_streams"] <= maximum for e in aggregate)


def test_defaults_and_invalid_settings():
    assert transport.copy_limits(None) == (2, 16)
    for key in ("model_transfer_streams_per_host", "model_transfer_max_streams"):
        for value in (0, -1, True, 1.5, "2", None):
            with pytest.raises(ModelArtifactError, match=key):
                transport.builtin_copy(request(config={key: value}), Mock(), {})


def test_batches_are_disjoint_and_small_files_are_batched(simulated):
    req = request(("a",))
    batches = list(transport._batches(req.targets[0], {f.path: f.size for f in req.manifest.files}))
    assert len(batches) == 7
    assert batches[-1].required_files == ("config.json", "tokenizer.json")
    assert all(len(batch.required_files) == 1 for batch in batches[:-1])


def test_failed_host_stops_new_batches_while_other_hosts_complete(simulated):
    calls = []

    def execute(host, script, **kwargs):
        target, files = json.loads(script)
        calls.append((target, files))
        return HostCommandResult(target, 12 if target == "a" else 0, stderr=b"connection unexpectedly closed")

    req = request(("a", "b"), config={"model_transfer_streams_per_host": 1})
    result = transport.builtin_copy(req, SimpleNamespace(execute=execute), {})
    assert result.outcomes == {"a": "failed", "b": "complete"}
    assert len([h for h, _ in calls if h == "a"]) == 1
    assert result.errors.keys() == {"a"}


@pytest.mark.parametrize("cancel", [True, False])
def test_operation_cancel_and_deadline_stop_active_workers_and_admission(simulated, cancel):
    active = set()
    started = threading.Event()
    lock = threading.Lock()
    calls = []
    state = OperationState(deadline=None if cancel else time.monotonic() + 0.25)

    def execute(host, script, *, cancelled, timeout, **kwargs):
        with lock:
            active.add(script)
            calls.append(script)
            if len(calls) == 4:
                started.set()
        try:
            assert started.wait(2)
            if cancel:
                state.cancel.set()
            end = time.monotonic() + timeout
            while not cancelled() and time.monotonic() < end:
                time.sleep(0.01)
            raise ModelArtifactError("model copy cancelled" if cancelled() else "model transfer deadline exceeded")
        finally:
            with lock:
                active.remove(script)

    with operation_scope(state=state):
        if cancel:
            with pytest.raises(ModelArtifactError, match="cancelled"):
                transport.builtin_copy(request(config={"model_transfer_max_streams": 4}), SimpleNamespace(execute=execute), {})
        else:
            result = transport.builtin_copy(request(config={"model_transfer_max_streams": 4}), SimpleNamespace(execute=execute), {})
            assert set(result.outcomes.values()) == {"failed"}
    assert not active
    assert len(calls) == 4


def test_rsync_bulk_connections_override_multiplexing_and_keep_trust():
    req = request(("a",))
    script = transport.copy_script(req, req.targets[0], {"ssh_options": ["-o", "ControlMaster=auto", "-o", "UserKnownHostsFile=/trust"]})
    argv = shlex.split(script.splitlines()[-1])
    ssh = shlex.split(argv[argv.index("-e") + 1])
    assert ssh.index("ControlMaster=no") < ssh.index("ControlMaster=auto")
    assert "ControlPath=none" in ssh
    assert "UserKnownHostsFile=/trust" in ssh
    assert "--partial-dir=.sparkrun-partial" in argv
    assert "--inplace" not in argv and "--whole-file" not in argv
    assert argv[-1].startswith("ib-a:")


@POSIX
@pytest.mark.parametrize("iteration", range(3))
def test_real_parallel_rsync_and_shared_destination_aliases(tmp_path, monkeypatch, iteration):
    monkeypatch.setattr(transport, "_BATCH_BYTES", 32)
    source, target = tmp_path / "source", tmp_path / "target"
    manifest = artifact(DATA)
    original = cache(source, manifest, DATA, shared=True)
    locations = {None: source, "a": target, "alias": target, "b": tmp_path / "second"}
    destinations = []

    class LocalHosts(ModelHostIO):
        def root(self, host, path):
            return str(locations[host])

        def execute(self, host, script, **kwargs):
            return super().execute(None, script, **kwargs)

        def lock(self, host, root, repo):
            return super().lock(None, root, repo)

    original_copy = transport.copy_script

    def copy(req, dest, ssh_kwargs, **kwargs):
        destinations.append((dest.host, dest.required_files))
        return original_copy(replace(req, source_host=None), replace(dest, host="localhost"), {}, **kwargs)

    monkeypatch.setattr(preparation, "ModelHostIO", LocalHosts)
    monkeypatch.setattr(transport, "copy_script", copy)
    prepared = preparation.prepare_model("org/model", ["a", "alias", "b"], cache_dir="/hf", manifest=manifest, offline=True, checksums=True)
    assert {binding.host for binding in prepared.bindings} == {"a", "alias", "b"}
    assert {host for host, _ in destinations} == {"a", "b"}
    for root in (target, locations["b"]):
        for name, data in DATA.items():
            path = root / manifest.relative_snapshot / name
            assert path.read_bytes() == data
            assert not path.is_symlink()
    assert all((original / name).is_symlink() for name in DATA)
    destinations.clear()
    preparation.prepare_model("org/model", ["a", "alias", "b"], cache_dir="/hf", manifest=manifest, offline=True, checksums=True)
    assert destinations == []


@POSIX
@pytest.mark.parametrize("cancel", [True, False])
def test_real_stream_cleans_processes_without_closing_borrowed_session(tmp_path, cancel):
    marker = tmp_path / "started"
    child = tmp_path / "child"
    script = f"(trap '' TERM; touch {shlex.quote(str(marker))}; exec sleep 60) &\necho $! > {shlex.quote(str(child))}\nwait\n"
    session = SshHostSession()
    try:
        with ModelHostIO(session=session) as io:

            def stop():
                return cancel and marker.exists()

            with pytest.raises(ModelArtifactError, match="cancelled|deadline"):
                io.execute("localhost", script, timeout=0.4, cancelled=stop)
            assert not session._processes
            assert session.execute("localhost", ["true"]).returncode == 0
        assert session.execute("localhost", ["true"]).returncode == 0
        pid = int(child.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            import psutil

            assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        session.close()


def test_cluster_transfer_settings_roundtrip_and_override(tmp_path):
    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterDistributionConfig, ModelDistributionPrefs
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.api._wire import to_wire

    prefs = ModelDistributionPrefs.from_dict({"streams_per_host": 3, "max_streams": 8})
    assert prefs.to_dict() == {"streams_per_host": 3, "max_streams": 8}
    assert not prefs.is_default()
    assert to_wire(prefs)["max_streams"] == 8
    config = SparkrunConfig(config_path=tmp_path / "config.yaml")
    config.set("model_transfer_streams_per_host", 1)
    cluster = ClusterDefinition(name="test", hosts=["a"], distribution=ClusterDistributionConfig(model=prefs))
    assert transport.copy_limits(config.for_cluster(cluster)) == (3, 8)
    assert transport.copy_limits(config) == (1, 16)


@POSIX
def test_publication_leases_identify_aliases_but_not_distinct_caches(tmp_path):
    with ModelHostIO() as io:
        first = io.lock("localhost", str(tmp_path / "one"), "org/model")
        alias = io.lock("127.0.0.1", str(tmp_path / "one"), "org/model")
        second = io.lock("localhost", str(tmp_path / "two"), "org/model")
        assert first == alias
        assert first != second
        assert len(io.session._processes) == 2


def test_retry_progress_never_goes_backwards_or_exceeds_total():
    from sparkrun.models._copy_progress import CopyProgress

    events = []
    with operation_scope(state=OperationState(events=events.append)):
        req = request(("a",))
        progress = CopyProgress(req, {0: (req.targets[0], 100)})
        feed = progress.parser(0)
        progress.start(0)
        for data in (b"\r 80 80% 1MB/s\r", b"\r 20 20% 1MB/s\r", b"\r 1,200 100% 1MB/s\n"):
            feed(data)
            progress.emit(force=True)
        progress.finish(0, True)
        progress.emit(force=True, outcomes={"a": "complete"}, status="succeeded")
    totals = [e["current"] for e in events if "host" not in e]
    assert totals == [80, 80, 100, 100]


@POSIX
def test_stream_drains_large_output_and_delivers_large_scripts():
    # Exceeds OS argv/pipe capacity; no blocked stdin writer or stderr reader.
    script = ("# filler\n" * 40000) + "head -c 200000 /dev/zero\nhead -c 200000 /dev/zero >&2\n"
    seen = []
    with ModelHostIO() as io:
        result = io.execute(None, script, timeout=10, cancelled=lambda: False, stdout=lambda chunk: seen.append(len(chunk)))
    assert result.returncode == 0
    assert sum(seen) == 200000
    assert len(result.stdout) == len(result.stderr) == 65536


@pytest.mark.parametrize("exception", [KeyboardInterrupt, RuntimeError])
def test_callback_failure_joins_workers_and_marks_feedback_failed(simulated, exception):
    events = []

    def fail(*args):
        raise exception("callback broke")

    with operation_scope(state=OperationState(events=events.append)):
        req = request(("a",), progress=fail)
        with pytest.raises(exception, match="callback broke"):
            transport.builtin_copy(req, SimpleNamespace(execute=lambda *a, **kw: HostCommandResult("a", 0)), {})
    final = [e for e in events if "host" not in e][-1]
    assert final["status"] == "failed"
    assert final["metrics"]["active_streams"] == 0


@POSIX
def test_interrupted_remote_protocol_copy_keeps_and_reuses_partial_file(tmp_path, monkeypatch):
    # Execute the real rsync remote protocol locally, through a stand-in for
    # ssh. No cluster credentials/network service needed; unlike local rsync,
    # this uses delta transfer and partial-dir reuse.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    ssh = bindir / "ssh"
    ssh.write_text('#!/bin/sh\nwhile [ "$#" -gt 0 ] && [ "$1" != rsync ]; do shift; done\nexec "$@"\n')
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    data = {"model.safetensors": b"abc12345" * (256 * 1024)}
    manifest = artifact(data)
    source, dest = tmp_path / "source cache", tmp_path / "destination cache"
    cache(source, manifest, data)
    target = ModelTarget("remote", str(dest), "ib-remote", tuple(data))
    req = ModelCopyRequest(manifest, None, str(source), (target,), "resume")
    script = transport.copy_script(req, target, {})
    stopped = threading.Event()

    def progress(chunk):
        if b"%" in chunk:
            stopped.set()

    with ModelHostIO() as io:
        with pytest.raises(ModelArtifactError, match="cancelled"):
            io.execute(
                None,
                script.replace("rsync ", "rsync --bwlimit=512 ", 1),
                timeout=15,
                cancelled=stopped.is_set,
                stdout=progress,
            )
        partial = dest / manifest.relative_snapshot / ".sparkrun-partial/model.safetensors"
        assert 0 < partial.stat().st_size < len(data["model.safetensors"])
        result = io.execute(None, script.replace("rsync ", "rsync --stats ", 1), timeout=15, cancelled=lambda: False)
        assert result.returncode == 0, result.stderr
        import re

        matched = re.search(rb"Matched data: ([0-9,]+) bytes", result.stdout)
        assert matched is not None and int(matched[1].replace(b",", b"")) > 0
        assert io.observe("localhost", manifest, str(dest), checksums=True).complete
