"""Public resolved-target contracts for TUI clients and finite log export."""

from dataclasses import FrozenInstanceError
import os
import subprocess
import sys
import threading
from unittest.mock import Mock

import pytest

from sparkrun import api
from sparkrun.core.cluster_status import ClusterStatus
from sparkrun.core.log_source import LogSource, source_sort_key
from sparkrun.core.operation import OperationState
from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata
from sparkrun.orchestration import logs


CID = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"


@pytest.fixture
def open_target(tmp_path, monkeypatch):
    monkeypatch.setattr(DockerExecutor, "query_status", lambda *a, **kw: ClusterStatus())

    def resolve(*, runtime="sglang", hosts=("h1", "User@H2", "h3"), **kwargs):
        recipe = Recipe({"sparkrun_version": "2", "runtime": runtime, "model": "test/model"})
        save_job_metadata(CID, recipe, list(hosts), cache_dir=str(tmp_path))
        return api.open_log_target(CID, cache_dir=str(tmp_path), **kwargs)

    return resolve


def test_target_metadata_is_immutable_and_resolution_opens_no_readers(open_target, monkeypatch):
    spawn = Mock(side_effect=AssertionError("resolution spawned a log reader"))
    monkeypatch.setattr(logs, "_spawn", spawn)
    target = open_target(runtime="vllm-ray", scope="all")
    assert target.cluster_id == CID
    assert target.runtime == "vllm-ray"
    assert isinstance(target.sources, tuple)
    assert target.sources[0].container == CID + "_head"
    with pytest.raises(FrozenInstanceError):
        target.runtime = "changed"
    spawn.assert_not_called()


@pytest.mark.parametrize("host", ["User@H2", "H2", "h2", "other@h2"])
def test_host_filter_preserves_worker_rank_and_connection_identity(open_target, host):
    target = open_target(host=host)
    (source,) = target.sources
    assert (source.host, source.rank, source.container) == ("User@H2", 1, CID + "_node_1")


def test_container_filter_selects_worker_despite_default_head_scope(open_target):
    target = open_target(container=CID + "_node_2")
    assert [(s.host, s.rank) for s in target.sources] == [("h3", 2)]


def test_filters_intersect_and_misses_preserve_metadata(open_target, tmp_path):
    with pytest.raises(api.JobNotFound, match="No log sources match"):
        open_target(host="h1", container=CID + "_node_2")
    assert load_job_metadata(CID, cache_dir=str(tmp_path)) is not None


def test_normalized_ambiguous_host_requires_exact_identity(open_target):
    with pytest.raises(api.SparkrunError, match="Ambiguous log host"):
        open_target(hosts=("alice@h1", "bob@h1"), host="h1")
    target = open_target(hosts=("alice@h1", "bob@h1"), host="bob@h1")
    assert [s.host for s in target.sources] == ["bob@h1"]


@pytest.mark.parametrize("kwargs", [{"host": ""}, {"host": "  "}, {"container": ""}, {"container": "\t"}, {"scope": "bogus"}])
def test_invalid_selectors_fail_eagerly(open_target, kwargs):
    with pytest.raises((ValueError, api.SparkrunError)):
        open_target(**kwargs)


@pytest.mark.parametrize("method", ["read", "follow", "page", "logs"])
@pytest.mark.parametrize("override", [False, True])
def test_cancel_inheritance_and_explicit_override(open_target, monkeypatch, tmp_path, method, override):
    sctx = api.default_sctx()
    sctx.operation = OperationState()
    explicit = threading.Event() if override else None
    expected = explicit if override else sctx.operation.cancel
    target = open_target(sctx=sctx)
    calls = []

    def read(*args, **kwargs):
        calls.append(kwargs)
        yield from ()

    monkeypatch.setattr(logs, "read_log_sources", read)
    if method == "logs":
        list(api.logs(CID, cache_dir=str(tmp_path), sctx=sctx, cancel=explicit))
    elif method == "page":
        target.page(target.sources[0], skip=0, cancel=explicit)
    else:
        list(getattr(target, method)(cancel=explicit))
    assert calls[0]["cancel"] is expected
    assert calls[0]["check"] is True
    if method == "follow":
        assert calls[0]["follow"] is True and calls[0]["tail"] == 200


def test_open_target_cancel_overrides_operation_for_future_reads(open_target, monkeypatch):
    sctx = api.default_sctx()
    sctx.operation = OperationState()
    explicit = threading.Event()
    target = open_target(sctx=sctx, cancel=explicit)
    capture = Mock(return_value=iter(()))
    monkeypatch.setattr(logs, "read_log_sources", capture)
    list(target.read())
    assert capture.call_args.kwargs["cancel"] is explicit


@pytest.mark.parametrize("skip,count", [(0, 5), (8, 5), (10, 5), (20, 5), (0, 0), (0, 20), (3, 4)])
def test_page_short_logs_and_boundaries_using_real_shared_reader(open_target, monkeypatch, skip, count):
    target = open_target(host="h2")
    rows = [str(n) for n in range(10)]

    def build(executor, source, *, follow, tail, ssh_kwargs):
        assert not follow
        assert tail == skip + count
        assert source is target.sources[0]
        return [sys.executable, "-c", "print(" + repr("\n".join(rows[-tail:])) + ")"]

    monkeypatch.setattr(logs, "build_read_command", build)
    result = target.page(target.sources[0], skip=skip, count=count)
    end = max(0, len(rows) - skip)
    assert [line.text for line in result] == rows[max(0, end - count) : end]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"skip": -1},
        {"skip": True},
        {"skip": 0, "count": -1},
        {"skip": 0, "count": 1.5},
        {"skip": 0, "timeout": 0},
        {"skip": 0, "timeout": float("nan")},
    ],
)
def test_page_validation(open_target, kwargs):
    target = open_target()
    with pytest.raises(ValueError):
        target.page(target.sources[0], **kwargs)


def test_page_rejects_source_outside_target(open_target):
    target = open_target()
    with pytest.raises(ValueError, match="does not belong"):
        target.page(LogSource("arbitrary-host", "arbitrary-container"), skip=0)


@pytest.mark.parametrize(
    "script,error", [("import sys; sys.exit(7)", subprocess.CalledProcessError), ("import time; time.sleep(60)", subprocess.TimeoutExpired)]
)
def test_page_propagates_failures_and_timeout(open_target, monkeypatch, script, error):
    target = open_target()
    monkeypatch.setattr(logs, "build_read_command", lambda *a, **kw: [sys.executable, "-c", script])
    with pytest.raises(error):
        target.page(target.sources[0], skip=0, timeout=0.5)


def test_cancelled_page_discards_partial_results(open_target, monkeypatch):
    cancel = threading.Event()
    target = open_target(cancel=cancel)

    def read(*args, **kwargs):
        yield api.LogLine("h1", "c", "partial")
        cancel.set()

    monkeypatch.setattr(logs, "read_log_sources", read)
    assert target.page(target.sources[0], skip=0) == []
    assert cancel.is_set()


def test_rank_order_is_shared_and_deterministic():
    sources = [
        LogSource("z", "worker", role="worker"),
        LogSource("a", "rank2", rank=2),
        LogSource("z", "head", role="head"),
        LogSource("a", "rank1", rank=1),
    ]
    assert [s.container for s in sorted(sources, key=source_sort_key)] == ["head", "rank1", "rank2", "worker"]


def test_cli_reports_checked_reader_failure(monkeypatch, capsys):
    from sparkrun.cli._stop_logs import _render_logs

    def failed(*args, **kwargs):
        yield api.LogLine("h1", "c", "remote diagnostic")
        raise subprocess.CalledProcessError(7, ["reader"])

    monkeypatch.setattr(api, "logs", failed)
    with pytest.raises(SystemExit) as error:
        _render_logs(api.default_sctx(), cluster_id=CID, hosts=["h1"], scope="head", lines=10, follow=False)
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "remote diagnostic" in output.out
    assert "Log reader exited with status 7" in output.err


def test_cli_rejects_negative_tail_before_reading():
    from click.testing import CliRunner
    from sparkrun.cli._stop_logs import logs_cmd as command

    result = CliRunner().invoke(command, [CID, "--tail", "-1"])
    assert result.exit_code == 2
    assert "Invalid value" in result.output and "--tail" in result.output


@pytest.mark.parametrize("follow", [False, True])
@pytest.mark.skipif(os.name == "nt", reason="Local executor fixture requires POSIX bash/tail")
def test_local_target_reads_entire_log_when_tail_is_none(tmp_path, monkeypatch, follow):
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.executors._base import ExecutorConfig

    path = tmp_path / "model.log"
    expected = ["line-%d" % n for n in range(30)]
    path.write_text("\n".join(expected) + "\n")
    executor = LocalExecutor(ExecutorConfig(log_file=str(path)))
    source = LogSource("localhost", "local-reader")
    target = api.LogTarget("cid", "sglang", (source,), executor, {})
    monkeypatch.setattr("sparkrun.orchestration.ssh.should_run_locally", lambda *a: True)
    reader = target.read(tail=None, follow=follow)
    try:
        assert [next(reader).text for _ in expected] == expected
        if not follow:
            assert list(reader) == []
    finally:
        reader.close()


@pytest.fixture
def diagnostic_ssh(tmp_path, monkeypatch):
    """Exercise real command/SSH quoting and distinct transport stderr locally."""
    if os.name == "nt":
        pytest.skip("Remote POSIX shell emulation requires bash/tail")
    ssh = tmp_path / "ssh.py"
    ssh.write_text(
        "import subprocess, sys\n"
        "print('SSH warning', file=sys.stderr, flush=True)\n"
        "sys.exit(subprocess.call(' '.join(sys.argv[1:]), shell=True))\n"
    )
    monkeypatch.setattr("sparkrun.orchestration.ssh.should_run_locally", lambda *a: False)
    monkeypatch.setattr("sparkrun.orchestration.ssh.build_ssh_cmd", lambda *a, **kw: [sys.executable, str(ssh)])


@pytest.mark.parametrize("skip,count", [(0, 5), (5, 5), (8, 5), (10, 5)])
def test_page_offsets_exclude_ssh_diagnostics(tmp_path, diagnostic_ssh, caplog, skip, count):
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.executors._base import ExecutorConfig

    path = tmp_path / "model.log"
    rows = ["line-%d" % n for n in range(10)]
    path.write_text("\n".join(rows) + "\n")
    executor = LocalExecutor(ExecutorConfig(log_file=str(path)))
    source = LogSource("remote", "local-reader")
    target = api.LogTarget("cid", "sglang", (source,), executor, {})
    page = target.page(source, skip=skip, count=count)
    end = max(0, len(rows) - skip)
    assert [line.text for line in page] == rows[max(0, end - count) : end]
    assert "SSH warning" in caplog.text


@pytest.mark.parametrize("follow", [False, True])
def test_docker_workload_stderr_survives_transport_diagnostic_separation(tmp_path, monkeypatch, diagnostic_ssh, follow):
    from sparkrun.core.log_source import MODE_STDOUT

    # A Docker stand-in reproduces docker logs' two output descriptors. The
    # real DockerExecutor builds the command, which runs through the SSH fixture.
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nprintf 'workload stdout\\n'\nprintf 'workload stderr\\n' >&2\n")
    docker.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)
    source = LogSource("remote", "docker-reader", mode=MODE_STDOUT)
    target = api.LogTarget("cid", "vllm-ray", (source,), DockerExecutor(), {})
    assert [line.text for line in target.read(follow=follow)] == ["workload stdout", "workload stderr"]
    assert [line.text for line in target.page(source, skip=1, count=1)] == ["workload stdout"]
