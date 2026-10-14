"""Teardown must run on the substrate that launched the workload.

Every stop path used to emit ``docker rm -f`` regardless of executor.  For a
``local`` executor workload — a native process with a pidfile, no container
anywhere — Docker truthfully answers "no such container", the verification
pass sees nothing surviving, and ``stop`` reports success while the process
keeps serving.  Observed live: a vLLM server survived ``stop --all``, ``stop
<cluster_id>`` *and* had its job metadata deleted as a "confirmed" teardown.

These tests pin the seam that fixes it: ``Executor.teardown_script`` is the
one source of a teardown script, and every dispatch path reaches the right
executor.  The local-executor cases run against **real processes** rather than
a mocked shell, because the bug was precisely that a plausible-looking script
did nothing to the process.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

import pytest

from sparkrun.orchestration.executors._base import ExecutorConfig
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.executors.local import LocalExecutor
from sparkrun.orchestration.teardown import parse_teardown_removed

CID = "sparkrun_aaaabbbbccccdddd_eeeeffff0000"
NAME = CID + "_solo"


def _local_executor(tmp_path) -> LocalExecutor:
    return LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path / "pids"), log_dir=str(tmp_path / "logs")))


def _run(script: str) -> "tuple[int, str, str]":
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    return proc.returncode, proc.stdout, proc.stderr


def _spawn_group(tmp_path, name: str = NAME) -> int:
    """Launch a real workload through the executor's own ``run_cmd``.

    Uses the launch path under test rather than an approximation of it, and
    reads back the pidfile it writes.  The payload backgrounds a second
    sleeper so the test can tell a process-*group* kill from merely
    signalling the direct child — the distinction ``setsid`` +
    ``kill -- -<pgid>`` exists for.

    Because ``run_cmd`` backgrounds the workload and the launching shell then
    exits, the workload is orphaned to init exactly as it is when the launch
    arrives over SSH.  That matters here: a workload left as a *child of
    pytest* would linger as an unreaped zombie after a successful kill, and
    zombies answer both ``kill -0`` and ``pgrep``.
    """
    executor = _local_executor(tmp_path)
    script = executor.run_cmd(image="", command="sleep 300 & sleep 300", container_name=name)
    subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30, check=True)

    pid_file = tmp_path / "pids" / ("%s.pid" % name)
    pid = int(pid_file.read_text().strip())
    # Wait for bash to fork the background sleeper so a group assertion isn't
    # racing the setup.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and len(_group_members(pid)) < 2:
        time.sleep(0.05)
    return pid


def _group_members(pgid: int) -> list[str]:
    """Live PIDs in process group *pgid* (``setsid`` makes pgid == the pid).

    Zombies are excluded, and that exclusion is the point: ``os.kill(pid, 0)``
    and ``pgrep`` both succeed for a defunct process, so either would report a
    successfully-killed workload as a survivor.
    """
    proc = subprocess.run(["ps", "-o", "pid=,stat=", "-g", str(pgid)], capture_output=True, text=True)
    members = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and "Z" not in parts[1]:
            members.append(parts[0])
    return members


def _reap(pgid: int) -> None:
    """Best-effort cleanup so a failing assertion can't leak a sleeper."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# The local executor's teardown must actually kill the process
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="process-group teardown is POSIX-only")
def test_local_teardown_kills_the_real_process_group(tmp_path):
    """The regression, end to end: a live native workload must not survive.

    ``docker rm -f <name>`` exits 0 for a container that does not exist, so
    the old teardown reported this exact workload as removed and verified
    gone while it kept running.
    """
    pid = _spawn_group(tmp_path)
    try:
        assert len(_group_members(pid)) >= 2, "workload failed to start; test cannot prove anything"

        rc, out, err = _run(_local_executor(tmp_path).teardown_script([NAME]))

        assert rc == 0, err
        assert parse_teardown_removed(out) == 1
        # The whole group goes, not just the direct child — that is why
        # run_cmd uses setsid and stop_cmd signals the negative pid.
        assert _group_members(pid) == [], "the workload survived its own teardown"
        assert not (tmp_path / "pids" / ("%s.pid" % NAME)).exists()
    finally:
        _reap(pid)


@pytest.mark.skipif(os.name != "posix", reason="process-group teardown is POSIX-only")
def test_local_teardown_does_not_count_a_stale_pidfile(tmp_path):
    """A pidfile whose process is already gone is residue, not a removal.

    It is still cleaned up — teardown is idempotent and prunes what it finds —
    but reporting it as a stopped workload would inflate the count exactly
    like ``docker rm -f`` on an absent container did.
    """
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir(parents=True)
    # A PID that cannot be running: 0 is never a valid process to signal.
    (pid_dir / ("%s.pid" % NAME)).write_text("999999999\n")

    rc, out, err = _run(_local_executor(tmp_path).teardown_script([NAME]))

    assert rc == 0, err
    assert parse_teardown_removed(out) == 0
    assert not (pid_dir / ("%s.pid" % NAME)).exists()


def test_local_teardown_of_absent_workload_is_success(tmp_path):
    """Teardown is idempotent: nothing to stop is success, not failure."""
    rc, out, err = _run(_local_executor(tmp_path).teardown_script([NAME]))

    assert rc == 0, err
    assert parse_teardown_removed(out) == 0


@pytest.mark.skipif(os.name != "posix", reason="process-group teardown is POSIX-only")
def test_local_teardown_fails_when_the_workload_survives(tmp_path):
    """A surviving workload must fail the teardown, naming it on stderr.

    The generic composition's verification pass is what makes a `local`
    teardown's exit code meaningful; without it a caller cannot tell a real
    stop from a no-op — the disease this whole seam exists to cure.
    """
    pid = _spawn_group(tmp_path)
    try:
        # Neuter only the kill, leaving the census and verification intact
        # (and the retry sleep, so the test doesn't pay the grace period).
        script = _local_executor(tmp_path).teardown_script([NAME])
        script = script.replace("kill -TERM", "true kill -TERM").replace("kill -KILL", "true kill -KILL")
        script = script.replace("sleep 1", "true")

        rc, _out, err = _run(script)

        assert rc == 1
        assert NAME in err
        assert (tmp_path / "pids" / ("%s.pid" % NAME)).exists(), "failed teardown must preserve recovery state"
        assert _group_members(pid) != []
    finally:
        _reap(pid)


# ---------------------------------------------------------------------------
# Executor selection on the dispatch paths
# ---------------------------------------------------------------------------


def test_docker_executor_teardown_checks_the_daemon(tmp_path):
    """Docker overrides the generic composition to verify docker answered.

    An absent/dead docker makes every existence probe report "not present",
    which the generic verification would read as a successful teardown.
    """
    script = DockerExecutor().teardown_script([NAME]).replace("docker ", "docker-does-not-exist ")

    rc, _out, _err = _run(script)

    assert rc == 1


def test_cleanup_containers_by_host_uses_the_supplied_executor(monkeypatch, tmp_path):
    """The dispatch primitive runs *this* executor's script, not docker's."""
    from sparkrun.orchestration import primitives

    scripts: dict[str, str] = {}

    def fake_run(host, command, **kwargs):
        from sparkrun.orchestration.ssh import RemoteResult

        scripts[host] = command
        return RemoteResult(host=host, returncode=0, stdout="sparkrun_removed=1\n", stderr="")

    monkeypatch.setattr(primitives, "run_command_on_host", fake_run)

    primitives.cleanup_containers_by_host({"h1": [NAME]}, executor=_local_executor(tmp_path))

    assert "docker" not in scripts["h1"]
    assert "kill -TERM" in scripts["h1"]


def test_cleanup_containers_by_host_defaults_to_docker(monkeypatch):
    """No executor named → the historical docker teardown, unchanged."""
    from sparkrun.orchestration import primitives

    scripts: dict[str, str] = {}

    def fake_run(host, command, **kwargs):
        from sparkrun.orchestration.ssh import RemoteResult

        scripts[host] = command
        return RemoteResult(host=host, returncode=0, stdout="sparkrun_removed=1\n", stderr="")

    monkeypatch.setattr(primitives, "run_command_on_host", fake_run)

    primitives.cleanup_containers_by_host({"h1": [NAME]})

    assert "docker rm -f" in scripts["h1"]


def test_api_stop_tears_down_with_the_launching_executor(monkeypatch, tmp_path):
    """``stop <cluster_id>`` uses the executor recorded in the job metadata.

    The executor was already resolved here for ``enumerate_containers`` and
    then thrown away, which is how a ``local`` job got a docker teardown.
    """
    import sparkrun.api as api
    from sparkrun.orchestration.ssh import RemoteResult

    monkeypatch.setattr(
        "sparkrun.orchestration.job_metadata.load_job_metadata",
        lambda cid, cache_dir=None, **kw: {"hosts": ["h1"], "executor": "local", "ssh_user": "alice"},
    )

    seen: dict = {}

    def fake_cleanup(host_containers, ssh_kwargs=None, dry_run=False, max_workers=None, *, executor=None):
        seen["executor"] = executor
        return {h: RemoteResult(host=h, returncode=0, stdout="sparkrun_removed=1\n", stderr="") for h in host_containers}

    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", fake_cleanup)

    api.stop(workload_id=CID, cache_dir=str(tmp_path))

    assert isinstance(seen["executor"], LocalExecutor)


def test_api_stop_discovers_the_executor_when_metadata_is_gone(monkeypatch, tmp_path):
    """With no metadata, ask the running workload rather than defaulting.

    Job metadata does go missing — an interrupted launch, a cleared cache, or
    a previous "successful" stop that deleted the record of a workload it
    never actually stopped.  Falling back to the default executor there means
    a teardown aimed at the wrong substrate, which exits 0 having done
    nothing: the recovery path for the original bug would itself be broken.
    """
    import sparkrun.api as api
    from sparkrun.core.cluster_status import ClusterStatus, ContainerDetail, HostOccupancy, RunningWorkload
    from sparkrun.orchestration.ssh import RemoteResult

    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda cid, cache_dir=None, **kw: None)
    monkeypatch.setattr(
        "sparkrun.api._status.status",
        lambda hosts, **kw: ClusterStatus(
            hosts=(
                HostOccupancy(
                    host="h1",
                    workloads=(
                        RunningWorkload(
                            cluster_id=CID,
                            containers=(
                                ContainerDetail(name=NAME, role="solo", status="Up (pid 7)", image="(local process)", executor="local"),
                            ),
                        ),
                    ),
                    used_slots=1,
                ),
            ),
        ),
    )

    seen: dict = {}

    def fake_cleanup(host_containers, ssh_kwargs=None, dry_run=False, max_workers=None, *, executor=None):
        seen["executor"] = executor
        return {h: RemoteResult(host=h, returncode=0, stdout="sparkrun_removed=1\n", stderr="") for h in host_containers}

    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", fake_cleanup)

    api.stop(workload_id=CID, hosts=("h1",), cache_dir=str(tmp_path))

    assert isinstance(seen["executor"], LocalExecutor)


def test_api_stop_executor_discovery_tolerates_a_failed_sweep(monkeypatch, tmp_path):
    """A sweep that raises must not block the stop — it falls back to the chain."""
    import sparkrun.api as api
    from sparkrun.orchestration.ssh import RemoteResult

    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda cid, cache_dir=None, **kw: None)

    def boom(hosts, **kw):
        raise OSError("host unreachable")

    monkeypatch.setattr("sparkrun.api._status.status", boom)

    seen: dict = {}

    def fake_cleanup(host_containers, ssh_kwargs=None, dry_run=False, max_workers=None, *, executor=None):
        seen["executor"] = executor
        return {h: RemoteResult(host=h, returncode=0, stdout="sparkrun_removed=0\n", stderr="") for h in host_containers}

    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", fake_cleanup)

    result = api.stop(workload_id=CID, hosts=("h1",), cache_dir=str(tmp_path))

    assert isinstance(seen["executor"], DockerExecutor)
    assert result.hosts_failed == ()


# ---------------------------------------------------------------------------
# stop --all across a mixed cluster
# ---------------------------------------------------------------------------


def _mixed_report():
    """One host running a docker container *and* a native local process.

    Not contrived: docker and local share the ``"host"`` status scope, so
    ``api.status`` merges both into one snapshot and ``stop --all`` is
    expected to tear both down.
    """
    from sparkrun.core.cluster_manager import ClusterSoloEntry, ClusterStatusResult

    local_cid = "sparkrun_1111222233334444_55556666"
    return ClusterStatusResult(
        groups={},
        solo_entries=[
            ClusterSoloEntry(cluster_id=CID, host="h1", name=NAME, status="Up", image="img", meta={}),
            ClusterSoloEntry(
                cluster_id=local_cid,
                host="h1",
                name=local_cid + "_solo",
                status="Up (pid 42)",
                image="(local process)",
                meta={},
            ),
        ],
        errors={},
        idle_hosts=[],
        pending_ops=[],
        total_containers=2,
        host_count=1,
        container_executors={("h1", NAME): "docker", ("h1", local_cid + "_solo"): "local"},
    )


def test_stop_all_dispatches_each_workload_to_its_own_executor(monkeypatch, tmp_path):
    """A mixed host gets one teardown per executor, each with only its own names."""
    import sparkrun.api as api
    from sparkrun.orchestration.ssh import RemoteResult

    calls: list[tuple[str, list[str]]] = []

    def fake_cleanup(host_containers, ssh_kwargs=None, dry_run=False, max_workers=None, *, executor=None):
        name = type(executor).__name__ if executor is not None else "default"
        calls.append((name, sorted(host_containers["h1"])))
        return {h: RemoteResult(host=h, returncode=0, stdout="sparkrun_removed=1\n", stderr="") for h in host_containers}

    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", fake_cleanup)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.remove_job_metadata", lambda cid, cache_dir=None, **kw: None)

    result = api.stop_all(["h1"], cache_dir=str(tmp_path), discovered=_mixed_report())

    by_executor = dict(calls)
    assert by_executor["DockerExecutor"] == [NAME]
    assert by_executor["LocalExecutor"] == ["sparkrun_1111222233334444_55556666_solo"]
    # Both executors' counts are reported, not just the last one's.
    assert result.containers_removed == 2
    assert result.hosts_failed == {}


def test_stop_all_keeps_metadata_when_one_executors_teardown_fails(monkeypatch, tmp_path):
    """A host is only "stopped" if *every* executor on it confirmed.

    Otherwise a surviving native process would have its job metadata deleted
    because the docker half of the same host succeeded — stranding a live
    workload with no way to find it again.
    """
    import sparkrun.api as api
    from sparkrun.orchestration.ssh import RemoteResult

    def fake_cleanup(host_containers, ssh_kwargs=None, dry_run=False, max_workers=None, *, executor=None):
        if isinstance(executor, LocalExecutor):
            return {h: RemoteResult(host=h, returncode=1, stderr="workloads still present", stdout="") for h in host_containers}
        return {h: RemoteResult(host=h, returncode=0, stdout="sparkrun_removed=1\n", stderr="") for h in host_containers}

    removed: list[str] = []
    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", fake_cleanup)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.remove_job_metadata", lambda cid, cache_dir=None, **kw: removed.append(cid))

    result = api.stop_all(["h1"], cache_dir=str(tmp_path), discovered=_mixed_report())

    assert "h1" in result.hosts_failed
    assert removed == []
    assert result.success is False


def test_stop_all_unattributed_containers_use_the_default_executor(monkeypatch, tmp_path):
    """No attribution (an older snapshot) must not skip the host entirely."""
    import sparkrun.api as api
    from sparkrun.core.cluster_manager import ClusterSoloEntry, ClusterStatusResult
    from sparkrun.orchestration.ssh import RemoteResult

    report = ClusterStatusResult(
        groups={},
        solo_entries=[ClusterSoloEntry(cluster_id=CID, host="h1", name=NAME, status="Up", image="img", meta={})],
        errors={},
        idle_hosts=[],
        pending_ops=[],
        total_containers=1,
        host_count=1,
    )

    seen: dict = {}

    def fake_cleanup(host_containers, ssh_kwargs=None, dry_run=False, max_workers=None, *, executor=None):
        seen["executor"] = executor
        seen["names"] = host_containers["h1"]
        return {h: RemoteResult(host=h, returncode=0, stdout="sparkrun_removed=1\n", stderr="") for h in host_containers}

    monkeypatch.setattr("sparkrun.orchestration.primitives.cleanup_containers_by_host", fake_cleanup)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.remove_job_metadata", lambda cid, cache_dir=None, **kw: None)

    api.stop_all(["h1"], cache_dir=str(tmp_path), discovered=report)

    assert seen["executor"] is None  # → the primitive's docker default
    assert seen["names"] == [NAME]


# ---------------------------------------------------------------------------
# Supporting machinery
# ---------------------------------------------------------------------------


def test_merge_teardown_results_sums_counts_and_keeps_failures():
    """Per host: counts add up, and one failure fails the host."""
    from sparkrun.orchestration.primitives import merge_teardown_results
    from sparkrun.orchestration.ssh import RemoteResult

    a = {"h1": RemoteResult(host="h1", returncode=0, stdout="sparkrun_removed=2\n", stderr="")}
    b = {"h1": RemoteResult(host="h1", returncode=0, stdout="sparkrun_removed=3\n", stderr="")}

    merged = merge_teardown_results(a, b)
    assert parse_teardown_removed(merged["h1"].stdout) == 5
    assert merged["h1"].success

    failed = merge_teardown_results(a, {"h1": RemoteResult(host="h1", returncode=1, stdout="", stderr="boom")})
    assert not failed["h1"].success
    assert "boom" in failed["h1"].stderr


def test_status_attributes_each_container_to_its_reporting_executor():
    """The docker+local merge must not lose which executor saw what."""
    from sparkrun.core.cluster_status import (
        ClusterStatus,
        ContainerDetail,
        HostOccupancy,
        RunningWorkload,
        attribute_executor,
    )

    def snap(name, executor_name):
        return attribute_executor(
            ClusterStatus(
                hosts=(
                    HostOccupancy(
                        host="h1",
                        workloads=(
                            RunningWorkload(
                                cluster_id="cid_" + executor_name,
                                containers=(ContainerDetail(name=name, role="solo", status="Up", image="i"),),
                            ),
                        ),
                        used_slots=1,
                    ),
                ),
                executor=executor_name,
            ),
            executor_name,
        )

    merged = ClusterStatus.merge([snap("d_solo", "docker"), snap("l_solo", "local")])
    attribution = {c.name: c.executor for entry in merged.hosts for w in entry.workloads for c in w.containers}

    assert attribution == {"d_solo": "docker", "l_solo": "local"}


def test_classify_carries_attribution_into_the_display_result(tmp_path):
    """``ClusterStatusResult`` exposes (host, container) → executor for teardown."""
    from sparkrun.core.cluster_manager import classify_cluster_status
    from sparkrun.core.cluster_status import ClusterStatus, ContainerDetail, HostOccupancy, RunningWorkload

    snapshot = ClusterStatus(
        hosts=(
            HostOccupancy(
                host="h1",
                workloads=(
                    RunningWorkload(
                        cluster_id=CID,
                        containers=(ContainerDetail(name=NAME, role="solo", status="Up", image="i", executor="local"),),
                    ),
                ),
                used_slots=1,
            ),
        ),
    )

    result = classify_cluster_status(snapshot, cache_dir=str(tmp_path), host_list=["h1"])

    assert result.container_executors == {("h1", NAME): "local"}
