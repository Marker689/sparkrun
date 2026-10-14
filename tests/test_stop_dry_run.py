"""``stop --dry-run`` previews a teardown; ``workload_id`` is the canonical stop id."""

import pytest
from click.testing import CliRunner

from sparkrun import api


def _no_teardown(*args, **kwargs):
    raise AssertionError("dry run must not tear anything down")


def test_api_stop_dry_run_resolves_without_teardown(tmp_path, monkeypatch, idle_occupancy):
    from sparkrun.api import _stop
    from sparkrun.orchestration.executors.docker import DockerExecutor

    monkeypatch.setattr(_stop, "_stop_containers", _no_teardown)
    monkeypatch.setattr(DockerExecutor, "stop_workload", _no_teardown)
    result = api.stop(workload_id="sparkrun_previewonly", hosts=("h1", "h2"), cache_dir=str(tmp_path), dry_run=True)
    assert result.dry_run is True
    assert result.containers_removed == 0
    assert result.hosts_targeted == ("h1", "h2")
    assert result.workload_id == "sparkrun_previewonly"


def test_api_stop_has_no_cluster_id_keyword():
    with pytest.raises(TypeError):
        api.stop(hosts=("h1",), **{"cluster_id": "sparkrun_legacyspelling"})


def test_cli_stop_target_dry_run_does_not_stop(monkeypatch):
    from sparkrun.cli import main

    calls = []

    def fake_stop(**kwargs):
        calls.append(kwargs)
        return api.StopResult(workload_id=kwargs["workload_id"], hosts_targeted=("h1",), containers_removed=0, dry_run=kwargs["dry_run"])

    monkeypatch.setattr(api, "stop", fake_stop)
    result = CliRunner().invoke(main, ["stop", "sparkrun_0123456789ab", "--hosts", "h1", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert calls and calls[0]["dry_run"] is True
    assert "[dry-run] Would stop" in result.output
    assert "Workload stopped" not in result.output
