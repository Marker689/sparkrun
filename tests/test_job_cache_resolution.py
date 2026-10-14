"""Metadata entry points share configured storage without requiring plugin startup."""

from unittest.mock import Mock

import pytest
import yaml

from sparkrun import api
from sparkrun.application import initialize
from sparkrun.core import bootstrap
from sparkrun.core.config import SparkrunConfig, resolve_configured_cache_dir
from sparkrun.orchestration import job_metadata
from test_proxy_management_initialization import prepare_application


@pytest.fixture(params=[False, True], ids=["sparkrun", "alternate"])
def job(tmp_path, monkeypatch, request):
    config, identity = prepare_application(tmp_path, monkeypatch, alternate=request.param, enabled=False)
    root = tmp_path / "configured-cache"
    data = yaml.safe_load(config.read_text())
    data["cache_dir"] = str(root)
    config.write_text(yaml.safe_dump(data))
    from sparkrun.core.application_profile import get_application_profile

    cluster_id = get_application_profile().resource_namespace + "_fixture"
    path = root / "jobs" / "fixture.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({"cluster_id": cluster_id, "distribution": identity, "hosts": ["fixture-host"], "executor": "docker"}))
    return config, root, cluster_id, path


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("operation", ["list_jobs", "stop", "logs"])
def test_implicit_context_finds_configured_jobs(job, monkeypatch, initialized, operation):
    _, root, cluster_id, _ = job
    context = initialize() if initialized else None
    if operation == "list_jobs":
        assert [j.workload_id for j in api.list_jobs()] == [cluster_id]
        if context is not None:
            assert api.list_jobs() == api.list_jobs(sctx=context)
    else:
        boundary = Mock(side_effect=RuntimeError("metadata resolved"))
        monkeypatch.setattr("sparkrun.api._resolve.resolve_cluster_for_job", boundary)
        # Stop at host resolution; never contact a host or stop a real workload.
        with pytest.raises(RuntimeError, match="metadata resolved"):
            getattr(api, operation)(workload_id=cluster_id)
        assert boundary.call_args.kwargs["meta"]["cluster_id"] == cluster_id
        if context is not None:
            with pytest.raises(RuntimeError, match="metadata resolved"):
                getattr(api, operation)(workload_id=cluster_id, sctx=context)
    assert (bootstrap._variables is not None) == initialized
    assert resolve_configured_cache_dir() == root


def test_explicit_cache_wins_over_context_and_configuration(job, tmp_path):
    _, root, cluster_id, _ = job
    context = initialize()
    other = tmp_path / "other-cache"
    assert api.list_jobs(cache_dir=other, sctx=context) == []
    assert job_metadata.load_job_metadata(cluster_id, cache_dir=str(other), sctx=context) is None
    assert resolve_configured_cache_dir(other, config=context.config) == other
    assert resolve_configured_cache_dir(config=context.config) == root


def test_passive_metadata_remains_available_after_plugin_failure(job, monkeypatch):
    _, _, cluster_id, path = job
    monkeypatch.setattr(bootstrap, "_register_plugins", Mock(side_effect=RuntimeError("broken plugin")))
    with pytest.raises(RuntimeError, match="broken plugin"):
        initialize()
    assert [j.workload_id for j in api.list_jobs()] == [cluster_id]
    assert job_metadata.load_job_metadata(cluster_id)["hosts"] == ["fixture-host"]
    job_metadata.remove_job_metadata(cluster_id)
    assert not path.exists()


def test_bad_configuration_is_not_silently_replaced_by_default_storage(job, tmp_path):
    config, _, _, _ = job
    config.write_text("invalid: [\n")
    with pytest.raises(yaml.YAMLError):
        api.list_jobs()
    # Explicit recovery storage does not need to read that configuration.
    assert api.list_jobs(cache_dir=tmp_path / "explicit") == []


def test_cache_property_programming_errors_propagate(job, monkeypatch):
    context = initialize()

    def broken(self):
        raise RuntimeError("config bug")

    monkeypatch.setattr(SparkrunConfig, "cache_dir", property(broken))
    with pytest.raises(RuntimeError, match="config bug"):
        api.list_jobs(sctx=context)
    with pytest.raises(RuntimeError, match="config bug"):
        job_metadata.load_job_metadata(job[2], sctx=context)
