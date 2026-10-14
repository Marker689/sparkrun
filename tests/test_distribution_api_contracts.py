"""Regression tests for application boundaries and benchmark retry contracts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
from unittest.mock import Mock

import pytest

from test_benchmark_startup_collection import bench_env as bench_env


@pytest.mark.parametrize("unit_present", [False, True])
@pytest.mark.parametrize("foreign_artifact", ["recipe", "service_cluster", "cluster"])
def test_uninstall_preserves_foreign_service_files(tmp_path, unit_present, foreign_artifact):
    from sparkrun.orchestration.systemd_service import (
        render_uninstall_script as _render_uninstall_script,
        service_artifacts as _service_artifacts,
    )

    paths = dict(
        zip(("recipe", "service_cluster", "cluster"), map(Path, _service_artifacts("review", "review", str(tmp_path))), strict=True)
    )
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# sparkrun.distribution=%s\n" % ("jetsonrun" if name == foreign_artifact else "sparkrun"))
    unit = tmp_path / "sparkrun-review.service"
    if unit_present:
        unit.write_text("# sparkrun.distribution=sparkrun\n")
    original = {p: p.read_bytes() for p in [*paths.values(), unit] if p.exists()}
    script = _render_uninstall_script("review", "review", str(tmp_path)).replace("/etc/systemd/system/", str(tmp_path) + "/")
    calls = tmp_path / "systemctl-calls"
    result = subprocess.run(
        ["bash", "-c", 'systemctl() { echo called >> "%s"; };\n' % calls + script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode != 0 and "another application" in result.stderr
    assert not calls.exists()
    assert all(p.read_bytes() == value for p, value in original.items())


@pytest.mark.parametrize("legacy", [False, True])
def test_service_cleanup_and_privileged_install_after_cleanup(tmp_path, legacy):
    from sparkrun.orchestration.systemd_service import (
        render_install_script as _render_install_script,
        render_sudo_install_script as _render_sudo_install_script,
        render_uninstall_script as _render_uninstall_script,
        service_artifacts as _service_artifacts,
    )

    def execute(script):
        script = script.replace("/etc/systemd/system/", str(tmp_path) + "/")
        return subprocess.run(["bash", "-c", "systemctl() { :; };\n" + script], capture_output=True, text=True, timeout=10)

    assert execute(_render_install_script("review", "name: review", "hosts: []", "review", str(tmp_path))).returncode == 0
    unit = "# sparkrun.distribution=sparkrun\n[Unit]\nDescription=sparkrun inference: review\n"
    install = _render_sudo_install_script("review", unit, user_home=str(tmp_path), cluster_name="review")
    assert execute(install).returncode == 0
    if legacy:
        for p in [*map(Path, _service_artifacts("review", "review", str(tmp_path))), tmp_path / "sparkrun-review.service"]:
            p.write_text(p.read_text().replace("# sparkrun.distribution=sparkrun\n", ""))
    assert execute(_render_uninstall_script("review", "review", str(tmp_path))).returncode == 0
    assert not (tmp_path / ".config/sparkrun/services/review").exists()
    assert not (tmp_path / ".config/sparkrun/clusters/review.yaml").exists()
    # The privileged phase must not install an orphan unit if cleanup occurred
    # after the unprivileged install phase released its lock.
    assert execute(install).returncode != 0
    assert not (tmp_path / "sparkrun-review.service").exists()


def test_uninstall_serializes_with_user_install(tmp_path):
    import fcntl
    import os
    from sparkrun.orchestration.systemd_service import (
        render_install_script as _render_install_script,
        render_uninstall_script as _render_uninstall_script,
    )

    install = _render_install_script("review", "name: review", "hosts: []", "review", str(tmp_path))
    uninstall = _render_uninstall_script("review", "review", str(tmp_path))
    # Hold exactly the shared inode. Neither generated script may progress to
    # writing files or systemctl while another operation holds it.
    fd = os.open(tmp_path, os.O_RDONLY)
    processes = []
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        for script in (install, uninstall):
            script = script.replace("/etc/systemd/system/", str(tmp_path) + "/")
            proc = subprocess.Popen(["bash", "-c", "systemctl() { :; };\n" + script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(proc)
            with pytest.raises(subprocess.TimeoutExpired):
                proc.wait(timeout=0.1)
        assert not (tmp_path / ".config").exists()
    finally:
        os.close(fd)
        for proc in processes:
            proc.communicate(timeout=10)
            assert proc.returncode == 0


@pytest.fixture
def integration_registry(monkeypatch):
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})


def _completed_state(sctx, settings):
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from sparkrun.core.benchmark_integrations import STATE_KEY

    state = BenchmarkRunState(
        benchmark_id="bench_contract",
        cluster_id="job",
        recipe_qualified_name="recipe",
        framework="test-bench",
        profile=None,
        base_args={},
        schedule=[{}],
        completed_indices=[0],
        extras={
            "measurement_complete": True,
            "benchmark_category": "performance",
            STATE_KEY: {"review": {"settings": settings, "data": {}}},
        },
    )
    state.save(str(sctx.config.cache_dir))
    (state.state_dir(str(sctx.config.cache_dir)) / "result.yaml").write_text("requests_per_second: 12\n")
    return state


def test_resume_validates_merged_settings_without_measurement_defaults(integration_registry):
    from sparkrun.api import resume_benchmark, BenchmarkFailed, BenchmarkIntegrationFailed
    from sparkrun.api._context import default_sctx
    from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration

    sctx = default_sctx()
    validated = []

    def validate(ctx):
        validated.append(dict(ctx.settings))
        if not ctx.settings.get("endpoint"):
            raise BenchmarkFailed("endpoint required")

    prepare = Mock(side_effect=AssertionError("Resume cannot reselect measurement defaults"))
    register_benchmark_integration(BenchmarkIntegration("review", prepare=prepare, validate=validate))
    state = _completed_state(sctx, {"endpoint": "https://example.invalid", "verbose": False})
    assert resume_benchmark(state.benchmark_id, sctx=sctx, integrations={"review": {"verbose": True}}).success
    assert validated[-1] == {"endpoint": "https://example.invalid", "verbose": True}
    assert resume_benchmark(state.benchmark_id, sctx=sctx).success
    assert validated[-1] == validated[-2]
    with pytest.raises(BenchmarkIntegrationFailed, match="endpoint required"):
        resume_benchmark(state.benchmark_id, sctx=sctx, integrations={"review": {"endpoint": None}})
    assert resume_benchmark(state.benchmark_id, sctx=sctx).success
    assert validated[-1] == {"endpoint": "https://example.invalid", "verbose": True}
    prepare.assert_not_called()


def test_publication_error_retains_measurement_provenance(integration_registry):
    from sparkrun.api import resume_benchmark, BenchmarkIntegrationFailed
    from sparkrun.api._context import default_sctx
    from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration

    sctx = default_sctx()

    def complete(ctx):
        if ctx.settings.get("fail"):
            raise RuntimeError("publication unavailable")

    register_benchmark_integration(
        BenchmarkIntegration("review", prepare=lambda o, c: replace(o, profile="default-profile"), on_complete=complete)
    )
    state = _completed_state(sctx, {})
    successful = resume_benchmark(state.benchmark_id, sctx=sctx)
    with pytest.raises(BenchmarkIntegrationFailed) as caught:
        resume_benchmark(state.benchmark_id, sctx=sctx, integrations={"review": {"fail": True}})
    failed = caught.value.result
    assert successful.results == failed.results
    assert successful.profile is failed.profile is None
    assert successful.metadata == failed.metadata == {"framework": "test-bench", "profile": None, "bench_args": {}}


@pytest.mark.parametrize("explicit_hosts", [None, ("override-host",)])
def test_benchmark_retains_inline_cluster_policy(bench_env, monkeypatch, explicit_hosts):
    from sparkrun.api import benchmark
    from sparkrun.core.cluster_manager import ClusterDefinition

    env = bench_env
    inline = ClusterDefinition(
        name="inline", hosts=["inline-host"], user="inline-user", executor="local", plugins={"example": {"enabled": True}}
    )
    lookup = Mock(side_effect=AssertionError("Inline clusters must not be reloaded"))
    monkeypatch.setattr(env.sctx.cluster_manager, "get", lookup)
    assert benchmark(replace(env.options, cluster=inline, hosts=explicit_hosts), sctx=env.sctx).success
    cluster = env.run.call_args.args[0].cluster
    assert cluster.hosts == list(explicit_hosts or inline.hosts)
    assert cluster.user == "inline-user" and cluster.executor == "local"
    assert cluster.plugins == inline.plugins
    assert env.run.call_args.kwargs["sctx"].config.ssh_user == "inline-user"
    assert env.sctx.config.ssh_user != "inline-user"
    lookup.assert_not_called()


def test_required_integration_uses_api_error_before_planning(monkeypatch):
    from sparkrun.api import run, RunOptions, IntegrationUnavailable, SparkrunError
    from sparkrun.api._context import default_sctx
    from sparkrun.core import installed_plugins

    sctx = default_sctx()
    monkeypatch.setattr(
        installed_plugins,
        "_inventory",
        [
            installed_plugins.InstalledIntegration(
                name="required-review",
                module="missing",
                package=None,
                version=None,
                selected=True,
                selection_source="profile",
                required=True,
                failure="not installed",
            )
        ],
    )
    plan = Mock(side_effect=AssertionError("Cannot plan with missing required integration"))
    monkeypatch.setattr("sparkrun.api._run._build_plan", plan)
    with pytest.raises(IntegrationUnavailable, match="required-review") as caught:
        run(RunOptions(recipe="unused"), sctx=sctx)
    assert isinstance(caught.value, SparkrunError)
    assert isinstance(caught.value.__cause__, installed_plugins.RequiredIntegrationError)
    plan.assert_not_called()


def test_copied_config_has_one_distinct_controller(tmp_path):
    from sparkrun.application import get_controller_identity
    from sparkrun.core.config import SparkrunConfig

    first = tmp_path / "first"
    first.mkdir()
    before = get_controller_identity(SparkrunConfig(first / "config.yaml"))
    shutil.copytree(first, tmp_path / "second")
    copied_config = SparkrunConfig(tmp_path / "second/config.yaml")
    with ThreadPoolExecutor(max_workers=8) as pool:
        copies = list(pool.map(get_controller_identity, [copied_config] * 16))
    assert all(c == copies[0] for c in copies)
    assert before.scope_key != copies[0].scope_key
    assert get_controller_identity(SparkrunConfig(first / "config.yaml")) == before
    assert not before.owns(copies[0].labels())
    shutil.move(tmp_path / "second", tmp_path / "third")
    moved = get_controller_identity(SparkrunConfig(tmp_path / "third/config.yaml"))
    assert moved.scope_key not in {before.scope_key, copies[0].scope_key}


def test_variables_injection_is_first_initialization_only():
    from scitrera_app_framework import Variables
    from sparkrun.application import initialize

    variables = Variables()
    context = initialize(variables=variables)
    assert context.variables is variables
    assert initialize().variables is variables
    assert initialize(variables=variables).variables is variables
    with pytest.raises(RuntimeError, match="first initialization"):
        initialize(variables=Variables())
    assert initialize().variables is context.variables
