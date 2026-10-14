"""``sparkrun.api.systemd_export``: the console-free half of ``export systemd``."""

from __future__ import annotations

import inspect

import pytest
import yaml

from sparkrun.api import JobNotFound, SparkrunError
from sparkrun.api import systemd_export as svc
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.ssh import RemoteResult


@pytest.fixture
def recipe(tmp_path):
    path = tmp_path / "svc-test.yaml"
    path.write_text(
        yaml.safe_dump({"model": "test/model", "runtime": "vllm", "container": "test/image", "defaults": {"tensor_parallel": 2}}),
        encoding="utf-8",
    )
    loaded = Recipe.load(path)
    loaded.resolve()
    return loaded


@pytest.fixture
def lab():
    return ClusterDefinition(
        name="lab",
        hosts=["h1", "h2", "h3", "h4"],
        description="the lab",
        user="ops",
        executor="local",
        executor_config={"privileged": True},
        scheduler="occupancy-sparse",
        sync_source="spark_vllm_docker:/somewhere/.env",
    )


def _cluster_doc(plan):
    return yaml.safe_load(plan.cluster_yaml)


def test_plan_carries_the_source_cluster_and_bakes_no_placement(recipe, lab):
    """A tp=2 recipe on 4 hosts keeps all 4: the unit's `run` places at each start."""
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, dry_run=True))

    doc = _cluster_doc(plan)
    assert doc["name"] == plan.cluster_name == "%s-systemd" % recipe.slug
    assert doc["hosts"] == ["h1", "h2", "h3", "h4"]
    assert doc["user"] == "ops"
    assert doc["executor"] == "local"
    assert doc["executor_config"] == {"privileged": True}
    assert doc["scheduler"] == "occupancy-sparse"
    # Provenance of the source, not of this copy.
    assert "sync_source" not in doc and "description" not in doc
    assert plan.head_host == "h1"
    assert plan.ssh_user == "ops" and plan.ssh_kwargs["ssh_user"] == "ops"
    assert plan.service == "sparkrun-%s" % recipe.slug
    assert plan.unit_path == "/etc/systemd/system/sparkrun-%s.service" % recipe.slug


def test_plan_does_not_schedule(recipe, lab, monkeypatch):
    import sparkrun.api as api

    monkeypatch.setattr(api, "schedule", lambda *a, **kw: pytest.fail("export must not decide placement"))
    svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, dry_run=True))


def test_explicit_hosts_replace_the_cluster_hosts_but_keep_its_definition(recipe, lab):
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, hosts=("h3", "h4"), dry_run=True))
    doc = _cluster_doc(plan)
    assert doc["hosts"] == ["h3", "h4"] and doc["scheduler"] == "occupancy-sparse"
    assert plan.head_host == "h3"


def test_cluster_env_is_resolved_because_the_env_file_is_control_side(recipe, tmp_path):
    env_file = tmp_path / "secrets.env"
    env_file.write_text("TOKEN=abc\n", encoding="utf-8")
    cluster = ClusterDefinition(name="lab", hosts=["h1"], user="ops", env={"HF_TOKEN": "${TOKEN}"}, env_file=str(env_file))

    doc = _cluster_doc(svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=cluster, dry_run=True)))

    assert doc["env"] == {"HF_TOKEN": "abc"}
    assert "env_file" not in doc


def test_ssh_user_falls_back_to_the_local_login_not_user_env(recipe, monkeypatch):
    monkeypatch.setenv("USER", "root")
    monkeypatch.setattr("getpass.getuser", lambda: "alice")
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, hosts=("h1",), dry_run=True))
    assert plan.ssh_user == "alice"
    assert _cluster_doc(plan)["user"] == "alice"


def test_unknown_ssh_user_is_refused_rather_than_guessed(recipe, monkeypatch):
    def no_login():
        raise OSError("no login name")

    monkeypatch.setattr("getpass.getuser", no_login)
    with pytest.raises(SparkrunError, match="SSH user"):
        svc.plan(svc.SystemdExportOptions(recipe=recipe, hosts=("h1",), dry_run=True))


@pytest.mark.parametrize("name", ["bad name", "x';rm -rf /;'", "-lead", "a/b"])
def test_service_name_is_validated(recipe, name):
    with pytest.raises(SparkrunError, match="Invalid service name"):
        svc.plan(svc.SystemdExportOptions(recipe=recipe, hosts=("h1",), service_name=name, dry_run=True))


def test_exactly_one_source_is_required(recipe):
    with pytest.raises(SparkrunError):
        svc.plan(svc.SystemdExportOptions(hosts=("h1",)))
    with pytest.raises(SparkrunError):
        svc.plan(svc.SystemdExportOptions(recipe=recipe, job="sparkrun_abc", hosts=("h1",)))


def test_job_source_uses_recorded_hosts_cluster_and_overrides(recipe, lab, monkeypatch):
    meta = {
        "recipe_state": recipe.__getstate__(),
        "overrides": {"port": 9999},
        "effective_container_image": "test/image@sha256:" + "a" * 64,
        "hosts": ["h2", "h3"],
        "cluster": "lab",
    }
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda cid, cache_dir=None: meta)
    monkeypatch.setattr("sparkrun.api._resolve.resolve_cluster", _resolver_with(lab))

    plan = svc.plan(svc.SystemdExportOptions(job="sparkrun_aabbccdd", dry_run=True))

    assert plan.overrides == {"port": 9999}
    assert plan.recipe.container.endswith("a" * 64)
    doc = _cluster_doc(plan)
    assert doc["hosts"] == ["h2", "h3"]  # the job's own placement, as it ran
    assert doc["user"] == "ops" and doc["scheduler"] == "occupancy-sparse"


def _resolver_with(cluster):
    from sparkrun.api import _resolve

    real = _resolve.resolve_cluster

    def resolve(cluster_input=None, hosts_input=None, **kw):
        if cluster_input == cluster.name:
            return _resolve._replace_cluster_hosts(cluster, list(hosts_input)) if hosts_input is not None else cluster
        return real(cluster_input, hosts_input, **kw)

    return resolve


def test_missing_job_metadata_raises_job_not_found(monkeypatch):
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda cid, cache_dir=None: None)
    with pytest.raises(JobNotFound):
        svc.plan(svc.SystemdExportOptions(job="sparkrun_aabbccdd", hosts=("h1",), dry_run=True))


def test_render_and_install_steps(recipe, lab, monkeypatch):
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab))
    monkeypatch.setattr(
        "sparkrun.orchestration.systemd_service.detect_remote_command", lambda host, kw, dry_run=False: ("/opt/bin/sparkrun", "/home/ops")
    )
    host = svc.locate_host(plan)
    artifacts = svc.render(plan, host)
    assert "User=ops" in artifacts.unit_text and "--cluster %s" % plan.cluster_name in artifacts.unit_text
    assert plan.cluster_yaml.strip() in artifacts.install_script
    assert artifacts.unit_text in artifacts.sudo_install_script

    calls = []

    def remote(host, script, **kw):
        calls.append(("user", host, kw.get("ssh_user")))
        return RemoteResult(host=host, returncode=1, stdout="", stderr="disk full")

    def sudo(host, script, password, **kw):
        calls.append(("sudo", host, password))
        return RemoteResult(host=host, returncode=0, stdout="enabled", stderr="")

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", remote)
    monkeypatch.setattr("sparkrun.orchestration.sudo.run_sudo_script_on_host", sudo)

    with pytest.raises(svc.SystemdExportError) as err:
        svc.install_files(plan, artifacts)
    assert err.value.detail == "disk full" and "h1" in str(err.value)
    assert svc.install_unit(plan, artifacts, sudo_password="pw") == "enabled"
    svc.start(plan, sudo_password="pw")
    assert calls == [("user", "h1", "ops"), ("sudo", "h1", "pw"), ("sudo", "h1", "pw")]


def test_dry_run_uninstall_does_not_connect(recipe, lab, monkeypatch):
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_command", lambda *a, **kw: pytest.fail("dry run connected"))
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, dry_run=True))
    script = svc.render_uninstall(plan)
    assert "/home/ops/.config/sparkrun/services/%s" % plan.slug in script


def test_module_is_console_free():
    source = inspect.getsource(svc)
    assert "import click" not in source and "sys.exit" not in source and "print(" not in source
