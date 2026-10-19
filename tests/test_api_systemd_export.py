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


# --- issue #323: executor settings survive into the service recipe ----------------------------


def _load(tmp_path, data, name="svc-test"):
    path = tmp_path / ("%s.yaml" % name)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    loaded = Recipe.load(path)
    loaded.resolve()
    return loaded


def _baked(plan):
    return yaml.safe_load(plan.recipe_yaml)


_BASE = {"model": "test/model", "runtime": "vllm", "container": "test/image", "defaults": {"tensor_parallel": 2}}


def test_an_empty_entrypoint_reaches_the_service_recipe(tmp_path, lab):
    """The recipe-level fix for a consuming ENTRYPOINT must survive export, empty string and all."""
    recipe = _load(tmp_path, {**_BASE, "executor_config": {"entrypoint": "", "shm_size": "16g"}})
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, dry_run=True))

    baked = _baked(plan)
    assert baked["executor_config"] == {"entrypoint": "", "shm_size": "16g"}
    assert Recipe(baked).executor_config["entrypoint"] == ""


def test_executor_overrides_go_to_executor_config_not_defaults(recipe, lab):
    """``-o entrypoint=''`` configures the container; in ``defaults`` it would be an unmapped serve flag."""
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, overrides={"entrypoint": "", "port": 9000}, dry_run=True))

    baked = _baked(plan)
    assert baked["executor_config"] == {"entrypoint": ""}
    assert baked["defaults"]["port"] == 9000 and "entrypoint" not in baked["defaults"]


def test_job_source_bakes_the_recorded_executor_layer(recipe, lab, monkeypatch):
    """A job launched with ``-o entrypoint=''`` exports a service that launches the same way."""
    meta = {
        "recipe_state": recipe.__getstate__(),
        "overrides": {"port": 9999},
        "executor_overrides": {"entrypoint": ""},
        # The resolved config carries every layer; it is not what gets baked.
        "executor_config": {"entrypoint": "", "auto_remove": True, "user": "1000:1000"},
        "hosts": ["h2", "h3"],
        "cluster": "lab",
    }
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda cid, cache_dir=None: meta)
    monkeypatch.setattr("sparkrun.api._resolve.resolve_cluster", _resolver_with(lab))

    baked = _baked(svc.plan(svc.SystemdExportOptions(job="sparkrun_aabbccdd", dry_run=True)))
    assert baked["executor_config"] == {"entrypoint": ""}
    assert baked["defaults"]["port"] == 9999


# --- trust: the baked recipe is a local file on the head node ------------------------------------


def _untrusted(tmp_path, data):
    loaded = _load(tmp_path, data)
    loaded.source_registry = "third-party"
    loaded.source_registry_url = "https://example.invalid/recipes.git"
    return loaded


@pytest.mark.parametrize(
    "extra",
    [{"pre_exec": ["echo hi"]}, {"executor_config": {"privileged": True}}],
    ids=["hooks", "isolation"],
)
def test_installing_untrusted_hooks_or_isolation_overrides_needs_trust(tmp_path, lab, extra, monkeypatch):
    """Preview stays possible (that is how the recipe gets audited); the install is what needs trust."""
    from sparkrun.api import TrustRejected

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", lambda *a, **kw: pytest.fail("refused install connected"))
    recipe = _untrusted(tmp_path, {**_BASE, **extra})
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, dry_run=True))
    assert "--trust" in plan.trust_required and set(extra) <= set(_baked(plan))
    host = svc.SystemdHost(executable="/opt/bin/sparkrun", user_home="/home/ops")
    with pytest.raises(TrustRejected, match="--trust"):
        svc.install_files(plan, svc.render(plan, host))

    trusted = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, trust=True, dry_run=True))
    assert trusted.trust_required is None


def test_an_untrusted_recipe_with_only_an_empty_entrypoint_needs_no_trust(tmp_path, lab):
    recipe = _untrusted(tmp_path, {**_BASE, "executor_config": {"entrypoint": ""}})
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, dry_run=True))
    assert _baked(plan)["executor_config"] == {"entrypoint": ""} and plan.trust_required is None


# --- --pin ----------------------------------------------------------------------------------------

_SHA = "sha256:" + "a" * 64
_COMMIT = "c" * 40


@pytest.fixture
def resolvers(monkeypatch):
    from sparkrun.containers import digest as digest_mod
    from sparkrun.models import revision as revision_mod

    calls = {}

    def record(name, value):
        def fn(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            return value

        return fn

    monkeypatch.setattr(digest_mod, "resolve_registry_digest", record("registry", _SHA))
    monkeypatch.setattr(digest_mod, "resolve_host_digest", record("hosts", _SHA))
    monkeypatch.setattr(revision_mod, "resolve_hub_commit", record("hub", _COMMIT))
    monkeypatch.setattr(revision_mod, "resolve_cached_commit", record("cached", _COMMIT))
    return calls


def test_pin_is_off_by_default(recipe, lab, resolvers):
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, dry_run=True))
    assert plan.pins == () and _baked(plan)["container"] == "test/image"
    assert "model_revision" not in _baked(plan) and not resolvers


def test_pin_resolves_mutable_references_online(recipe, lab, resolvers):
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, pin=True, dry_run=True))

    baked = _baked(plan)
    assert baked["container"] == "test/image@" + _SHA
    assert baked["model_revision"] == _COMMIT
    assert baked["metadata"]["pinned"]["mode"] == "online"
    assert {p.field: p.source for p in plan.pins} == {"container": "registry", "model_revision": "hub"}
    assert "hosts" not in resolvers and "cached" not in resolvers


def test_pin_keeps_references_that_are_already_pinned(tmp_path, lab, resolvers):
    image = "test/image@" + "sha256:" + "f" * 64
    recipe = _load(tmp_path, {**_BASE, "container": image, "model_revision": "e" * 40})
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, pin=True, dry_run=True))

    baked = _baked(plan)
    assert baked["container"] == image and baked["model_revision"] == "e" * 40
    assert plan.pins == () and not resolvers


@pytest.mark.parametrize("offline", [False, True])
def test_pin_preserves_conditions_and_pins_every_override_image(tmp_path, lab, resolvers, offline):
    from sparkrun.core.recipe_overrides import OverrideContext, apply_recipe_override_layers

    pinned = "test/pinned@sha256:" + "f" * 64
    overrides = [
        {"when": {"nodes": 1}, "container": "test/solo:latest"},
        {"when": {"nodes": 2}, "container": "test/pair:latest"},
        {"when": {"nodes": 3}, "container": pinned},
        {"when": {"nodes": 4}, "defaults": {"gpu_memory_utilization": 0.8}},
    ]
    recipe = _load(tmp_path, {**_BASE, "overrides": overrides})
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, pin=True, offline=offline, dry_run=True))
    baked = _baked(plan)
    expected = ["test/solo@" + _SHA, "test/pair@" + _SHA, pinned, baked["container"]]
    for nodes, image in enumerate(expected, 1):
        restarted = Recipe.from_dict(baked)
        apply_recipe_override_layers(restarted, OverrideContext(shape={"nodes": nodes}))
        assert restarted.container == image
    assert [o["when"] for o in baked["overrides"]] == [o["when"] for o in overrides]
    assert baked["overrides"][3]["defaults"] == overrides[3]["defaults"]
    assert {p.field for p in plan.pins} == {"container", "model_revision", "overrides[0].container", "overrides[1].container"}
    calls = resolvers["hosts" if offline else "registry"]
    assert {args[0] for args, _ in calls} == {_BASE["container"], "test/solo:latest", "test/pair:latest"}
    if offline:
        assert all(list(args[1]) == lab.hosts for args, _ in calls)
        assert "registry" not in resolvers


def test_offline_pin_requires_every_candidate_host(recipe, lab, resolvers):
    """The service is not placed yet, so any candidate may run it: all of them must agree."""
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, pin=True, offline=True, dry_run=True))

    assert {p.source for p in plan.pins} == {"hosts"}
    ((args, kwargs),) = resolvers["hosts"]
    assert list(args[1]) == ["h1", "h2", "h3", "h4"] and args[2]["ssh_user"] == "ops"
    assert list(resolvers["cached"][0][0][2]) == ["h1", "h2", "h3", "h4"]
    assert "registry" not in resolvers and "hub" not in resolvers


def test_offline_without_pin_is_refused(recipe, lab):
    with pytest.raises(SparkrunError, match="pin"):
        svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, offline=True, dry_run=True))


def test_pin_writes_in_a_platform_default_image_only_when_hosts_agree(tmp_path, lab, resolvers, monkeypatch):
    from sparkrun.core.bootstrap import get_runtime

    recipe = _load(tmp_path, {key: value for key, value in _BASE.items() if key != "container"})
    runtime_cls = type(get_runtime(recipe.runtime))
    monkeypatch.setattr(runtime_cls, "resolve_container", lambda self, recipe, host_hardware=None: "vendor/default:1")
    plan = svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, pin=True, dry_run=True))
    assert _baked(plan)["container"] == "vendor/default@" + _SHA

    images = iter(["vendor/a:1", "vendor/b:1", "vendor/a:1", "vendor/a:1"])
    monkeypatch.setattr(runtime_cls, "resolve_container", lambda self, recipe, host_hardware=None: next(images))
    with pytest.raises(SparkrunError, match="different images"):
        svc.plan(svc.SystemdExportOptions(recipe=recipe, cluster=lab, pin=True, dry_run=True))
