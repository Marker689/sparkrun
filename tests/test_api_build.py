"""Preparation-only API: same asset pipeline, no inference lifecycle."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

import sparkrun.api as api
from sparkrun.builders.base import BuilderPlugin
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.distribution import DistributionError, TransferModeResult

IMAGE = "registry.example/model@sha256:" + "a" * 64


def recipe(**extra):
    return Recipe.from_dict(
        {
            "recipe_version": "2",
            "runtime": "vllm",
            "model": "org/model",
            "model_revision": "target-sha",
            "container": IMAGE,
            "min_nodes": 4,
            "max_nodes": 4,
            "defaults": {"tensor_parallel": 4},
            **extra,
        }
    )


@pytest.fixture
def io(monkeypatch, v):
    calls = []
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    def forbidden(*args, **kwargs):
        pytest.fail("build reached inference lifecycle or occupancy")

    for path in [
        "sparkrun.api._hosts.resolve_effective_hosts",
        "sparkrun.api._status.status",
        "sparkrun.core.launcher.launch_inference",
        "sparkrun.core.launcher.post_launch_lifecycle",
        "sparkrun.core.execution.resolve_recipe_execution",
        "sparkrun.orchestration.job_metadata.save_job_metadata",
        "sparkrun.orchestration.primitives.try_clear_page_cache",
        "sparkrun.api._run._evict_superseded_deployments",
    ]:
        monkeypatch.setattr(path, forbidden)
    monkeypatch.setattr(VllmDistributedRuntime, "run", forbidden)
    monkeypatch.setattr("sparkrun.core.launcher._verify_mount_sources", lambda *a, **k: None)
    monkeypatch.setattr("sparkrun.orchestration.distribution.resolve_auto_transfer_mode", lambda mode, *a, **k: TransferModeResult("local"))
    monkeypatch.setattr("sparkrun.core.image_preparation._resolve_host_image_id", lambda host, image, *args: image)

    def distribute(rec, image, hosts, cache, config, dry_run, **kwargs):
        calls.append(("distribution", rec, image, hosts, cache, config, dry_run, kwargs))
        if kwargs.get("after_container_sync"):
            kwargs["after_container_sync"]()
        return None, {}, {}, {}

    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", distribute)
    monkeypatch.setattr("sparkrun.core.asset_preparation.prepare_tuning", lambda *a, **k: calls.append(("tuning", a, k)))
    return calls


def options(rec=None, **kwargs):
    return api.BuildOptions(
        recipe=rec if rec is not None else recipe(),
        cluster=ClusterDefinition("busy", ["h1", "h2", "h3", "h4", "h5"], user="alice", cache_dir="/target/hf"),
        **kwargs,
    )


def test_plan_build_keeps_all_hosts_without_occupancy_or_node_trimming(io):
    value = options()
    plan = api.plan_build(value)
    assert plan.host_list == ("h1", "h2", "h3", "h4", "h5")
    assert plan.needs_image
    assert plan.executor_target.executor == "docker"
    assert not hasattr(plan, "placement") and not hasattr(plan, "cluster_id")
    assert io == []


def test_can_prepare_one_host_of_tp4_recipe(io):
    plan = api.plan_build(options(hosts=("h2",)))
    assert plan.host_list == ("h2",)
    assert plan.cluster.user == "alice"


def test_build_reuses_plan_and_stages_images_models_tuning(io, monkeypatch):
    value = options()
    plan = api.plan_build(value)

    def forbidden(*a, **k):
        pytest.fail("re-planned supplied build plan")

    monkeypatch.setattr("sparkrun.api._build.plan_build", forbidden)
    result = api.build(value, plan=plan)
    assert result.host_list == plan.host_list
    assert result.container_image == IMAGE
    assert result.images_by_node == (IMAGE,) * 5
    assert result.models == ("org/model",)
    assert result.effective_cache_dir == "/target/hf"
    assert result.environment_file is None
    assert not hasattr(result, "cluster_id")
    assert [c[0] for c in io] == ["distribution", "tuning"]
    assert io[0][5].ssh_user == "alice"
    assert io[0][7]["skip_model"] is False
    assert io[1][2]["strict"] is True
    assert result.timeline["spans"][0]["name"] == "build"


def test_builder_output_and_auxiliary_models_are_staged_without_mutating_recipe(io, monkeypatch):
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    seen = []

    class Builder(BuilderPlugin):
        builder_name = "eugr"

        def prepare(self, image, rec, hosts, **kwargs):
            seen.append((image, tuple(hosts), dict(kwargs)))
            return "local/derived:built"

    monkeypatch.setattr("sparkrun.core.bootstrap.get_builder", lambda *a, **k: Builder())
    monkeypatch.setattr(
        VllmDistributedRuntime, "prepare", lambda self, rec, *a, **k: rec.distribution_config.add_model("org/draft", revision="draft-sha")
    )
    rec = recipe(builder="eugr")
    before = deepcopy(rec.distribution_config)
    value = options(rec, rebuild=True)
    plan = api.plan_build(value)
    result = api.build(value, plan=plan)
    assert result.container_image == "local/derived:built"
    assert result.models == ("org/model", "org/draft")
    assert io[0][1].distribution_config.models.entries[1].revision == "draft-sha"
    assert io[0][2] == "local/derived:built"
    assert seen[0][2]["builder_context"]["engine"] == "vllm"
    assert io[0][1].builder_config["rebuild"] is True
    assert rec.distribution_config == before
    assert rec.builder_config.get("rebuild") is None
    assert len(plan.recipe.distribution_config.models.entries) == 1


def test_native_uv_venv_provisions_environment_without_resolving_images(io, monkeypatch, tmp_path):
    scripts = []

    def provision(hosts, script, **kwargs):
        scripts.append((hosts, script, kwargs))
        return [SimpleNamespace(host=h, returncode=0, stdout="", stderr="") for h in hosts]

    monkeypatch.setattr("sparkrun.builders.uv_venv.run_remote_scripts_parallel", provision)
    monkeypatch.setattr("sparkrun.core.image_preparation.resolve_runtime_image_plan", lambda *a, **k: pytest.fail("native image planning"))
    rec = recipe(
        container="",
        executor="local",
        builder="uv-venv",
        builder_config={
            "venv_path": str(tmp_path / "venv"),
            "python": "3.12",
            "requirements": ["vllm"],
        },
    )
    result = api.build(options(rec, hosts=("localhost",)))
    assert scripts[0][0] == ["localhost"]
    assert "uv pip install" in scripts[0][1]
    assert "vllm serve" not in scripts[0][1]
    assert scripts[0][2]["allow_local"] and scripts[0][2]["session_guard"]
    assert result.executor == "local"
    assert result.container_image is None and result.images_by_node == ()
    assert result.environment_file == str(tmp_path / "venv" / "sparkrun-env.sh")
    assert io[0][7]["skip_container"] is True
    assert result.models == ("org/model",)


def test_native_provision_failure_aborts_before_distribution(io, monkeypatch):
    monkeypatch.setattr(
        "sparkrun.builders.uv_venv.run_remote_scripts_parallel",
        lambda *a, **k: [
            SimpleNamespace(host="h1", returncode=1, stdout="", stderr="dependency failed"),
        ],
    )
    rec = recipe(container="", executor="local", builder="uv-venv", builder_config={"requirements": ["vllm"]})
    with pytest.raises(api.SparkrunError, match="dependency failed"):
        api.build(options(rec))
    assert io == []


def test_dry_run_does_not_probe_resident_images(io, monkeypatch):
    monkeypatch.setattr("sparkrun.core.image_preparation._resolve_host_image_id", lambda *a: pytest.fail("live image probe"))
    result = api.build(options(dry_run=True))
    assert result.dry_run and result.images_by_node == (IMAGE,) * 5
    assert io[0][6] is True
    assert io[1][2]["dry_run"] is True


@pytest.mark.parametrize("local_model", [False, True])
def test_preplaced_or_disabled_models_skip_download(io, local_model):
    value = options(recipe(model="/weights/model") if local_model else None)
    if not local_model:
        value.cluster.distribution.model.enabled = False
    result = api.build(value)
    assert io[0][7]["skip_model"] is True
    assert result.models == ()


def test_model_transfer_failure_is_not_success(io, monkeypatch):
    def fail(*a, **k):
        raise DistributionError("Model distribution failed on h3")

    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", fail)
    with pytest.raises(api.SparkrunError, match="h3") as caught:
        api.build(options())
    assert isinstance(caught.value.__cause__, DistributionError)
    assert io == []


def test_interrupt_propagates(io, monkeypatch):
    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", interrupt)
    with pytest.raises(KeyboardInterrupt):
        api.build(options())


def test_changed_options_rejected_with_supplied_plan(io):
    value = options()
    plan = api.plan_build(value)
    with pytest.raises(ValueError, match="same BuildOptions"):
        api.build(replace(value, hosts=("h1",)), plan=plan)


def test_empty_targets_rejected(io):
    with pytest.raises(api.HostsUnreachable, match="at least one"):
        api.plan_build(options(hosts=()))


def test_no_inference_or_recipe_shell_hooks_run(io):
    rec = recipe(pre_exec=["exit 99"], post_exec=["exit 99"], post_commands=["exit 99"])
    assert api.build(options(rec)).models == ("org/model",)


def test_build_hooks_are_opt_in_and_never_use_execution_strategy(io):
    from sparkrun.core.recipe_items import FunctionalRecipeItemHandler, register_recipe_item, unregister_recipe_item

    calls = []

    def prepare(context):
        calls.append(context.options.dry_run)
        context.recipe.distribution_config.add_model("org/extra", revision="extra-sha")
        return {"snapshot_driver": "n580"}

    register_recipe_item("testbuild", FunctionalRecipeItemHandler(lambda v, r: v), owner="test.build", build_preparation=prepare)
    try:
        api.build(options())
        assert calls == []
        rec = recipe(testbuild={})
        result = api.build(options(rec, builder_context={"snapshot_driver": "n610"}))
        assert calls == [False]
        assert result.models == ("org/model", "org/extra")
        assert len(rec.distribution_config.models.entries) == 1
    finally:
        unregister_recipe_item("testbuild", owner="test.build")


def test_untrusted_build_hook_rejected_before_staging(io):
    from sparkrun.core.recipe_items import FunctionalRecipeItemHandler, register_recipe_item, unregister_recipe_item

    register_recipe_item(
        "testbuild",
        FunctionalRecipeItemHandler(lambda v, r: v),
        owner="test.build",
        build_preparation=lambda ctx: pytest.fail("untrusted hook executed"),
    )
    try:
        rec = recipe(testbuild={})
        rec.is_url_sourced = True
        with pytest.raises(api.TrustRejected):
            api.build(options(rec))
        assert io == []
    finally:
        unregister_recipe_item("testbuild", owner="test.build")


def test_build_preparation_must_affect_fingerprint():
    from sparkrun.core.recipe_items import FunctionalRecipeItemHandler, register_recipe_item

    with pytest.raises(ValueError, match="must affect the fingerprint"):
        register_recipe_item(
            "testbuild",
            FunctionalRecipeItemHandler(lambda v, r: v),
            owner="test.build",
            build_preparation=lambda ctx: {},
            affects_fingerprint=False,
        )


def test_executor_override_used_for_preflight_without_changing_recipe(io, monkeypatch):
    mounts = []
    monkeypatch.setattr("sparkrun.core.launcher._verify_mount_sources", lambda *a, **k: mounts.append(k))
    rec = recipe(executor="docker")
    result = api.build(options(rec, executor="local"))
    assert result.executor == "local" and result.container_image is None
    assert mounts[0]["overrides"]["executor"] == "local"
    assert rec.executor == "docker"
    assert io[0][7]["skip_container"] is True


def test_distribution_uses_cluster_transport_and_model_preferences(io):
    value = options(preserve_model_perms=False, skip_model_fan_out=True)
    value.cluster.mgmt_interface = "eth-mgmt"
    value.cluster.transfer_interface = "eth-data"
    api.build(value)
    passed = io[0][7]
    assert passed["cluster_name"] == "busy"
    assert passed["mgmt_interface"] == "eth-mgmt"
    assert passed["transfer_interface"] == "eth-data"
    assert passed["prefs"].preserve_perms is False
    assert passed["prefs"].skip_fan_out is True


def test_unsupported_executor_rejected_before_target_resolution(io, monkeypatch):
    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", lambda **k: SimpleNamespace(executor_name="k8s"))
    with pytest.raises(api.SparkrunError, match="executor 'k8s'"):
        api.build(options())
    assert io == []


def test_result_excludes_models_with_no_selected_targets(io):
    rec = recipe(
        distribution_config={
            "models": {
                "entries": [
                    {"name": "org/head", "target": [0]},
                    {"name": "org/elsewhere", "target": [5]},
                ]
            },
        }
    )
    result = api.build(options(rec, hosts=("h1",)))
    assert result.models == ("org/head",)


def test_build_and_run_options_share_target_fields_and_defaults():
    from dataclasses import fields

    from sparkrun.api._models import TargetOptions

    shared = {f.name: (f.default, f.default_factory) for f in fields(TargetOptions)}
    for cls in (api.BuildOptions, api.RunOptions):
        own = {f.name: (f.default, f.default_factory) for f in fields(cls)}
        assert {k: own[k] for k in shared} == shared
    assert "offline" in shared and api.BuildOptions("r").offline is None
    with pytest.raises(TypeError):
        api.BuildOptions("r", ("h1",))  # only recipe is positional


def test_build_options_lift_executor_overrides_like_run():
    value = api.BuildOptions("r", overrides={"entrypoint": "", "port": 9000}, executor_config={"shm_size": "1g"})
    assert value.overrides == {"port": 9000}
    assert value.executor_config == {"shm_size": "1g", "entrypoint": ""}
    assert value.executor_overrides() == {"shm_size": "1g", "entrypoint": ""}


@pytest.mark.parametrize(
    "flag,cluster_offline,expected", [(None, True, True), (False, True, False), (True, None, True), (None, None, False)]
)
def test_plan_build_resolves_offline_like_plan(io, flag, cluster_offline, expected):
    value = options(offline=flag)
    value.cluster.offline = cluster_offline
    assert api.plan_build(value).offline.offline is expected


def test_offline_build_preflights_and_keeps_builder_and_distribution_local(io, monkeypatch):
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    seen, preflights = [], []

    class Builder(BuilderPlugin):
        builder_name = "eugr"

        def prepare(self, image, rec, hosts, **kwargs):
            seen.append(dict(kwargs))
            return image

    monkeypatch.setattr("sparkrun.core.bootstrap.get_builder", lambda *a, **k: Builder())
    monkeypatch.setattr(VllmDistributedRuntime, "prepare", lambda *a, **k: None)
    monkeypatch.setattr("sparkrun.core.launcher._offline_preflight", lambda *a, **k: preflights.append(k))
    # A caller's builder_context cannot switch an offline build back online.
    api.build(options(recipe(builder="eugr"), offline=True, builder_context={"offline": False}))
    assert len(preflights) == 1 and preflights[0]["skip_images"] is False
    assert seen[0]["builder_context"]["offline"] is True
    assert io[0][7]["offline"] is True


def test_online_build_has_no_offline_preflight(io, monkeypatch):
    monkeypatch.setattr("sparkrun.core.launcher._offline_preflight", lambda *a, **k: pytest.fail("online build preflighted"))
    api.build(options())
    assert io[0][7]["offline"] is False


def test_offline_build_loads_named_recipe_offline(io, monkeypatch, tmp_path):
    import yaml

    path = tmp_path / "named.yaml"
    path.write_text(yaml.safe_dump(recipe().to_dict()))
    loads, real = [], Recipe.load.__func__

    def load(cls, *args, **kwargs):
        loads.append(kwargs.get("offline"))
        return real(cls, *args, **kwargs)

    monkeypatch.setattr(Recipe, "load", classmethod(load))
    api.plan_build(options(str(path), offline=True))
    assert loads == [True]
