"""Image defaults must agree across per-host planning and image preparation."""

from _runtime_fixtures import StubRuntime

from unittest.mock import Mock

import pytest

from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.image_preparation import prepare_images
from sparkrun.core.images import ImagePlanError, resolve_runtime_image_plan
from sparkrun.core.recipe import Recipe, RecipeError
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
from sparkrun.runtimes.vllm_ray import VllmRayRuntime


def _cluster():
    return ClusterDefinition(
        name="mixed",
        hosts=["spark", "generic"],
        hosts_hardware={
            "spark": HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10")]),
            "generic": HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h200")]),
        },
    )


def _recipe(**fields):
    return Recipe.from_dict({"recipe_version": "2", "model": "org/model", "runtime": "vllm-distributed", **fields})


def test_platform_defaults_are_used_per_host_and_preserve_declarations():
    recipe = _recipe()
    cluster = _cluster()
    runtime = VllmDistributedRuntime()
    from sparkrun.orchestration.job_metadata import generate_intent_id

    before = generate_intent_id(recipe)
    planned = resolve_runtime_image_plan(recipe, runtime, cluster.hosts, cluster=cluster)
    prepared = prepare_images(recipe, runtime, cluster.hosts, cluster=cluster, run_builder=False)
    assert planned.images_by_node == (
        "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest",
        "vllm/vllm-openai:latest",
    )
    assert prepared.image_plan == planned
    assert planned.declared == ()
    # Distribution derives the per-node transfer entries, but not recipe image declarations.
    assert recipe.container == "" and recipe.containers == []
    assert generate_intent_id(recipe) == before


def test_explicit_recipe_image_bypasses_default_hook(monkeypatch):
    runtime = VllmDistributedRuntime()
    default = Mock(side_effect=AssertionError("explicit image must win"))
    monkeypatch.setattr(runtime, "default_image_for", default)
    recipe = _recipe(container="org/explicit:tag")
    cluster = _cluster()
    planned = resolve_runtime_image_plan(recipe, runtime, cluster.hosts, cluster=cluster)
    assert planned.images_by_node == ("org/explicit:tag",) * 2
    default.assert_not_called()


def test_explicit_host_images_need_no_runtime_default(monkeypatch):
    runtime = VllmDistributedRuntime()
    monkeypatch.setattr(runtime, "default_image_for", Mock(side_effect=AssertionError("unused default")))
    recipe = _recipe(containers=[{"host": "generic", "image": "org/tuned:tag"}])
    assert resolve_runtime_image_plan(recipe, runtime, ["generic"], cluster=_cluster()).images_by_node == ("org/tuned:tag",)


def test_per_host_image_overrides_only_its_host():
    recipe = _recipe(containers=[{"host": "spark", "image": "org/tuned:tag"}])
    cluster = _cluster()
    planned = resolve_runtime_image_plan(recipe, VllmDistributedRuntime(), cluster.hosts, cluster=cluster)
    assert planned.images_by_node == ("org/tuned:tag", "vllm/vllm-openai:latest")
    assert planned.declared == (("spark", "org/tuned:tag"),)


def test_homogeneous_runtime_rejects_differing_platform_defaults():
    cluster = _cluster()
    with pytest.raises(ImagePlanError, match="requires one image"):
        resolve_runtime_image_plan(_recipe(), VllmRayRuntime(), cluster.hosts, cluster=cluster)
    planned = resolve_runtime_image_plan(_recipe(container="org/shared:tag"), VllmRayRuntime(), cluster.hosts, cluster=cluster)
    assert planned.images_by_node == ("org/shared:tag",) * 2


def test_builder_rejects_differing_sources_before_preparation(monkeypatch):
    builder = Mock(transforms_image=True)
    monkeypatch.setattr("sparkrun.core.bootstrap.get_builder", lambda *args: builder)
    cluster = _cluster()
    with pytest.raises(RecipeError, match="builder requires one source image"):
        prepare_images(_recipe(builder="test"), VllmDistributedRuntime(), cluster.hosts, cluster=cluster)
    builder.prepare.assert_not_called()


def test_unknown_hardware_requires_explicit_image():
    runtime = VllmDistributedRuntime()
    unknown = HostHardware()
    assert runtime.resolve_container(_recipe(), host_hardware=unknown) == ""


def test_absent_default_requires_an_explicit_image():
    runtime = StubRuntime()
    assert runtime.resolve_container(_recipe()) == ""
    with pytest.raises(ImagePlanError, match="No container image"):
        resolve_runtime_image_plan(_recipe(), runtime, ["host"])


def test_default_hook_errors_are_visible(monkeypatch):
    platform = Mock()
    platform.default_image.side_effect = RuntimeError("broken platform")
    monkeypatch.setattr("sparkrun.platforms.resolve_platform", lambda hardware, **kwargs: platform)
    with pytest.raises(RuntimeError, match="broken platform"):
        VllmDistributedRuntime().resolve_container(_recipe(), host_hardware=_cluster().hardware_for("spark"))


class _NoDefaultRuntime(VllmDistributedRuntime):
    def default_image_for(self, host_hardware=None):
        raise AssertionError("unused default image hook")


@pytest.mark.parametrize("images", [("org/resident:tag",) * 2, ("org/a:tag", "org/b:tag")])
def test_prepared_images_replace_unused_declarations_and_defaults(images):
    from dataclasses import replace
    from sparkrun.api import materialize
    from test_api_materialize import _fixture

    options, plan, sctx = _fixture()
    plan.recipe.container = ""
    plan = replace(plan, runtime=_NoDefaultRuntime())
    prepared = prepare_images(plan.recipe, plan.runtime, list(plan.host_list), run_builder=False, images_by_node=images)
    spec = materialize(options, plan=plan, sctx=sctx, images_by_node=images)
    assert prepared.images_by_node == tuple(unit.image for unit in spec.units) == images


def test_uniform_prepared_images_replace_differing_ray_defaults():
    cluster = _cluster()
    images = ("org/shared:tag",) * 2
    prepared = prepare_images(_recipe(), VllmRayRuntime(), cluster.hosts, cluster=cluster, run_builder=False, images_by_node=images)
    assert prepared.images_by_node == images


@pytest.mark.parametrize("images", [("org/a", "org/b"), ("org/a",), ("org/a", None), ("org/a", " "), "xx"])
def test_prepared_images_share_alignment_type_and_runtime_validation(images):
    from dataclasses import replace
    from sparkrun.api import materialize
    from test_api_materialize import _fixture

    class UniformRuntime(VllmDistributedRuntime):
        supports_heterogeneous_images = False

    options, plan, sctx = _fixture()
    plan = replace(plan, runtime=UniformRuntime())
    with pytest.raises(ImagePlanError):
        materialize(options, plan=plan, sctx=sctx, images_by_node=images)
    with pytest.raises(RecipeError):
        prepare_images(plan.recipe, plan.runtime, list(plan.host_list), images_by_node=images, run_builder=False)


def test_preparation_reuses_recipe_without_retaining_distribution_targets():
    from copy import deepcopy
    from sparkrun.orchestration.distribution import _resolve_targets

    recipe = _recipe()
    original = deepcopy(recipe.__getstate__())
    cluster = _cluster()
    runtime = VllmDistributedRuntime()
    for hosts in (cluster.hosts, ["generic"], list(reversed(cluster.hosts))):
        prepared = prepare_images(recipe, runtime, hosts, cluster=cluster, run_builder=False)
        transferred = {
            host: entry.name for entry in prepared.container_distribution.entries for host in _resolve_targets(entry.target, hosts)
        }
        assert transferred == dict(zip(hosts, prepared.images_by_node, strict=True))
        assert recipe.__getstate__() == original


def test_explicit_container_distribution_is_preserved():
    from copy import deepcopy

    recipe = _recipe(distribution_config={"containers": {"enabled": False, "entries": [{"name": "org/manual", "target": [0]}]}})
    assert recipe.distribution_config.containers.explicit
    original = deepcopy(recipe.distribution_config)
    prepared = prepare_images(recipe, VllmDistributedRuntime(), _cluster().hosts, cluster=_cluster(), run_builder=False)
    assert prepared.container_distribution is None
    assert recipe.distribution_config == original


def test_containerless_preparation_runs_environment_builder_without_image(monkeypatch):
    recipe = _recipe(builder="uv-venv")
    builder = Mock(transforms_image=False)
    builder.prepare.return_value = ""
    monkeypatch.setattr("sparkrun.core.bootstrap.get_builder", lambda *args: builder)
    prepared = prepare_images(recipe, _NoDefaultRuntime(), ["localhost"], needs_image=False)
    assert prepared.image_plan is None and prepared.source_image is None
    assert prepared.images_by_node == () and prepared.container_distribution is None
    assert builder.prepare.call_args.args[0] == ""


def test_containerless_launch_reuses_executor_and_never_resolves_image(monkeypatch, tmp_path):
    from test_launcher import _builder_phase_harness
    from sparkrun.application import initialize
    from sparkrun.core.launcher import launch_inference
    from sparkrun.orchestration.executors.local import LocalExecutor

    _builder_phase_harness(monkeypatch, tmp_path)
    sctx = initialize(config_path=tmp_path / "config.yaml")
    recipe = _recipe(executor="local", defaults={"tensor_parallel": 1})
    runtime = _NoDefaultRuntime()
    runtime.run = Mock(return_value=0)
    distribution = Mock(return_value=(None, {}, {}, {}))
    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", distribution)
    result = launch_inference(
        recipe=recipe,
        runtime=runtime,
        host_list=["localhost"],
        overrides={},
        config=sctx.config,
        v=sctx.variables,
        is_solo=True,
        dry_run=True,
        sync_tuning=False,
        trust=True,
    )
    assert result.rc == 0 and result.container_image == ""
    assert isinstance(runtime.run.call_args.kwargs["executor"], LocalExecutor)
    assert runtime.run.call_args.kwargs["image"] == ""
    assert distribution.call_args.kwargs["skip_container"] is True


def test_container_launch_still_requires_image_before_distribution(monkeypatch, tmp_path):
    from test_launcher import _builder_phase_harness
    from sparkrun.application import initialize
    from sparkrun.core.launcher import launch_inference
    from sparkrun.core.recipe import RecipeError

    _builder_phase_harness(monkeypatch, tmp_path)
    sctx = initialize(config_path=tmp_path / "config.yaml")
    recipe = _recipe(executor="docker", defaults={"tensor_parallel": 1})
    runtime = VllmDistributedRuntime()
    monkeypatch.setattr(runtime, "default_image_for", lambda *args, **kwargs: "")
    distribution = Mock(side_effect=AssertionError("invalid image must not reach distribution"))
    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", distribution)
    with pytest.raises(RecipeError, match="No container image"):
        launch_inference(
            recipe=recipe,
            runtime=runtime,
            host_list=["localhost"],
            overrides={},
            config=sctx.config,
            v=sctx.variables,
            is_solo=True,
            dry_run=True,
            sync_tuning=False,
            trust=True,
        )
    distribution.assert_not_called()


@pytest.mark.parametrize("command", [None, "", "   "])
@pytest.mark.usefixtures("external_control")
def test_launcher_rejects_invalid_runtime_command_before_submission(tmp_path, monkeypatch, command):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.launcher import launch_inference
    from sparkrun.runtimes.modular_max import ModularMaxRuntime

    runtime = ModularMaxRuntime()
    monkeypatch.setattr(runtime, "generate_command", Mock(return_value=command))
    submit = Mock(side_effect=AssertionError("invalid command submitted"))
    monkeypatch.setattr(runtime, "run", submit)
    with pytest.raises(ValueError, match="non-empty serve command"):
        launch_inference(
            recipe=_recipe(runtime="modular-max", executor="local"),
            runtime=runtime,
            host_list=["localhost"],
            overrides={},
            config=SparkrunConfig(tmp_path / "config.yaml"),
            is_solo=True,
            dry_run=True,
            sync_tuning=False,
        )
    submit.assert_not_called()
