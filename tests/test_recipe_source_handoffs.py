"""Recipe provenance and failure contracts survive frontend/API handoffs."""

from unittest.mock import Mock

import pytest
import yaml

from sparkrun import api
from sparkrun.api._resolve import resolve_recipe
from sparkrun.cli._common import _load_recipe
from sparkrun.core.launcher import resolve_recipe_trust
from sparkrun.core.recipe import Recipe, RecipeError
from sparkrun.core.registry import RegistryEntry
from test_catalog_source_boundaries import context as context, recipe_data, shared_cache


def load_recipe(loader, selection, context):
    if loader == "catalog":
        return api.resolve_catalog_recipe(selection, sctx=context)[0]
    if loader == "cli":
        return _load_recipe(context.config, selection, resolve=False)[0]
    if loader == "core":
        from sparkrun.core.resolve import load_recipe as core_load_recipe

        return core_load_recipe(context.config, selection, resolve=False)[0]
    return resolve_recipe(selection, sctx=context)


@pytest.mark.parametrize("loader", ["catalog", "api", "cli", "core"])
@pytest.mark.parametrize("alias", [False, True])
def test_import_provenance_survives_path_handoff(context, loader, alias):
    context.registry_manager._save_registries([], pending_bootstrap_urls=[])
    details = api.import_recipe(yaml.safe_dump(recipe_data(pre_exec=["echo needs-trust"])), sctx=context)
    selection = details["source_path"]
    if alias:
        from pathlib import Path

        path = context.config.config_path.parent / "alias.yaml"
        path.symlink_to(Path(selection))
        selection = str(path)
    recipe = load_recipe(loader, selection, context)
    assert recipe.is_url_sourced and recipe.source_registry is None
    assert not resolve_recipe_trust(recipe, False, sctx=context)
    assert resolve_recipe_trust(recipe, True, sctx=context)
    assert resolve_recipe(recipe, sctx=context) is recipe
    assert recipe.is_url_sourced
    assert not api.get_recipe_details(details["reference"], sctx=context)["trusted"]


@pytest.mark.parametrize("loader", ["catalog", "api", "cli", "core"])
def test_local_recipe_still_inherits_local_trust(context, loader):
    context.registry_manager._save_registries([], pending_bootstrap_urls=[])
    path = context.config.config_path.parent / "local.yaml"
    path.write_text(yaml.safe_dump(recipe_data(pre_exec=["echo locally-authored"])))
    recipe = load_recipe(loader, str(path), context)
    assert recipe.source_registry is None and not recipe.is_url_sourced
    assert resolve_recipe_trust(recipe, False, sctx=context)


@pytest.mark.parametrize("loader", ["catalog", "api", "cli", "core"])
def test_named_shared_clone_keeps_selected_registry(context, loader):
    entries = shared_cache(context, overlapping=True)
    for entry in entries:
        recipe = load_recipe(loader, "@%s/sample" % entry.name, context)
        assert recipe.source_registry == entry.name and recipe.source_registry_url == entry.url
        assert resolve_recipe_trust(recipe, False, sctx=context) is entry.trusted


@pytest.mark.parametrize("change", ["replace", "remove", "disable", "untrust", "missing_url"])
def test_preloaded_recipe_cannot_inherit_unrelated_or_unavailable_trust(context, change):
    entries = shared_cache(context)
    if change != "replace":
        api.configure_registry("trust", "first", acknowledge_trust=True, sctx=context)
    recipe, _ = api.resolve_catalog_recipe("@first/sample", sctx=context)
    assert resolve_recipe_trust(recipe, False, sctx=context) is (change != "replace")
    if change == "replace":
        replacement = RegistryEntry(name="first", url="https://example.invalid/replacement.git", subpath="recipes", trusted=True)
        context.registry_manager._save_registries([replacement, entries[1]], pending_bootstrap_urls=[])
    elif change == "missing_url":
        recipe.source_registry_url = None
    else:
        api.configure_registry(change, "first", sctx=context)
    assert resolve_recipe(recipe, sctx=context) is recipe
    assert not resolve_recipe_trust(recipe, False, sctx=context)
    assert resolve_recipe_trust(recipe, True, sctx=context)


def test_same_registry_can_be_intentionally_trusted_after_loading(context):
    shared_cache(context)
    recipe, _ = api.resolve_catalog_recipe("@first/sample", sctx=context)
    assert not resolve_recipe_trust(recipe, False, sctx=context)
    api.configure_registry("trust", "first", acknowledge_trust=True, sctx=context)
    assert resolve_recipe_trust(recipe, False, sctx=context)


def test_public_run_rejects_import_before_preparation_and_launch(context, monkeypatch):
    from sparkrun.core.hardware import default_dgx_spark_hardware
    from sparkrun.core.cluster_status import empty_status

    monkeypatch.setattr(
        "sparkrun.core.limits.resolved_hardware_for_scheduling",
        lambda cluster, hosts: {host: default_dgx_spark_hardware() for host in hosts},
    )
    monkeypatch.setattr("sparkrun.api.status", lambda hosts, **kwargs: empty_status(hosts))
    context.registry_manager._save_registries([], pending_bootstrap_urls=[])
    details = api.import_recipe(yaml.safe_dump(recipe_data(pre_exec=["echo needs-trust"])), sctx=context)
    forbidden = Mock(side_effect=AssertionError("untrusted recipe reached preparation/launch"))
    monkeypatch.setattr("sparkrun.core.execution.resolve_recipe_execution", forbidden)
    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", forbidden)
    # Not a dry run: a dry run executes nothing and reports instead of refusing
    # (tests/test_recipe_hook_trust.py).  A real plan probes its hosts.
    monkeypatch.setattr("sparkrun.core.hardware_probe.probe_hosts", lambda hosts, **kwargs: {})
    with pytest.raises(api.TrustRejected, match=r"Recipe hooks \(pre_exec\) from an untrusted source require explicit authorization"):
        api.run(api.RunOptions(recipe=details["source_path"], hosts=("localhost",), solo=True), sctx=context)
    forbidden.assert_not_called()


@pytest.mark.parametrize("operation", [api.plan, api.run])
@pytest.mark.parametrize("content", ["command", "yaml", "min_nodes"])
def test_public_recipe_data_errors_stay_in_api_family(context, operation, content):
    context.registry_manager._save_registries([], pending_bootstrap_urls=[])
    path = context.config.config_path.parent / "bad.yaml"
    data = recipe_data(**({"command": ["invalid"]} if content == "command" else {"min_nodes": "invalid"}))
    path.write_text("model: [" if content == "yaml" else yaml.safe_dump(data))
    with pytest.raises(api.SparkrunError, match="Recipe is invalid") as error:
        operation(api.RunOptions(recipe=str(path), hosts=("localhost",), dry_run=True), sctx=context)
    assert not isinstance(error.value, api.RecipeNotFound)
    assert isinstance(error.value.__cause__, (RecipeError, ValueError, yaml.YAMLError))


@pytest.mark.parametrize("preloaded", [False, True])
def test_recipe_resolution_errors_with_overrides_stay_typed(context, monkeypatch, preloaded):
    context.registry_manager._save_registries([], pending_bootstrap_urls=[])
    path = context.config.config_path.parent / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe_data()))
    recipe_input = Recipe.load(path, resolve=False) if preloaded else str(path)
    cause = RecipeError("invalid resolver input")
    resolver = Mock(side_effect=cause)
    monkeypatch.setattr(Recipe, "resolve", resolver)
    with pytest.raises(api.SparkrunError, match="Recipe is invalid") as error:
        api.plan(api.RunOptions(recipe=recipe_input, hosts=("localhost",), overrides={"port": 9000}, dry_run=True), sctx=context)
    assert error.value.__cause__ is cause
    resolver.assert_called_once_with({"port": 9000})


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(2), RuntimeError("programming bug")])
def test_recipe_decoder_does_not_swallow_interrupts_or_unexpected_errors(context, monkeypatch, failure):
    recipe = Recipe(recipe_data())
    monkeypatch.setattr(recipe, "resolve", Mock(side_effect=failure))
    with pytest.raises(type(failure)) as error:
        resolve_recipe(recipe, overrides={}, sctx=context)
    assert error.value is failure
