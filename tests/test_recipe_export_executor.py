"""Recipe export keeps the executor settings a launch depends on (issue #323)."""

from __future__ import annotations

import yaml

from sparkrun.core.recipe import Recipe

_BASE = {"model": "org/model", "runtime": "vllm-distributed", "container": "org/img:1", "defaults": {"port": 8000}}


def _round_trip(recipe: Recipe, **kwargs) -> Recipe:
    return Recipe(yaml.safe_load(recipe.export(**kwargs)))


def test_an_empty_entrypoint_survives_export():
    """``entrypoint: ""`` clears a consuming image ENTRYPOINT; dropping it as falsy breaks the launch."""
    recipe = Recipe({**_BASE, "executor_config": {"entrypoint": "", "shm_size": "16g"}})
    assert recipe.to_dict()["executor_config"] == {"entrypoint": "", "shm_size": "16g"}
    assert _round_trip(recipe).executor_config == {"entrypoint": "", "shm_size": "16g"}


def test_executor_selector_and_capabilities_survive_export():
    recipe = Recipe({**_BASE, "executor": "local", "capabilities": ["tools"], "unsupported_capabilities": ["vision"]})
    exported = _round_trip(recipe)
    assert exported.executor == "local"
    assert exported.capabilities == ["tools"] and exported.unsupported_capabilities == ["vision"]


def test_a_recipe_without_executor_settings_exports_none():
    data = Recipe(dict(_BASE)).to_dict()
    assert "executor" not in data and "executor_config" not in data and "capabilities" not in data


def test_executor_overrides_merge_over_the_recipes_own():
    recipe = Recipe({**_BASE, "executor_config": {"entrypoint": "/init", "shm_size": "8g"}})
    data = recipe.to_dict(overrides={"port": 9000}, executor_overrides={"entrypoint": "", "executor": "docker"})
    assert data["executor_config"] == {"entrypoint": "", "shm_size": "8g"}
    assert data["executor"] == "docker"
    assert data["defaults"]["port"] == 9000 and "entrypoint" not in data["defaults"]
