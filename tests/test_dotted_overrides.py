"""``-o root.field=value`` sets one field of a mapping-valued default."""

from __future__ import annotations

import json

import pytest

from sparkrun.core.dotted_overrides import DottedOverrideError, fold_dotted_overrides, is_dotted_override
from sparkrun.core.recipe import Recipe
from sparkrun.core.resolve import apply_recipe_overrides
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

SPEC = {
    "method": "dflash",
    "model": "org/drafter",
    "revision": "abc123",
    "num_speculative_tokens": 7,
    "draft_load_config": {"load_format": "auto"},
}


def _recipe(**defaults) -> Recipe:
    return Recipe.from_dict({"model": "org/model", "runtime": "vllm", "defaults": {"port": 8000, **defaults}})


def _spec_in(command: str) -> dict:
    marker = "--speculative-config '"
    start = command.index(marker) + len(marker)
    return json.loads(command[start : command.index("'", start)])


def _command(recipe: Recipe, options: tuple[str, ...]) -> str:
    recipe, overrides = apply_recipe_overrides(options, recipe=recipe)
    return VllmDistributedRuntime().generate_command(recipe, overrides, is_cluster=False)


@pytest.mark.parametrize(
    ("key", "dotted"),
    [("a.b", True), ("a.b.c", True), ("env.X", False), ("_x.y", False), ("plain", False), ("a.", False), (".a", False)],
)
def test_is_dotted_override(key, dotted):
    assert is_dotted_override(key) is dotted


def test_one_field_is_replaced_and_the_rest_kept():
    spec = _spec_in(_command(_recipe(speculative_config=SPEC), ("speculative_config.method=mtp",)))
    assert spec == {**SPEC, "method": "mtp"}


def test_nested_field_and_coercion():
    spec = _spec_in(
        _command(
            _recipe(speculative_config=SPEC),
            ("speculative_config.draft_load_config.load_format=dummy", "speculative_config.num_speculative_tokens=3"),
        )
    )
    assert spec["draft_load_config"] == {"load_format": "dummy"}
    assert spec["num_speculative_tokens"] == 3


def test_json_string_root_is_refined():
    """Older recipes spell speculative_config as a JSON string."""
    spec = _spec_in(_command(_recipe(speculative_config=json.dumps(SPEC)), ("speculative_config.method=mtp",)))
    assert spec == {**SPEC, "method": "mtp"}


def test_dotted_refines_a_whole_value_override():
    options = ('speculative_config={"method": "eagle", "num_speculative_tokens": 2}', "speculative_config.num_speculative_tokens=4")
    spec = _spec_in(_command(_recipe(speculative_config=SPEC), options))
    assert spec == {"method": "eagle", "num_speculative_tokens": 4}


def test_json_leaf_is_structure():
    spec = _spec_in(_command(_recipe(speculative_config=SPEC), ('speculative_config.draft_load_config={"load_format": "x"}',)))
    assert spec["draft_load_config"] == {"load_format": "x"}


def test_recipe_defaults_are_not_mutated():
    recipe = _recipe(speculative_config=SPEC)
    _command(recipe, ("speculative_config.method=mtp",))
    assert recipe.defaults["speculative_config"] == SPEC


def test_folds_over_a_later_defaults_value():
    """Folding happens at chain build, so a matched `overrides:` layer is refined, not replaced."""
    recipe, overrides = apply_recipe_overrides(("speculative_config.method=mtp",), recipe=_recipe(speculative_config=SPEC))
    recipe.defaults["speculative_config"] = {"method": "eagle", "num_speculative_tokens": 1}
    assert recipe.build_config_chain(overrides).get("speculative_config") == {"method": "mtp", "num_speculative_tokens": 1}


def test_drafter_detection_sees_the_override():
    recipe, _ = apply_recipe_overrides(("speculative_config.model=org/other-drafter",), recipe=_recipe(speculative_config=SPEC))
    assert VllmDistributedRuntime().detect_spec_config_draft(recipe) == ("org/other-drafter", "abc123")


@pytest.mark.parametrize("option", ["speculative_confg.method=x", "max_model_len.x=1", "port.x=1"])
def test_path_without_a_mapping_is_refused(option):
    with pytest.raises(DottedOverrideError, match="Invalid --option"):
        apply_recipe_overrides((option,), recipe=_recipe(speculative_config=SPEC))


def test_path_through_a_scalar_field_is_refused():
    with pytest.raises(DottedOverrideError, match="speculative_config.method"):
        fold_dotted_overrides({"speculative_config.method.x": 1}, {"speculative_config": SPEC}.get)


def test_env_prefix_still_routes_to_container_env():
    recipe, overrides = apply_recipe_overrides(("env.FOO=bar",), recipe=_recipe())
    assert recipe.env["FOO"] == "bar"
    assert "env.FOO" not in overrides
