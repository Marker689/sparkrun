"""Tests for ``distribution_config`` parsing, esp. per-subkey defaulting.

An omitted ``models`` / ``containers`` subkey falls back to the auto-default
single entry (``{model}`` / ``{container}``); a subkey that is present is
honored literally (including an explicit empty ``entries`` or ``enabled: false``).
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from sparkrun.core.recipe import Recipe


def _recipe(dist_cfg=None):
    d = {
        "recipe_version": "2",
        "name": "dc",
        "model": "org/primary",
        "runtime": "vllm-distributed",
        "container": "img:latest",
    }
    if dist_cfg is not None:
        d["distribution_config"] = dist_cfg
    return Recipe.from_dict(d)


def _names(rc):
    return [e.name for e in rc.entries]


def test_no_block_uses_default_both():
    dc = _recipe().distribution_config
    assert _names(dc.models) == ["{model}"]
    assert _names(dc.containers) == ["{container}"]


def test_models_present_containers_defaulted():
    """The headline ergonomics fix: add a model, don't re-list the container."""
    dc = _recipe({"models": {"entries": [{"name": "{model}"}, {"name": "aux/backbone"}]}}).distribution_config
    assert _names(dc.models) == ["{model}", "aux/backbone"]
    # containers omitted → auto-default entry, not empty
    assert _names(dc.containers) == ["{container}"]


def test_containers_present_models_defaulted():
    dc = _recipe({"containers": {"entries": [{"name": "custom:img"}]}}).distribution_config
    assert _names(dc.models) == ["{model}"]
    assert _names(dc.containers) == ["custom:img"]


def test_present_empty_entries_is_honored_not_defaulted():
    """An explicit empty list means 'distribute nothing', not 'use default'."""
    dc = _recipe({"models": {"entries": []}}).distribution_config
    assert _names(dc.models) == []
    # the omitted containers subkey still defaults
    assert _names(dc.containers) == ["{container}"]


def test_present_disabled_subkey_is_honored():
    dc = _recipe({"models": {"enabled": False, "entries": [{"name": "{model}"}]}}).distribution_config
    assert dc.models.enabled is False
    assert _names(dc.models) == ["{model}"]


def test_string_shorthand_entries():
    dc = _recipe({"models": {"entries": ["{model}", "aux/backbone"]}}).distribution_config
    assert _names(dc.models) == ["{model}", "aux/backbone"]


def test_target_defaults_to_all_nodes():
    dc = _recipe({"models": {"entries": [{"name": "aux/backbone"}]}}).distribution_config
    assert dc.models.entries[0].target == [-1]


def test_provided_block_marked_externally_provided():
    dc = _recipe({"models": {"entries": [{"name": "{model}"}]}}).distribution_config
    assert dc.externally_provided is True


def test_resolve_substitutes_placeholders_across_defaulted_subkey():
    r = _recipe({"models": {"entries": [{"name": "{model}"}, {"name": "aux/backbone"}]}})
    dc = r.distribution_config.resolve(r, resolved_container=r.container, overrides={})
    assert _names(dc.models) == ["org/primary", "aux/backbone"]
    # the defaulted container subkey still resolves {container}
    assert _names(dc.containers) == ["img:latest"]


# --- per-entry revision ---------------------------------------------------
#
# A commit SHA is only meaningful in the repo it came from.  The recipe's
# top-level `model_revision` is stamped onto the auto-generated entry, and
# nothing else inherits it — a draft model added by runtime.prepare() is a
# different repo, and pinning it to the served model's SHA fails the download
# with "Revision Not Found" after the served model has already synced.


def _rev_recipe(model_revision=None, dist_cfg=None):
    d = {
        "recipe_version": "2",
        "name": "dc",
        "model": "org/primary",
        "runtime": "vllm-distributed",
        "container": "img:latest",
    }
    if model_revision is not None:
        d["model_revision"] = model_revision
    if dist_cfg is not None:
        d["distribution_config"] = dist_cfg
    return Recipe.from_dict(d)


def _revs(rc):
    return [e.revision for e in rc.entries]


def test_model_revision_stamped_on_auto_default_entry():
    dc = _rev_recipe(model_revision="deadbeef").distribution_config
    assert _revs(dc.models) == ["deadbeef"]


def test_model_revision_stamped_on_inherited_default_when_models_omitted():
    """`models` omitted → the inherited default entry still carries the pin."""
    dc = _rev_recipe("deadbeef", {"containers": {"entries": [{"name": "custom:img"}]}}).distribution_config
    assert _revs(dc.models) == ["deadbeef"]


def test_added_model_does_not_inherit_recipe_revision():
    """The bug: a draft model pinned to the served model's SHA 404s."""
    r = _rev_recipe(model_revision="deadbeef")
    r.distribution_config.add_model("org/draft")
    entries = {e.name: e.revision for e in r.distribution_config.models.entries}
    assert entries == {"{model}": "deadbeef", "org/draft": None}


def test_add_model_pins_its_own_revision():
    r = _rev_recipe(model_revision="deadbeef")
    r.distribution_config.add_model("org/draft", revision="cafe1234")
    entries = {e.name: e.revision for e in r.distribution_config.models.entries}
    assert entries == {"{model}": "deadbeef", "org/draft": "cafe1234"}


def test_add_model_pin_upgrades_an_earlier_unpinned_entry():
    """A later caller with a revision outranks an earlier unpinned add.

    Dedup used to return on the first name match, so whichever call happened to
    run first won — and an unpinned add silently discarded a pin that arrived
    afterwards, fetching the draft model unpinned.
    """
    r = _rev_recipe(model_revision="deadbeef")
    r.distribution_config.add_model("org/draft")
    r.distribution_config.add_model("org/draft", revision="cafe1234")
    entries = {e.name: e.revision for e in r.distribution_config.models.entries}
    assert entries == {"{model}": "deadbeef", "org/draft": "cafe1234"}


def test_add_model_without_a_pin_never_clears_one():
    """The reverse order must not un-pin what a previous caller pinned."""
    r = _rev_recipe(model_revision="deadbeef")
    r.distribution_config.add_model("org/draft", revision="cafe1234")
    r.distribution_config.add_model("org/draft")
    entries = {e.name: e.revision for e in r.distribution_config.models.entries}
    assert entries == {"{model}": "deadbeef", "org/draft": "cafe1234"}


def test_hand_written_entries_are_authoritative_about_revision():
    """A recipe that lists its own entries states every revision it wants."""
    dc = _rev_recipe(
        "deadbeef",
        {"models": {"entries": [{"name": "{model}"}, {"name": "aux/backbone", "revision": "abc123"}]}},
    ).distribution_config
    assert _revs(dc.models) == [None, "abc123"]


def test_revision_survives_state_round_trip():
    """Job metadata re-reads the recipe; an unpinned draft must stay unpinned."""
    r = _rev_recipe(model_revision="deadbeef")
    r.distribution_config.add_model("org/draft")
    restored = Recipe._deserialize(r.__getstate__())
    entries = {e.name: e.revision for e in restored.distribution_config.models.entries}
    assert entries == {"{model}": "deadbeef", "org/draft": None}


def test_resource_entry_types_survive_roundtrip_and_keep_model_revisions():
    from dataclasses import asdict
    from sparkrun.core.recipe import DistributionConfig, DistributionModelEntry, DistributionContainerEntry

    original = _recipe().distribution_config
    original.add_model("org/draft", revision="draft-sha")
    restored = DistributionConfig.from_dict(asdict(original))
    assert all(isinstance(entry, DistributionModelEntry) for entry in restored.models.entries)
    assert all(isinstance(entry, DistributionContainerEntry) for entry in restored.containers.entries)
    assert restored.models.entries[-1].revision == "draft-sha"
    restored.add_model("org/draft", revision="updated-sha")
    assert restored.models.entries[-1].revision == "updated-sha"


def test_cross_kind_entries_fail_at_construction_and_after_mutation():
    import pytest
    from sparkrun.core.recipe import DistributionConfig, DistributionResourceConfig, DistributionContainerEntry, RecipeError

    wrong = DistributionContainerEntry("image:tag")
    with pytest.raises(RecipeError, match="models.*DistributionModelEntry"):
        DistributionConfig(models=DistributionResourceConfig(entries=[wrong]))
    with pytest.raises(RecipeError, match="DistributionModelEntry"):
        DistributionConfig.from_dict({"models": {"entries": [wrong]}})
    config = DistributionConfig()
    config.models.entries.append(wrong)
    with pytest.raises(RecipeError, match="models.*DistributionModelEntry"):
        config.resolve(_recipe())


def test_additional_models_preserve_defaults_and_runtime_pins():
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    recipe = _rev_recipe(
        "primary-sha",
        {"additional_models": ["org/tokenizer", {"name": "org/draft", "revision": "extra-sha", "target": [0]}]},
    )
    recipe.defaults["speculative_config"] = {"model": "org/draft", "revision": "runtime-sha", "num_speculative_tokens": 3}
    VllmDistributedRuntime().prepare(recipe, ["h1"])
    before = deepcopy(recipe.distribution_config)

    resolved = recipe.distribution_config.finalize(recipe, resolved_container=recipe.container)

    assert [(e.name, e.revision, e.target) for e in resolved.models.entries] == [
        ("org/primary", "primary-sha", [-1]),
        ("org/draft", "runtime-sha", [-1]),
        ("org/tokenizer", None, [-1]),
        ("org/draft", "extra-sha", [0]),
    ]
    assert _names(resolved.containers) == ["img:latest"]
    assert resolved.additional_models == []
    assert recipe.distribution_config == before
    assert not before.models.explicit and not before.containers.explicit


@pytest.mark.parametrize("entries", [[], ["org/replacement"]])
@pytest.mark.parametrize("enabled", [False, True])
def test_additional_models_append_to_explicit_models_without_restoring_defaults(entries, enabled):
    recipe = _recipe({"models": {"entries": entries, "enabled": enabled}, "additional_models": ["org/extra"]})
    resolved = recipe.distribution_config.finalize(recipe)
    assert _names(resolved.models) == entries + ["org/extra"]
    assert resolved.models.enabled is enabled
    assert resolved.models.explicit


@pytest.mark.parametrize("extras", [None, []])
def test_empty_additional_models_keep_defaults(extras):
    recipe = _recipe({"additional_models": extras})
    assert _names(recipe.distribution_config.finalize(recipe).models) == ["org/primary"]


@pytest.mark.parametrize(
    "extras, message",
    [
        ("org/model", "must be a list"),
        ({"entries": []}, "must be a list"),
        ([42], "must be a model name or an entry mapping"),
        ([{}], "name must be a non-empty string"),
        ([" "], "name must be a non-empty string"),
        ([{"name": 12}], "name must be a non-empty string"),
        ([{"name": "org/model", "target": 0}], "target must be a list of node indices"),
        ([{"name": "org/model", "target": [True]}], "target must be a list of node indices"),
        ([{"name": "org/model", "revision": 123}], "revision must be a non-empty string or null"),
    ],
)
def test_invalid_additional_models_report_the_field(extras, message):
    from sparkrun.core.recipe import RecipeError

    with pytest.raises(RecipeError, match="distribution_config.additional_models.*" + message):
        _recipe({"additional_models": extras})


def test_finalization_deduplicates_resolved_names_and_unions_targets():
    recipe = _recipe(
        {
            "models": {"entries": [{"name": "{model}", "revision": "sha", "target": [0, 0]}]},
            "additional_models": [
                {"name": "org/primary", "revision": "sha", "target": [1, 0]},
                {"name": "org/primary", "revision": "other-sha", "target": [1]},
                "org/primary",
                "org/gguf:Q4_K_M",
                "org/gguf:Q8_0",
                "org/gguf:Q4_K_M",
            ],
            "containers": {
                "entries": [
                    {"name": "{container}", "target": [0, 0]},
                    {"name": "built:tag", "target": [1, 0]},
                    {"name": "", "target": [2]},
                    {"name": "helper:tag", "target": [0, 0]},
                ]
            },
        }
    )
    before = deepcopy(recipe.distribution_config)
    resolved = recipe.distribution_config.finalize(recipe, resolved_container="built:tag")
    assert [(e.name, e.revision, e.target) for e in resolved.models.entries] == [
        ("org/primary", "sha", [0, 1]),
        ("org/primary", "other-sha", [1]),
        ("org/primary", None, [-1]),
        ("org/gguf:Q4_K_M", None, [-1]),
        ("org/gguf:Q8_0", None, [-1]),
    ]
    assert [(e.name, e.target) for e in resolved.containers.entries] == [("built:tag", [0, 1, 2]), ("helper:tag", [0])]
    assert resolved.finalize(recipe, resolved_container="built:tag") == resolved
    assert recipe.distribution_config == before


@pytest.mark.parametrize("all_nodes", [{}, {"target": []}, {"target": [-1]}, {"target": [-1, 0]}])
@pytest.mark.parametrize("all_first", [False, True])
def test_all_node_targets_dominate_in_either_order(all_nodes, all_first):
    entries = [{"name": "org/model", "target": [1]}, {"name": "org/model", **all_nodes}]
    if all_first:
        entries.reverse()
    recipe = _recipe({"models": {"entries": []}, "additional_models": entries, "containers": {"entries": entries}})
    resolved = recipe.distribution_config.finalize(recipe)
    assert [(e.name, e.target) for e in resolved.models.entries] == [("org/model", [-1])]
    assert [(e.name, e.target) for e in resolved.containers.entries] == [("org/model", [-1])]


def test_finalization_can_be_repeated_with_different_launch_values():
    recipe = _recipe({"additional_models": ["{model}", "{aux_model}"]})
    before = deepcopy(recipe.distribution_config)
    first = recipe.distribution_config.finalize(recipe, resolved_container="first:tag", overrides={"aux_model": "org/aux-first"})
    recipe.model = "org/second"
    second = recipe.distribution_config.finalize(recipe, resolved_container="second:tag", overrides={"aux_model": "org/aux-second"})
    assert _names(first.models) == ["org/primary", "org/aux-first"]
    assert _names(second.models) == ["org/second", "org/aux-second"]
    assert _names(first.containers) == ["first:tag"]
    assert _names(second.containers) == ["second:tag"]
    assert recipe.distribution_config == before


def test_additional_models_survive_state_and_yaml_round_trips():
    import yaml
    from dataclasses import asdict
    from sparkrun.core.recipe import DistributionConfig

    recipe = _rev_recipe("primary-sha", {"additional_models": ["org/tokenizer", {"name": "org/draft", "revision": "draft-sha"}]})
    recipe.distribution_config.add_model("org/runtime", revision="runtime-sha")
    restored = Recipe._deserialize(recipe.__getstate__())
    assert restored.distribution_config == recipe.distribution_config
    assert DistributionConfig.from_dict(asdict(recipe.distribution_config)) == recipe.distribution_config

    exported = yaml.safe_load(recipe.export())
    assert set(exported["distribution_config"]) == {"additional_models"}
    reloaded = Recipe.from_dict(exported)
    assert reloaded.distribution_config.additional_models == recipe.distribution_config.additional_models
    assert not reloaded.distribution_config.models.explicit
    assert not reloaded.distribution_config.containers.explicit
    assert [(e.name, e.revision) for e in reloaded.distribution_config.finalize(reloaded).models.entries] == [
        ("org/primary", "primary-sha"),
        ("org/tokenizer", None),
        ("org/draft", "draft-sha"),
    ]
    # An exported additive recipe still picks up a changed primary pin.
    exported["model_revision"] = "new-primary-sha"
    assert Recipe.from_dict(exported).distribution_config.models.entries[0].revision == "new-primary-sha"


@pytest.mark.parametrize("kind", ["models", "containers"])
def test_export_preserves_only_explicit_distribution_sections(kind):
    recipe = _recipe({kind: {"enabled": False, "entries": []}, "additional_models": ["org/extra"]})
    data = recipe.to_dict()
    assert data["distribution_config"] == {
        kind: {"enabled": False, "entries": []},
        "additional_models": [{"name": "org/extra", "target": [], "revision": None}],
    }
    assert Recipe.from_dict(data).distribution_config == recipe.distribution_config


def test_old_distribution_state_without_additional_models_loads():
    recipe = _recipe()
    state = recipe.__getstate__()
    state["distribution_config"].pop("additional_models")
    assert Recipe._deserialize(state).distribution_config == recipe.distribution_config


@pytest.mark.parametrize("child_extras, expected", [(["org/child"], ["org/child"]), ([], []), (None, [])])
def test_include_replaces_additional_models_list(tmp_path, child_extras, expected):
    import yaml

    (tmp_path / "base.yaml").write_text(yaml.safe_dump(_recipe({"additional_models": ["org/base"]}).to_dict()))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({"include": "base", "distribution_config": {"additional_models": child_extras}}))
    recipe = Recipe.load(tmp_path / "child.yaml")
    assert _names(recipe.distribution_config.finalize(recipe).models) == ["org/primary", *expected]


def test_additional_models_reject_cross_kind_mutations():
    from sparkrun.core.recipe import DistributionContainerEntry, RecipeError

    recipe = _recipe()
    recipe.distribution_config.additional_models.append(DistributionContainerEntry("image:tag"))
    with pytest.raises(RecipeError, match="additional_models.*DistributionModelEntry"):
        recipe.distribution_config.finalize(recipe)
