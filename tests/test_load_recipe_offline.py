"""Recipe loading honours ``offline`` on every path that loads by name (B12)."""

from __future__ import annotations


import pytest
import yaml
from click.testing import CliRunner

from sparkrun.core import resolve
from sparkrun.core.recipe import Recipe, RecipeError

_RECIPE = {"recipe_version": "2", "model": "org/model", "runtime": "vllm", "container": "org/image:tag"}


@pytest.fixture
def recipe_file(tmp_path):
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(_RECIPE))
    return path


@pytest.fixture
def load_spy(monkeypatch):
    seen = []
    real = Recipe.load.__func__

    def load(cls, *args, **kwargs):
        seen.append(kwargs.get("offline"))
        return real(cls, *args, **kwargs)

    monkeypatch.setattr(Recipe, "load", classmethod(load))
    return seen


@pytest.fixture
def no_telemetry_side_effects(monkeypatch):
    from sparkrun.core.application_profile import env_name

    monkeypatch.setattr("sparkrun.models.hub.disable_hub_metadata", lambda: None)
    monkeypatch.setenv(env_name("NO_TELEMETRY"), "0")  # restored after _enter_offline_mode sets it


@pytest.mark.parametrize("offline", [True, False])
def test_core_load_recipe_threads_offline_to_recipe_load(v, recipe_file, load_spy, offline):
    from sparkrun.core.config import SparkrunConfig

    recipe, path, _ = resolve.load_recipe(SparkrunConfig(), str(recipe_file), resolve=False, offline=offline)
    assert path is not None and recipe.model == "org/model"
    assert load_spy == [offline]


def test_core_load_recipe_url_is_cache_only_offline(v, recipe_file, load_spy, monkeypatch):
    from sparkrun.core.config import SparkrunConfig

    fetches = []
    monkeypatch.setattr("sparkrun.core.recipe.fetch_and_cache_recipe", lambda url, **kwargs: fetches.append(kwargs) or recipe_file)
    recipe, _, _ = resolve.load_recipe(SparkrunConfig(), "https://spark-arena.com/r.yaml", offline=True)
    assert fetches == [{"allow_untrusted_host": False, "offline": True}]
    assert load_spy == [True]
    assert recipe.source_path == "https://spark-arena.com/r.yaml"


@pytest.mark.parametrize("offline,refreshes", [(True, 0), (False, 1)])
def test_core_load_recipe_refreshes_registries_on_a_miss_only_online(v, monkeypatch, offline, refreshes):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.registry import RegistryManager

    updates, notices = [], []
    monkeypatch.setattr(RegistryManager, "update", lambda self, *a: updates.append(a))
    with pytest.raises(RecipeError):
        resolve.load_recipe(SparkrunConfig(), "no-such-recipe", offline=offline, retry_after_update=True, on_refresh=notices.append)
    assert len(updates) == refreshes and len(notices) == refreshes


def test_cli_wrapper_delegates_offline_to_core(v, recipe_file, monkeypatch):
    from sparkrun.cli._common import _load_recipe
    from sparkrun.core.config import SparkrunConfig

    calls = []
    monkeypatch.setattr(resolve, "load_recipe", lambda config, name, **kwargs: calls.append(kwargs) or ("r", "p", "m"))
    assert _load_recipe(SparkrunConfig(), str(recipe_file), resolve=False, offline=True) == ("r", "p", "m")
    assert calls[0]["offline"] is True and calls[0]["resolve"] is False


def test_cli_wrapper_exits_on_missing_recipe(v):
    from sparkrun.cli._common import _load_recipe
    from sparkrun.core.config import SparkrunConfig

    with pytest.raises(SystemExit):
        _load_recipe(SparkrunConfig(), "no-such-recipe", offline=True)


@pytest.mark.parametrize(
    "flag,cluster_offline,expected", [(None, True, True), (False, True, False), (True, None, True), (None, None, False)]
)
def test_benchmark_recipe_lookup_follows_offline_resolution(v, flag, cluster_offline, expected):
    from sparkrun.api import BenchmarkOptions, default_sctx
    from sparkrun.api._benchmark import _loads_offline

    sctx = default_sctx()
    sctx.cluster_manager.create(name="lab", hosts=["h1"])
    if cluster_offline is not None:
        sctx.cluster_manager.update("lab", offline=cluster_offline)
    assert _loads_offline(BenchmarkOptions(recipe="r", cluster="lab", offline=flag), sctx) is expected


def test_export_realize_offline_loads_recipe_offline(v, recipe_file, no_telemetry_side_effects, monkeypatch):
    from sparkrun import api
    from sparkrun.cli import main

    calls = []
    real = resolve.load_recipe
    monkeypatch.setattr(resolve, "load_recipe", lambda *a, **kw: calls.append(kw.get("offline")) or real(*a, **kw))

    def realize(options, **kwargs):
        raise api.SparkrunError("stop after loading")

    monkeypatch.setattr(api, "realize_recipe", realize)
    result = CliRunner().invoke(main, ["export", "recipe", str(recipe_file), "--realize", "--offline", "--hosts", "h1"])
    assert "stop after loading" in result.output
    assert calls == [True]
