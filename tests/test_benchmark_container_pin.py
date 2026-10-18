"""Tests for container image SHA pinning (P1) and long-term archival ref (P2).

Covers:
- resolve_image_sha helper unit tests
- BenchmarkRunState extras round-trip for SHA and longterm_ref
- BenchmarkExecution.generate_metadata prefers persisted longterm_image_ref over
  live builder resolution
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

from sparkrun.benchmarking.base import BenchmarkExecution
from sparkrun.benchmarking.run_state import BenchmarkRunState
from sparkrun.orchestration.primitives import resolve_image_sha
from sparkrun.orchestration.ssh import RemoteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_remote_result(returncode: int, stdout: str, host: str = "host1") -> RemoteResult:
    return RemoteResult(host=host, returncode=returncode, stdout=stdout, stderr="")


def _make_state(bid: str = "bench_aabbccddeeff") -> BenchmarkRunState:
    return BenchmarkRunState(
        benchmark_id=bid,
        cluster_id="cluster-abc",
        recipe_qualified_name="@registry/my-recipe",
        framework="llama-benchy",
        profile="default",
        base_args={"pp": [2048]},
        schedule=[{"depth": 0, "concurrency": 1}],
    )


# ---------------------------------------------------------------------------
# resolve_image_sha unit tests
# ---------------------------------------------------------------------------


def test_resolve_image_sha_returns_sha_on_success(monkeypatch):
    """Returns sha256:... string when docker inspect succeeds on the first host."""
    expected = "sha256:abcdef1234567890"
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda host, cmd, **kw: _make_remote_result(0, expected + "\n"),
    )
    result = resolve_image_sha("myimage:latest", ["host1"])
    assert result == expected


def test_resolve_image_sha_returns_none_on_failure(monkeypatch):
    """Returns None when docker inspect exits non-zero on all hosts."""
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda host, cmd, **kw: _make_remote_result(1, ""),
    )
    result = resolve_image_sha("myimage:latest", ["host1", "host2"])
    assert result is None


def test_resolve_image_sha_returns_none_dry_run(monkeypatch):
    """Returns None immediately on dry_run without calling run_command_on_host."""
    called = []
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda *a, **kw: called.append(True) or _make_remote_result(0, "sha256:abc"),
    )
    result = resolve_image_sha("myimage:latest", ["host1"], dry_run=True)
    assert result is None
    assert not called, "run_command_on_host should not be called in dry_run mode"


def test_resolve_image_sha_returns_none_empty_hosts(monkeypatch):
    """Returns None when hosts list is empty."""
    called = []
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda *a, **kw: called.append(True) or _make_remote_result(0, "sha256:abc"),
    )
    result = resolve_image_sha("myimage:latest", [])
    assert result is None
    assert not called


def test_resolve_image_sha_iterates_hosts(monkeypatch):
    """First host fails (rc!=0), second succeeds — second host's SHA is returned."""
    expected = "sha256:deadbeef"
    responses = {
        "host1": _make_remote_result(1, "", host="host1"),
        "host2": _make_remote_result(0, expected + "\n", host="host2"),
    }
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda host, cmd, **kw: responses[host],
    )
    result = resolve_image_sha("myimage:latest", ["host1", "host2"])
    assert result == expected


def test_resolve_image_sha_rejects_non_sha256_output(monkeypatch):
    """Returns None when docker inspect output does not start with 'sha256:'."""
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda host, cmd, **kw: _make_remote_result(0, "not-a-sha\n"),
    )
    result = resolve_image_sha("myimage:latest", ["host1"])
    assert result is None


def test_resolve_image_sha_skips_exception_host(monkeypatch):
    """Host raising an exception is skipped; next host is tried."""
    expected = "sha256:cafebabe"

    def _side_effect(host, cmd, **kw):
        if host == "host1":
            raise OSError("connection refused")
        return _make_remote_result(0, expected + "\n", host=host)

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_command_on_host", _side_effect)
    result = resolve_image_sha("myimage:latest", ["host1", "host2"])
    assert result == expected


# ---------------------------------------------------------------------------
# BenchmarkRunState extras round-trip
# ---------------------------------------------------------------------------


def test_sha_pin_persists_in_state_extras(tmp_path: Path):
    """container_image_sha survives a save/load cycle in state.extras."""
    state = _make_state()
    state.extras["container_image_sha"] = "sha256:abc123"
    state.save(str(tmp_path))

    loaded = BenchmarkRunState.load(state.benchmark_id, str(tmp_path))
    assert loaded is not None
    assert loaded.extras.get("container_image_sha") == "sha256:abc123"


def test_longterm_ref_persists_in_state_extras(tmp_path: Path):
    """container_image_longterm_ref survives a save/load cycle in state.extras."""
    state = _make_state()
    state.extras["container_image_longterm_ref"] = "ghcr.io/x/y@sha256:def456"
    state.extras["container_image_longterm_pinned"] = True
    state.save(str(tmp_path))

    loaded = BenchmarkRunState.load(state.benchmark_id, str(tmp_path))
    assert loaded is not None
    assert loaded.extras.get("container_image_longterm_ref") == "ghcr.io/x/y@sha256:def456"
    assert loaded.extras.get("container_image_longterm_pinned") is True


# ---------------------------------------------------------------------------
# BenchmarkExecution.generate_metadata — longterm_image_ref preference
# ---------------------------------------------------------------------------


def _make_minimal_recipe(container: str = "myimage:latest") -> MagicMock:
    from sparkrun.core.recipe import Recipe

    recipe = Recipe({"name": "r", "container": container, "model": "org/model", "runtime": "vllm-distributed"})
    recipe.source_registry = "reg"
    recipe.source_registry_url = ""
    return recipe


def _make_bench_result_with_launch(recipe: MagicMock, builder_return: tuple) -> BenchmarkExecution:
    """BenchmarkExecution with a launch_result whose builder returns builder_return."""
    builder = MagicMock()
    builder.resolve_long_term_image.return_value = builder_return

    launch_result = MagicMock()
    launch_result.recipe = recipe
    launch_result.overrides = {}
    launch_result.cluster_id = "cluster-abc"
    launch_result.host_list = ["host1"]
    launch_result.container_image = recipe.container
    launch_result.runtime_info = {}
    launch_result.builder = builder

    br = BenchmarkExecution()
    br.launch_result = launch_result
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    br.start_time = now
    br.end_time = now
    br.framework = MagicMock()
    br.framework.framework_name = "llama-benchy"
    br.profile = "default"
    br.benchmark_args = {}
    return br


def test_generate_metadata_prefers_persisted_longterm_ref():
    """When longterm_image_ref is pre-set, generate_metadata uses it and does
    not call builder.resolve_long_term_image."""
    recipe = _make_minimal_recipe("myimage:latest")
    persisted_ref = "ghcr.io/x/y@sha256:persisted"
    live_ref = "ghcr.io/x/y@sha256:live-different"

    br = _make_bench_result_with_launch(recipe, (live_ref, True))
    br.longterm_image_ref = persisted_ref
    br.longterm_image_pinned = True

    meta = br.generate_metadata()

    assert meta["recipe"]["container"] == persisted_ref
    assert meta["recipe"]["container_pinned"] is True
    # Builder should NOT have been called since we had a persisted ref
    br.launch_result.builder.resolve_long_term_image.assert_not_called()


def test_generate_metadata_falls_back_to_builder_when_no_persisted_ref():
    """When longterm_image_ref is None, generate_metadata falls back to
    builder.resolve_long_term_image."""
    recipe = _make_minimal_recipe("myimage:latest")
    live_ref = "ghcr.io/x/y@sha256:ddd"

    br = _make_bench_result_with_launch(recipe, (live_ref, True))
    # longterm_image_ref is None by default

    meta = br.generate_metadata()

    assert meta["recipe"]["container"] == live_ref
    assert meta["recipe"]["container_pinned"] is True
    br.launch_result.builder.resolve_long_term_image.assert_called_once()


def test_generate_metadata_unpinned_builder_leaves_raw_container():
    """When builder returns pinned=False, recipe_container stays as-is."""
    recipe = _make_minimal_recipe("myimage:latest")

    br = _make_bench_result_with_launch(recipe, ("ghcr.io/x/y@sha256:ddd", False))
    meta = br.generate_metadata()

    # pinned=False from builder means we don't override recipe_container
    assert meta["recipe"]["container_pinned"] is False
    assert meta["recipe"]["container"] == recipe.container


def _rich_recipe(executor_config):
    from sparkrun.core.recipe import Recipe

    recipe = Recipe(
        {
            "name": "r",
            "container": "myimage:latest",
            "model": "org/model",
            "runtime": "vllm-distributed",
            "executor_config": executor_config,
            "capabilities": ["tools"],
        }
    )
    recipe.source_registry, recipe.source_registry_url = "reg", ""
    return recipe


def test_published_recipe_carries_portable_executor_settings():
    """A published recipe is also the prototype someone else launches: ``entrypoint: ""``
    decides whether the image starts at all, so it is part of the record (#323)."""
    import yaml

    from sparkrun.core.recipe import Recipe

    recipe = _rich_recipe({"entrypoint": "", "ipc": "host", "shm_size": "32g"})
    meta = _make_bench_result_with_launch(recipe, ("myimage:latest", False)).generate_metadata()
    published = yaml.safe_load(meta["recipe"]["text"])
    assert published["executor_config"] == {"entrypoint": "", "ipc": "host", "shm_size": "32g"}
    assert published["capabilities"] == ["tools"]
    assert Recipe(published).executor_config["entrypoint"] == ""


def test_published_recipe_omits_host_specific_executor_settings():
    """Host paths and site metadata stay out; a trailing comment says what was dropped."""
    import yaml

    recipe = _rich_recipe(
        {
            "entrypoint": "",
            "volumes": ["/home/someone/models:/models:ro", "/scratch"],
            "user": "0:0",
            "devices": ["/dev/infiniband"],
            "kubeconfig": "/home/someone/.kube/config",
        }
    )
    meta = _make_bench_result_with_launch(recipe, ("myimage:latest", False)).generate_metadata()
    text = meta["recipe"]["text"]
    assert "/home/someone" not in text
    # Running as root is part of the run, so ``user`` is kept.
    assert yaml.safe_load(text)["executor_config"] == {"entrypoint": "", "user": "0:0"}
    assert "#   volumes: <host path>:/models:ro" in text and "#   volumes: <host path>:/scratch" in text
    assert "#   devices" in text and "#   kubeconfig" in text
    assert meta["recipe"]["effective_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_published_recipe_without_executor_settings_is_unchanged():
    """Recipes that declare no executor settings keep the text and hashes they had before."""
    from sparkrun.benchmarking.metadata import public_recipe_text

    recipe = _make_minimal_recipe()
    meta = _make_bench_result_with_launch(recipe, ("myimage:latest", False)).generate_metadata()
    assert meta["recipe"]["hash"] == hashlib.sha256(public_recipe_text(recipe.export()).encode("utf-8")).hexdigest()
    assert "executor" not in meta["recipe"]["text"]


def test_every_executor_config_field_is_classified_for_publication():
    """A new ExecutorConfig field must be declared portable or host-specific, or it would be
    silently dropped from (or, worse, wrongly kept in) every published recipe."""
    import dataclasses

    from sparkrun.benchmarking.metadata import HOST_SPECIFIC_EXECUTOR_KEYS, PORTABLE_EXECUTOR_KEYS
    from sparkrun.orchestration.executors._base import ExecutorConfig

    fields = {field.name for field in dataclasses.fields(ExecutorConfig)}
    assert fields - PORTABLE_EXECUTOR_KEYS - HOST_SPECIFIC_EXECUTOR_KEYS == set()
    assert not PORTABLE_EXECUTOR_KEYS & HOST_SPECIFIC_EXECUTOR_KEYS
