"""Tests for the KV cache sizing seam (``sparkrun.models.kv``).

The MLA and dense *numbers* are covered in ``test_vram.py``; this file covers
the seam itself — resolution order, the field declaration that drives the
plumbing, and whether a new architecture can actually be added without editing
core.  That last one is the whole point of the abstraction, so it is tested
end-to-end rather than asserted in a docstring.
"""

from __future__ import annotations

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.models.kv import (
    ArchField,
    ArchInfo,
    KVCacheStrategy,
    KVDetection,
    KVSizing,
    arch_field,
    arch_fields,
    arch_marker_names,
    extract_arch_fields,
    is_kv_layout,
    is_valid_kv_dtype,
    list_kv_strategies,
    register_kv_strategy,
    resolve_kv_strategy,
)
from sparkrun.models.vram import estimate_vram, extract_model_info
import sparkrun.models.kv as kv_registry


@pytest.fixture
def clean_registry():
    """Restore the process-global strategy registry after a test mutates it."""
    saved = list(kv_registry._REGISTRY)
    yield
    kv_registry._REGISTRY[:] = saved


class TestResolutionOrder:
    """Most specific claim wins; dense terminates the search."""

    def test_dense_is_last(self):
        assert [s.name for s in list_kv_strategies()][-1] == "dense"

    def test_unremarkable_model_resolves_to_dense(self):
        strategy, detection = resolve_kv_strategy(ArchInfo(num_layers=32, num_kv_heads=8, head_dim=128, model_type="qwen3"))
        assert strategy.name == "dense"
        assert detection.source == "fallback"

    @pytest.mark.parametrize(
        ("arch", "source"),
        [
            pytest.param(ArchInfo(extra={"kv_lora_rank": 512}), "architecture", id="marker"),
            pytest.param(ArchInfo(kv_dtype="fp8_ds_mla"), "kv_layout", id="layout"),
            pytest.param(ArchInfo(model_type="deepseek_v3"), "model_type", id="family"),
        ],
    )
    def test_mla_claims_and_reports_why(self, arch, source):
        strategy, detection = resolve_kv_strategy(arch)
        assert strategy.name == "mla"
        assert detection.source == source

    def test_resolution_always_terminates(self):
        """Even an empty ArchInfo resolves — callers never handle a None."""
        strategy, detection = resolve_kv_strategy(ArchInfo())
        assert strategy.name == "dense"
        assert detection is not None


class TestArchInfo:
    def test_extra_shadows_nothing_universal(self):
        arch = ArchInfo(head_dim=128, extra={"kv_lora_rank": 512})
        assert arch.get("head_dim") == 128
        assert arch.get("kv_lora_rank") == 512
        assert arch.get("nonexistent") is None

    def test_present_but_none_reads_as_absent(self):
        """Callers build `extra` from a plain metadata sweep, misses included."""
        arch = ArchInfo(num_layers=61, extra={"kv_lora_rank": None, "compress_ratios": None})
        assert arch.get("kv_lora_rank") is None
        assert arch.get("kv_lora_rank", 7) == 7
        assert arch.get("num_layers") == 61


class TestArchFieldDeclaration:
    def test_every_declared_field_is_unique(self):
        names = [f.name for f in arch_fields()]
        assert len(names) == len(set(names))

    def test_markers_match_declarations(self):
        assert arch_marker_names() == {f.name for f in arch_fields()}

    def test_lookup_by_name(self):
        assert arch_field("kv_lora_rank").kind is int
        assert arch_field("compress_ratios").kind is list
        assert arch_field("no_such_field") is None

    @pytest.mark.parametrize(
        ("value", "ok"),
        [
            pytest.param(512, True, id="positive-int"),
            pytest.param(0, False, id="zero"),
            pytest.param(-1, False, id="negative"),
            # bool is an int subclass, so `kv_lora_rank: true` would coerce to a
            # one-element-wide latent if this were not rejected explicitly.
            pytest.param(True, False, id="bool"),
            pytest.param("512", False, id="string"),
            pytest.param(None, True, id="absent"),
        ],
    )
    def test_int_validation(self, value, ok):
        assert (arch_field("kv_lora_rank").validate(value) is None) is ok

    @pytest.mark.parametrize(
        ("value", "ok"),
        [
            pytest.param([4, 128], True, id="ints"),
            pytest.param([], False, id="empty"),
            pytest.param([4, True], False, id="contains-bool"),
            pytest.param("4,128", False, id="string"),
            pytest.param(None, True, id="absent"),
        ],
    )
    def test_list_validation(self, value, ok):
        assert (arch_field("compress_ratios").validate(value) is None) is ok

    def test_coercion(self):
        assert arch_field("kv_lora_rank").coerce(512.0) == 512
        assert arch_field("compress_ratios").coerce((4, 128)) == [4, 128]


class TestKvDtypes:
    def test_packed_layouts_are_layouts_not_element_widths(self):
        assert is_kv_layout("fp8_ds_mla")
        assert is_kv_layout("nvfp4-ds-mla")
        assert not is_kv_layout("bfloat16")

    def test_validity_covers_both_kinds(self):
        assert is_valid_kv_dtype("fp8_ds_mla")
        assert is_valid_kv_dtype("bf16")
        assert not is_valid_kv_dtype("bogus")


# ---------------------------------------------------------------------------
# A new architecture, added the way a real one would be.
# ---------------------------------------------------------------------------


class _SlidingWindowStrategy(KVCacheStrategy):
    """A windowed cache: sublinear in context length, and it saturates.

    Deliberately the shape the per-token contract *cannot* express — every
    layer caches ``min(max_model_len, window)`` tokens — because that is what
    the next real architecture (Gemma 3, Llama-4 iRoPE) looks like.
    """

    name = "sliding"
    label = "Sliding-window attention"
    priority = 5
    replicates_kv = False

    def arch_fields(self):
        return (ArchField("sliding_window", hf_keys=("sliding_window",), doc="Window size in tokens."),)

    def detect(self, arch):
        if arch.get("sliding_window"):
            return KVDetection(source="architecture")
        return None

    def size(self, arch, *, max_model_len):
        window = arch.get("sliding_window")
        per_token = 2.0 * arch.num_layers * arch.num_kv_heads * arch.head_dim * 2.0
        cached = min(max_model_len, window) if max_model_len else window
        return KVSizing(total_bytes=per_token * cached, per_token_bytes=None)

    def tokens_for_budget(self, arch, sizing, budget_bytes):
        # Saturates at the window: a linear extrapolation would promise a
        # context length the cache cannot actually hold.
        return arch.get("sliding_window")


class TestAddingAnArchitecture:
    """Registering a strategy is the *only* edit a new architecture needs."""

    def test_registration_respects_priority(self, clean_registry):
        register_kv_strategy(_SlidingWindowStrategy())
        names = [s.name for s in list_kv_strategies()]
        assert names == ["sliding", "mla", "hybrid", "dense"]
        assert names[-1] == "dense", "dense must stay the terminator"

    def test_re_registering_a_name_replaces_it(self, clean_registry):
        register_kv_strategy(_SlidingWindowStrategy())
        register_kv_strategy(_SlidingWindowStrategy())
        assert [s.name for s in list_kv_strategies()].count("sliding") == 1

    def test_its_field_joins_the_declaration(self, clean_registry):
        assert "sliding_window" not in arch_marker_names()
        register_kv_strategy(_SlidingWindowStrategy())
        assert "sliding_window" in arch_marker_names()
        assert arch_field("sliding_window").hf_keys == ("sliding_window",)

    def test_extraction_picks_it_up_with_no_core_edit(self, clean_registry):
        cfg = {"num_hidden_layers": 26, "num_key_value_heads": 4, "head_dim": 256, "sliding_window": 4096, "model_type": "gemma3_text"}
        assert "sliding_window" not in extract_model_info(cfg)
        register_kv_strategy(_SlidingWindowStrategy())
        assert extract_model_info(cfg)["sliding_window"] == 4096
        assert extract_arch_fields(cfg, {})["sliding_window"] == 4096

    def test_the_estimator_uses_it(self, clean_registry):
        register_kv_strategy(_SlidingWindowStrategy())
        kwargs = dict(model_vram=10.0, num_layers=26, num_kv_heads=4, head_dim=256, max_model_len=131_072)
        windowed = estimate_vram(**kwargs, arch={"sliding_window": 4096})
        dense = estimate_vram(**kwargs)

        assert windowed.kv_arch == "sliding"
        assert windowed.kv_arch_label == "Sliding-window attention"
        # The window, not max_model_len, sets the size: 32x shorter here.
        assert windowed.kv_cache_total_gb == pytest.approx(dense.kv_cache_total_gb / 32)

    def test_a_nonlinear_cache_is_not_extrapolated(self, clean_registry):
        """The budget inversion is the strategy's, so it can refuse to be linear."""
        register_kv_strategy(_SlidingWindowStrategy())
        est = estimate_vram(
            total_gpu_memory_gb=61.3,
            model_vram=10.0,
            num_layers=26,
            num_kv_heads=4,
            head_dim=256,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            arch={"sliding_window": 4096},
        )
        # Ample budget, but the cache cannot hold more than one window.
        assert est.max_context_tokens == 4096
        # A cache that is not linear in length reports no per-token figure
        # rather than one that would be wrong if multiplied out.
        assert est.kv_cache_per_token_bytes is None

    def test_recipe_metadata_round_trips_it(self, clean_registry, monkeypatch, tmp_path):
        """Read, HF fill-in, estimator hand-off and write-back are all sweeps.

        The old code hand-listed each field at all four sites; this asserts the
        loop covers a field none of that code has ever heard of.
        """
        register_kv_strategy(_SlidingWindowStrategy())
        cfg = {
            "model_type": "gemma3_text",
            "torch_dtype": "bfloat16",
            "num_hidden_layers": 26,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "sliding_window": 4096,
        }
        monkeypatch.setattr("sparkrun.models.vram.fetch_model_config", lambda *a, **k: cfg)
        monkeypatch.setattr("sparkrun.models.vram.fetch_safetensors_size", lambda *a, **k: None)
        monkeypatch.setattr("sparkrun.models.vram.fetch_safetensors_params", lambda *a, **k: 4_000_000_000)

        recipe = Recipe.from_dict(
            {"name": "T", "model": "google/gemma-3-4b-it", "defaults": {"max_model_len": 131_072}},
        )
        est = recipe.estimate_vram(auto_detect=True)
        assert est.kv_arch == "sliding"
        # Written back, so a second estimate needs no refetch...
        assert recipe.metadata["sliding_window"] == 4096
        # ...and reaches the same verdict from metadata alone.
        assert recipe.estimate_vram(auto_detect=False).kv_arch == "sliding"

    def test_validation_covers_it_too(self, clean_registry):
        recipe = Recipe.from_dict(
            {"name": "T", "model": "m", "container": "c", "runtime": "vllm-distributed", "metadata": {"sliding_window": -1}},
        )
        assert not any("sliding_window" in i for i in recipe.validate())
        register_kv_strategy(_SlidingWindowStrategy())
        assert any("metadata.sliding_window -1 must be a positive integer" in i for i in recipe.validate())
