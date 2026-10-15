"""Hybrid attention + recurrent KV sizing (NemotronH, Qwen3-Next / Qwen3.5)."""

from __future__ import annotations

import pytest

from sparkrun.models.kv import ArchInfo, resolve_kv_strategy
from sparkrun.models.kv.hybrid import HybridKVStrategy, layer_layout
from sparkrun.models.vram import estimate_vram, extract_model_info

# NemotronH Mamba2 dimensions shared by Nemotron-3-Super and Nemotron-3.5-Bolt.
_MAMBA = {"mamba_num_heads": 128, "mamba_head_dim": 64, "ssm_state_size": 128, "n_groups": 8, "conv_kernel": 4}
# One Mamba2 layer: conv (4-1) * (128*64 + 2*8*128) * 2 B + ssm 128*64*128 * 4 B.
_MAMBA_LAYER_BYTES = 3 * (8192 + 2048) * 2 + 128 * 64 * 128 * 4


def _bolt_config():
    """The shape of nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB: text config under
    ``llm_config``, depth stated only as ``layers_block_type``."""
    layout = (["mamba", "moe"] * 5 + ["attention"]) * 8
    return {
        "model_type": "nemotron_h_omni",
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "llm_config": {
            "model_type": "nemotron_h_puzzle",
            "dtype": "bfloat16",
            "layers_block_type": layout,
            "num_key_value_heads": 2,
            "num_attention_heads": 32,
            "head_dim": 128,
            "hidden_size": 4096,
            "max_position_embeddings": 262144,
            "mamba_ssm_cache_dtype": "float32",
            **_MAMBA,
        },
    }


class TestLayout:
    @pytest.mark.parametrize(
        "cfg,expected",
        [
            pytest.param({"hybrid_override_pattern": "MEM*E"}, ["M", "E", "M", "*", "E"], id="pattern"),
            pytest.param({"layers_block_type": ["mamba", "attention"]}, ["mamba", "attention"], id="block-type"),
            pytest.param({"layers_block_type": "['mamba', 'attention']"}, ["mamba", "attention"], id="stringified"),
            pytest.param({"layer_types": ["linear_attention", "full_attention"]}, ["linear_attention", "full_attention"], id="types"),
            pytest.param({}, None, id="absent"),
        ],
    )
    def test_every_spelling_is_read(self, cfg, expected):
        assert layer_layout(cfg) == expected

    def test_a_layout_without_recurrent_layers_is_not_hybrid(self):
        # gpt-oss / Gemma: sliding + full attention, no recurrent state.
        assert HybridKVStrategy().extract({"layer_types": ["sliding_attention", "full_attention"]}, {}) == {}


class TestExtraction:
    def test_bolt_config_is_fully_sized(self):
        info = extract_model_info(_bolt_config())
        assert info["num_layers"] == 88  # derived from the layout list
        assert info["attention_layers"] == 8
        assert info["recurrent_state_bytes"] == 40 * _MAMBA_LAYER_BYTES
        assert (info["num_kv_heads"], info["head_dim"]) == (2, 128)

    def test_nemotron_pattern(self):
        cfg = {"num_hidden_layers": 6, "hybrid_override_pattern": "MEM*EM", **_MAMBA}
        found = HybridKVStrategy().extract(cfg, {})
        assert found == {"attention_layers": 1, "recurrent_state_bytes": 3 * _MAMBA_LAYER_BYTES}

    def test_gated_delta_net_state(self):
        cfg = {
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_cache_dtype": "bfloat16",
        }
        per_layer = 3 * (2 * 16 * 128 + 32 * 128) * 2 + 32 * 128 * 128 * 2
        assert HybridKVStrategy().extract(cfg, {}) == {"attention_layers": 1, "recurrent_state_bytes": 3 * per_layer}

    def test_unknown_recurrent_dims_still_count_attention_layers(self):
        assert HybridKVStrategy().extract({"hybrid_override_pattern": "MM*"}, {}) == {"attention_layers": 1}


class TestSizing:
    def test_mla_still_claims_first(self):
        strategy, _ = resolve_kv_strategy(ArchInfo(extra={"kv_lora_rank": 512, "attention_layers": 4}))
        assert strategy.name == "mla"

    def test_only_attention_layers_cache_per_token(self):
        kwargs = dict(model_vram=10.0, num_layers=88, num_kv_heads=2, head_dim=128, kv_dtype="fp8", max_model_len=262_144)
        hybrid = estimate_vram(**kwargs, arch={"attention_layers": 8, "recurrent_state_bytes": 1024**3})
        dense = estimate_vram(**kwargs)
        assert hybrid.kv_arch == "hybrid"
        assert hybrid.kv_cache_per_token_bytes == pytest.approx(dense.kv_cache_per_token_bytes * 8 / 88)
        assert hybrid.kv_cache_total_gb == pytest.approx(2 * 8 * 2 * 128 * 262_144 / 1024**3 + 1.0)
        assert not hybrid.kv_estimate_is_floor

    def test_missing_state_marks_the_estimate_a_floor(self):
        est = estimate_vram(model_vram=10.0, num_layers=4, num_kv_heads=2, head_dim=128, max_model_len=1024, arch={"attention_layers": 1})
        assert est.kv_estimate_is_floor

    def test_budget_inversion_subtracts_the_state_once(self):
        est = estimate_vram(
            model_vram=10.0,
            num_layers=88,
            num_kv_heads=2,
            head_dim=128,
            kv_dtype="fp8",
            max_model_len=262_144,
            total_gpu_memory_gb=20.0,
            gpu_memory_utilization=1.0,
            arch={"attention_layers": 8, "recurrent_state_bytes": 1024**3},
        )
        per_token = 2 * 8 * 2 * 128
        assert est.max_context_tokens == int((10.0 - 1.0) * 1024**3 / per_token)
