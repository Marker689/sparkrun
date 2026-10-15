"""Hybrid attention + recurrent (Mamba2 / GatedDeltaNet) KV cache sizing.

NemotronH and Qwen3-Next-style models interleave a few full-attention layers
with many recurrent layers. Only the attention layers keep a per-token KV cache.
A recurrent layer keeps a fixed-size state per *sequence*, however long the
context. The dense formula multiplies across every layer, which overestimates
these models several times over (Nemotron-3-Super: 88 layers, 8 attention, ~11x).

Configs spell the layer layout three ways, and all three are read:

- ``layers_block_type``: a list (``"mamba"`` / ``"moe"`` / ``"attention"``),
  the NemotronH Puzzle / Omni spelling.
- ``hybrid_override_pattern``: a string, one character per layer (``M`` mamba,
  ``E`` moe, ``-`` mlp, ``*`` attention), the NemotronH spelling.
- ``layer_types``: a list (``"full_attention"`` / ``"linear_attention"``), the
  Qwen3-Next / Qwen3.5 spelling.

A layout with no recurrent layer is not hybrid and is left to the next strategy.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from sparkrun.models.dtypes import kv_bytes_per_element
from sparkrun.models.kv._base import ArchField, ArchInfo, KVCacheStrategy, KVDetection, KVSizing

__all__ = ["HYBRID_ARCH_FIELDS", "HybridKVStrategy", "layer_layout"]

_ATTENTION = frozenset({"attention", "full_attention", "*"})
_RECURRENT = frozenset({"mamba", "linear_attention", "M"})

#: Element width for the recurrent state when the config does not name one.
#: vLLM's ``mamba_ssm_cache_dtype`` defaults to the model dtype, but several
#: configs pin float32. Counting fp32 overstates a bf16 state by a few MB per
#: sequence, which refuses a borderline placement rather than OOMing it.
_STATE_BYTES_DEFAULT = 4.0
#: Element width for the convolution state, which follows the model dtype.
_CONV_BYTES = 2.0

HYBRID_ARCH_FIELDS: tuple[ArchField, ...] = (
    ArchField(
        "attention_layers",
        positive=False,
        doc="Layers keeping a per-token KV cache in a hybrid attention/recurrent model. Switches KV sizing to the hybrid path.",
    ),
    ArchField(
        "recurrent_state_bytes",
        doc="Recurrent (Mamba2 / GatedDeltaNet) state per sequence, in bytes, summed over the recurrent layers.",
    ),
)


def layer_layout(cfg: Mapping[str, Any]) -> list[str] | None:
    """The per-layer type list, from whichever spelling the config uses."""
    pattern = cfg.get("hybrid_override_pattern")
    if isinstance(pattern, str) and pattern:
        return list(pattern)
    for key in ("layers_block_type", "layer_types"):
        value = cfg.get(key)
        if isinstance(value, str):
            # Some exports stringify the list.
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                continue
        if isinstance(value, Sequence) and not isinstance(value, str) and value and all(isinstance(v, str) for v in value):
            return list(value)
    return None


def _positive(cfg: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _state_bytes_per_element(cfg: Mapping[str, Any]) -> float:
    return kv_bytes_per_element(str(cfg.get("mamba_ssm_cache_dtype") or "")) or _STATE_BYTES_DEFAULT


def _mamba2_layer_bytes(cfg: Mapping[str, Any]) -> float | None:
    """One Mamba2 layer's conv + SSM state, as vLLM's ``mamba2_state_shape``."""
    heads = _positive(cfg, "mamba_num_heads")
    head_dim = _positive(cfg, "mamba_head_dim")
    state = _positive(cfg, "ssm_state_size", "state_size")
    kernel = _positive(cfg, "conv_kernel", "mamba_d_conv")
    if not (heads and head_dim and state and kernel):
        return None
    groups = _positive(cfg, "n_groups", "mamba_n_groups") or 1
    conv = (kernel - 1) * (heads * head_dim + 2 * groups * state) * _CONV_BYTES
    ssm = heads * head_dim * state * _state_bytes_per_element(cfg)
    return conv + ssm


def _gated_delta_net_layer_bytes(cfg: Mapping[str, Any]) -> float | None:
    """One GatedDeltaNet layer's conv + recurrent state, as vLLM's ``gated_delta_net_state_shape``."""
    k_heads = _positive(cfg, "linear_num_key_heads")
    v_heads = _positive(cfg, "linear_num_value_heads")
    k_dim = _positive(cfg, "linear_key_head_dim")
    v_dim = _positive(cfg, "linear_value_head_dim")
    kernel = _positive(cfg, "linear_conv_kernel_dim")
    if not (k_heads and v_heads and k_dim and v_dim and kernel):
        return None
    conv = (kernel - 1) * (2 * k_heads * k_dim + v_heads * v_dim) * _CONV_BYTES
    temporal = v_heads * k_dim * v_dim * _state_bytes_per_element(cfg)
    return conv + temporal


class HybridKVStrategy(KVCacheStrategy):
    """KV for the attention layers only, plus a per-sequence recurrent state."""

    name: ClassVar[str] = "hybrid"
    label: ClassVar[str | None] = "Hybrid attention/recurrent (KV on attention layers only, plus per-sequence state)"
    # After MLA, which refuses hybrid linear/MLA layouts rather than guess.
    priority: ClassVar[int] = 50
    replicates_kv: ClassVar[bool] = False

    def arch_fields(self) -> tuple[ArchField, ...]:
        return HYBRID_ARCH_FIELDS

    def extract(self, cfg: Mapping[str, Any], core: Mapping[str, Any]) -> dict[str, Any]:
        layout = layer_layout(cfg)
        if not layout:
            return {}
        recurrent = sum(1 for t in layout if t in _RECURRENT)
        if not recurrent:
            return {}
        found: dict[str, Any] = {"attention_layers": sum(1 for t in layout if t in _ATTENTION)}
        per_layer = _mamba2_layer_bytes(cfg) or _gated_delta_net_layer_bytes(cfg)
        if per_layer:
            found["recurrent_state_bytes"] = int(per_layer * recurrent)
        return found

    def detect(self, arch: ArchInfo) -> KVDetection | None:
        if arch.get("attention_layers") is None:
            return None
        return KVDetection(source="architecture")

    def size(self, arch: ArchInfo, *, max_model_len: int | None) -> KVSizing:
        layers = arch.get("attention_layers")
        if not (arch.num_kv_heads and arch.head_dim):
            missing = [n for n, v in (("num_kv_heads", arch.num_kv_heads), ("head_dim", arch.head_dim)) if not v]
            return KVSizing(
                total_bytes=None,
                unsizable_reason="Missing architecture info (%s); KV cache estimate unavailable" % ", ".join(missing),
            )
        bpe = kv_bytes_per_element(arch.kv_dtype or "")
        if bpe is None:
            return KVSizing(total_bytes=None, unsizable_reason="Unknown KV cache dtype %r" % arch.kv_dtype)

        per_token = 2.0 * layers * arch.num_kv_heads * arch.head_dim * bpe
        state = arch.get("recurrent_state_bytes")
        return KVSizing(
            total_bytes=per_token * max_model_len + (state or 0) if max_model_len else None,
            per_token_bytes=per_token,
            # Without the recurrent state the figure omits a per-sequence cache.
            is_floor=state is None,
        )

    def tokens_for_budget(self, arch: ArchInfo, sizing: KVSizing, budget_bytes: float) -> int | None:
        """Tokens left after one sequence's recurrent state, which does not grow with length."""
        per_token = sizing.per_token_bytes
        if not per_token or per_token <= 0:
            return None
        return max(int((budget_bytes - (arch.get("recurrent_state_bytes") or 0)) / per_token), 0)
