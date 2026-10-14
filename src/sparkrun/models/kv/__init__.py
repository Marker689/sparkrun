"""KV cache sizing strategies, and the registry that selects one.

Adding an architecture means adding a module here and registering it — nothing
in :mod:`sparkrun.models.vram`, :mod:`sparkrun.core.recipe` or the CLI needs to
learn its name.  See :mod:`sparkrun.models.kv._base` for the contract.

Resolution is **order-sensitive**: strategies are tried by
:attr:`~sparkrun.models.kv._base.KVCacheStrategy.priority`, most specific first,
and the first to claim the model wins.  That is why this is an in-process
ordered registry rather than a SAF extension point — SAF selects plugins by
exact name, which is the wrong question here.  The same reasoning keeps
``platforms/`` in-process.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sparkrun.models.dtypes import dtype_key, kv_bytes_per_element
from sparkrun.models.kv._base import ArchField, ArchInfo, KVCacheStrategy, KVDetection, KVSizing
from sparkrun.models.kv.dense import DenseKVStrategy
from sparkrun.models.kv.mla import MlaKVStrategy

__all__ = [
    "ArchField",
    "ArchInfo",
    "KVCacheStrategy",
    "KVDetection",
    "KVSizing",
    "arch_field",
    "arch_fields",
    "arch_marker_names",
    "extract_arch_fields",
    "is_kv_layout",
    "is_valid_kv_dtype",
    "kv_layout_names",
    "list_kv_strategies",
    "register_kv_strategy",
    "resolve_kv_strategy",
]

_REGISTRY: list[KVCacheStrategy] = [MlaKVStrategy(), DenseKVStrategy()]


def register_kv_strategy(strategy: KVCacheStrategy) -> None:
    """Add a KV sizing strategy, keeping the registry in priority order.

    The escape hatch for out-of-tree plugins (a vendor fork with its own packed
    slot layout), called from a plugin module's ``register(v)`` hook alongside
    :func:`~sparkrun.platforms.register_platform`.  Registering the same
    strategy class under its name again is a no-op; a different strategy under
    a registered name (including a built-in's) raises ``PluginConflictError``.
    """
    from sparkrun.core.registration import claim_registration

    existing = next((s for s in _REGISTRY if s.name == strategy.name), None)
    if claim_registration(existing, strategy, description="KV cache strategy", key=strategy.name, equal=lambda a, b: type(a) is type(b)):
        return
    _REGISTRY.append(strategy)
    _REGISTRY.sort(key=lambda s: s.priority)


def list_kv_strategies() -> tuple[KVCacheStrategy, ...]:
    """Every registered strategy, most specific first."""
    return tuple(sorted(_REGISTRY, key=lambda s: s.priority))


def resolve_kv_strategy(arch: ArchInfo) -> tuple[KVCacheStrategy, KVDetection]:
    """Select the strategy that sizes *arch*.

    Always resolves: :class:`~sparkrun.models.kv.dense.DenseKVStrategy` sits
    last and claims everything.
    """
    for strategy in list_kv_strategies():
        detection = strategy.detect(arch)
        if detection is not None:
            return strategy, detection
    # Unreachable while dense is registered, but a plugin could have replaced it.
    dense = DenseKVStrategy()
    return dense, dense.detect(arch) or KVDetection(source="fallback")


# -- field declarations -----------------------------------------------


def arch_fields() -> tuple[ArchField, ...]:
    """Every architecture parameter any strategy consumes, deduped by name.

    The single source of truth for HuggingFace extraction, recipe ``metadata``
    keys, their validation, and the post-estimate write-back.  Keeping those
    four in step by hand is what made adding one architecture a five-site edit.
    """
    seen: dict[str, ArchField] = {}
    for strategy in list_kv_strategies():
        for f in strategy.arch_fields():
            seen.setdefault(f.name, f)
    return tuple(seen.values())


def arch_field(name: str) -> ArchField | None:
    """Look one declared field up by name."""
    for f in arch_fields():
        if f.name == name:
            return f
    return None


def arch_marker_names() -> frozenset[str]:
    """Names whose presence marks a config as belonging to a specific architecture.

    Used to decide whether to descend into a multimodal wrapper's nested
    ``text_config``: a wrapper can be complete for the universal keys while
    hiding every architecture marker below, and that is indistinguishable from
    an ordinary model unless the nested config is consulted.
    """
    return frozenset(f.name for f in arch_fields())


def extract_arch_fields(cfg: Mapping[str, Any], core: Mapping[str, Any]) -> dict[str, Any]:
    """Run every strategy's extraction over one HuggingFace config dict.

    *core* carries the already-extracted universal keys, which a strategy may
    need for a derivation (MLA resolves its latent width partly from
    ``head_dim``).  Higher-priority strategies win a contested field name.
    """
    found: dict[str, Any] = {}
    for strategy in list_kv_strategies():
        for key, value in strategy.extract(cfg, core).items():
            found.setdefault(key, value)
    return found


# -- dtypes -----------------------------------------------------------


def kv_layout_names() -> frozenset[str]:
    """Packed KV slot layout names claimed by any strategy (e.g. ``fp8_ds_mla``)."""
    names: set[str] = set()
    for strategy in list_kv_strategies():
        names |= strategy.kv_layouts()
    return frozenset(names)


def is_kv_layout(dtype: str) -> bool:
    """Whether *dtype* names a packed slot layout rather than an element width.

    A layout identifies the architecture on its own — ``kv_dtype: fp8_ds_mla``
    says "MLA" without any config markers — which is what makes it one of the
    signals that architecture detection has already been satisfied.
    """
    return dtype_key(dtype) in kv_layout_names()


def is_valid_kv_dtype(dtype: str) -> bool:
    """Whether *dtype* is something a recipe may name as its KV cache dtype.

    Either a dtype with a per-element width, or a packed slot layout some
    strategy claims.  Recipe validation asks this rather than enumerating
    architectures, so a new packed layout does not need a second call site.
    """
    return kv_bytes_per_element(dtype) is not None or dtype_key(dtype) in kv_layout_names()
