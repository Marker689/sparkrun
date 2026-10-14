"""The one definition of how a launch unit's container environment is layered.

The solo launch (``RuntimePlugin._run_solo``), the cluster launch
(``ClusterContext.build``) and launch materialization (``api.materialize``)
all compose a container environment from the same tiers. They used to write
that merge three times, in three different orders, so ``api.materialize``
could report an environment the launch never set. Every caller now goes
through :func:`compose_launch_env`.
"""

from __future__ import annotations

from collections.abc import Mapping

from sparkrun.utils import merge_env


def compose_launch_env(
    *,
    comm: Mapping[str, str] | None = None,
    runtime_cache: Mapping[str, str] | None = None,
    runtime_common: Mapping[str, str] | None = None,
    runtime_mode: Mapping[str, str] | None = None,
    platform: Mapping[str, str] | None = None,
    recipe: Mapping[str, str] | None = None,
    templates: Mapping[str, str] | None = None,
    runtime_extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge the container env tiers, lowest first (later tiers win).

    1. *comm* — per-host communication env (NCCL / GLOO / ``NODE_IP``). The
       launch paths leave this ``None``: the executor layers ``nccl_env``
       beneath the container env itself, and this order mirrors that.
    2. *runtime_cache* — ``XDG_CACHE_HOME`` plus the runtime's cache paths
       under ``/cache/runtime``. Low so anything else may repoint them.
    3. *runtime_common* then *runtime_mode* — ``get_common_env()`` and the
       solo or cluster env: the runtime's hardware-agnostic defaults.
    4. *platform* — hardware defaults from ``default_env`` (e.g.
       ``PYTORCH_CUDA_ALLOC_CONF`` on GB10). Above the runtime's generic
       defaults (the more specific fact wins) but the lowest tier a user
       could have written, so cluster and recipe env always beat it.
    5. *recipe* — cluster ``env`` / ``env_file`` under recipe ``env`` (already
       carrying ``-e`` overrides), followed by *templates*, the rendered
       values of the recipe's templated keys.
    6. *runtime_extra* — ``get_extra_env()``, last. It carries
       ``HF_HOME`` / ``HF_HUB_CACHE``, which must beat the ``XDG_CACHE_HOME``
       catch-all or the model cache relocates off its own mount.
    """
    return merge_env(
        dict(comm or {}),
        dict(runtime_cache or {}),
        dict(runtime_common or {}),
        dict(runtime_mode or {}),
        dict(platform or {}),
        dict(recipe or {}),
        dict(templates or {}),
        dict(runtime_extra or {}),
    )
