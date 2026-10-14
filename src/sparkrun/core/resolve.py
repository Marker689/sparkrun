"""Shared, console-free input-resolution logic for sparkrun.

This module hosts the *pure* resolvers that used to live in
``sparkrun.cli._common`` but are needed by the library API as well.  None
of these helpers call ``click.echo`` / ``sys.exit`` / ``click.prompt``;
failures raise ordinary exceptions (``RecipeError`` subclasses) so both
the CLI (which translates to ``click.echo`` + ``sys.exit``) and the API
(which surfaces typed ``SparkrunError`` subclasses) can build on the same
core.

The CLI keeps thin wrappers in ``cli/_common.py`` that add the
interactive concerns (disambiguation prompts, untrusted-host
confirmation, registry-refresh retry) and translate exceptions into
console output + exit codes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe, RecipeAmbiguousError

logger = logging.getLogger(__name__)

#: ``-o key=value`` keys that configure the *executor* rather than the serve
#: command.  :func:`split_executor_overrides` lifts them out of a launch's
#: overrides into its executor config at the API boundary (``RunOptions``), so
#: every frontend (``run``, ``benchmark``, ``proxy load``, ``export recipe
#: --realize``, library callers) honors them.  Left in the overrides they would
#: be reported as unmapped serve flags and dropped.
EXECUTOR_OVERRIDE_KEYS = frozenset(
    {
        "auto_remove",
        "restart_policy",
        "privileged",
        "gpus",
        "ipc",
        "shm_size",
        "network",
        "user",
        "security_opt",
        "cap_add",
        "ulimit",
        "devices",
        "memory_limit",
        # ``-o entrypoint=''`` clears a consuming image ENTRYPOINT (one that
        # parses sparkrun's appended ``bash -c`` as its own flags) without
        # having to fork a third-party recipe just to add two lines of
        # executor_config.  Empty string is the meaningful value here and
        # survives ``coerce_value``; see ExecutorConfig.entrypoint.
        "entrypoint",
    }
)


def split_executor_overrides(overrides) -> tuple[dict, dict]:
    """Split *overrides* into ``(serve_overrides, executor_overrides)``.

    Keys in :data:`EXECUTOR_OVERRIDE_KEYS` go to the second mapping.  The input
    is not modified.
    """
    serve: dict = {}
    executor: dict = {}
    for key, value in (overrides or {}).items():
        (executor if key in EXECUTOR_OVERRIDE_KEYS else serve)[key] = value
    return serve, executor


def apply_recipe_overrides(
    options,
    tensor_parallel=None,
    pipeline_parallel=None,
    data_parallel=None,
    gpu_mem=None,
    max_model_len=None,
    image=None,
    *,
    recipe: Recipe,
    **kwargs,
) -> tuple[Recipe, dict]:
    """Build the overrides dict, apply it to *recipe*, and resolve the runtime.

    Returns ``(recipe, overrides)`` — *recipe* is returned to make the
    mutation explicit (runtime may change based on overrides).

    The required *recipe* is resolved with ``recipe.resolve(overrides)`` so
    that overrides can influence runtime resolution (e.g.
    ``distributed_executor_backend=ray`` switches vllm-distributed to
    vllm-ray).

    Pure logic: no console I/O.  *options* is a tuple of ``key=value``
    strings (the CLI ``--option/-o`` form); malformed entries raise
    ``ValueError`` (the CLI wrapper validates and reports these before
    calling this function, so they should not reach here in practice).
    """
    from sparkrun.utils import coerce_value

    if recipe is None:
        raise ValueError("Recipe overrides require a recipe")

    overrides: dict = {}
    for opt in options or ():
        if "=" not in opt:
            raise ValueError("--option must be key=value, got: %s" % opt)
        key, _, value = opt.partition("=")
        key = key.strip()
        if not key:
            raise ValueError("--option has empty key: %s" % opt)
        overrides[key] = coerce_value(value.strip())

    if tensor_parallel is not None:
        overrides["tensor_parallel"] = tensor_parallel
    if pipeline_parallel is not None:
        overrides["pipeline_parallel"] = pipeline_parallel
    if data_parallel is not None:
        overrides["data_parallel"] = data_parallel
    if gpu_mem is not None:
        overrides["gpu_memory_utilization"] = gpu_mem
    if max_model_len is not None:
        overrides["max_model_len"] = max_model_len
    if image:
        recipe.container = image
        # A matched `overrides:` container layer must not replace an explicit --image.
        recipe._cli_image = True
        if recipe.containers:
            # An explicit --image is a whole-launch override: anything subtler
            # (override the fallback but keep the machine-specific images) would
            # silently leave most nodes on the image the user just replaced.
            logger.info(
                "--image %s overrides the recipe's per-machine `containers:` block (%d entr%s); every node will run this image.",
                image,
                len(recipe.containers),
                "y" if len(recipe.containers) == 1 else "ies",
            )
            recipe.containers = []

    for k, v in kwargs.items():
        if v is not None:
            overrides[k] = v

    # Apply env.* overrides to recipe.env directly
    for k, v in list(overrides.items()):
        if k.startswith("env."):
            recipe.env[k[4:]] = str(v)
            recipe._cli_env_keys.add(k[4:])
            del overrides[k]

    # Resolve runtime with overrides visible to resolvers
    recipe.resolve(overrides)

    return recipe, overrides


def apply_env_overrides(recipe, env_pairs) -> dict[str, str]:
    """Apply ``KEY=VALUE`` container-env overrides to *recipe*.

    The peer of ``-o env.KEY=VALUE`` (handled inside
    :func:`apply_recipe_overrides`) for the direct ``-e/--env`` form, with one
    deliberate difference: values are taken **verbatim**. The ``-o`` path routes
    through ``coerce_value``, which turns ``-o env.X=true`` into the string
    ``"True"`` — right for a serve flag, wrong for an env var a runtime parses
    itself.

    Only the first ``=`` splits, so ``FOO=a=b`` and
    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` survive intact, and
    ``FOO=`` sets an empty value (the way to blank out a lower-tier default).

    Pure logic: no console I/O. Malformed entries raise ``ValueError``; the CLI
    wrapper translates that into a ``UsageError``.

    Returns the mapping that was applied (for logging / testing).
    """
    applied: dict[str, str] = {}
    for pair in env_pairs or ():
        if "=" not in pair:
            raise ValueError("--env must be KEY=VALUE, got: %s" % pair)
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise ValueError("--env has empty key: %s" % pair)
        applied[key] = value
    if recipe is not None and applied:
        recipe.env.update(applied)
        # Matched `overrides:` env layers must not replace what -e set.
        recipe._cli_env_keys.update(applied)
    return applied


def recipe_name_looks_like_path(name: str) -> bool:
    """Return True when *name* looks like a filesystem path.

    Used to short-circuit registry refresh retries for obvious path inputs,
    since updating remote registries cannot help resolve a missing local file.
    """
    if not name:
        return False
    if name.startswith("@"):  # @registry/recipe is a registry reference, not a path
        return False
    if name.startswith((".", "/", "~")):
        return True
    if name.endswith((".yaml", ".yml")):
        return True
    return False


def load_recipe(
    config: "SparkrunConfig",
    recipe_name: str,
    *,
    resolve: bool = True,
    offline: bool = False,
    allow_untrusted_host: bool = False,
    choose_ambiguous: "Callable[[RecipeAmbiguousError], tuple[Path, str | None]] | None" = None,
    retry_after_update: bool = False,
    on_refresh: "Callable[[str], None] | None" = None,
):
    """Find and load a recipe without any console I/O or exits.

    The one implementation behind both the API and the CLI's interactive
    ``cli/_common.py:_load_recipe`` wrapper, which supplies the prompts through
    the hooks below.  Expands shortcuts, fetches URL-sourced recipes, looks up
    the recipe across configured registries and CWD recipes, and tags it with
    its source registry.

    Args:
        offline: Load from what is already here: a URL recipe comes from the
            local cache only, registries are never refreshed on a miss, and a
            foreign recipe format reads local facts only.
        allow_untrusted_host: Fetch a URL recipe from a host off the allowlist
            (the CLI asks first).
        choose_ambiguous: Called with a :class:`RecipeAmbiguousError` to pick
            one match, returning ``(path, registry_name)``; ``None`` raises.
        retry_after_update: When the lookup misses, refresh the registries
            once and retry (never offline, never for a path-like name).
        on_refresh: Called with the name before that refresh (for a notice).

    Returns ``(recipe, recipe_path, registry_mgr)``.

    Raises:
        RecipeUntrustedHostError: URL host is off the allowlist.
        RecipeAmbiguousError: Name matches multiple registries and no
            *choose_ambiguous* was given.
        RecipeError: Recipe not found or failed to load.
    """
    from sparkrun.core.recipe import (
        Recipe,
        RecipeAmbiguousError,
        RecipeError,
        discover_cwd_recipes,
        expand_recipe_shortcut,
        fetch_and_cache_recipe,
        find_recipe,
        is_recipe_url,
    )

    from sparkrun.core._recipe_source import recipe_registry_entry, tag_recipe_source
    from sparkrun.utils import parse_scoped_name

    recipe_name = expand_recipe_shortcut(recipe_name)

    if is_recipe_url(recipe_name):
        logger.debug("Loading recipe from URL: %s", recipe_name)
        cached_path = fetch_and_cache_recipe(recipe_name, allow_untrusted_host=allow_untrusted_host, offline=offline)
        registry_mgr = config.get_registry_manager()
        registry_mgr.ensure_initialized()
        recipe = Recipe.load(cached_path, resolve=resolve, registry_manager=registry_mgr, allow_local_includes=False, offline=offline)
        recipe.source_path = recipe_name
        # URL-sourced recipes are never auto-trusted (see
        # core.launcher.resolve_recipe_trust): their hooks require --trust
        # or interactive confirmation.
        tag_recipe_source(recipe, None, config=config, external=True)
        return recipe, cached_path, registry_mgr

    registry_mgr = config.get_registry_manager()
    registry_mgr.ensure_initialized()

    selected_registry, _ = parse_scoped_name(recipe_name)
    retried = False
    while True:
        try:
            recipe_path = find_recipe(recipe_name, registry_manager=registry_mgr, local_files=discover_cwd_recipes())
            break
        except RecipeAmbiguousError as err:
            if choose_ambiguous is None:
                raise
            recipe_path, selected_registry = choose_ambiguous(err)
            break
        except RecipeError:
            if retried or not retry_after_update or offline or recipe_name_looks_like_path(recipe_name):
                raise
            retried = True
            if on_refresh is not None:
                on_refresh(recipe_name)
            # A scoped name (@registry/...) refreshes only that registry.
            try:
                registry_mgr.update(selected_registry) if selected_registry else registry_mgr.update()
            except Exception as update_err:
                logger.debug("Registry update failed during retry: %s", update_err)

    recipe = Recipe.load(recipe_path, resolve=resolve, registry_manager=registry_mgr, offline=offline)
    registry = recipe_registry_entry(recipe_path, registry_mgr, registry_name=selected_registry)
    tag_recipe_source(recipe, registry, config=config)
    return recipe, recipe_path, registry_mgr


__all__ = ["EXECUTOR_OVERRIDE_KEYS", "apply_recipe_overrides", "load_recipe", "split_executor_overrides"]
