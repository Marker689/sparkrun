"""Pin an exported recipe's container image and model revision."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sparkrun.api._errors import SparkrunError
from sparkrun.api._models import RunOptions, RunPlan

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe


@dataclass(frozen=True)
class Pin:
    """One reference that was pinned: *field* went from *before* to *after*, read from *source*."""

    field: str
    before: str
    after: str
    source: str


@dataclass(frozen=True)
class PinScope:
    """What a pin is resolved against.

    :attr:`hosts` are the machines an offline pin reads (every one must hold
    the same build); an online pin asks the registry and the Hub and reads
    none of them. :attr:`default_image` is written in, pinned, when the recipe
    leaves its container to the platform.
    """

    recipe: "Recipe"
    hosts: tuple[str, ...]
    ssh_kwargs: dict[str, Any]
    default_image: str | None = None
    cache_dir: str | None = None
    local_cache_dir: str | None = None


def _ssh_kwargs(user: str | None, config) -> dict:
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    kwargs = build_ssh_kwargs(config)
    if user:
        kwargs = {**kwargs, "ssh_user": user}
    return kwargs


def _pull_ref(image: str, recipe, config, v) -> str:
    """The registry ref the launch would pull for *image*, through the recipe's builder."""
    if not recipe.builder:
        return image
    from sparkrun.core.bootstrap import get_builder

    try:
        builder = get_builder(recipe.builder, v)
    except Exception as error:
        raise SparkrunError("cannot pin %s: builder %r is unavailable (%s)" % (image, recipe.builder, error)) from error
    ref = builder.pull_ref(image, recipe, config)
    if ref is None:
        raise SparkrunError(
            "cannot pin %s: the %r builder runs a locally built image, which has no registry digest. "
            "Use --no-pin, or publish the image and name it in the recipe" % (image, recipe.builder)
        )
    return ref


def _pin_image(image: str, scope: PinScope, *, offline: bool, config, v) -> tuple[str, str]:
    from sparkrun.containers.digest import DigestResolutionError, is_pinned, pinned_ref, resolve_host_digest, resolve_registry_digest

    if is_pinned(image):
        return image, "already pinned"
    ref = _pull_ref(image, scope.recipe, config, v)
    if is_pinned(ref):
        return ref, "already pinned"
    try:
        if offline:
            digest = resolve_host_digest(ref, list(scope.hosts), scope.ssh_kwargs)
            return pinned_ref(ref, digest), "hosts"
        return pinned_ref(ref, resolve_registry_digest(ref)), "registry"
    except DigestResolutionError as error:
        raise SparkrunError("cannot pin %s: %s" % (ref, error)) from error


def _pin_revision(model: str, revision: str | None, scope: PinScope, *, offline: bool, config) -> tuple[str, str]:
    from sparkrun.models.revision import (
        RevisionResolutionError,
        resolve_cached_commit,
        resolve_hub_commit,
        resolve_local_cached_commit,
    )

    try:
        if not offline:
            return resolve_hub_commit(model, revision, config=config), "hub"
        try:
            return (
                resolve_cached_commit(model, revision, list(scope.hosts), ssh_kwargs=scope.ssh_kwargs, cache_dir=scope.cache_dir),
                "hosts",
            )
        except RevisionResolutionError as error:
            # Hosts that lack the model entirely get it from this machine in the
            # push transfer modes, so this machine's cache is an existing
            # resource too. A *disagreement* between hosts is never papered over.
            local = resolve_local_cached_commit(model, revision, scope.local_cache_dir) if error.missing else None
            if local is None:
                raise
            return local, "control machine cache"
    except RevisionResolutionError as error:
        raise SparkrunError("cannot pin %s: %s" % (model, error)) from error


def _set_after(data: dict[str, Any], key: str, value: Any, *, after: str) -> None:
    """Set *key*, placing it right after *after* when it is new (export key order is part of readability)."""
    if key in data or after not in data:
        data[key] = value
        return
    items = list(data.items())
    data.clear()
    for existing, existing_value in items:
        data[existing] = existing_value
        if existing == after:
            data[key] = value


def pin_realized_recipe(data: dict[str, Any], run_plan: RunPlan, options: RunOptions, *, offline: bool, sctx) -> tuple[Pin, ...]:
    """Pin a realized recipe (see :func:`pin_recipe_data`) against the hosts *run_plan* placed it on."""
    recipe = run_plan.recipe
    default_image = None
    if not data.get("container"):
        default_image = run_plan.runtime.resolve_container(
            recipe, host_hardware=run_plan.host_hardware.get(run_plan.host_list[0]) if run_plan.host_list else None
        )
    scope = PinScope(
        recipe=recipe,
        hosts=tuple(run_plan.host_list),
        ssh_kwargs=_ssh_kwargs(run_plan.cluster.user, sctx.config),
        default_image=default_image,
        cache_dir=options.cache_dir,
        local_cache_dir=options.local_cache_dir,
    )
    return pin_recipe_data(data, scope, offline=offline, sctx=sctx)


def pin_recipe_data(data: dict[str, Any], scope: PinScope, *, offline: bool, sctx) -> tuple[Pin, ...]:
    """Rewrite *data* (an exported recipe mapping) so its image and model revision are immutable.

    The image goes through the recipe's builder first (:meth:`BuilderPlugin.pull_ref`),
    so an eugr ``:latest`` sentinel pins the GHCR nightly the launch would
    actually pull. A container the recipe leaves to the platform default
    (:attr:`PinScope.default_image`) is written in, pinned. References that are
    already immutable are left as they are. Raises :class:`SparkrunError` for
    anything that cannot be pinned: a pin that quietly kept a mutable
    reference would be worse than no pin.
    """
    from sparkrun.models.revision import is_commit, is_hub_model

    config, v = sctx.config, sctx.variables
    pins: list[Pin] = []

    image = data.get("container") or scope.default_image
    if image:
        pinned, source = _pin_image(image, scope, offline=offline, config=config, v=v)
        if pinned != image or "container" not in data:
            data["container"] = pinned
            pins.append(Pin("container", image, pinned, source))
    for index, entry in enumerate(data.get("containers") or []):
        before = entry.get("image")
        if before:
            pinned, source = _pin_image(before, scope, offline=offline, config=config, v=v)
            if pinned != before:
                entry["image"] = pinned
                pins.append(Pin("containers[%d].image" % index, before, pinned, source))

    model, revision = data.get("model") or "", data.get("model_revision")
    if is_hub_model(model) and not is_commit(revision):
        commit, source = _pin_revision(model, revision, scope, offline=offline, config=config)
        _set_after(data, "model_revision", commit, after="model")
        pins.append(Pin("model_revision", revision or "main", commit, source))

    metadata = data.setdefault("metadata", {})
    metadata["pinned"] = {
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "offline" if offline else "online",
        "from": {pin.field: pin.before for pin in pins},
    }
    return tuple(pins)


def unpinned_references(data: dict[str, Any]) -> list[str]:
    """References a pin does not cover, for the caller to report."""
    notes = []
    spec = (data.get("defaults") or {}).get("speculative_config")
    if not isinstance(spec, dict):
        return notes
    draft = spec.get("model")
    if isinstance(draft, str) and draft and draft != data.get("model") and not spec.get("revision"):
        notes.append("speculative_config.model %s (draft model revision is not pinned)" % draft)
    return notes
