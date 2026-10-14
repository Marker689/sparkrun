"""Native serving dialects of a recipe's runtime, resolved without planning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sparkrun.api._models import RecipeProtocols

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.base import RuntimePlugin

#: What every sparkrun runtime serves; the answer when a runtime claims nothing.
DEFAULT_PROTOCOL = "openai"


def recipe_protocols(recipe: "Recipe", *, sctx: "SparkrunContext | None" = None) -> RecipeProtocols:
    """Return the dialects and capabilities *recipe*'s runtime serves natively.

    Resolves only the runtime plugin: no planning, no SSH, no Hub lookups.
    A gateway uses it to pick the upstream wire format for a binding before
    (or without) launching it.

    Raises:
        SparkrunError: The recipe's runtime is not registered.
    """
    from sparkrun.api._resolve import resolve_runtime

    return runtime_protocols(resolve_runtime(recipe, sctx=sctx), recipe)


def runtime_protocols(runtime: "RuntimePlugin", recipe: "Recipe") -> RecipeProtocols:
    """:func:`recipe_protocols` for an already-resolved runtime (internal)."""
    protocols = tuple(runtime.native_protocols(recipe) or ()) or (DEFAULT_PROTOCOL,)
    return RecipeProtocols(protocols=protocols, capabilities=tuple(runtime.native_capabilities(recipe) or ()))
