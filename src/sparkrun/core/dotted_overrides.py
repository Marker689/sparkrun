"""Dotted ``-o`` keys that set one field inside a mapping-valued default.

``-o speculative_config.method=dflash`` overrides a single field of the
``speculative_config`` mapping and keeps the rest. A dotted key stays in the
overrides dict as typed; it is folded into its root only where the config
chain is built (:meth:`sparkrun.core.recipe.Recipe.build_config_chain`,
:meth:`~sparkrun.core.recipe.Recipe._effective_default`). Folding when ``-o``
is parsed would be too early: matched recipe ``overrides:`` layers are written
into ``defaults`` later (in ``api.plan``), and a merged copy taken before that
would replace the layer's value instead of refining it.

``env.KEY`` is not a path. :func:`sparkrun.core.resolve.apply_recipe_overrides`
routes it to the container env before any of this runs.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from typing import Any

#: Prefixes routed somewhere other than the config chain, never folded as paths.
ROUTED_PREFIXES = ("env.",)


class DottedOverrideError(ValueError):
    """A dotted override cannot be applied to the value it addresses."""


def is_dotted_override(key: str) -> bool:
    """True for a key addressing a field inside a mapping (``root.field``)."""
    if "." not in key or key.startswith("_") or key.startswith(ROUTED_PREFIXES):
        return False
    return all(key.split("."))


def dotted_root(key: str) -> str:
    """The config key a dotted override modifies (``speculative_config``)."""
    return key.split(".", 1)[0]


def _as_mapping(value: Any, what: str) -> dict:
    """Return a mutable copy of *value* as a dict, accepting a JSON-object string."""
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    raise DottedOverrideError("cannot set a field inside %s: its value %r is not a mapping" % (what, value))


def _leaf_value(value: Any) -> Any:
    """A JSON object or array typed as a leaf is structure, not a string."""
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def fold_dotted_overrides(overrides: Mapping[str, Any] | None, lookup: Callable[[str], Any]) -> dict[str, Any]:
    """Return *overrides* with every dotted key merged into its root.

    The base for a root is the override's own whole value when one was given
    (``-o speculative_config={...}``), else ``lookup(root)`` (the lower
    layers of the chain); a root with neither is refused. Missing
    intermediate fields are created. Dotted keys apply in the
    order given, shallowest first, so ``a.b={...}`` then ``a.b.c=1`` refines
    rather than discards. Dotted keys are removed from the result. Raises
    :class:`DottedOverrideError` when a path runs through a non-mapping.
    """
    result = dict(overrides or {})
    dotted = [k for k in result if is_dotted_override(k)]
    if not dotted:
        return result
    roots: dict[str, dict] = {}
    for key in sorted(dotted, key=lambda k: k.count(".")):
        value = result.pop(key)
        root, *path = key.split(".")
        if root not in roots:
            base = result[root] if root in result else lookup(root)
            if base is None:
                # Nothing to refine: a typo'd root, or a scalar flag with no value
                # here. Building a mapping from nothing would render `{"x": 1}`.
                raise DottedOverrideError(
                    "'%s' sets a field of '%s', which has no mapping value to modify; "
                    "set the whole value instead, e.g. -o '%s={...}'" % (key, root, root)
                )
            roots[root] = _as_mapping(base, "'%s'" % root)
        node = roots[root]
        for depth, part in enumerate(path[:-1], start=1):
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            elif not isinstance(child, dict):
                child = node[part] = _as_mapping(child, "'%s'" % ".".join([root, *path[:depth]]))
            node = child
        node[path[-1]] = _leaf_value(value)
    result.update(roots)
    return result
