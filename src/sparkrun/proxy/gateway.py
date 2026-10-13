"""Gateway selection — which implementation backs ``sparkrun proxy``.

A *gateway* is the process that fronts every discovered inference endpoint
behind one OpenAI-compatible API.  Core ships one implementation,
:class:`~sparkrun.plugins.litellm.engine.ProxyEngine` (LiteLLM); others register here.

Implementations become resolvable via :func:`register_gateway`, which is what
lets one live outside the ``sparkrun.proxy`` tree entirely.  The registry is
in-process rather than a SAF extension point because an engine is *constructed
with arguments* (host, port, master key, state dir) rather than resolved as a
stateless singleton — the same reason ``platforms`` and ``models.kv`` stayed
in-process.  Registration carries a **loader** rather than the class, so
a plugin engine can import this module without a cycle and a registration
never drags in the implementation it names.

Explicitly enabling a gateway feature runs its plugin's lifecycle hook and
persists ``proxy.gateway`` in ``proxy.yaml``. Installation and plugin loading
changes preserve that selection and the other providers' capability settings.

Two mechanisms, deliberately separate:

- **Availability** — each gateway declares a feature flag
  (``gateway.<name>``, see :mod:`sparkrun.core.features`).  ``gateway.litellm``
  defaults on for every channel. Separately installed plugins declare their
  own availability; enabling a flag does not install an implementation.
- **Selection** — exactly one gateway is used at a time.  That is arbitrated
  *here*, at resolution: an explicit name (``proxy.gateway`` in ``proxy.yaml``,
  or a caller argument) must be known and enabled; with no name, the default
  wins when enabled, else the single remaining enabled gateway, else the caller
  is told to name one.  The flag registry has no notion of mutually-exclusive
  flags, so nothing stops a user enabling two — resolution refuses to guess
  instead.  This mirrors ``_default_executor_name`` in
  :mod:`sparkrun.orchestration.executor`.
"""

from __future__ import annotations

from sparkrun.core.registration import enlist_registry_state

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from sparkrun.core.config import SparkrunConfig

logger = logging.getLogger(__name__)

#: Gateway used when nothing names one.
DEFAULT_GATEWAY = "litellm"

#: Known gateway name -> the feature flag gating its availability.
#:
#: Maintained by :func:`register_gateway`; read by availability resolution.
GATEWAY_FEATURE_FLAGS: dict[str, str] = {}

enlist_registry_state(globals(), "GATEWAY_FEATURE_FLAGS")

#: Known gateway name -> zero-arg callable returning its engine class.
_GATEWAY_LOADERS: dict[str, "Callable[[], type]"] = {}
_GATEWAY_OWNERS: dict[str, object] = {}

enlist_registry_state(globals(), "_GATEWAY_LOADERS", "_GATEWAY_OWNERS")


class GatewayError(RuntimeError):
    """Base class for gateway selection failures."""


class GatewayUnavailableError(GatewayError):
    """A gateway was requested (or needed) but is unknown or disabled.

    :attr:`gateway` names what was asked for (``None`` when the failure is
    "nothing is enabled"); :attr:`available` lists the enabled gateways.
    """

    def __init__(self, message: str, *, gateway: str | None = None, available: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.gateway = gateway
        self.available = available


class AmbiguousGatewayError(GatewayError):
    """Several gateways are enabled and none was named.

    :attr:`available` carries the candidates.
    """

    def __init__(self, message: str, *, available: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.available = available


def register_gateway(name: str, *, feature_flag: str, loader: "Callable[[], type]") -> None:
    """Make gateway *name* resolvable.

    Args:
        name: Selector used by ``proxy.gateway`` in ``proxy.yaml``.
        feature_flag: Flag gating availability; resolution refuses a gateway
            whose flag is off rather than silently falling back.
        loader: Zero-arg callable returning the engine class.  Deferred so a
            registration can name a class this module must not import at
            module scope, and so registering costs nothing at import time.

    Repeated registration of the same provider is idempotent. Distinct
    providers must use distinct names, rather than depending on import order.
    """
    existing = _GATEWAY_LOADERS.get(name)
    if existing is not None:

        def provider(value):
            return (getattr(value, "__module__", None), getattr(value, "__qualname__", None))

        same_provider = existing is loader or provider(existing) == provider(loader) != (None, None)
        if not same_provider or GATEWAY_FEATURE_FLAGS.get(name) != feature_flag:
            from sparkrun.core.installed_plugins import PluginConflictError

            raise PluginConflictError("Gateway %r is claimed by both %r and %r" % (name, provider(existing), provider(loader)))
        return
    GATEWAY_FEATURE_FLAGS[name] = feature_flag
    _GATEWAY_LOADERS[name] = loader
    from sparkrun.core.plugin_identity import current_registration_owner

    if current_registration_owner() is not None:
        _GATEWAY_OWNERS[name] = current_registration_owner()


def gateway_class(name: str) -> type:
    """Return the engine class implementing gateway *name*.

    The one place a gateway name becomes an implementation.

    Raises:
        GatewayUnavailableError: No implementation is registered.  Deliberately
            distinct from "disabled": a name can be known to the flag registry
            while its plugin failed to load, and telling someone to enable a
            flag that is already on is a dead end.
    """
    from sparkrun.core.in_tree_plugins import plugin_application_profile_failure

    failure = plugin_application_profile_failure(name)
    if failure:
        raise GatewayUnavailableError(failure, gateway=name)
    loader = _GATEWAY_LOADERS.get(name)
    if loader is None:
        raise GatewayUnavailableError(
            "No implementation registered for gateway %r" % name,
            gateway=name,
            available=tuple(sorted(_GATEWAY_LOADERS)),
        )
    return loader()


def gateway_feature_flag(name: str) -> str | None:
    """Return the feature flag gating gateway *name*, or ``None`` if unknown."""
    return GATEWAY_FEATURE_FLAGS.get(name)


def is_gateway_enabled(name: str, *, config: "SparkrunConfig | None" = None) -> bool:
    """True when gateway *name* is known and its feature flag resolves on.

    Args:
        name: Gateway name (e.g. ``"litellm"``).
        config: Optional config to bind the flag resolution to.  When
            ``None`` the standard config path is read — the auto-discover
            daemon and other ``sctx``-less callers land here.
    """
    from sparkrun.core.features import is_feature_enabled

    if not is_feature_enabled("gateway", config=config):
        return False
    flag = gateway_feature_flag(name)
    if flag is None:
        return False
    if config is not None:
        return config.is_feature_enabled(flag)

    from sparkrun.core.features import feature_gate_enabled

    return feature_gate_enabled(flag)


def list_gateways(*, config: "SparkrunConfig | None" = None) -> list[str]:
    """Return the enabled gateway names, sorted."""
    return sorted(name for name in GATEWAY_FEATURE_FLAGS if is_gateway_enabled(name, config=config))


def resolve_gateway(name: str | None = None, *, config: "SparkrunConfig | None" = None) -> str:
    """Resolve which gateway to use.

    Args:
        name: Explicitly requested gateway (from ``proxy.gateway`` in
            ``proxy.yaml`` or a caller argument).  ``None`` / empty means
            "pick the default".
        config: Optional config binding for flag resolution.

    Returns:
        The resolved gateway name.

    Raises:
        GatewayUnavailableError: *name* is unknown or its flag is off, or no
            gateway is enabled at all.
        AmbiguousGatewayError: no name was given, the default is disabled, and
            more than one other gateway is enabled.
    """
    from sparkrun.core.features import is_feature_enabled

    if not is_feature_enabled("gateway", config=config):
        raise GatewayUnavailableError("Gateway support is disabled. Enable it with: sparkrun setup features enable gateway", gateway=name)
    enabled = list_gateways(config=config)

    if name:
        from sparkrun.core.in_tree_plugins import plugin_application_profile_failure

        failure = plugin_application_profile_failure(name)
        if failure:
            raise GatewayUnavailableError(failure, gateway=name)
        if is_gateway_enabled(name, config=config):
            return name
        raise GatewayUnavailableError(
            _unavailable_message(name, enabled),
            gateway=name,
            available=tuple(enabled),
        )

    if DEFAULT_GATEWAY in enabled:
        return DEFAULT_GATEWAY

    if not enabled:
        raise GatewayUnavailableError(
            "No inference gateway is enabled. Inspect sparkrun plugins list, enable a provider plugin, "
            "then select it with: sparkrun setup features enable gateway.<name>",
            available=(),
        )

    if len(enabled) == 1:
        return enabled[0]

    raise AmbiguousGatewayError(
        "Several gateways are enabled (%s) and the default (%s) is disabled. "
        "Name one with 'gateway: <name>' under 'proxy:' in proxy.yaml." % (", ".join(enabled), DEFAULT_GATEWAY),
        available=tuple(enabled),
    )


def require_gateway_enabled(name: str, *, config: "SparkrunConfig | None" = None) -> None:
    """Raise unless gateway *name* is known and enabled.

    The enforcement point for a gateway implementation's own bring-up (see
    :meth:`sparkrun.plugins.litellm.engine.ProxyEngine.start`).  Deliberately *not*
    applied to teardown, status, or the auto-discover daemon's restart path:
    a proxy that is already running must stay stoppable and manageable even
    after its flag has been turned off — the same rule
    ``cleanup_cluster_transport`` follows for transports.
    """
    from sparkrun.core.in_tree_plugins import plugin_application_profile_failure

    from sparkrun.core.features import is_feature_enabled

    if not is_feature_enabled("gateway", config=config):
        raise GatewayUnavailableError("Gateway support is disabled. Enable it with: sparkrun setup features enable gateway", gateway=name)
    failure = plugin_application_profile_failure(name)
    if failure:
        raise GatewayUnavailableError(failure, gateway=name)
    if is_gateway_enabled(name, config=config):
        return
    enabled = list_gateways(config=config)
    raise GatewayUnavailableError(
        _unavailable_message(name, enabled),
        gateway=name,
        available=tuple(enabled),
    )


def _unavailable_message(name: str, enabled: list[str]) -> str:
    """Build the 'gateway X is not usable' message, with the remedy."""
    flag = gateway_feature_flag(name)
    if flag is None:
        from sparkrun.core.features import DECLARED_FEATURE_OWNERS

        owner = DECLARED_FEATURE_OWNERS.get("gateway." + name)
        if owner is not None:
            return (
                "Gateway %r is unavailable. Inspect sparkrun plugins list; enable its plugin with "
                "sparkrun plugins enable %s, then select it with sparkrun setup features enable gateway.%s" % (name, owner.plugin_id, name)
            )
        known = ", ".join(sorted(GATEWAY_FEATURE_FLAGS)) or "(none)"
        return "Unknown gateway %r. Known gateways: %s" % (name, known)
    detail = "Gateway %r is disabled. Enable it with: sparkrun setup features enable %s" % (name, flag)
    if enabled:
        detail += " (currently enabled: %s)" % ", ".join(enabled)
    return detail


__all__ = [
    "DEFAULT_GATEWAY",
    "GATEWAY_FEATURE_FLAGS",
    "AmbiguousGatewayError",
    "GatewayError",
    "GatewayUnavailableError",
    "gateway_class",
    "gateway_feature_flag",
    "is_gateway_enabled",
    "list_gateways",
    "register_gateway",
    "require_gateway_enabled",
    "resolve_gateway",
]
