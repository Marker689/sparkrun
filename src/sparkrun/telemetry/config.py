"""Telemetry runtime configuration."""

from __future__ import annotations

import logging
import os
from uuid import uuid4

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.application_profile import get_application_profile, product_env

logger = logging.getLogger(__name__)

NO_TELEMETRY_ENV = "SPARKRUN_NO_TELEMETRY"
TELEMETRY_ENDPOINT_ENV = "SPARKRUN_TELEMETRY_ENDPOINT"
TELEMETRY_KEY_ENV = "SPARKRUN_TELEMETRY_KEY"
TELEMETRY_TIMEOUT_ENV = "SPARKRUN_TELEMETRY_TIMEOUT"

TELEMETRY_ENABLED_KEY = "telemetry.enabled"
TELEMETRY_INSTALLATION_ID_KEY = "telemetry.installation_id"

DEFAULT_TELEMETRY_ENDPOINT = "https://telemetry.sparkrun.dev/"
DEFAULT_TELEMETRY_KEY = "sparkrun-telemetry-v1"
TELEMETRY_AUTH_HEADER = "x-sparkrun-telemetry-key"
DEFAULT_TIMEOUT_SECONDS = 0.75

_TRUE_VALUES = {"1", "true", "yes", "on", "enable", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disable", "disabled"}


def parse_bool(value: bool | str | int | float | None) -> bool | None:
    """Parse common boolean spellings, returning None for unset/unknown."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def env_telemetry_override() -> bool | None:
    """Return the environment-level telemetry decision, if one is set.

    ``SPARKRUN_NO_TELEMETRY`` is the single per-process override: a truthy value
    opts out, a falsy value forces telemetry on, and an unset or unrecognized
    value defers to the persisted config.
    """
    if get_application_profile().id != "sparkrun" and parse_bool(os.environ.get(NO_TELEMETRY_ENV)) is True:
        return False  # shared hard test/CI containment, never an opt-in alias
    no_telemetry = parse_bool(product_env("NO_TELEMETRY"))
    if no_telemetry is None:
        return None
    return not no_telemetry


def telemetry_enabled(config: SparkrunConfig) -> bool:
    """Return whether telemetry should be emitted for this process.

    Anything that is not a real :class:`SparkrunConfig` fails closed, ahead of
    every other layer.  No production path can reach that branch — each caller
    sources the config from ``SparkrunContext.config`` or constructs one
    directly, and the class has no subclasses — but a *test double* does, and
    the resolution below fails open for it: ``parse_bool()`` on a mock yields an
    unrecognized string, returns None, and falls through to the unset default of
    True.  A mock config therefore used to *enable* telemetry, which is how mock
    objects were posted to the live collector on three separate occasions.

    Checked before the environment override deliberately: ``SPARKRUN_NO_TELEMETRY=0``
    forces telemetry on, and must not be able to resurrect this case.
    """
    if not isinstance(config, SparkrunConfig):
        logger.debug("Telemetry disabled: config is a %s, not a SparkrunConfig", type(config).__name__)
        return False

    if get_application_profile().id != "sparkrun" and not telemetry_endpoint():
        return False
    env_override = env_telemetry_override()
    if env_override is not None:
        return env_override
    configured = parse_bool(config.get(TELEMETRY_ENABLED_KEY))
    if configured is not None:
        return configured
    return get_application_profile().telemetry_enabled


def persistent_telemetry_setting(config: SparkrunConfig) -> bool | None:
    """Return the persisted telemetry setting, ignoring environment overrides."""
    return parse_bool(config.get(TELEMETRY_ENABLED_KEY))


def set_persistent_telemetry(config: SparkrunConfig, enabled: bool) -> None:
    """Persist the telemetry enablement preference."""
    config.set(TELEMETRY_ENABLED_KEY, bool(enabled))
    config.save()


def telemetry_endpoint() -> str:
    return product_env("TELEMETRY_ENDPOINT", get_application_profile().telemetry_endpoint or "").strip()


def telemetry_key() -> str:
    return product_env("TELEMETRY_KEY", get_application_profile().telemetry_key).strip() or DEFAULT_TELEMETRY_KEY


def telemetry_timeout() -> float:
    raw = os.environ.get(TELEMETRY_TIMEOUT_ENV)
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return max(0.05, min(timeout, 5.0))


def ensure_installation_id(config: SparkrunConfig) -> str:
    """Return a persistent anonymous installation id, creating one if needed."""
    raw = config.get(TELEMETRY_INSTALLATION_ID_KEY)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    installation_id = str(uuid4())
    try:
        config.set(TELEMETRY_INSTALLATION_ID_KEY, installation_id)
        config.save()
    except OSError:
        logger.debug("Failed to persist telemetry installation id", exc_info=True)
    return installation_id


def worker_environment() -> dict[str, str]:
    """Forward explicit telemetry policy/routing into independently supervised workers."""
    from sparkrun.core.application_profile import env_name

    profile = get_application_profile()
    names = {NO_TELEMETRY_ENV, TELEMETRY_TIMEOUT_ENV}
    for setting in ("NO_TELEMETRY", "TELEMETRY_ENDPOINT", "TELEMETRY_KEY"):
        names.add(env_name(setting))
        names.update(profile.env_aliases.get(setting, ()))
    return {name: os.environ[name] for name in sorted(names) if name in os.environ}
