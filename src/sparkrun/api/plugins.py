"""Plugin management without application bootstrap (including recovery)."""

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.plugin_lifecycle import change_plugin


def enable(name, *, config=None, dry_run=False):
    """Enable loading, retaining capability settings; run the explicit enable hook."""
    return change_plugin(config if config is not None else SparkrunConfig(), name, True, dry_run=dry_run)


def disable(name, *, config=None, dry_run=False):
    """Disable loading, retaining settings and allowing broken-plugin recovery."""
    return change_plugin(config if config is not None else SparkrunConfig(), name, False, dry_run=dry_run)
