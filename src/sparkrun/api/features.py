"""Explicit feature actions shared by the CLI and embedded applications."""

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.plugin_lifecycle import change_feature


def _config(config, sctx):
    """Explicit *config* wins, then the shared context's, then a fresh load."""
    if config is not None:
        return config
    return sctx.config if sctx is not None else SparkrunConfig()


def enable(name, *, config=None, dry_run=False, sctx=None):
    return change_feature(_config(config, sctx), name, True, dry_run=dry_run)


def disable(name, *, config=None, dry_run=False, sctx=None):
    return change_feature(_config(config, sctx), name, False, dry_run=dry_run)


def reset(name, *, config=None, dry_run=False, sctx=None):
    return change_feature(_config(config, sctx), name, None, dry_run=dry_run)
