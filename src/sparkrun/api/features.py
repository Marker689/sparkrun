"""Explicit feature actions shared by the CLI and embedded applications."""

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.plugin_lifecycle import change_feature


def enable(name, *, config=None, dry_run=False):
    return change_feature(config if config is not None else SparkrunConfig(), name, True, dry_run=dry_run)


def disable(name, *, config=None, dry_run=False):
    return change_feature(config if config is not None else SparkrunConfig(), name, False, dry_run=dry_run)


def reset(name, *, config=None, dry_run=False):
    return change_feature(config if config is not None else SparkrunConfig(), name, None, dry_run=dry_run)
