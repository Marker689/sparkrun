"""Compatibility alias for the canonical top-level plugins group."""

import click
from .._plugins import plugins, plugins_list as setup_plugins_list
from . import setup

# Share callbacks and commands, but not visibility: the canonical group is public.
setup_plugins = click.Group(
    "plugins", callback=plugins.callback, commands=plugins.commands, invoke_without_command=True, hidden=True, help=plugins.help
)
setup.add_command(setup_plugins)

__all__ = ["setup_plugins", "setup_plugins_list"]
