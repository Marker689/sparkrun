"""Explicit service entry point; help remains server-import free."""

from sparkrun.core.application_profile import initialize_child_application_profile

initialize_child_application_profile()
from sparkrun.cli._daemon import daemon  # noqa: E402

daemon()
