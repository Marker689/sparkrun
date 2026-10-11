"""Optional Kubernetes executor, setup commands, and JobSet launch integration."""

from __future__ import annotations

from sparkrun.core.features import FeatureFlag

from sparkrun import __version__ as __version__

SPARKRUN_PLUGIN_API_VERSION = 1

FEATURE_FLAG = "integration.k8s"

# The loader registers these only after the application enables this plugin.
FEATURE_DEFINITIONS = (
    FeatureFlag("executor.k8s", "Experimental Kubernetes (kubectl) executor", default=True),
    FeatureFlag("cli.setup.k8s", "Kubernetes cluster setup commands", default=True),
    FeatureFlag("api.run.k8s", "Launch Kubernetes workloads through JobSet", default=True),
)


def _load_setup_command():
    from .cli import build_setup_command

    return build_setup_command()


def _run(options, sctx, *, plan, started_at, before_start):
    from .run import run_k8s

    return run_k8s(options, sctx, plan=plan, started_at=started_at, before_start=before_start)


def register(v):
    from .daemon import register as register_daemon

    register_daemon()
    from sparkrun.core.cli_registry import register_cli_command
    from sparkrun.core.run_handlers import RunHandler, register_run_handler

    register_cli_command(_load_setup_command, name="k8s", parent=("setup",))
    register_run_handler(RunHandler("k8s", _run, feature_flag="api.run.k8s"))
