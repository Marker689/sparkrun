"""First-party cross-cutting integrations, and the plugin author import surface.

Two roles share this package:

* **The supported import surface for plugin authors.** Everything a plugin
  registers with, subclasses, or receives from a registration hook is
  re-exported here and listed in :data:`__all__`. Import from
  ``sparkrun.plugins``; the defining modules (``sparkrun.core.*``,
  ``sparkrun.proxy.*`` and so on) are internal and may move. Operational APIs a
  plugin *calls* live in the separately documented ``sparkrun.api`` and
  ``sparkrun.application``. See ``docs/PLUGINS.md``.
* **The home of bundled integrations.** Each subpackage is one integration,
  discovered and registered at bootstrap by
  :func:`sparkrun.core.in_tree_plugins.load_in_tree_plugins` — the same
  registration path out-of-tree plugins take, so a first-party integration has
  no capability an external one lacks.

This package is deliberately *not* the home for every first-party plugin. A
runtime belongs in ``sparkrun.runtimes``, an executor in
``sparkrun.orchestration.executors``, and so on: those map cleanly onto one
extension point and keep their own scan in :mod:`sparkrun.core.bootstrap`.
What lives here is what spans several at once and is only coherent as a single
removable unit — an integration contributing, say, a backend implementation, a
hidden CLI command, and the wire protocol that backend calls back through, none
of which is "a runtime" or "an executor" on its own.

Every in-process registry exported here follows one conflict rule: registering
an equal value again is a no-op; anything else raises
:class:`PluginConflictError`.
"""

from sparkrun.benchmarking.base import BenchmarkingPlugin
from sparkrun.benchmarking.metadata import public_recipe_text
from sparkrun.benchmarking.scheduler import BenchTask
from sparkrun.core.benchmark_integrations import (
    BenchmarkDefaults,
    BenchmarkIntegration,
    BenchmarkIntegrationContext,
    BenchmarkMeasurement,
    BenchmarkReporter,
    BenchmarkStateInfo,
    register_benchmark_integration,
)
from sparkrun.core.build_preparation import BuildContext
from sparkrun.core.cli_registry import CliOptionSpec, register_cli_command, register_cli_options
from sparkrun.core.daemon_registry import DaemonOp, DaemonRoute, register_plugin_daemon_op, register_plugin_daemon_route
from sparkrun.core.env_templates import env_template_launch_fields, render_env_template, render_launch_env
from sparkrun.core.features import FeatureFlag
from sparkrun.core.hardware_probe_extensions import register_hardware_probe
from sparkrun.core.image_distribution import (
    IMAGE_DISTRIBUTION_API_VERSION,
    IMAGE_PULL_API_VERSION,
    IMAGE_RUNTIME_API_VERSION,
    ImageCopyRequest,
    ImageCopyResult,
    ImageDistributionFailed,
    ImageDistributionUnsupported,
    ImagePullRequest,
    has_image_distribution_provider,
    image_distribution_operation,
    register_image_distribution_provider,
    resolve_distributed_image,
    try_image_copy,
    try_image_pull,
)
from sparkrun.core.log_source import LogResource, LogSourceChanged
from sparkrun.core.model_distribution import (
    MODEL_DISTRIBUTION_API_VERSION,
    MODEL_PULL_API_VERSION,
    ModelCopyRequest,
    ModelDistributionFailed,
    ModelDistributionUnsupported,
    ModelPullRequest,
    ModelTarget,
    ModelTransferResult,
    register_model_distribution_provider,
)
from sparkrun.core.plugin_identity import application_daemon_registration
from sparkrun.core.recipe_formats import RecipeFormat, register_recipe_format
from sparkrun.core.recipe_items import FunctionalRecipeItemHandler, RecipeItemHandler, register_recipe_item
from sparkrun.core.registration import PluginConflictError, enlist_registry_state, register_unique
from sparkrun.core.registry import RegistryEntry
from sparkrun.core.registry_defaults import register_default_registry
from sparkrun.core.run_handlers import RunHandler, register_run_handler
from sparkrun.core.setup_steps import register_setup_constraint, register_setup_step
from sparkrun.core.workload_lifecycle import WORKLOAD_LIFECYCLE_API_VERSION, WorkloadLifecycleProvider, register_workload_lifecycle
from sparkrun.diagnostics.plugin import (
    DIAGNOSTICS_API_VERSION,
    DiagnosticSection,
    DiagnosticsReporter,
    PluginDiagnostics,
    register_plugin_diagnostics,
)
from sparkrun.feedback import FEEDBACK_API_VERSION, FeedbackEvent, FeedbackReporter, get_feedback
from sparkrun.models.artifacts import ModelArtifactFile, ModelArtifactManifest
from sparkrun.models.kv import KVCacheStrategy, register_kv_strategy
from sparkrun.orchestration.executor import Executor, ExecutorConfig, ExecutorTarget
from sparkrun.orchestration.hooks import HookLaunchContext, run_pre_exec
from sparkrun.orchestration.job_metadata import save_job_metadata
from sparkrun.platforms import register_platform
from sparkrun.platforms.base import HardwarePlatformPlugin
from sparkrun.plugin_lifecycle import PLUGIN_LIFECYCLE_API_VERSION, LifecycleContext, LifecyclePlan, select_gateway
from sparkrun.proxy.contracts import (
    GatewayAdminToken,
    GatewayConsole,
    GatewayConsoleCredentials,
    GatewayOperationError,
    GatewayQueryError,
    ProxyModel,
)
from sparkrun.proxy.gateway import register_gateway
from sparkrun.proxy.supervisor import GatewaySupervisor
from sparkrun.runtimes.base import RuntimePlugin

__all__ = [
    "BenchTask",
    "BenchmarkDefaults",
    "BenchmarkIntegration",
    "BenchmarkIntegrationContext",
    "BenchmarkMeasurement",
    "BenchmarkReporter",
    "BenchmarkStateInfo",
    "BenchmarkingPlugin",
    "BuildContext",
    "CliOptionSpec",
    "DIAGNOSTICS_API_VERSION",
    "DaemonOp",
    "DaemonRoute",
    "DiagnosticSection",
    "DiagnosticsReporter",
    "Executor",
    "ExecutorConfig",
    "ExecutorTarget",
    "FEEDBACK_API_VERSION",
    "FeatureFlag",
    "FeedbackEvent",
    "FeedbackReporter",
    "FunctionalRecipeItemHandler",
    "GatewayAdminToken",
    "GatewayConsole",
    "GatewayConsoleCredentials",
    "GatewayOperationError",
    "GatewayQueryError",
    "GatewaySupervisor",
    "HardwarePlatformPlugin",
    "HookLaunchContext",
    "IMAGE_DISTRIBUTION_API_VERSION",
    "IMAGE_PULL_API_VERSION",
    "IMAGE_RUNTIME_API_VERSION",
    "ImageCopyRequest",
    "ImageCopyResult",
    "ImageDistributionFailed",
    "ImageDistributionUnsupported",
    "ImagePullRequest",
    "KVCacheStrategy",
    "LifecycleContext",
    "LifecyclePlan",
    "LogResource",
    "LogSourceChanged",
    "MODEL_DISTRIBUTION_API_VERSION",
    "MODEL_PULL_API_VERSION",
    "ModelArtifactFile",
    "ModelArtifactManifest",
    "ModelCopyRequest",
    "ModelDistributionFailed",
    "ModelDistributionUnsupported",
    "ModelPullRequest",
    "ModelTarget",
    "ModelTransferResult",
    "PLUGIN_LIFECYCLE_API_VERSION",
    "PluginConflictError",
    "PluginDiagnostics",
    "ProxyModel",
    "RecipeFormat",
    "RecipeItemHandler",
    "RegistryEntry",
    "RunHandler",
    "RuntimePlugin",
    "WORKLOAD_LIFECYCLE_API_VERSION",
    "WorkloadLifecycleProvider",
    "application_daemon_registration",
    "enlist_registry_state",
    "env_template_launch_fields",
    "get_feedback",
    "has_image_distribution_provider",
    "image_distribution_operation",
    "public_recipe_text",
    "register_benchmark_integration",
    "register_cli_command",
    "register_cli_options",
    "register_default_registry",
    "register_gateway",
    "register_hardware_probe",
    "register_image_distribution_provider",
    "register_kv_strategy",
    "register_model_distribution_provider",
    "register_platform",
    "register_plugin_daemon_op",
    "register_plugin_daemon_route",
    "register_plugin_diagnostics",
    "register_recipe_format",
    "register_recipe_item",
    "register_run_handler",
    "register_setup_constraint",
    "register_setup_step",
    "register_unique",
    "register_workload_lifecycle",
    "render_env_template",
    "render_launch_env",
    "resolve_distributed_image",
    "run_pre_exec",
    "save_job_metadata",
    "select_gateway",
    "try_image_copy",
    "try_image_pull",
]
