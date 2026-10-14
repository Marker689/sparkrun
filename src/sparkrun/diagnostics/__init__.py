"""Sparkrun field diagnostics — host and run diagnostic collection.

The shared bundle lifecycle is exposed through ``sparkrun.api.collect_diagnostics``
and ``sparkrun.api.capture_diagnostics``. The components below are lower-level
collectors and writers; they do not own the public bundle lifecycle.

Components:
    - :func:`collect_spark_diagnostics` — collect hardware/firmware/network/Docker
      info from one or more hosts.
    - :func:`summarize_host_diagnostics` — condense one host's raw probe output
      into a display summary (platform, OS, CPU/RAM, GPU + driver, CUDA, Docker).
    - :class:`NDJSONWriter` — append-only, immediate-flush NDJSON writer.
    - :class:`RunDiagnosticsCollector` — wraps a ``sparkrun run`` lifecycle with
      phase timing, error capture, and log collection.
"""

from __future__ import annotations

from sparkrun.diagnostics.ndjson_writer import NDJSONWriter
from sparkrun.diagnostics.spark_collector import (
    collect_config_diagnostics,
    collect_spark_diagnostics,
    collect_sudo_diagnostics,
    summarize_host_diagnostics,
)
from sparkrun.diagnostics.run_collector import RunDiagnosticsCollector
from sparkrun.diagnostics.plugin import (
    DIAGNOSTICS_API_VERSION,
    DiagnosticSection,
    DiagnosticsReporter,
    PluginDiagnostics,
    register_plugin_diagnostics,
)

__all__ = [
    "DIAGNOSTICS_API_VERSION",
    "DiagnosticSection",
    "DiagnosticsReporter",
    "PluginDiagnostics",
    "register_plugin_diagnostics",
    "NDJSONWriter",
    "RunDiagnosticsCollector",
    "collect_config_diagnostics",
    "collect_spark_diagnostics",
    "collect_sudo_diagnostics",
    "summarize_host_diagnostics",
]
