#!/usr/bin/env python3
"""Offline incremental reporter benchmark; no host probes, transfers, or user config.

Run with the repository environment and redirect JSON to retain a baseline.
This measures reporting overhead, not end-to-end transfer performance.
"""

from pathlib import Path
import json
import platform
import statistics
import tempfile
import time
import timeit

from sparkrun.core.operation import OperationState
from sparkrun.core.plugin_identity import PluginIdentity, plugin_registration
from sparkrun.diagnostics import DiagnosticSection, register_plugin_diagnostics
from sparkrun.diagnostics._plugin_sink import PluginSink
from sparkrun.diagnostics.ndjson_writer import NDJSONWriter
from sparkrun.diagnostics.plugin import DISABLED
from sparkrun.diagnostics.redaction import Redactor


def main():
    with plugin_registration(PluginIdentity("diagnostics-benchmark", __name__, ("benchmark", __name__, None))):
        declaration = register_plugin_diagnostics([DiagnosticSection("phase")])
    builder_calls = 0

    def build():
        nonlocal builder_calls
        builder_calls += 1
        return {"observations": ["a" * 1024, "b" * 1024], "receivers": 4, "transport": "http2-direct"}

    def guarded():
        if DISABLED.enabled:
            DISABLED.snapshot("phase", build)

    disabled = {}
    for name, action in (("guarded", guarded), ("direct", lambda: DISABLED.snapshot("phase", build)), ("bind", declaration.bind)):
        disabled[name + "_ns"] = statistics.median(timeit.repeat(action, number=100_000, repeat=5)) * 1e9 / 100_000
    assert builder_calls == 0
    wall, cpu, sizes = [], [], []
    with tempfile.TemporaryDirectory(prefix="sparkrun-diagnostics-perf-") as root:
        for index in range(20):
            path = Path(root) / ("%d.ndjson" % index)
            state = OperationState()
            state.secrets.update("benchmark-secret-%d" % number for number in range(32))
            redact = Redactor()
            redact.operation_secrets = state.secrets
            with NDJSONWriter(path, exclusive=True, redact=redact, max_bytes=16 * 1024 * 1024) as writer:
                sink = PluginSink(writer, state, max_bytes=16 * 1024 * 1024)
                state.diagnostics_sink = sink
                from sparkrun.core.operation import operation_scope

                with operation_scope(state=state):
                    reporter = declaration.bind()
                for _ in range(24):
                    started_wall, started_cpu = time.perf_counter_ns(), time.process_time_ns()
                    assert reporter.snapshot("phase", build)
                    cpu.append(time.process_time_ns() - started_cpu)
                    wall.append(time.perf_counter_ns() - started_wall)
                sink.close()
                assert not sink.partial
                state.diagnostics_sink = None
            sizes.append(path.stat().st_size)
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.system(),
                "architecture": platform.machine(),
                "sink": "synchronous temporary local file, flush per record, no fsync",
                "disabled": disabled,
                "disabled_builder_calls": 0,
                "enabled": {
                    "snapshots": len(wall),
                    "payload_text_bytes": 2048,
                    "registered_secrets": 32,
                    "median_wall_us": statistics.median(wall) / 1000,
                    "p95_wall_us": sorted(wall)[int(len(wall) * 0.95)] / 1000,
                    "median_cpu_us": statistics.median(cpu) / 1000,
                    "mean_capture_bytes": statistics.mean(sizes),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
