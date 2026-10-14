# SPDX-FileCopyrightText: 2026 Spark Arena
# SPDX-License-Identifier: Apache-2.0

"""Optional host diagnostics: static declarations, operation-local reporters."""

_factory = None


def register():
    global _factory
    import sparkrun.plugins as api

    if getattr(api, "DIAGNOSTICS_API_VERSION", None) != 1:
        _factory = None
        return
    _factory = api.register_plugin_diagnostics([
        api.DiagnosticSection('source_selection'),
        api.DiagnosticSection('route_selection'),
        api.DiagnosticSection('receiver_failure', priority="terminal"),
        api.DiagnosticSection('transfer_result', priority="terminal"),
    ])


def bind(*, feedback=None, sctx=None, workload_id=None):
    if _factory is None:
        return None
    return _factory.bind(feedback=feedback, sctx=sctx, workload_id=workload_id)
