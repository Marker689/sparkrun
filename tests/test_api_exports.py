"""Public ``sparkrun.api`` surface: returned types, experimental helpers, trimmed exports."""

from __future__ import annotations

import subprocess
import sys

import pytest

from sparkrun import api


@pytest.mark.parametrize(
    "name,module",
    [
        ("ClusterStatus", "sparkrun.core.cluster_status"),
        ("HostOccupancy", "sparkrun.core.cluster_status"),
        ("RunningWorkload", "sparkrun.core.cluster_status"),
        ("GpuOccupancy", "sparkrun.core.cluster_status"),
        ("ClusterDefinition", "sparkrun.core.cluster_manager"),
        ("SchedulingRequest", "sparkrun.core.scheduler"),
        ("SchedulingResult", "sparkrun.core.scheduler"),
    ],
)
def test_returned_types_are_reexported_as_the_core_classes(name, module):
    import importlib

    assert name in api.__all__
    assert getattr(api, name) is getattr(importlib.import_module(module), name)


def test_returned_types_are_imported_lazily():
    script = (
        "import sys, sparkrun.api as a\n"
        "assert 'sparkrun.core.cluster_status' not in sys.modules\n"
        "from sparkrun.api import ClusterStatus\n"
        "assert ClusterStatus.__module__ == 'sparkrun.core.cluster_status'\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        api.NotARealName  # noqa: B018


def test_resolve_sctx_is_public_and_the_plugin_uses_it():
    from sparkrun.api import _context
    from sparkrun.plugins.k8s.api import _logs, _ops

    assert api.resolve_sctx is _context.resolve_sctx and "resolve_sctx" in api.__all__
    assert _ops.resolve_sctx is api.resolve_sctx and _logs.resolve_sctx is api.resolve_sctx


def test_internal_helpers_left_the_public_lists():
    from sparkrun.api import setup

    assert "discover_endpoints" not in api.__all__
    assert "PostLaunchResult" in api.__all__
    for name in ("build_authorized_key_script", "build_collect_key_script", "build_install_keys_script"):
        assert name not in setup.__all__ and not hasattr(setup, name)


def test_run_plan_runtime_is_typed():
    from dataclasses import fields

    # Postponed annotation (TYPE_CHECKING import); pyright checks the resolution.
    assert {f.name: f.type for f in fields(api.RunPlan)}["runtime"].strip("'") == "RuntimePlugin"
