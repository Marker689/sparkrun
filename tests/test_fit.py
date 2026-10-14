"""Tests for sparkrun.models.fit (Phase 3 of hardware abstraction)."""

from __future__ import annotations

import pytest

from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.layout import Placement, RecipeLayout
from sparkrun.core.parallelism import ParallelismConfig
from sparkrun.schedulers.greedy import pack
from sparkrun.models.fit import FitResult, check_fit
from sparkrun.models.vram import DEFAULT_VRAM_GB, DGX_SPARK_VRAM_GB, VRAMEstimate


def _estimate(per_gpu_gb: float, tp: int = 1) -> VRAMEstimate:
    """Minimal VRAMEstimate stand-in for fit tests."""
    return VRAMEstimate(
        model_weights_gb=per_gpu_gb * 0.9,
        kv_cache_per_token_bytes=None,
        kv_cache_total_gb=None,
        total_per_gpu_gb=per_gpu_gb,
        max_model_len=None,
        tensor_parallel=tp,
    )


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------


def test_default_vram_gb_alias_matches_dgx_constant():
    """DEFAULT_VRAM_GB and the legacy DGX_SPARK_VRAM_GB alias point at the same value."""
    assert DEFAULT_VRAM_GB == DGX_SPARK_VRAM_GB == 121.0


# --------------------------------------------------------------------------
# Homogeneous DGX — matches the Spark platform's fit budget
# --------------------------------------------------------------------------


def test_check_fit_dgx_three_hosts_fits():
    """A 50 GB-per-GPU model fits on a 3× DGX Spark cluster.

    DGX Spark GB10 applies the platform usable-memory cap of 0.90, so the fit
    decision is made against 121 × 0.90 = 108.9 GB usable (nominal stays 121).
    """
    cluster = ClusterDefinition(name="dgx", hosts=["s1", "s2", "s3"])
    placement = pack(ParallelismConfig(tensor_parallel=3), cluster.hosts)

    result = check_fit(_estimate(per_gpu_gb=50.0, tp=3), cluster, placement)

    assert result.ok is True
    assert set(result.hosts_used) == {"s1", "s2", "s3"}
    for detail in result.per_host.values():
        assert detail.accelerator_memory_gb == pytest.approx(108.9)  # usable = 121 × 0.90
        assert detail.nominal_memory_gb == 121.0
        assert detail.max_gpu_memory_utilization == 0.90
        assert detail.headroom_gb == pytest.approx(58.9)
        assert detail.ok is True


def test_check_fit_dgx_matches_the_spark_fit_budget():
    """For homogeneous DGX, check_fit.ok mirrors the Spark platform's fit budget."""
    from sparkrun.platforms.dgx_spark import DGX_SPARK_MEMORY_GB, DGX_SPARK_SCHEDULING_FRACTION

    cluster = ClusterDefinition(name="dgx", hosts=["s1"])
    placement = pack(ParallelismConfig(), cluster.hosts)

    fits = _estimate(per_gpu_gb=80.0)
    exceeds = _estimate(per_gpu_gb=200.0)

    def within_spark_budget(est):
        return est.total_per_gpu_gb <= est.fit_budget_gb(DGX_SPARK_MEMORY_GB, DGX_SPARK_SCHEDULING_FRACTION)

    assert within_spark_budget(fits) is True
    assert check_fit(fits, cluster, placement).ok is True
    assert within_spark_budget(exceeds) is False
    assert check_fit(exceeds, cluster, placement).ok is False


def test_check_fit_cap_flips_borderline_model():
    """A 110 GB model fits nominal 121 GB but exceeds the 0.90 GB10 cap (108.9)."""
    cluster = ClusterDefinition(name="dgx", hosts=["s1"])
    placement = pack(ParallelismConfig(), cluster.hosts)

    est = _estimate(per_gpu_gb=110.0)
    result = check_fit(est, cluster, placement)
    assert result.ok is False
    assert result.per_host["s1"].max_gpu_memory_utilization == 0.90
    assert result.per_host["s1"].headroom_gb < 0

    # Explicit cluster-wide cap of 1.0 restores the full 121 GB → fits again.
    uncapped = ClusterDefinition(name="dgx", hosts=["s1"], max_gpu_memory_utilization=1.0)
    assert check_fit(est, uncapped, placement).ok is True


def test_check_fit_unknown_memory_still_ok_with_warning():
    """A host whose accelerator declares no memory_gb is reported ok + warning."""
    cluster = ClusterDefinition(
        name="mystery",
        hosts=["s1"],
        hosts_hardware={"s1": HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="unknown")])},
    )
    placement = pack(ParallelismConfig(), cluster.hosts)
    result = check_fit(_estimate(per_gpu_gb=999.0), cluster, placement)
    assert result.ok is True
    assert result.per_host["s1"].accelerator_memory_gb is None
    assert any("s1" in w for w in result.warnings)


# --------------------------------------------------------------------------
# Multi-GPU hosts (H100 80 GB, H200 141 GB, MI300X 192 GB)
# --------------------------------------------------------------------------


def _h100_8gpu_cluster() -> ClusterDefinition:
    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h100", count=8, memory_gb=80.0)])
    return ClusterDefinition(name="h100", hosts=["h100-box"], hosts_hardware={"h100-box": hw})


def _h200_8gpu_cluster() -> ClusterDefinition:
    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h200", count=8, memory_gb=141.0)])
    return ClusterDefinition(name="h200", hosts=["h200-box"], hosts_hardware={"h200-box": hw})


def test_check_fit_h100_within_budget():
    cluster = _h100_8gpu_cluster()
    placement = pack(ParallelismConfig(tensor_parallel=8), cluster.hosts, host_hardware=cluster.hosts_hardware)
    result = check_fit(_estimate(per_gpu_gb=60.0, tp=8), cluster, placement)
    assert result.ok is True
    assert result.per_host["h100-box"].accelerator_memory_gb == 80.0
    assert result.per_host["h100-box"].ranks_assigned == 8


def test_check_fit_h100_exceeds_per_gpu_budget():
    cluster = _h100_8gpu_cluster()
    placement = pack(ParallelismConfig(tensor_parallel=8), cluster.hosts, host_hardware=cluster.hosts_hardware)
    result = check_fit(_estimate(per_gpu_gb=100.0, tp=8), cluster, placement)
    assert result.ok is False
    detail = result.per_host["h100-box"]
    assert detail.ok is False
    assert detail.headroom_gb == -20.0


def test_check_fit_h200_room_for_kv_cache():
    cluster = _h200_8gpu_cluster()
    placement = pack(ParallelismConfig(tensor_parallel=8), cluster.hosts, host_hardware=cluster.hosts_hardware)
    result = check_fit(_estimate(per_gpu_gb=120.0, tp=8), cluster, placement)
    assert result.ok is True
    assert result.per_host["h200-box"].headroom_gb == 21.0


# --------------------------------------------------------------------------
# Heterogeneous cluster with explicit layout
# --------------------------------------------------------------------------


def test_check_fit_heterogeneous_per_host_outcome():
    """Layout-placed mixed cluster yields per-host pass/fail."""
    spark_hw = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10", memory_gb=121.0)])
    h100_hw = HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h100", count=2, memory_gb=80.0)])
    cluster = ClusterDefinition(
        name="mix",
        hosts=["spark", "h100"],
        hosts_hardware={"spark": spark_hw, "h100": h100_hw},
    )
    layout = RecipeLayout(
        placements=[
            Placement(host="spark", ranks=(0,)),
            Placement(host="h100", ranks=(1, 2), local_gpus=(0, 1)),
        ]
    )
    placement = pack(
        ParallelismConfig(tensor_parallel=3),
        cluster.hosts,
        host_hardware=cluster.hosts_hardware,
        layout=layout,
    )
    # 100 GB-per-GPU fits the Spark (121 GB) but not the H100 (80 GB).
    result = check_fit(_estimate(per_gpu_gb=100.0, tp=3), cluster, placement)
    assert result.ok is False
    assert result.per_host["spark"].ok is True
    assert result.per_host["h100"].ok is False
    assert result.per_host["h100"].ranks_assigned == 2


# --------------------------------------------------------------------------
# Unknown memory metadata - warning, not failure
# --------------------------------------------------------------------------


def test_check_fit_unknown_memory_warns_but_ok():
    """A host with no memory_gb declared returns ok=True + a warning."""
    hw = HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300x")])  # no memory_gb
    cluster = ClusterDefinition(name="amd", hosts=["box"], hosts_hardware={"box": hw})
    placement = pack(ParallelismConfig(), cluster.hosts, host_hardware=cluster.hosts_hardware)

    result = check_fit(_estimate(per_gpu_gb=300.0), cluster, placement)

    assert result.ok is True
    assert result.per_host["box"].ok is True
    assert result.per_host["box"].accelerator_memory_gb is None
    assert result.per_host["box"].headroom_gb is None
    assert any("box" in w for w in result.warnings)


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def test_fit_result_to_dict_round_trips_fields():
    cluster = ClusterDefinition(name="dgx", hosts=["s1"])
    placement = pack(ParallelismConfig(), cluster.hosts)
    result = check_fit(_estimate(per_gpu_gb=10.0), cluster, placement)

    d = result.to_dict()
    assert d["ok"] is True
    assert "s1" in d["per_host"]
    assert d["per_host"]["s1"]["vram_per_rank_gb"] == 10.0
    # Usable memory after the DGX GB10 0.90 cap; nominal is carried separately.
    assert d["per_host"]["s1"]["accelerator_memory_gb"] == pytest.approx(108.9)
    assert d["per_host"]["s1"]["nominal_memory_gb"] == 121.0
    assert d["per_host"]["s1"]["max_gpu_memory_utilization"] == 0.90
    assert isinstance(result, FitResult)


def test_detected_gb10_without_memory_agrees_across_fit_scheduling_and_display():
    from sparkrun.core.fingerprint import parse_fingerprint_output, build_host_hardware
    from sparkrun.core.limits import resolved_hardware_for_scheduling
    from sparkrun.utils.cli_formatters import _resolve_target_accelerator

    hardware = build_host_hardware(
        parse_fingerprint_output("NVIDIA_GPU_COUNT=1\nNVIDIA_GPU_0_NAME=NVIDIA GB10\nNVIDIA_GPU_0_MEMORY_MIB=N/A\nIB_PRESENT=1\n")
    )
    assert hardware.accelerators[0].memory_gb is None
    # A persisted probe must benefit without changing or re-fingerprinting inventory.
    saved = HostHardware.from_dict(hardware.to_dict())
    cluster = ClusterDefinition(name="sparks", hosts=["s1", "s2"], hosts_hardware={"s1": saved, "s2": saved})
    resolved = resolved_hardware_for_scheduling(cluster, cluster.hosts)
    placement = pack(ParallelismConfig(tensor_parallel=2), cluster.hosts, host_hardware=resolved)
    fit = check_fit(_estimate(77.71, tp=2), cluster, placement)
    assert fit.ok
    assert fit.warnings == []
    for host, detail in fit.per_host.items():
        assert detail.nominal_memory_gb == 121.0
        assert detail.accelerator_memory_gb == pytest.approx(108.9)
        assert resolved[host].usable_gpu_memory_slots() == pytest.approx([108.9])
    assert _resolve_target_accelerator(cluster, placement) == (121.0, "gb10")
    assert saved.to_dict() == hardware.to_dict()
    assert not check_fit(_estimate(110, tp=2), cluster, placement).ok


def test_explicit_gb10_memory_wins_over_platform_default():
    cluster = ClusterDefinition(
        name="limited",
        hosts=["s1"],
        hosts_hardware={"s1": HostHardware(accelerators=[AcceleratorSpec("nvidia", "gb10", memory_gb=96)])},
    )
    placement = pack(ParallelismConfig(), cluster.hosts)
    detail = check_fit(_estimate(90), cluster, placement).per_host["s1"]
    assert detail.nominal_memory_gb == 96
    assert detail.accelerator_memory_gb == pytest.approx(86.4)
    assert not detail.ok
