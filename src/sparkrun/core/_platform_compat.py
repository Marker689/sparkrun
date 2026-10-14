"""Deprecated single-platform API adapters; never used for target resolution."""

from __future__ import annotations

import warnings


def legacy_capacity(name: str) -> float:
    from sparkrun.platforms.dgx_spark import DGX_SPARK_MEMORY_GB, DGX_SPARK_SCHEDULING_FRACTION

    warnings.warn(
        "%s is a legacy Spark-only value; resolve the target's platform capacity and policy instead" % name,
        DeprecationWarning,
        stacklevel=3,
    )
    return DGX_SPARK_SCHEDULING_FRACTION if name == "DGX_SPARK_SCHEDULING_FRACTION" else DGX_SPARK_MEMORY_GB
