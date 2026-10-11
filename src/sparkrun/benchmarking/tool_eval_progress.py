"""Advisory tool-eval-bench JSONL observations, independent of execution state.

The v2.6/v2.8 CLI protocol has no trial identifiers. Multiple trials therefore
count received results without claiming an exact total or identifying trials.
"""

from __future__ import annotations

from collections import Counter
import json
from typing import Any, TypeGuard

_MAX_LINE = 65536
_MAX_SCENARIOS = 100000
_VERDICTS = {"pass", "partial", "fail"}


def _number(value: Any) -> bool:
    return type(value) in (int, float) and 0 <= value <= 1e308


def _text(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and 0 < len(value) <= 512 and value.isprintable()


def scenario_rows(consolidated: dict[str, Any]) -> list[tuple[Any, ...]]:
    scores = consolidated.get("scores")
    rows = scores.get("scenario_results", []) if isinstance(scores, dict) else []
    if not isinstance(rows, list):
        return []
    return [
        (row["scenario_id"], row.get("status", ""), row.get("points"), row.get("duration_seconds"), row.get("failure_kind") or "")
        for row in rows
        if isinstance(row, dict) and _text(row.get("scenario_id"))
    ]


def row_from_feedback(payload: dict[str, Any]) -> tuple[Any, ...] | None:
    metrics = payload.get("metrics", {})
    if metrics.get("event") != "scenario_result":
        return None
    return (metrics["scenario_id"], metrics["verdict"], metrics["points"], metrics["duration_seconds"], "")


def format_cell(value, column) -> str:
    if value is None:
        return ""
    if column.name == "Duration" and _number(value):
        return f"{value:.2f}s"
    return str(value)


def result_summary(consolidated: dict[str, Any]) -> str:
    rows = scenario_rows(consolidated)
    if not rows:
        return ""
    counts = Counter(row[1] for row in rows)
    parts = [f"{len(rows)} scenarios", *(f"{counts[key]} {key}" for key in ("pass", "partial", "fail"))]
    score = consolidated.get("final_score", consolidated.get("scores", {}).get("final_score"))
    if _number(score):
        parts.append(f"score {score:g}")
    stats = consolidated.get("trial_statistics")
    if isinstance(stats, dict) and stats.get("trials", 1) > 1:
        parts.append(f"table: last trial; aggregate statistics: {stats['trials']} trials (saved JSON)")
    return "; ".join(parts)


class ToolEvalProgressParser:
    """One parser per subprocess attempt; never reads paths from child output."""

    source = "benchmark.tool-eval-bench"

    def __init__(self, args: dict[str, Any]):
        trials = args.get("trials", 1)
        self.trials = trials if type(trials) is int and trials > 0 else 1
        self.total: int | None = None
        self._inconsistent_total = False
        self._active: set[str] = set()
        self._results: dict[str, tuple[Any, ...]] = {}
        self._received = 0

    def _observation(self, message, *, stage="scenarios", status="running", **metrics) -> dict[str, Any]:
        return {
            "message": message,
            "stage": stage,
            "status": status,
            "current": self._received if self.trials > 1 else len(self._results),
            "total": self.total,
            "total_kind": "exact" if self.total is not None else "unknown",
            "unit": "scenario results" if self.trials > 1 else "scenarios",
            "metrics": {"active": len(self._active), "trials": self.trials, **metrics},
        }

    def start(self) -> dict[str, Any]:
        return self._observation("Preparing tool evaluation suite", stage="preparation")

    def feed(self, line: str) -> dict[str, Any] | None:
        if len(line) > _MAX_LINE:
            return None
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            return None
        if not isinstance(event, dict):
            return None
        kind = event.get("event")
        if kind == "benchmark_complete":
            # Upstream can still exit nonzero (e.g. --fail-on-safety).
            return self._observation("Finalizing tool evaluation suite", stage="finalization", event=kind)
        if kind not in ("scenario_start", "scenario_result"):
            return None
        scenario, index, total = event.get("scenario_id"), event.get("index"), event.get("total")
        if not _text(scenario) or type(total) is not int or not 1 <= total <= _MAX_SCENARIOS:
            return None
        if type(index) is not int or not 0 <= index < total:
            return None
        if kind == "scenario_result":
            verdict, points, duration = event.get("status"), event.get("points"), event.get("duration_seconds")
            if not isinstance(verdict, str) or verdict not in _VERDICTS or not _number(points) or not _number(duration):
                return None
        if self.trials == 1:
            if self.total is not None and self.total != total:
                self._inconsistent_total = True
            self.total = None if self._inconsistent_total else total
        if kind == "scenario_start":
            if scenario in self._active or (self.trials == 1 and scenario in self._results):
                return None
            if len(self._active) >= _MAX_SCENARIOS:
                return None
            self._active.add(scenario)
            return self._observation(f"Running {scenario}", event=kind, scenario_id=scenario)
        row = (scenario, verdict, points, duration)
        if self.trials == 1 and self._results.get(scenario) == row:
            return None
        if len(self._results) >= _MAX_SCENARIOS and scenario not in self._results:
            return None
        self._results[scenario] = row
        self._active.discard(scenario)
        self._received += 1
        if self.total is not None and len(self._results) > self.total:
            self.total = None
            self._inconsistent_total = True
        return self._observation(
            f"{scenario}: {verdict}", event=kind, scenario_id=scenario, verdict=verdict, points=points, duration_seconds=duration
        )

    def finish(self, result: dict[str, Any] | None, *, success: bool) -> dict[str, Any]:
        # Only the assigned artifact, accepted by the scheduler, can reconcile
        # dropped callbacks. Never multiply last-trial rows by configured trials.
        if success and result is not None:
            from sparkrun.benchmarking.tool_eval_bench import _has_measurements

            success = _has_measurements(result)
        if success and result is not None and self.trials == 1:
            rows = scenario_rows(result)
            if rows:
                self._results = {row[0]: row[:4] for row in rows}
                self.total = len(self._results)
        self._active.clear()
        return self._observation(
            "Tool evaluation suite completed" if success else "Tool evaluation suite failed",
            stage="finalization",
            status="succeeded" if success else "failed",
        )
