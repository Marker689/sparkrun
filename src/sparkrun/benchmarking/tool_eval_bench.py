"""tool-eval-bench benchmarking framework plugin for sparkrun.

Runs `SeraphimSerapis/tool-eval-bench`_ against an OpenAI-compatible
endpoint to measure tool-call correctness across a deterministic suite of
scenarios (69 by default, plus 28 optional Hard Mode scenarios).

.. _SeraphimSerapis/tool-eval-bench: https://github.com/SeraphimSerapis/tool-eval-bench
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import shutil
from logging import Logger
from typing import Any, TYPE_CHECKING

from scitrera_app_framework import Variables

from sparkrun.benchmarking.base import BenchmarkingPlugin

if TYPE_CHECKING:
    from sparkrun.benchmarking.scheduler import BenchTask

logger = logging.getLogger(__name__)

# Pinned upstream ref. Bump as needed; users can override per-call with
# ``-b ref=<tag-or-branch>`` (the value is consumed by the plugin and not
# forwarded to the subprocess).
_DEFAULT_REF = "v2.8.0"
_GIT_URL = "https://github.com/SeraphimSerapis/tool-eval-bench"

# CLI value shapes in v2.8.0. All other values are scalar; commas in labels,
# JSON and paths must not become additional command arguments.
_LIST_ARGS = {"scenarios", "categories"}
_CSV_ARGS = {"depth", "concurrency", "spec_prompts", "mmlu_subjects"}
_REPEAT_ARGS = {"scenario_pack", "header"}
_HIDDEN_ARGS = {"ref", "framework_pinned_version"}
_REMOVED_ARGS = {"experimental_async", "llm_judge", "perf_legacy", "perf_legacy_only"}
_PERF_TRIGGER_ARGS = {"perf"}
_BOOL_ARGS = {
    "include_held_out",
    "needle",
    "needle_only",
    "decision_bench",
    "decision_bench_only",
    "decision_live",
    "dry_run",
    "fail_on_safety",
    "gsm8k",
    "gsm8k_only",
    "gsm8k_shuffle",
    "hardmode",
    "hardmode_only",
    "history",
    "ifeval",
    "ifeval_only",
    "json",
    "leaderboard",
    "mmlu",
    "mmlu_only",
    "no_live",
    "no_preflight",
    "no_probe_engine",
    "no_think",
    "no_warmup",
    "pack_only",
    "perf",
    "perf_only",
    "probe",
    "redact_url",
    "short",
    "skip_coherence",
    "skip_tool_eval",
    "spec_bench",
    "spec_live",
    "weight_by_difficulty",
}
_INT_ARGS = {
    "variant_seed",
    "spec_runs",
    "needle_depths",
    "needle_lengths",
    "benchy_runs",
    "context_size",
    "gsm8k_limit",
    "gsm8k_shots",
    "ifeval_limit",
    "max_turns",
    "mmlu_limit",
    "mmlu_shots",
    "parallel",
    "pp",
    "seed",
    "sweep_steps",
    "tg",
    "top_k",
    "trials",
}
_FLOAT_ARGS = {
    "decision_live_interval",
    "alpha",
    "baseline_tgs",
    "context_pressure",
    "error_rate",
    "min_p",
    "repeat_penalty",
    "spec_live_interval",
    "temperature",
    "timeout",
    "top_p",
}
# Upstream's standalone modes exit before writing --json-file. Do not start
# inference for work that cannot satisfy the scheduler's artifact contract.
_NO_ARTIFACT_MODES = {
    "needle_only",
    "decision_bench_only",
    "decision_live",
    "version",
    "compare",
    "export",
    "perf_only",
    "gsm8k_only",
    "mmlu_only",
    "ifeval_only",
    "skip_tool_eval",
    "spec_bench",
    "spec_live",
    "context_pressure_sweep",
    "dry_run",
    "history",
    "leaderboard",
    "probe",
}
_INFRASTRUCTURE_FAILURES = {"timeout", "connection_error", "server_error"}


def _validate_run_args(args: dict[str, Any]) -> None:
    removed = _REMOVED_ARGS.intersection(args)
    if removed:
        raise ValueError("tool-eval-bench v2.6 removed arguments: %s" % ", ".join(sorted(removed)))
    unsupported = sorted(key for key in _NO_ARTIFACT_MODES if args.get(key) is not None and args.get(key) is not False)
    if unsupported:
        raise ValueError(
            "tool-eval-bench modes do not provide scheduled JSON artifacts: %s. "
            "Use the llama-benchy framework for performance-only work or invoke the standalone tool directly." % ", ".join(unsupported)
        )


def _has_measurements(result: dict[str, Any]) -> bool:
    """Quality failures are measurements; infrastructure-only suites are not.

    Partial completion follows upstream: accept at least one graded scenario,
    preserving excluded_scenarios/completion_rate in the original envelope.
    """
    scores = result.get("scores")
    if not isinstance(scores, dict):
        return False
    maximum = scores.get("max_points")
    rows = scores.get("scenario_results")
    if (
        not isinstance(maximum, (int, float))
        or isinstance(maximum, bool)
        or not math.isfinite(maximum)
        or maximum <= 0
        or not isinstance(rows, list)
    ):
        return False
    excluded = scores.get("excluded_scenarios") or []
    return any(
        isinstance(row, dict)
        and isinstance(row.get("scenario_id"), str)
        and row["scenario_id"]
        and row["scenario_id"] not in excluded
        and row.get("status") in {"pass", "partial", "fail"}
        and row.get("failure_kind") not in _INFRASTRUCTURE_FAILURES
        for row in rows
    )


class ToolEvalBenchFramework(BenchmarkingPlugin):
    """tool-eval-bench framework — deterministic tool-call correctness.

    Invokes the pinned tool's ``run --json-file`` command and preserves its
    versioned result envelope. Synthesizes per-scenario CSV for result export.
    """

    framework_name = "tool-eval-bench"
    categories = ("tools",)
    default_args: dict[str, Any] = {
        "ref": _DEFAULT_REF,
        "parallel": 1,
        "timeout": 120,
        "max_turns": 8,
        "temperature": 0.0,
    }
    passthrough_args: set[str] = set()

    def initialize(self, v: Variables, logger_arg: Logger) -> ToolEvalBenchFramework:
        return self

    def check_prerequisites(self) -> list[str]:
        missing: list[str] = []
        if shutil.which("uvx") is None:
            missing.append("uvx not found on PATH. Install uv: https://docs.astral.sh/uv/getting-started/installation/")
        return missing

    def apply_session_warmup_state(self, run_args: dict[str, Any], *, is_first_task: bool) -> dict[str, Any]:
        """tool-eval-bench: warmup + coherence checks run once per session on
        the first task; subsequent tasks suppress them via ``no_warmup`` /
        ``skip_coherence``.
        """
        out = dict(run_args)
        if not is_first_task:
            out.setdefault("no_warmup", True)
            out.setdefault("skip_coherence", True)
        return out

    def build_benchmark_command(
        self,
        target_url: str,
        model: str,
        args: dict[str, Any],
        result_file: str | None = None,
    ) -> list[str]:
        """Assemble the ``uvx tool-eval-bench`` argv.

        ``--json-file`` produces the scheduler-owned artifact. Standalone
        command construction without a result file uses JSON on stdout.

        ``--model`` is the name sent in each request, so *model* arrives already
        resolved to the served name (``base.resolve_request_model``).  An
        explicit ``model`` in *args* — ``-b model=<name>`` — overrides it: it is
        the escape hatch for an endpoint whose served name sparkrun cannot see,
        and it used to be dropped on the floor, leaving the flag looking
        accepted while every scenario asked for something else (issue #298).
        """
        _validate_run_args(args)
        requested_model = args.get("model")
        if requested_model is not None:
            model = str(requested_model)
        ref = args.get("framework_pinned_version") or args.get("ref") or _DEFAULT_REF
        wants_perf = any(args.get(k) for k in _PERF_TRIGGER_ARGS)
        if wants_perf:
            from_spec = "tool-eval-bench[perf] @ git+%s@%s" % (_GIT_URL, ref)
        else:
            from_spec = "git+%s@%s" % (_GIT_URL, ref)

        cmd = [
            "uvx",
            "--from",
            from_spec,
            "tool-eval-bench",
            "run",
            "--base-url",
            target_url,
            "--model",
            model,
            "--json",
        ]

        if result_file:
            cmd.extend(["--json-file", result_file])

        for key, value in args.items():
            if key in _HIDDEN_ARGS or key in {"base_url", "model", "json", "json_file"} or value is None:
                continue
            flag = "--" + key.replace("_", "-")
            if key in _BOOL_ARGS:
                if type(value) is not bool:
                    raise ValueError("tool-eval-bench argument %r requires a boolean" % key)
                if value:
                    cmd.append(flag)
            elif isinstance(value, bool):
                raise ValueError("tool-eval-bench argument %r is not a boolean flag" % key)
            elif key in _REPEAT_ARGS:
                for item in value if isinstance(value, list) else [value]:
                    cmd.extend([flag, str(item)])
            elif isinstance(value, list):
                if key in _LIST_ARGS:
                    cmd.extend([flag, *(str(item) for item in value)])
                elif key in _CSV_ARGS:
                    cmd.extend([flag, ",".join(str(item) for item in value)])
                else:
                    raise ValueError("tool-eval-bench argument %r requires a scalar value" % key)
            elif key == "backend_kwargs" and isinstance(value, dict):
                cmd.extend([flag, json.dumps(value, separators=(",", ":"))])
            else:
                cmd.extend([flag, str(value)])

        return cmd

    def build_task_list(self, base_args: dict[str, Any], schedule: list[dict[str, Any]] | None) -> list[BenchTask]:
        """Run the complete tool suite as one resumable invocation."""
        from copy import deepcopy
        from sparkrun.benchmarking.scheduler import BenchTask
        from sparkrun.core.benchmark_profiles import BenchmarkError

        entries = [{}] if schedule is None else schedule
        if len(entries) != 1 or not isinstance(entries[0], dict):
            raise BenchmarkError("tool-eval-bench requires one schedule entry for its complete suite")
        entry = deepcopy(entries[0])
        run_args = {**deepcopy(base_args), **deepcopy(entry)}
        try:
            _validate_run_args(run_args)
        except ValueError as error:
            raise BenchmarkError(str(error)) from error
        return [BenchTask(0, "tool evaluation suite", run_args, entry)]

    def consolidate_per_task_results(self, per_task_jsons: list[dict[str, Any]]) -> dict[str, Any]:
        """Keep the tool's original JSON schema for the single suite artifact."""
        if len(per_task_jsons) > 1:
            raise ValueError("tool-eval-bench expects one suite artifact")
        return per_task_jsons[0] if per_task_jsons else {}

    def interpret_arg(self, key: str, value: str) -> Any:
        from sparkrun.utils import coerce_value

        if key == "header":
            # Header values may contain commas; repeat via a YAML/API list.
            return value
        if key in _LIST_ARGS | _CSV_ARGS | _REPEAT_ARGS:
            return [item.strip() for item in value.split(",")]
        if key in _BOOL_ARGS:
            parsed = coerce_value(value)
            if type(parsed) is not bool:
                raise ValueError("tool-eval-bench argument %r requires true or false" % key)
            return parsed
        if key in _INT_ARGS:
            return int(value)
        if key in _FLOAT_ARGS:
            return float(value)
        return value

    def consolidated_coverage_keys(self, consolidated: dict[str, Any]) -> set[int]:
        return {0} if _has_measurements(consolidated) else set()

    def measured_nothing(self, parsed: dict[str, Any]) -> bool:
        return not _has_measurements(parsed.get("json", {}))

    def progress_parser(self, args):
        from sparkrun.benchmarking.tool_eval_progress import ToolEvalProgressParser

        return ToolEvalProgressParser(args)

    def progress_table_spec(self):
        from sparkrun.benchmarking.base import ProgressColumn, ProgressTableSpec
        from sparkrun.benchmarking.tool_eval_progress import scenario_rows, row_from_feedback, format_cell, result_summary

        return ProgressTableSpec(
            columns=[
                ProgressColumn("Scenario", "left"),
                ProgressColumn("Verdict", "left"),
                ProgressColumn("Points"),
                ProgressColumn("Duration"),
                ProgressColumn("Failure kind", "left"),
            ],
            rows_from_consolidated=scenario_rows,
            row_key=lambda row: row[0],
            format_cell=format_cell,
            row_from_feedback=row_from_feedback,
            summary_from_consolidated=result_summary,
        )

    def estimate_test_count(self, args: dict[str, Any]) -> int | None:
        # Informational only: progress totals always come from observed events.
        if args.get("scenario_pack") or args.get("categories"):
            return None
        scenarios = args.get("scenarios")
        if isinstance(scenarios, list) and scenarios:
            count = len(set(scenarios))
        else:
            ref = args.get("framework_pinned_version") or args.get("ref") or _DEFAULT_REF
            if ref not in {"v2.6.0", "v2.8.0"}:
                return None
            hardmode = 28 if ref == "v2.8.0" else 19
            if args.get("hardmode_only"):
                count = hardmode
            else:
                count = (15 if args.get("short") else 69) + (hardmode if args.get("hardmode") else 0)
        trials = args.get("trials", 1)
        return count * trials if type(trials) is int and trials > 0 else None

    def parse_results(
        self,
        stdout: str,
        stderr: str,
        result_file: str | None = None,
    ) -> dict[str, Any]:
        from pathlib import Path

        json_data: dict[str, Any] = {}
        text = Path(result_file).read_text() if result_file else (stdout or "").strip()
        if text:
            try:
                json_data = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse tool-eval-bench JSON output")

        if not isinstance(json_data, dict):
            raise ValueError("tool-eval-bench result must be a JSON object")
        schema = json_data.get("schema_version")
        if schema is not None and schema != "1":
            raise ValueError("Unsupported tool-eval-bench result schema: %r" % schema)
        scores = json_data.get("scores", {})
        scenario_results = scores.get("scenario_results", []) if isinstance(scores, dict) else []
        csv_text = scenario_results_to_csv(scenario_results) if scenario_results else ""

        return {
            "json": json_data,
            "csv": csv_text,
            "stdout": stdout,
        }


_CSV_HEADERS = [
    "scenario_id",
    "status",
    "points",
    "duration_seconds",
    "ttft_ms",
    "turn_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "summary",
    "failure_kind",
    "turn_budget_exceeded",
]


def scenario_results_to_csv(scenario_results: list[dict[str, Any]]) -> str:
    """Render tool-eval-bench scenario results as CSV.

    Optional fields (``ttft_ms``, token counts) emit as empty strings when
    absent, mirroring the convention used by ``llama_benchy.json_to_csv``.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_HEADERS, extrasaction="ignore")
    writer.writeheader()
    for r in scenario_results:
        row = {h: r.get(h, "") for h in _CSV_HEADERS}
        # Strip newlines so each scenario is one CSV row in grep-style consumers.
        if isinstance(row.get("summary"), str):
            row["summary"] = row["summary"].replace("\n", " ").strip()
        writer.writerow(row)
    return buf.getvalue()
