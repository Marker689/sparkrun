"""Tests for the tool-eval-bench benchmarking framework plugin."""

from __future__ import annotations

import json
import shutil

import pytest
from unittest.mock import patch

from sparkrun.benchmarking.tool_eval_bench import (
    ToolEvalBenchFramework,
    scenario_results_to_csv,
)


def test_framework_name():
    fw = ToolEvalBenchFramework()
    assert fw.framework_name == "tool-eval-bench"


def test_default_args():
    fw = ToolEvalBenchFramework()
    defaults = fw.get_default_args()
    assert "backend" not in defaults
    assert defaults["ref"] == "v2.8.0"
    assert defaults["timeout"] == 120
    assert defaults["parallel"] == 1
    assert defaults["max_turns"] == 8
    assert defaults["temperature"] == 0.0


def test_build_command_basic():
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://localhost:8000/v1",
        model="org/model",
        args={"backend": "vllm", "parallel": 2},
    )
    assert cmd[0] == "uvx"
    assert "--from" in cmd
    assert "tool-eval-bench" in cmd
    assert "--base-url" in cmd
    assert "http://localhost:8000/v1" in cmd
    assert "--model" in cmd
    assert "org/model" in cmd
    assert "--json" in cmd
    assert "--backend" in cmd
    assert "vllm" in cmd
    assert "--parallel" in cmd
    assert "2" in cmd


def test_build_command_default_ref_pinned():
    """The git ref pin appears in the --from spec."""
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={},
    )
    from_idx = cmd.index("--from")
    spec = str(cmd[from_idx + 1])
    assert "git+https://github.com/SeraphimSerapis/tool-eval-bench@" in spec


def test_build_command_ref_override():
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={"ref": "main"},
    )
    from_idx = cmd.index("--from")
    spec = str(cmd[from_idx + 1])
    assert spec.endswith("@main")
    # `ref` is consumed by the plugin and must not be forwarded as a flag.
    assert "--ref" not in cmd


def test_build_command_perf_extra_activated():
    """Setting perf=True switches the --from spec to the [perf] extra."""
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={"perf": True},
    )
    from_idx = cmd.index("--from")
    spec = str(cmd[from_idx + 1])
    assert "tool-eval-bench[perf]" in spec
    assert "--perf" in cmd


def test_build_command_perf_extra_not_activated_by_default():
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={},
    )
    from_idx = cmd.index("--from")
    spec = str(cmd[from_idx + 1])
    assert "[perf]" not in spec


def test_build_command_bool_flag():
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={"no_think": True, "short": True, "hardmode": False},
    )
    assert "--no-think" in cmd
    assert "--short" in cmd
    assert "--hardmode" not in cmd


def test_build_command_list_args():
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={"categories": ["A", "B", "K"], "scenarios": ["TC-01", "TC-07"]},
    )
    cat_idx = cmd.index("--categories")
    assert cmd[cat_idx + 1] == "A"
    assert cmd[cat_idx + 2] == "B"
    assert cmd[cat_idx + 3] == "K"

    sc_idx = cmd.index("--scenarios")
    assert cmd[sc_idx + 1] == "TC-01"
    assert cmd[sc_idx + 2] == "TC-07"


def test_build_command_kebab_case_keys():
    """snake_case arg keys render as --kebab-case flags."""
    fw = ToolEvalBenchFramework()
    cmd = fw.build_benchmark_command(
        target_url="http://h:8000/v1",
        model="org/m",
        args={"max_turns": 16, "reference_date": "2026-04-25"},
    )
    assert "--max-turns" in cmd
    assert "16" in cmd
    assert "--reference-date" in cmd
    assert "2026-04-25" in cmd


def test_interpret_arg_list():
    fw = ToolEvalBenchFramework()
    assert fw.interpret_arg("categories", "A,B,K") == ["A", "B", "K"]
    assert fw.interpret_arg("scenarios", "TC-01,TC-07") == ["TC-01", "TC-07"]


def test_interpret_arg_bool():
    fw = ToolEvalBenchFramework()
    assert fw.interpret_arg("no_think", "true") is True
    assert fw.interpret_arg("short", "false") is False


def test_interpret_arg_scalar():
    fw = ToolEvalBenchFramework()
    assert fw.interpret_arg("parallel", "4") == 4
    assert fw.interpret_arg("temperature", "0.7") == 0.7
    assert fw.interpret_arg("backend", "sglang") == "sglang"


def test_estimate_test_count_default():
    fw = ToolEvalBenchFramework()
    assert fw.estimate_test_count({}) == 69


def test_estimate_test_count_short():
    fw = ToolEvalBenchFramework()
    assert fw.estimate_test_count({"short": True}) == 15


def test_estimate_test_count_hardmode():
    fw = ToolEvalBenchFramework()
    assert fw.estimate_test_count({"hardmode": True}) == 97


def test_estimate_test_count_explicit_scenarios():
    fw = ToolEvalBenchFramework()
    assert fw.estimate_test_count({"scenarios": ["TC-01", "TC-02", "TC-03"]}) == 3


def test_check_prerequisites_with_uvx():
    fw = ToolEvalBenchFramework()
    with patch.object(shutil, "which", return_value="/usr/bin/uvx"):
        assert fw.check_prerequisites() == []


def test_check_prerequisites_without_uvx():
    fw = ToolEvalBenchFramework()
    with patch.object(shutil, "which", return_value=None):
        missing = fw.check_prerequisites()
    assert len(missing) == 1
    assert "uvx" in missing[0].lower()


def test_parse_results_with_scenarios():
    fw = ToolEvalBenchFramework()
    payload = {
        "run_id": "2026-04-25T12-00-00Z_abc123",
        "status": "completed",
        "config": {"model": "org/m", "backend": "vllm"},
        "scores": {
            "final_score": 87,
            "total_points": 120,
            "max_points": 138,
            "rating": "★★★★",
            "category_scores": [
                {
                    "category": "A",
                    "label": "Basic Tool Selection",
                    "earned": 10,
                    "max": 10,
                    "percent": 100,
                    "pass_count": 5,
                    "partial_count": 0,
                    "fail_count": 0,
                },
            ],
            "scenario_results": [
                {
                    "scenario_id": "TC-01",
                    "status": "pass",
                    "points": 2,
                    "summary": "Selected get_weather correctly",
                    "duration_seconds": 1.23,
                    "ttft_ms": 95.1,
                    "turn_count": 1,
                    "prompt_tokens": 200,
                    "completion_tokens": 50,
                    "total_tokens": 250,
                },
                {
                    "scenario_id": "TC-02",
                    "status": "fail",
                    "points": 0,
                    "summary": "Hallucinated tool name",
                    "duration_seconds": 0.8,
                    "turn_count": 1,
                },
            ],
        },
        "metadata": {},
    }
    results = fw.parse_results(json.dumps(payload), "")

    assert results["json"]["scores"]["final_score"] == 87
    assert results["stdout"] == json.dumps(payload)

    csv_lines = results["csv"].strip().splitlines()
    assert len(csv_lines) == 3  # header + 2 scenarios
    assert csv_lines[0].startswith("scenario_id,status,points,")
    assert "TC-01" in csv_lines[1]
    assert "pass" in csv_lines[1]
    assert "TC-02" in csv_lines[2]
    assert "fail" in csv_lines[2]


def test_parse_results_empty_stdout():
    fw = ToolEvalBenchFramework()
    results = fw.parse_results("", "")
    assert results["json"] == {}
    assert results["csv"] == ""


def test_parse_results_invalid_json():
    fw = ToolEvalBenchFramework()
    results = fw.parse_results("not json at all", "")
    assert results["json"] == {}
    assert results["csv"] == ""
    assert results["stdout"] == "not json at all"


def test_scenario_results_to_csv_strips_newlines_in_summary():
    rows = [
        {
            "scenario_id": "TC-01",
            "status": "pass",
            "points": 2,
            "summary": "line one\nline two",
            "duration_seconds": 1.0,
            "turn_count": 1,
        },
    ]
    csv_text = scenario_results_to_csv(rows)
    # Newline-stripped summary should not introduce extra CSV rows.
    assert csv_text.count("\n") == 2  # one for header, one for row, no embedded newline
    assert "line one line two" in csv_text


def test_scenario_results_to_csv_missing_optional_fields():
    rows = [
        {
            "scenario_id": "TC-05",
            "status": "partial",
            "points": 1,
            "summary": "ok-ish",
            "duration_seconds": 0.5,
            "turn_count": 2,
            # ttft_ms and token counts intentionally omitted
        },
    ]
    csv_text = scenario_results_to_csv(rows)
    lines = csv_text.strip().splitlines()
    assert len(lines) == 2
    # ttft_ms column exists but value is empty
    header = lines[0].split(",")
    row = lines[1].split(",")
    ttft_idx = header.index("ttft_ms")
    assert row[ttft_idx] == ""


def test_latest_command_uses_native_json_file_and_raw_values(tmp_path):
    fw = ToolEvalBenchFramework()
    output = str(tmp_path / "results with spaces.json")
    label = "owner's trial, $(not a shell)"
    command = fw.build_benchmark_command(
        "http://localhost:8000/v1",
        "org/model",
        {
            "label": label,
            "scenario_pack": ["/packs/one", "/packs/two with spaces"],
            "depth": [0, 4096],
            "concurrency": [1, 4],
            "spec_prompts": ["code", "structured"],
            "backend_kwargs": {"rate": 1, "text": "a,b"},
            "json_file": "/wrong/path",
            "format": "openai",
            "hardmode_only": True,
        },
        result_file=output,
    )
    assert command[:5] == ["uvx", "--from", "git+https://github.com/SeraphimSerapis/tool-eval-bench@v2.8.0", "tool-eval-bench", "run"]
    assert command[command.index("--label") + 1] == label
    assert command[command.index("--json-file") + 1] == output
    assert command.count("--json-file") == 1 and "/wrong/path" not in command
    assert command.count("--scenario-pack") == 2
    assert command[command.index("--depth") + 1] == "0,4096"
    assert command[command.index("--concurrency") + 1] == "1,4"
    assert json.loads(command[command.index("--backend-kwargs") + 1]) == {"rate": 1, "text": "a,b"}


def test_interpret_arg_preserves_scalar_commas():
    fw = ToolEvalBenchFramework()
    assert fw.interpret_arg("label", "first, second") == "first, second"
    assert fw.interpret_arg("backend_kwargs", '{"a":1,"b":2}') == '{"a":1,"b":2}'
    assert fw.interpret_arg("scenario_pack", "one,two") == ["one", "two"]
    assert fw.interpret_arg("hardmode_only", "true") is True


def test_single_suite_task_round_trips_without_mutating_arguments():
    fw = ToolEvalBenchFramework()
    base = fw.get_default_args()
    tasks = fw.build_task_list(base, [{"scenarios": ["TC-01"], "label": "trial"}])
    restored = fw.build_task_list(base, [task.schedule_entry for task in tasks])
    assert restored == tasks and tasks[0].run_args["ref"] == "v2.8.0"
    tasks[0].run_args["scenarios"].append("TC-02")
    assert "scenarios" not in base
    assert tasks[0].schedule_entry["scenarios"] == ["TC-01"]


def test_native_result_file_preserves_versioned_envelope(tmp_path):
    import csv
    import io

    fw = ToolEvalBenchFramework()
    payload = {
        "schema_version": "1",
        "tool_eval_bench_version": "2.6.0",
        "final_score": 0,
        "scores": {
            "completion_rate": 0.5,
            "excluded_scenarios": ["TC-02"],
            "scenario_results": [
                {"scenario_id": "TC-01", "status": "fail", "points": 0, "summary": "wrong tool"},
                {"scenario_id": "TC-02", "status": "fail", "points": 0, "failure_kind": "timeout"},
            ],
        },
        "trial_statistics": {"trials": 2},
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    result = fw.parse_results("diagnostic output is not JSON", "", str(path))
    assert result["json"] == payload
    assert list(csv.DictReader(io.StringIO(result["csv"])))[1]["failure_kind"] == "timeout"
    assert fw.consolidate_per_task_results([payload]) == payload


def test_unknown_result_schema_is_not_silently_interpreted():
    import pytest

    with pytest.raises(ValueError, match="Unsupported.*schema"):
        ToolEvalBenchFramework().parse_results('{"schema_version":"2"}', "")


def test_single_suite_rejects_unsupported_schedules():
    import pytest
    from sparkrun.core.benchmark_profiles import BenchmarkError

    for schedule in ([], [{}, {}]):
        with pytest.raises(BenchmarkError, match="one schedule entry"):
            ToolEvalBenchFramework().build_task_list({}, schedule)


@pytest.mark.parametrize("partial", [False, True])
def test_coverage_accepts_graded_zero_and_partial_infrastructure_results(partial):
    fw = ToolEvalBenchFramework()
    scores = {
        "final_score": 0,
        "max_points": 2,
        "completion_rate": 0.5 if partial else 1.0,
        "excluded_scenarios": ["TC-02"] if partial else [],
        "scenario_results": [{"scenario_id": "TC-01", "status": "fail", "failure_kind": "missing_step"}],
    }
    if partial:
        scores["scenario_results"].append({"scenario_id": "TC-02", "status": "fail", "failure_kind": "timeout"})
    parsed = fw.parse_results(json.dumps({"schema_version": "1", "scores": scores}), "")
    assert fw.consolidated_coverage_keys(parsed["json"]) == {0}
    assert not fw.measured_nothing(parsed)
    assert parsed["json"]["scores"] == scores


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"scores": {}},
        {"scores": {"max_points": 2, "scenario_results": []}},
        {"scores": {"max_points": 2, "scenario_results": [{"scenario_id": "TC-01", "status": "fail", "failure_kind": "timeout"}]}},
    ],
)
def test_coverage_rejects_missing_or_infrastructure_measurements(payload):
    fw = ToolEvalBenchFramework()
    parsed = fw.parse_results(json.dumps(payload), "")
    assert fw.consolidated_coverage_keys(parsed["json"]) == set()
    assert fw.measured_nothing(parsed)


@pytest.mark.parametrize(
    "key,value", [("label", "true"), ("label", "false"), ("label", "0012"), ("ref", "123"), ("db", "false"), ("scenario_pack", "123,true")]
)
def test_argument_shapes_preserve_text(key, value):
    expected = value.split(",") if key == "scenario_pack" else value
    assert ToolEvalBenchFramework().interpret_arg(key, value) == expected


def test_boolean_flags_reject_non_boolean_values():
    fw = ToolEvalBenchFramework()
    with pytest.raises(ValueError, match="true or false"):
        fw.interpret_arg("no_think", "perhaps")
    with pytest.raises(ValueError, match="boolean"):
        fw.build_benchmark_command("http://localhost/v1", "test/model", {"label": True})


@pytest.mark.parametrize(
    "args,expected",
    [
        ({"hardmode_only": True}, 28),
        ({"hardmode": True, "short": True}, 43),
        ({"ref": "v2.6.0", "hardmode": True}, 88),
        ({"ref": "v2.6.0", "hardmode_only": True}, 19),
        ({"ref": "main"}, None),
        ({"scenario_pack": ["custom"], "pack_only": True}, None),
        ({"scenarios": ["TC-01", "TC-01", "TC-02"], "trials": 2}, 4),
    ],
)
def test_estimates_respect_known_version_and_selection(args, expected):
    assert ToolEvalBenchFramework().estimate_test_count(args) == expected
