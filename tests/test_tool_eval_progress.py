"""Tool suite observations are advisory, bounded, and shared by every frontend."""

import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from sparkrun.api._benchmark import _CallbackProgressEmitter
from sparkrun.benchmarking._credentials import BenchmarkCredentials
from sparkrun.benchmarking._process import _Output
from sparkrun.benchmarking.progress_ui import BenchmarkProgressUI, _PlainUI, _RichUI
from sparkrun.benchmarking.run_state import BenchmarkRunState
from sparkrun.benchmarking.scheduler import run_schedule
from sparkrun.benchmarking.tool_eval_bench import ToolEvalBenchFramework
from sparkrun.benchmarking.tool_eval_progress import ToolEvalProgressParser, result_summary
from sparkrun.core.operation import OperationState, operation_scope


def event(kind="scenario_result", scenario="TC-01", index=0, total=2, **values):
    data = {"event": kind, "scenario_id": scenario, "index": index, "total": total}
    if kind == "scenario_result":
        data.update(status="pass", points=2, duration_seconds=0.25)
    return json.dumps({**data, **values}, ensure_ascii=False)


def artifact(*ids):
    return {
        "schema_version": "1",
        "tool_eval_bench_version": "2.8.0",
        "final_score": 100,
        "scores": {
            "max_points": 2 * len(ids),
            "scenario_results": [{"scenario_id": name, "status": "pass", "points": 2, "duration_seconds": 0.25} for name in ids],
        },
    }


def test_parallel_progress_counts_ids_not_completion_indices_and_reconciles_artifact():
    parser = ToolEvalProgressParser({})
    assert parser.start()["total"] is None
    parser.feed(event("scenario_start"))
    started = parser.feed(event("scenario_start", "TC-02", index=1))
    assert started["current"] == 0 and started["metrics"]["active"] == 2
    first = parser.feed(event(scenario="TC-02", index=0))
    assert first["current"] == 1 and first["metrics"]["active"] == 1
    assert parser.feed(event(scenario="TC-02", index=1)) is None
    changed = parser.feed(event(scenario="TC-02", status="partial", points=1))
    assert changed["current"] == 1
    # The child may omit a callback, and benchmark_complete cannot imply success.
    complete = parser.feed('{"event":"benchmark_complete","json_file":"/untrusted/path"}')
    assert complete["status"] == "running" and complete["current"] == 1
    final = parser.finish(artifact("TC-01", "TC-02"), success=True)
    assert final["current"] == final["total"] == 2
    assert final["metrics"]["active"] == 0 and final["status"] == "succeeded"


@pytest.mark.parametrize(
    "line",
    [
        "uv installation message",
        "{",
        "[]",
        "null",
        "[" * 1500,
        '{"event":"unknown"}',
        event(index=True),
        event(index=-1),
        event(index=2),
        event(total=-1),
        event(total=100001),
        event(total="2"),
        event(scenario=""),
        event(scenario="x" * 513),
        event(scenario="a\nb"),
        event(status=[]),
        event(status="new-verdict"),
        event(points=True),
        event(points=-1),
        event(duration_seconds=float("inf")),
        event(points=float("nan")),
        event(points=10**400),
        event() + " " * 65536,
    ],
)
def test_noise_and_malformed_events_do_not_change_progress(line):
    parser = ToolEvalProgressParser({})
    before = parser.start()
    assert parser.feed(line) is None
    assert parser.start() == before


def test_unknown_fields_ignored_and_inconsistent_totals_become_indeterminate():
    parser = ToolEvalProgressParser({})
    assert parser.feed(event(extra={"future": "field"}))["total"] == 2
    assert parser.feed(event(scenario="TC-02", total=3))["total_kind"] == "unknown"
    assert parser.feed(event(scenario="TC-03", total=3))["total"] is None


def test_multitrial_counts_received_results_without_inventing_trial_identity():
    parser = ToolEvalProgressParser({"trials": 2})
    for count in range(1, 5):
        value = parser.feed(event())
        assert value["current"] == count and value["total"] is None
        assert value["metrics"]["trials"] == 2
        assert "trial_index" not in value["metrics"]
    final = parser.finish(artifact("TC-01", "TC-02"), success=True)
    assert final["current"] == 4 and final["total_kind"] == "unknown"
    data = artifact("TC-01", "TC-02")
    data["trial_statistics"] = {"trials": 2, "mean": 100}
    assert "last trial" in result_summary(data)


def test_utf8_crlf_and_secret_fragments_redacted_before_parser():
    parser = ToolEvalProgressParser({})
    observed, logged = [], []

    def consume(line):
        logged.append(line)
        if value := parser.feed(line):
            observed.append(value)

    output = _Output(consume, BenchmarkCredentials("secret-key"))
    text = "plain noise\r\n" + event(scenario="café-secret-key") + "\r\n" + event(scenario="TC-02", index=1)
    for byte in text.encode():
        output.feed(bytes([byte]))
    output.feed(b"", final=True)
    assert len(logged) == 3 and len(observed) == 2
    assert observed[0]["metrics"]["scenario_id"] == "café-[REDACTED]"
    assert "secret-key" not in json.dumps(observed) + "".join(logged)
    assert observed[-1]["current"] == 2


def _schedule(tmp_path, monkeypatch, *, exit_code=0, subscriber=None, operation_subscriber=None):
    fw = ToolEvalBenchFramework()
    saved = artifact("TC-01", "TC-02")
    raw = "diagnostic\n" + event(scenario="TC-01") + '\n{"event":"benchmark_complete","json_file":"/wrong"}\n'

    def command(*args, result_file, **kwargs):
        return [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                f"sys.stderr.write({raw!r}); sys.stdout.write('stdout noise'); "
                f"Path({result_file!r}).write_text({json.dumps(saved)!r}); sys.exit({exit_code})"
            ),
        ]

    monkeypatch.setattr(fw, "build_benchmark_command", command)
    state = BenchmarkRunState(
        benchmark_id="bench_progress",
        cluster_id="cluster-test",
        recipe_qualified_name="test",
        framework=fw.framework_name,
        profile="default",
        base_args={},
        schedule=[{}],
    )
    state.save(str(tmp_path))
    callbacks, feedback = [], []
    emitter = _CallbackProgressEmitter(subscriber or callbacks.append)
    with operation_scope(state=OperationState(events=operation_subscriber or feedback.append)):
        with emitter.schedule_progress(total_tasks=1, benchmark_id=state.benchmark_id, fw=fw, title="test") as sink:
            result = run_schedule(
                fw,
                fw.build_task_list({}, None),
                state,
                target_url="http://localhost/v1",
                model="test",
                timeout=5,
                task_events=sink,
                cache_dir=str(tmp_path),
            )
    return result, callbacks, feedback


def test_scheduler_feedback_is_exactly_once_and_results_remain_authoritative(tmp_path, monkeypatch, caplog):
    caplog.set_level(25)
    result, callbacks, delivered = _schedule(tmp_path, monkeypatch)
    assert result.success and result.state.completed_indices == [0]
    feedback = [e for e in delivered if e["type"] == "feedback"]
    observations = [e.data["feedback"] for e in callbacks if e.kind == "task_progress"]
    assert len(feedback) == len(observations) == 4
    assert [{k: v for k, v in e.items() if k != "operation_id"} for e in feedback] == observations
    assert len({e["activity_id"] for e in observations}) == 1
    assert all(e["metrics"]["benchmark_id"] == "bench_progress" and e["metrics"]["task_index"] == 0 for e in observations)
    assert observations[-1]["current"] == 2  # Artifact fills the missing TC-02 callback.
    updates = [e for e in callbacks if e.kind == "results_update"]
    assert updates and all(e.data["results"] == artifact("TC-01", "TC-02") for e in updates)
    assert "TC-01: pass" not in caplog.text  # No duplicate feedback log messages.
    log = next(tmp_path.glob("benchmarks/bench_progress/runs/*.log")).read_text()
    assert "diagnostic" in log and "stdout noise" in log and "scenario_result" in log


def test_safety_exit_after_complete_event_stays_failed(tmp_path, monkeypatch):
    result, callbacks, _ = _schedule(tmp_path, monkeypatch, exit_code=2)
    assert not result.success and result.state.completed_indices == []
    observations = [e.data["feedback"] for e in callbacks if e.kind == "task_progress"]
    assert observations[-2]["status"] == "running"
    assert observations[-1]["status"] == "failed"
    assert result.consolidated == {}


def test_advisory_callback_errors_do_not_abort_measurement(tmp_path, monkeypatch):
    count = 0

    def callback(e):
        nonlocal count
        if e.kind == "task_progress":
            count += 1
            raise ValueError("broken UI")

    def broken_operation(e):
        raise ValueError("broken operation subscriber")

    result, _, _ = _schedule(tmp_path, monkeypatch, subscriber=callback, operation_subscriber=broken_operation)
    assert result.success and count == 4


def test_attempt_ids_and_counters_reset_on_retry(tmp_path, monkeypatch):
    from sparkrun.benchmarking.scheduler import _AttemptProgress

    sink = SimpleNamespace(observe_task=Mock())
    first = _AttemptProgress(ToolEvalBenchFramework(), {}, sink, "bench_test", 0)
    first.feed(event())
    first.finish()
    retry = _AttemptProgress(ToolEvalBenchFramework(), {}, sink, "bench_test", 0)
    retry.feed(event())
    values = [call.args[1] for call in sink.observe_task.call_args_list]
    assert values[0]["activity_id"] != values[3]["activity_id"]
    assert [v["current"] for v in values] == [0, 1, 1, 0, 1]


def test_daemon_does_not_duplicate_normalized_feedback():
    from sparkrun.daemon.ops import _benchmark_progress
    from sparkrun.api import ProgressEvent

    state = OperationState(events=Mock())
    sctx = SimpleNamespace(operation=state)
    _benchmark_progress(sctx, ProgressEvent("task_progress", {"feedback": {"type": "feedback"}}))
    state.events.assert_not_called()
    _benchmark_progress(sctx, ProgressEvent("task_end", {"success": True, "index": 0}))
    state.events.assert_called_once()
    assert state.events.call_args.args[0]["event"]["kind"] == "task_end"


def test_rich_unknown_total_no_empty_table_and_single_final_table():
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=120, height=20)
    fw = ToolEvalBenchFramework()
    with patch("sparkrun.benchmarking.progress_ui.Console", return_value=console):
        ui = _RichUI(total_tasks=1, benchmark_id="test", fw=fw)
    with ui:
        parser = ToolEvalProgressParser({})
        ui.observe_task(0, parser.start())
        assert ui._table.row_count == 0
        assert ui._progress.tasks[0].total is None
        console.print(ui._render_group())
        preparation = stream.getvalue()
        assert "0/1" not in preparation and "0.0%" not in preparation and "Scenario" not in preparation
        for i in range(25):
            observed = parser.feed(event(scenario=f"TC-{i:02}", index=i, total=25))
            ui.observe_task(0, observed)
        assert ui._table.row_count == ui._window_cap() == 12
        assert ui._progress.tasks[0].completed == 25
        ui.update_results_table(artifact("[bold]TC-01[/bold]"))
    output = stream.getvalue()
    assert "[bold]TC-01[/bold]" in output
    # Live is transient; its final table is not retained before the full table.
    assert ui._live.transient is True
    assert output.count("100") >= 1


def test_plain_redirected_output_no_empty_or_duplicate_table(capsys):
    fw = ToolEvalBenchFramework()
    ui = BenchmarkProgressUI(total_tasks=1, benchmark_id="test", fw=fw)
    assert isinstance(ui._backend, _PlainUI)
    with ui:
        parser = ToolEvalProgressParser({})
        ui.observe_task(0, parser.start())
        ui.observe_task(0, parser.feed(event()))
        ui.update_results_table(artifact("TC-01", "TC-02"))
        ui.update_results_table(artifact("TC-01", "TC-02"))
    output = capsys.readouterr().out
    assert output.count("Scenario") == 1
    assert output.count("TC-01") == output.count("TC-02") == 1
    assert "2 scenarios; 2 pass" in output and "\x1b[" not in output and "idx" not in output


def test_missing_stream_and_completed_artifact_render_final_rows(capsys):
    with _PlainUI(total_tasks=1, benchmark_id="saved", fw=ToolEvalBenchFramework()) as ui:
        ui.update_results_table(artifact("TC-01", "TC-02"))
    output = capsys.readouterr().out
    assert "TC-01" in output and "TC-02" in output and "Duration" in output


@pytest.mark.parametrize("ref,trials", [("v2.6.0", 1), ("v2.8.0", 1), ("v2.8.0", 2)])
def test_captured_upstream_protocol(ref, trials):
    from pathlib import Path

    parser = ToolEvalProgressParser({"trials": trials})
    path = Path(__file__).parent / "fixtures" / "tool_eval_bench" / f"{ref}-trials{trials}.jsonl"
    observed = [value for line in path.read_text().splitlines() if (value := parser.feed(line))]
    assert len(observed) == 4 * trials + 1
    results = [value for value in observed if value["metrics"].get("event") == "scenario_result"]
    assert results[-1]["current"] == 2 * trials
    assert results[-1]["total"] == (2 if trials == 1 else None)
    assert observed[-1]["stage"] == "finalization"
    assert all(value["status"] == "running" for value in observed)


def test_llama_benchy_shares_structured_task_feedback(tmp_path, monkeypatch):
    from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework
    from test_benchmark_scheduler import _make_state, _make_process_runner

    fw = LlamaBenchyFramework()
    tasks = fw.build_task_list({"depth": [0], "concurrency": [1], "pp": [128], "tg": [8], "runs": 1}, None)
    state = _make_state(tmp_path, n_tasks=1)
    state.framework = fw.framework_name
    callbacks, events = [], []
    monkeypatch.setattr("sparkrun.benchmarking.scheduler.run_benchmark_process", _make_process_runner([0]))
    emitter = _CallbackProgressEmitter(callbacks.append)
    with operation_scope(state=OperationState(events=events.append)):
        with emitter.schedule_progress(total_tasks=1, benchmark_id=state.benchmark_id, fw=fw, title="performance") as sink:
            result = run_schedule(
                fw, tasks, state, target_url="http://localhost/v1", model="test", timeout=5, task_events=sink, cache_dir=str(tmp_path)
            )
    assert result.success
    feedback = [e for e in events if e["type"] == "feedback"]
    assert [e["status"] for e in feedback] == ["running", "succeeded"]
    assert all(e["source"] == "benchmark.llama-benchy" and e["unit"] == "tasks" for e in feedback)
    assert [e["current"] for e in feedback] == [0, 1]
    assert len([e for e in callbacks if e.kind == "task_progress"]) == 2
    assert {"task_start", "task_end", "results_update"} <= {e.kind for e in callbacks}


@pytest.mark.parametrize("failure", ["timeout", "interrupt", "log_write"])
def test_process_failures_are_not_swallowed_by_observations(tmp_path, monkeypatch, failure):
    import subprocess

    def process(command, *, stdout, stderr, **kwargs):
        stderr(event() + "\n")
        if failure == "timeout":
            raise subprocess.TimeoutExpired("benchmark", 5)
        if failure == "interrupt":
            raise KeyboardInterrupt()
        raise OSError("log write failed")

    monkeypatch.setattr("sparkrun.benchmarking.scheduler.run_benchmark_process", process)
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            _schedule(tmp_path, monkeypatch)
    else:
        result, _, delivered = _schedule(tmp_path, monkeypatch)
        assert not result.success
        assert [e for e in delivered if e["type"] == "feedback"][-1]["status"] == "failed"
    state = BenchmarkRunState.load("bench_progress", str(tmp_path))
    assert not state.completed_indices
