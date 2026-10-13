"""UTF-8 artifacts through real measurement, API exports, and resume."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from sparkrun.api import BenchmarkFailed, benchmark, resume_benchmark
from sparkrun.benchmarking.run_state import BenchmarkRunState
from sparkrun.benchmarking.scheduler import BenchTask
from test_benchmark_startup_collection import bench_env as bench_env
from test_benchmark_api_contract import scheduled_env as scheduled_env


def test_benchmark_persistence_in_non_utf8_process(tmp_path):
    """Use a real locale boundary; also exercised on native Windows/macOS CI."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(Path(__file__).resolve()),
            "-k",
            "encoding_scenario",
            "--basetemp",
            str(tmp_path / "child"),
        ],
        env={**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "SPARKRUN_ENCODING_SCENARIO": "1"},
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, (result.stdout + result.stderr).decode("utf-8", errors="replace")


@pytest.mark.skipif(os.environ.get("SPARKRUN_ENCODING_SCENARIO") != "1", reason="run inside the non-UTF-8 controller subprocess")
@pytest.mark.parametrize("partial", [False, True])
def test_api_encoding_scenario(scheduled_env, partial):
    env = scheduled_env
    text = "café 世界 🚀"
    rows = {"speed": 42, "text": text}
    env.fw.build_task_list.return_value = [BenchTask(i, "task", {"slot": i}, {"slot": i}) for i in range(2)]
    env.fw.consolidated_coverage_keys.return_value = None
    env.fw.consolidated_coverage_keys.side_effect = None
    fail = partial

    def command(url, model, args, *, result_file):
        code = (
            "import sys; from pathlib import Path; "
            f"print({ascii(text)}, flush=True); "
            f"Path({str(result_file)!r}).write_bytes({json.dumps(rows, ensure_ascii=False).encode('utf-8')!r}); "
            f"sys.exit({1 if fail and args['slot'] == 1 else 0})"
        )
        return [sys.executable, "-c", code]

    env.fw.build_benchmark_command.side_effect = command
    env.fw.parse_results.side_effect = lambda stdout, stderr, **kw: {"json": json.loads(stdout), "csv": "text\n" + text + "\n"}
    options = replace(env.options, state_extras={"test.note": text})
    if partial:
        with pytest.raises(BenchmarkFailed, match="incomplete"):
            benchmark(options, sctx=env.sctx)
        state_path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
        state = BenchmarkRunState.load(state_path.parent.name, str(env.sctx.config.cache_dir), strict=True)
        assert state.completed_indices == [0] and state.failed_indices == [1]
        # Valid literal Unicode YAML must reload, not only ASCII-escaped YAML.
        state_path.write_text(yaml.safe_dump(yaml.safe_load(state_path.read_text(encoding="utf-8")), allow_unicode=True), encoding="utf-8")
        fail = False
        env.fw.build_benchmark_command.reset_mock()
        result = resume_benchmark(state.benchmark_id, sctx=env.sctx)
        assert env.fw.build_benchmark_command.call_count == 1
    else:
        result = benchmark(options, sctx=env.sctx)

    assert result.success and result.results["json"] == {"rows": [rows, rows]}
    state_dir = Path(result.state_dir)
    for log in state_dir.glob("runs/*.log"):
        assert text in log.read_text(encoding="utf-8")
    assert len(list(state_dir.glob("runs/*.log"))) == 2
    assert Path(result.outputs["csv"]).read_text(encoding="utf-8") == "text\n" + text + "\n"
    assert json.loads(Path(result.outputs["json"]).read_text(encoding="utf-8")) == result.results["json"]
    assert yaml.safe_load(Path(result.outputs["yaml"]).read_text(encoding="utf-8"))["sparkrun_benchmark"]["results"] == result.results
    state = BenchmarkRunState.load(result.benchmark_id, str(env.sctx.config.cache_dir), strict=True)
    assert state.extras["test.note"] == text

    (state_dir / "result.yaml").write_text(yaml.safe_dump(result.results, allow_unicode=True), encoding="utf-8")
    env.fw.build_benchmark_command.side_effect = AssertionError("complete resume must not remeasure")
    reused = resume_benchmark(result.benchmark_id, sctx=env.sctx)
    assert reused.success and reused.resumed and reused.results == result.results


@pytest.mark.skipif(os.environ.get("SPARKRUN_ENCODING_SCENARIO") != "1", reason="run inside the non-UTF-8 controller subprocess")
def test_framework_readers_encoding_scenario(tmp_path):
    from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework
    from sparkrun.benchmarking.tool_eval_bench import ToolEvalBenchFramework

    data = {"text": "café 世界 🚀"}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for framework in (LlamaBenchyFramework(), ToolEvalBenchFramework()):
        assert framework.parse_results("", "", result_file=str(path))["json"] == data
