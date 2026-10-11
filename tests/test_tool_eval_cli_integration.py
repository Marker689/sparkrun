"""Opt-in pinned upstream CLI contract, using only a local fake model endpoint.

Run with SPARKRUN_TEST_TOOL_EVAL_CLI=1. uvx must be installed and able to resolve
both pinned refs (or have them cached). Normal unit tests replay captured JSONL.
"""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading

import pytest

from sparkrun.benchmarking.tool_eval_bench import ToolEvalBenchFramework
from sparkrun.benchmarking.tool_eval_progress import ToolEvalProgressParser


@contextmanager
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, data, content_type="application/json"):
            body = data.encode() if isinstance(data, str) else json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond({"object": "list", "data": [{"id": "fixture", "object": "model", "owned_by": "local"}]})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = request.get("messages", [])
            if request.get("tool_choice") == "none":
                message = {"role": "assistant", "content": "Pi is approximately 3.14159."}
            elif any(item.get("role") == "tool" for item in messages):
                message = {"role": "assistant", "content": "Berlin is 8 degrees Celsius and overcast."}
            else:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fixture",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Berlin","units":"celsius"}',
                            },
                        }
                    ],
                }
            reason = "tool_calls" if "tool_calls" in message else "stop"
            usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
            if request.get("stream"):
                if "tool_calls" in message:
                    message["tool_calls"][0]["index"] = 0
                chunks = [
                    {
                        "id": "fixture",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "fixture",
                        "choices": [{"index": 0, "delta": message, "finish_reason": None}],
                    },
                    {
                        "id": "fixture",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "fixture",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
                        "usage": usage,
                    },
                ]
                self.respond("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n", "text/event-stream")
            else:
                self.respond(
                    {
                        "id": "fixture",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "fixture",
                        "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                        "usage": usage,
                    }
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.skipif(os.environ.get("SPARKRUN_TEST_TOOL_EVAL_CLI") != "1", reason="opt-in upstream CLI contract")
@pytest.mark.parametrize("ref,trials", [("v2.6.0", 1), ("v2.8.0", 1), ("v2.8.0", 2)])
def test_real_pinned_cli_schema_and_stderr(tmp_path, ref, trials):
    fw = ToolEvalBenchFramework()
    args = {
        **fw.get_default_args(),
        "ref": ref,
        "trials": trials,
        "scenarios": ["TC-01", "TC-44"],
        "parallel": 2,
        "no_preflight": True,
        "no_probe_engine": True,
        "no_warmup": True,
        "skip_coherence": True,
        "backend": "vllm",
        "format": "openai",
    }
    if ref == "v2.8.0":
        args.update(
            header=["X-Fixture=one,two", "X-Second=value"],
            session_header="X-Session",
            variant_seed=42,
            system_prompt="Local fixture",
            include_held_out=True,
        )
    result_file = tmp_path / "suite.json"
    with endpoint() as target:
        command = fw.build_benchmark_command(target, "fixture", args, str(result_file))
        result = subprocess.run(command, text=True, capture_output=True, timeout=90, cwd=tmp_path)
    (tmp_path / "stderr.jsonl").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    parsed = fw.parse_results(result.stdout, result.stderr, str(result_file))
    assert parsed["json"]["schema_version"] == "1"
    assert parsed["json"]["tool_eval_bench_version"] == ref.lstrip("v")
    assert not fw.measured_nothing(parsed)
    assert {row["scenario_id"] for row in parsed["json"]["scores"]["scenario_results"]} == {"TC-01", "TC-44"}
    assert parsed["json"]["final_score"] == 100
    parser = ToolEvalProgressParser(args)
    observations = [value for line in result.stderr.splitlines() if (value := parser.feed(line))]
    completed = [e for e in observations if e["metrics"].get("event") == "scenario_result"]
    assert len(completed) == 2 * trials
    assert completed[-1]["current"] == 2 * trials
    assert completed[-1]["total"] == (2 if trials == 1 else None)
    assert observations[-1]["stage"] == "finalization" and observations[-1]["status"] == "running"
    if capture := os.environ.get("SPARKRUN_TOOL_EVAL_CAPTURE_DIR"):
        # Developer-only protocol capture, stripping machine paths from the event.
        records = []
        for line in result.stderr.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("event") in {"scenario_start", "scenario_result", "benchmark_complete"}:
                if "json_file" in record:
                    record["json_file"] = "suite.json"
                records.append(record)
        destination = Path(capture)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{ref}-trials{trials}.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
