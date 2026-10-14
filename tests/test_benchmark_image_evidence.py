"""Current deployment evidence must agree before fresh or resumed measurement."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from test_benchmark_startup_collection import bench_env as bench_env
from test_benchmark_api_contract import scheduled_env as scheduled_env


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("private_handle", [False, True])
@pytest.mark.parametrize("case", ["agree", "public_missing", "metadata_missing", "digest_alias", "public_conflict", "metadata_conflict"])
def test_measurement_accepts_only_consistent_image_evidence(scheduled_env, monkeypatch, resumed, private_handle, case):
    from sparkrun import api
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from sparkrun.benchmarking.scheduler import BenchTask
    from sparkrun.orchestration.job_metadata import derive_cluster_id

    env = scheduled_env
    cid = derive_cluster_id(env.recipe, ["localhost"])
    env.launch.cluster_id = env.run.return_value.workload_id = cid
    monkeypatch.setattr("sparkrun.api._benchmark._resolve_running_deployment", lambda *a, **kw: (["localhost"], True, cid))
    original = "original/image@sha256:" + "a" * 64 if case == "digest_alias" else env.launch.container_image
    env.launch.container_image = original
    env.run.return_value.container_image = original
    metadata = {
        "hosts": ["localhost"],
        "port": 8000,
        "recipe_state": env.recipe.__getstate__(),
        "overrides": {},
        "effective_container_image": original,
    }
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **kw: dict(metadata))
    env.fw.build_task_list.return_value = [BenchTask(0, "first"), BenchTask(1, "second")]
    env.fw.task_coverage_key.side_effect = lambda task: task.label
    env.fw.consolidated_coverage_keys.side_effect = lambda result: {row["label"] for row in result.get("rows", [])}
    first_attempt = resumed
    commands = []

    def command(*args, result_file, **kwargs):
        idx = int(Path(result_file).stem)
        commands.append((idx, first_attempt))
        if idx == 1 and first_attempt:
            return [sys.executable, "-c", "raise SystemExit(7)"]
        row = {"label": "first" if idx == 0 else "second", "speed": 42}
        return [sys.executable, "-c", "from pathlib import Path; Path(%r).write_text(%r)" % (result_file, json.dumps(row))]

    env.fw.build_benchmark_command.side_effect = command
    saved = artifact = state_path = None
    if resumed:
        with pytest.raises(api.SparkrunError, match="incomplete"):
            api.benchmark(env.options, sctx=env.sctx)
        state_path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
        saved = BenchmarkRunState.load(state_path.parent.name, str(env.sctx.config.cache_dir))
        artifact = (state_path.parent / "runs/000.json").read_bytes()
    first_attempt = False
    if not private_handle:
        env.run.return_value.launch_result = None
    if case == "public_missing":
        env.run.return_value.container_image = None
    elif case == "metadata_missing":
        metadata.pop("effective_container_image")
    elif case == "digest_alias":
        env.run.return_value.container_image = "alias/image@sha256:" + "a" * 64
        if private_handle:
            env.launch.container_image = "third/image@sha256:" + "a" * 64
    elif case == "public_conflict":
        env.run.return_value.container_image = "different/image:B"
    elif case == "metadata_conflict":
        metadata["effective_container_image"] = "different/image:B"

    options = replace(env.options, resume=api.ResumeMode.IF_EXISTS) if resumed else env.options
    if case.endswith("conflict"):
        before = list(commands)
        with pytest.raises(api.SparkrunError, match="Conflicting container image evidence"):
            api.benchmark(options, sctx=env.sctx)
        assert commands == before
        if resumed:
            state = BenchmarkRunState.load(state_path.parent.name, str(env.sctx.config.cache_dir))
            assert state.extras["container_image"] == saved.extras["container_image"] == original
            assert state.measurement_spec == saved.measurement_spec
            assert state.completed_indices == [0]
            assert (state_path.parent / "runs/000.json").read_bytes() == artifact
        else:
            assert not list(env.sctx.config.cache_dir.glob("benchmarks/bench_*/runs/*.json"))
    else:
        result = api.benchmark(options, sctx=env.sctx)
        assert result.success
        assert commands == ([(0, True), (1, True), (1, False)] if resumed else [(0, False), (1, False)])
        expected = original if resumed else env.launch.container_image if private_handle else original
        assert result.container_image == expected
