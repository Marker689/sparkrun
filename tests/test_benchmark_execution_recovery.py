"""Process lifetime, completed-result reuse, and immutable measurement inputs."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from sparkrun.api import benchmark, resume_benchmark, BenchmarkFailed, SparkrunError, ResumeMode
from sparkrun.benchmarking._credentials import BenchmarkCredentials, resolve_credentials
from sparkrun.benchmarking._process import run_benchmark_process
from sparkrun.benchmarking.run_state import BenchmarkRunState, hold_state_dir
from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration
from test_benchmark_startup_collection import bench_env as bench_env
from test_benchmark_api_contract import scheduled_env as scheduled_env
from test_benchmark_auth_and_state import _interrupt_measurement


@pytest.fixture(autouse=True)
def integration_registry(monkeypatch):
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})


def _worker_command(pid_file, *, exit_wrapper=False):
    code = (
        "import subprocess, sys, time; from pathlib import Path; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "Path(%r).write_text(str(p.pid)); print('worker started', flush=True); "
        "time.sleep(%d)" % (str(pid_file), 0 if exit_wrapper else 30)
    )
    return [sys.executable, "-c", code]


def _assert_dead(pid_file):
    pid = int(pid_file.read_text())
    # A grandchild may briefly remain as an init-owned zombie after group kill.
    for _ in range(100):
        path = Path("/proc") / str(pid) / "stat"
        try:
            if path.read_text().split()[2] == "Z":
                return
        except (FileNotFoundError, ProcessLookupError):
            return  # the process may exit while /proc serves the read
        time.sleep(0.01)
    pytest.fail("benchmark worker is still running")


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("authenticated", [False, True])
def test_api_deadline_owns_worker_tree_and_releases_cleanup(request, scheduled, authenticated, tmp_path):
    env = request.getfixturevalue("scheduled_env" if scheduled else "bench_env")
    pid_file = tmp_path / "worker.pid"
    env.fw.build_benchmark_command.side_effect = lambda *a, **kw: _worker_command(pid_file, exit_wrapper=True)
    env.sctx.variables.set("RECOVERY_KEY", "process-secret")
    start = time.monotonic()
    with pytest.raises(BenchmarkFailed):
        benchmark(
            replace(env.options, timeout=1, exit_on_first_fail=True, api_key_env="RECOVERY_KEY" if authenticated else None),
            sctx=env.sctx,
        )
    assert time.monotonic() - start < 4
    env.stop.assert_called_once()
    _assert_dead(pid_file)
    if scheduled:
        state_file = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
        state = BenchmarkRunState.load(state_file.parent.name, str(env.sctx.config.cache_dir), strict=True)
        assert not state.completed_indices and state.failed_indices == [0]
        with hold_state_dir(state.benchmark_id, str(env.sctx.config.cache_dir)):
            pass


@pytest.mark.parametrize("error", [KeyboardInterrupt, OSError])
def test_process_callback_failure_terminates_workers(tmp_path, error):
    pid_file = tmp_path / "worker.pid"

    def output(line):
        raise error("consumer stopped")

    with pytest.raises(error, match="consumer stopped"):
        run_benchmark_process(_worker_command(pid_file), timeout=3, credentials=BenchmarkCredentials(), stdout=output)
    _assert_dead(pid_file)


def test_process_quiet_timeout_is_bounded():
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_benchmark_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=0.2,
            credentials=BenchmarkCredentials("command-secret"),
            stdout=lambda _: None,
        )
    assert time.monotonic() - started < 2


def test_single_task_drains_stderr_while_stdout_is_open(bench_env):
    env = bench_env
    original = env.fw.build_benchmark_command.side_effect

    def command(*args, **kwargs):
        cmd = original(*args, **kwargs)
        return [*cmd[:-1], "import sys; sys.stderr.write('x' * 1048576); " + cmd[-1]]

    env.fw.build_benchmark_command.side_effect = command
    result = benchmark(replace(env.options, timeout=3), sctx=env.sctx)
    assert result.success and result.results == env.rows
    log = next(Path(result.state_dir).glob("runs/*.log"))
    assert log.read_text() == "x" * 1048576


def test_output_redaction_covers_split_unicode_and_multiline_secret():
    from sparkrun.benchmarking._process import _Output

    output = []
    redactor = _Output(output.append, BenchmarkCredentials("sécret\nvalue"))
    for byte in "before sécret\nvalue after sécret\nvalue\nlast sé".encode():
        redactor.feed(bytes([byte]))
    redactor.feed(b"", final=True)
    assert "".join(output) == "before [REDACTED] after [REDACTED]\nlast sé"


@pytest.mark.parametrize("publication", [False, True])
def test_complete_reuse_needs_no_execution_and_matches_resume_by_id(scheduled_env, monkeypatch, publication):
    env = scheduled_env
    views = []
    register_benchmark_integration(BenchmarkIntegration("audit", on_complete=lambda ctx: views.append(ctx.result)))
    env.sctx.variables.set("RECOVERY_KEY", "expired-later")
    options = replace(env.options, api_key_env="RECOVERY_KEY", integrations={"audit": {}} if publication else {})
    first = benchmark(options, sctx=env.sctx)
    assert first.workload_id == env.run.return_value.workload_id
    before = (Path(first.state_dir) / "result.yaml").read_bytes()
    env.sctx.variables.set("RECOVERY_KEY", "")
    for operation in (env.run, env.stop, env.endpoint, env.probe, env.fw.build_benchmark_command, env.fw.check_prerequisites):
        operation.reset_mock()
        operation.side_effect = AssertionError("completed measurements must not execute")
    monkeypatch.setattr("sparkrun.api._benchmark.resolve_credentials", Mock(side_effect=AssertionError("no auth lookup")))
    implicit = benchmark(replace(options, resume=ResumeMode.IF_EXISTS), sctx=env.sctx)
    explicit = resume_benchmark(first.benchmark_id, sctx=env.sctx)
    assert implicit.results == explicit.results == first.results
    assert implicit.metadata == explicit.metadata
    assert implicit.outputs == explicit.outputs == first.outputs
    assert implicit.resumed and explicit.resumed
    assert implicit.already_complete is (not publication)
    assert (Path(first.state_dir) / "result.yaml").read_bytes() == before
    if publication:
        assert len(views) == 3
        assert views[0].completed_at == views[1].completed_at == views[2].completed_at


def test_saved_recipe_preserves_in_memory_inputs_and_actual_job_id(scheduled_env, monkeypatch):
    env = scheduled_env
    env.recipe.model = "edited/original-model"
    env.recipe.metadata["model_dtype"] = "bfloat16"
    env.recipe.defaults["custom_arg"] = 42
    env.launch.overrides = {"max_model_len": 4096}  # fixture launch reflects requested serving options
    path = _interrupt_measurement(env, recipe=env.recipe, overrides={"max_model_len": 4096})
    state = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir), strict=True)
    assert state.cluster_id == env.run.return_value.workload_id
    env.recipe.model = "changed/source-model"
    env.recipe.metadata["model_dtype"] = "float16"
    monkeypatch.setattr("sparkrun.core.resolve.load_recipe", Mock(side_effect=AssertionError("saved recipe must be self contained")))
    result = resume_benchmark(path.parent.name, sctx=env.sctx)
    assert result.success and result.benchmark_id == path.parent.name
    assert env.fw.build_benchmark_command.call_args.args[1] == "edited/original-model"
    exported = yaml.safe_load(Path(result.outputs["yaml"]).read_text())["sparkrun_benchmark"]
    assert exported["model"]["dtype"] == "bfloat16"
    assert "edited/original-model" in Path(result.outputs["yaml"]).read_text()
    assert yaml.safe_load(exported["recipe"]["text"])["defaults"]["max_model_len"] == 4096


@pytest.mark.parametrize("changed", [False, True])
def test_legacy_resume_verifies_original_identity_before_hooks(scheduled_env, changed):
    env = scheduled_env
    path = _interrupt_measurement(env)
    data = yaml.safe_load(path.read_text())
    del data["measurement_spec"]
    path.write_text(yaml.safe_dump(data))
    before = path.read_bytes()
    bound = Mock()
    register_benchmark_integration(BenchmarkIntegration("audit", on_bind=bound))
    env.fw.build_benchmark_command.reset_mock()
    if changed:
        env.recipe.defaults["dtype"] = "changed"
        with pytest.raises(SparkrunError, match="verify the original benchmark recipe"):
            resume_benchmark(path.parent.name, sctx=env.sctx, integrations={"audit": {}})
        bound.assert_not_called()
        env.fw.build_benchmark_command.assert_not_called()
        assert path.read_bytes() == before
    else:
        assert resume_benchmark(path.parent.name, sctx=env.sctx, export_files=False).success


@pytest.mark.parametrize("change", ["model", "runtime", "hosts", "spec_model", "spec_version", "spec_fingerprint", "job_config"])
def test_resume_rejects_changed_measurement_or_running_job_before_hooks(scheduled_env, monkeypatch, change):
    env = scheduled_env
    meta = {
        "hosts": ["localhost"],
        "port": 8000,
        "model": env.recipe.model,
        "runtime": env.recipe.runtime,
        "recipe_state": env.recipe.__getstate__(),
        "overrides": {},
    }
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **kw: meta)
    path = _interrupt_measurement(env)
    data = yaml.safe_load(path.read_text())
    if change == "spec_model":
        data["measurement_spec"]["recipe"]["model"] = "different/model"
    elif change == "spec_version":
        data["measurement_spec"]["version"] = True
    elif change == "spec_fingerprint":
        data["measurement_spec"]["fingerprint"] = "bad"
    elif change == "job_config":
        meta["overrides"]["dtype"] = "different"
    else:
        meta[change] = ["other-host"] if change == "hosts" else "different"
    path.write_text(yaml.safe_dump(data))
    before = path.read_bytes()
    bound = Mock()
    register_benchmark_integration(BenchmarkIntegration("audit", on_bind=bound))
    env.fw.build_benchmark_command.reset_mock()
    with pytest.raises(SparkrunError):
        resume_benchmark(path.parent.name, sctx=env.sctx, integrations={"audit": {}})
    bound.assert_not_called()
    env.fw.build_benchmark_command.assert_not_called()
    assert path.read_bytes() == before


def test_effective_job_configuration_can_rotate_credentials(scheduled_env, monkeypatch):
    env = scheduled_env
    env.recipe.defaults["api_key"] = "old-job-secret"
    env.launch.overrides = {"platform_default": True}
    meta = {"hosts": ["localhost"], "port": 8000, "recipe_state": env.recipe.__getstate__(), "overrides": dict(env.launch.overrides)}
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.load_job_metadata", lambda *a, **kw: meta)
    monkeypatch.setattr(env.launch.runtime, "resolve_api_key", lambda recipe, overrides: recipe.defaults.get("api_key"))
    path = _interrupt_measurement(env)
    meta["recipe_state"]["defaults"]["api_key"] = "new-job-secret"
    assert resume_benchmark(path.parent.name, sctx=env.sctx, export_files=False).success
    assert env.fw.build_benchmark_command.call_args.args[2]["api_key"] == "new-job-secret"
    assert "job-secret" not in path.read_text()


def test_arbitrary_recipe_named_strings_round_trip_all_owners(scheduled_env):
    env = scheduled_env
    payload = {"recipe_yaml": "an unfinished snippet: [", "effective_recipe_text": "another: ["}
    env.rows.update(payload)
    views = []

    def bind(ctx):
        ctx.data.update(payload)

    register_benchmark_integration(BenchmarkIntegration("audit", on_bind=bind, on_complete=lambda ctx: views.append(ctx.data.copy())))
    result = benchmark(replace(env.options, integrations={"audit": {}}, state_extras={"example.experiment": payload}), sctx=env.sctx)
    again = resume_benchmark(result.benchmark_id, sctx=env.sctx)
    state = BenchmarkRunState.load(result.benchmark_id, str(env.sctx.config.cache_dir), strict=True)
    assert state.extras["example.experiment"] == payload
    assert views == [payload, payload]
    assert again.results["rows"][0]["recipe_yaml"] == payload["recipe_yaml"]


@pytest.mark.parametrize("document", ["api_key: legacy-secret\nbroken: [", "legacy-secret", 123])
def test_arena_malformed_legacy_document_fails_without_secret_in_diagnostic(scheduled_env, monkeypatch, document):
    from sparkrun.plugins.sparkarena.integration import bind

    env = scheduled_env
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})
    first = benchmark(env.options, sctx=env.sctx)
    path = Path(first.state_dir) / "state.yaml"
    data = yaml.safe_load(path.read_text())
    data["extras"]["benchmark_integrations"] = {"arena": {"settings": {"local_test": True}, "data": {"effective_recipe_text": document}}}
    path.write_text(yaml.safe_dump(data))
    publish = Mock()
    register_benchmark_integration(BenchmarkIntegration("arena", on_bind=bind, on_complete=publish))
    with pytest.raises(SparkrunError, match="Saved Arena recipe is invalid") as error:
        resume_benchmark(first.benchmark_id, sctx=env.sctx)
    assert "legacy-secret" not in str(error.value)
    publish.assert_not_called()


@pytest.mark.parametrize("parser_available", [False, True])
def test_missing_credential_error_is_independent_of_optional_dotenv(scheduled_env, monkeypatch, parser_available):
    env = scheduled_env
    if parser_available:
        monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(dotenv_values=lambda *a, **kw: {}))
    else:
        monkeypatch.setitem(sys.modules, "dotenv", None)
    with pytest.raises(SparkrunError, match="RECOVERY_MISSING.*missing or empty"):
        benchmark(replace(env.options, api_key_env="RECOVERY_MISSING"), sctx=env.sctx)
    env.run.assert_not_called()


def test_available_dotenv_source_still_resolves_credential(scheduled_env, monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("RECOVERY_DOTENV=from-file")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(dotenv_values=lambda *a, **kw: {"RECOVERY_DOTENV": "from-file"}))
    assert resolve_credentials(scheduled_env.sctx.variables, api_key_env="RECOVERY_DOTENV").api_key == "from-file"


@pytest.mark.parametrize("entrypoint", ["implicit", "by_id"])
def test_completed_publication_failure_keeps_results_and_releases_lock(scheduled_env, entrypoint):
    from sparkrun.api import BenchmarkIntegrationFailed

    env = scheduled_env
    publish = Mock()
    register_benchmark_integration(BenchmarkIntegration("audit", on_complete=publish))
    options = replace(env.options, integrations={"audit": {}})
    first = benchmark(options, sctx=env.sctx)
    publish.side_effect = RuntimeError("publication unavailable")
    env.run.reset_mock()
    env.stop.reset_mock()
    env.fw.build_benchmark_command.reset_mock()
    with pytest.raises(BenchmarkIntegrationFailed, match="publication unavailable") as caught:
        if entrypoint == "implicit":
            benchmark(replace(options, resume=ResumeMode.IF_EXISTS), sctx=env.sctx)
        else:
            resume_benchmark(first.benchmark_id, sctx=env.sctx)
    assert caught.value.result.success and caught.value.result.results == first.results
    env.run.assert_not_called()
    env.stop.assert_not_called()
    env.fw.build_benchmark_command.assert_not_called()
    with hold_state_dir(first.benchmark_id, str(env.sctx.config.cache_dir)):
        pass


def test_implicit_legacy_backfill_captures_identity_before_image_pinning(scheduled_env):
    env = scheduled_env
    path = _interrupt_measurement(env)
    data = yaml.safe_load(path.read_text())
    del data["measurement_spec"]
    data["extras"]["container_image_sha"] = "test/image@sha256:abcd"
    path.write_text(yaml.safe_dump(data))
    command = env.fw.build_benchmark_command.side_effect
    env.fw.build_benchmark_command.side_effect = lambda *a, **kw: [sys.executable, "-c", "raise SystemExit(1)"]
    with pytest.raises(BenchmarkFailed, match="incomplete"):
        benchmark(replace(env.options, resume=ResumeMode.IF_EXISTS), sctx=env.sctx)
    state = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir), strict=True)
    assert state.measurement_spec
    assert env.run.call_args.args[0].overrides["image"] == "test/image@sha256:abcd"
    env.fw.build_benchmark_command.side_effect = command
    assert resume_benchmark(path.parent.name, sctx=env.sctx, export_files=False).success


@pytest.mark.parametrize("width", [1, 2, 7, 65536])
@pytest.mark.parametrize("ending", ["", "\n", "\r", "\r\n"])
def test_output_framing_preserves_payload_across_reads(width, ending):
    from sparkrun.benchmarking._process import _Output

    payload = "first\u2028second\u2029third\x85fourth\v\ffifth\r\nnext\rlast\nsecret é" + ending
    output = []
    reader = _Output(output.append, BenchmarkCredentials("secret é"))
    data = payload.encode()
    for start in range(0, len(data), width):
        reader.feed(data[start : start + width])
    reader.feed(b"", final=True)
    assert "".join(output) == payload.replace("secret é", "[REDACTED]")
    assert not reader.line


def test_process_preserves_unicode_separators_on_both_streams():
    payload = "first\u2028second\u2029third\x85fourth\v\ffifth\r\nnext\rlast\nsecret é"
    stdout, stderr = [], []
    code = "import os; data = %r; os.write(1, data); os.write(2, data)" % payload.encode()
    rc = run_benchmark_process(
        [sys.executable, "-c", code],
        timeout=3,
        credentials=BenchmarkCredentials("secret é"),
        stdout=stdout.append,
        stderr=stderr.append,
    )
    assert rc == 0
    assert "".join(stdout) == "".join(stderr) == payload.replace("secret é", "[REDACTED]")


def test_public_api_parses_json_with_literal_unicode_separators(bench_env):
    env = bench_env
    rows = {"text": "first\u2028second\u2029third\x85fourth"}
    env.fw.build_benchmark_command.side_effect = lambda *args, result_file, **kw: [
        sys.executable,
        "-c",
        "from pathlib import Path; Path(%r).write_text(%r)" % (result_file, json.dumps(rows, ensure_ascii=False)),
    ]
    result = benchmark(env.options, sctx=env.sctx)
    assert result.success and result.results == rows
    env.stop.assert_called_once()


@pytest.mark.parametrize("field", ["overrides", "bench_args", "integrations", "state_extras"])
@pytest.mark.parametrize("value", [None, [], [("tensor_parallel", 4)], "bad", {1: "bad"}])
def test_bad_benchmark_data_fails_before_hooks_or_launch(bench_env, field, value):
    env = bench_env
    prepare = Mock(side_effect=lambda defaults, context: defaults)
    register_benchmark_integration(BenchmarkIntegration("audit", prepare=prepare))
    options = replace(env.options, integrations={"audit": {}})
    with pytest.raises(SparkrunError, match="BenchmarkOptions." + field) as caught:
        benchmark(replace(options, **{field: value}), sctx=env.sctx)
    assert isinstance(caught.value.__cause__, TypeError)
    prepare.assert_not_called()
    env.run.assert_not_called()
    env.fw.build_benchmark_command.assert_not_called()


@pytest.mark.parametrize("settings", [None, [], "bad"])
def test_integration_settings_reject_non_mappings_before_hooks(bench_env, settings):
    prepare = Mock()
    register_benchmark_integration(BenchmarkIntegration("audit", prepare=prepare))
    with pytest.raises(SparkrunError, match="integrations.audit must be a mapping"):
        benchmark(replace(bench_env.options, integrations={"audit": settings}), sctx=bench_env.sctx)
    prepare.assert_not_called()
    bench_env.run.assert_not_called()


def test_benchmark_mapping_inputs_are_detached_before_hooks(scheduled_env):
    from types import MappingProxyType

    env = scheduled_env
    overrides, args, settings, extras = {"tensor_parallel": 1}, {"custom": [7]}, {"tags": ["original"]}, {"app.tags": ["original"]}

    def prepare(defaults, context):
        assert defaults.bench_args["custom"] == (7,)
        assert context.settings == {"tags": ["original"]}
        overrides["tensor_parallel"] = 9
        args["custom"].append(99)
        settings["tags"].append("changed")
        extras["app.tags"].append("changed")
        return defaults

    register_benchmark_integration(BenchmarkIntegration("audit", prepare=prepare))
    options = replace(
        env.options,
        recipe=env.recipe,
        overrides=MappingProxyType(overrides),
        bench_args=MappingProxyType(args),
        integrations=MappingProxyType({"audit": MappingProxyType(settings)}),
        state_extras=MappingProxyType(extras),
    )
    result = benchmark(options, sctx=env.sctx)
    assert result.success
    assert env.run.call_args.args[0].overrides["tensor_parallel"] == 1
    assert result.metadata["bench_args"]["custom"] == [7]
    state = BenchmarkRunState.load(result.benchmark_id, str(env.sctx.config.cache_dir), strict=True)
    assert state.extras["app.tags"] == ["original"]
    assert state.extras["benchmark_integrations"]["audit"]["settings"] == {"tags": ["original"]}


@pytest.mark.parametrize("settings", [[], {"audit": []}, {1: {}}])
def test_resume_rejects_invalid_integration_inputs_before_hooks(scheduled_env, settings):
    env = scheduled_env
    path = _interrupt_measurement(env)
    before = path.read_bytes()
    bind = Mock()
    register_benchmark_integration(BenchmarkIntegration("audit", on_bind=bind))
    env.fw.build_benchmark_command.reset_mock()
    with pytest.raises(SparkrunError, match="BenchmarkOptions.integrations"):
        resume_benchmark(path.parent.name, sctx=env.sctx, integrations=settings)
    bind.assert_not_called()
    env.fw.build_benchmark_command.assert_not_called()
    assert path.read_bytes() == before


def test_resume_and_initial_measurement_share_prerequisite_failure(scheduled_env):
    env = scheduled_env
    path = _interrupt_measurement(env)
    before = path.read_bytes()
    bind = Mock()
    register_benchmark_integration(BenchmarkIntegration("audit", on_bind=bind))
    env.fw.check_prerequisites.reset_mock()
    env.fw.check_prerequisites.return_value = ["missing measurement tool"]
    for operation in (env.run, env.stop, env.fw.build_benchmark_command):
        operation.reset_mock()
    failures = []
    for execute in (
        lambda: resume_benchmark(path.parent.name, sctx=env.sctx, integrations={"audit": {}}),
        lambda: benchmark(replace(env.options, integrations={"audit": {}}, bench_args={"variant": "fresh"}), sctx=env.sctx),
    ):
        with pytest.raises(BenchmarkFailed, match="missing measurement tool") as error:
            execute()
        failures.append(str(error.value))
    assert failures[0] == failures[1]
    assert env.fw.check_prerequisites.call_count == 2
    bind.assert_not_called()
    for operation in (env.run, env.stop, env.fw.build_benchmark_command):
        operation.assert_not_called()
    assert path.read_bytes() == before


def test_previews_skip_measurement_prerequisites(scheduled_env):
    env = scheduled_env
    path = _interrupt_measurement(env)
    before = path.read_bytes()
    env.fw.check_prerequisites.reset_mock()
    env.fw.check_prerequisites.side_effect = AssertionError("preview must not check execution prerequisites")
    assert resume_benchmark(path.parent.name, sctx=env.sctx, dry_run=True).success
    assert benchmark(replace(env.options, dry_run=True), sctx=env.sctx).success
    env.fw.check_prerequisites.assert_not_called()
    assert path.read_bytes() == before
