"""Lazy, bounded plugin evidence shares capture ownership and cannot change work."""

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event, Thread
from unittest.mock import Mock

import pytest

from sparkrun import api
from sparkrun.core.operation import OperationState
from sparkrun.core.installed_plugins import PluginConflictError
from sparkrun.core.plugin_identity import PluginIdentity, plugin_registration
from sparkrun.core.registration import registry_transaction
from sparkrun.diagnostics import DiagnosticSection, register_plugin_diagnostics
from sparkrun.diagnostics.plugin import DISABLED
from test_api_diagnostics import sctx as sctx, records


def register(name="example", sections=None):
    with plugin_registration(PluginIdentity(name, name, ("external", name, None))):
        return register_plugin_diagnostics(sections or [DiagnosticSection("state"), DiagnosticSection("result", priority="terminal")])


def test_disabled_reporter_never_evaluates_or_allocates_capture(sctx, monkeypatch):
    declaration = register()
    builder = Mock(side_effect=AssertionError("disabled builder"))
    sink = Mock(side_effect=AssertionError("disabled sink"))
    monkeypatch.setattr("sparkrun.diagnostics._plugin_sink.PluginSink", sink)
    assert sctx.operation is None
    reporter = declaration.bind(sctx=sctx)
    for _ in range(100):
        assert declaration.bind(sctx=sctx) is reporter is DISABLED
        assert not reporter.enabled and not reporter.snapshot("state", builder)
    assert sctx.operation is None
    builder.assert_not_called()
    sink.assert_not_called()


def test_registration_validation_and_rollback(sctx, tmp_path):
    with pytest.raises(ValueError, match="loader-owned"):
        register_plugin_diagnostics([DiagnosticSection("state")])
    with pytest.raises(ValueError):
        DiagnosticSection("host/variable")
    with pytest.raises(ValueError):
        DiagnosticSection("state", schema_version=True)
    with pytest.raises(ValueError):
        register(sections=[DiagnosticSection("same"), DiagnosticSection("same")])
    first = register("stable")
    assert register("stable") is first
    with pytest.raises(PluginConflictError):
        register("stable", [DiagnosticSection("different")])
    with pytest.raises(RuntimeError):
        with registry_transaction():
            stale = register("stale")
            raise RuntimeError("registration failed")
    with api.capture_diagnostics(tmp_path / "rollback.ndjson", sctx=sctx):
        assert stale.bind(sctx=sctx) is DISABLED
        assert first.bind(sctx=sctx).enabled


def test_records_are_owned_versioned_and_redacted_without_feedback_duplication(sctx, tmp_path):
    declaration = register(sections=[DiagnosticSection("state", schema_version=3)])
    path = tmp_path / "plugin.ndjson"
    events = []
    sctx.operation = OperationState(events=events.append)
    with api.capture_diagnostics(path, sctx=sctx) as capture:
        reporter = declaration.bind(sctx=sctx, activity_id="existing-activity", workload_id="workload")
        sctx.operation.secrets.add("late-secret")
        builder = Mock(return_value={"api_key": "private", "observations": ["Bearer token-value", "late-secret"]})
        assert reporter.snapshot("state", builder, host="host-a", stage="verify")
        assert "plugin_diagnostic" in path.read_text()  # flushed immediately
    assert not capture.result.partial
    assert not reporter.enabled and not reporter.snapshot("state", builder)
    assert reporter._sink.writer is reporter._sink.state is None
    builder.assert_called_once()
    data = records(path)
    entry = next(row for row in data if row["_type"] == "plugin_diagnostic")
    assert entry["plugin"] == "example" and entry["section"] == "state" and entry["schema_version"] == 3
    assert entry["operation_id"] == sctx.operation.id and entry["activity_id"] == "existing-activity"
    assert entry["workload_id"] == "workload" and entry["host"] == "host-a"
    assert entry["data"]["api_key"] == "[redacted]"
    for secret in ("private", "token-value", "late-secret"):
        assert secret not in path.read_text()
    assert len([row for row in data if row["_type"] == "plugin_diagnostics_metadata"]) == 1
    assert len([row for row in data if row["_type"] == "plugin_diagnostics_summary"]) == 1
    assert events == []


@pytest.mark.parametrize(
    "payload",
    [
        {"huge": "x" * 100_000},
        {"list": [0] * 100_000},
        {"number": 1 << 10_000},
        {"number": float("nan")},
        {"number": float("inf")},
        {"object": object()},
        {"generator": (x for x in range(10))},
        {str(x): x for x in range(65)},
    ],
)
def test_invalid_payloads_are_bounded_and_do_not_fail_work(sctx, tmp_path, payload):
    declaration = register()
    path = tmp_path / "invalid.ndjson"
    with api.capture_diagnostics(path, sctx=sctx) as capture:
        reporter = declaration.bind(sctx=sctx)
        assert not reporter.snapshot("state", lambda: payload)
    assert capture.result.partial
    assert records(path)[-1]["operation_success"] is True
    assert "plugin_diagnostic" not in [row["_type"] for row in records(path)]


def test_cycles_and_hostile_objects_are_not_formatted(sctx, tmp_path):
    class Hostile:
        def __str__(self):
            pytest.fail("must not stringify rejected payload")

    class HostileError(Exception):
        def __str__(self):
            pytest.fail("must not stringify producer error")

    cyclic = {}
    cyclic["cycle"] = cyclic
    with api.capture_diagnostics(tmp_path / "hostile.ndjson", sctx=sctx):
        reporter = register().bind(sctx=sctx)
        for value in (cyclic, {"hostile": Hostile()}):
            assert not reporter.snapshot("state", lambda value=value: value)
        assert not reporter.snapshot("state", Mock(side_effect=HostileError()))


def test_admission_stops_builders_and_reserves_terminal_count(sctx, tmp_path):
    declaration = register(
        sections=[*(DiagnosticSection("phase%d" % i) for i in range(4)), DiagnosticSection("result", priority="terminal")]
    )
    build = Mock(return_value={"value": 1})
    path = tmp_path / "limited.ndjson"
    with api.capture_diagnostics(path, sctx=sctx) as capture:
        reporter = declaration.bind(sctx=sctx)
        for section in ("phase0", "phase1", "phase2"):
            for _ in range(32):
                assert reporter.snapshot(section, build)
        for _ in range(100):
            assert not reporter.snapshot("phase3", build)
        assert build.call_count == 96
        assert reporter.snapshot("result", build)
    assert capture.result.partial and capture.result.truncated
    summary = next(row for row in records(path) if row["_type"] == "plugin_diagnostics_summary")
    assert summary["sections"]["phase3"]["dropped"] == 100
    assert len(capture.result.sections) < 10


def test_invalid_attempts_are_capped_before_reconstruction(sctx, tmp_path):
    build = Mock(return_value={"bad": object()})
    with api.capture_diagnostics(tmp_path / "attempts.ndjson", sctx=sctx):
        reporter = register().bind(sctx=sctx)
        for _ in range(100):
            assert not reporter.snapshot("state", build)
    assert build.call_count == 32


def test_payload_byte_reservations_are_shared_across_threads(sctx, tmp_path):
    declaration = register(sections=[DiagnosticSection("phase%d" % i, priority="terminal") for i in range(16)])
    path = tmp_path / "parallel.ndjson"
    with api.capture_diagnostics(path, sctx=sctx) as capture:
        reporter = declaration.bind(sctx=sctx)
        with ThreadPoolExecutor(8) as executor:
            accepted = list(executor.map(lambda i: reporter.snapshot("phase%d" % (i % 16), lambda: {"text": "x" * 2048}), range(150)))
        sink = reporter._sink
        assert sink.budget.used <= sink.budget.limit
        assert all(p.budget.used <= p.budget.limit for p in sink.participants.values())
    assert 0 < sum(accepted) < 150
    plugin_records = [row for row in records(path) if row["_type"].startswith("plugin_diagnostic")]
    assert sum(len((json.dumps(row) + "\n").encode()) for row in plugin_records) <= 256 * 1024
    assert capture.result.partial


def test_capture_close_does_not_wait_for_builder_or_write_late(sctx, tmp_path):
    entered, release = Event(), Event()
    results = []
    path = tmp_path / "closed.ndjson"

    def build():
        entered.set()
        assert release.wait(5)
        return {"late": True}

    try:
        with api.capture_diagnostics(path, sctx=sctx) as capture:
            reporter = register().bind(sctx=sctx)
            worker = Thread(target=lambda: results.append(reporter.snapshot("state", build)))
            worker.start()
            assert entered.wait(5)
        assert records(path)[-1]["_type"] == "bundle_complete"
        assert capture.result.partial
        before = path.read_bytes()
    finally:
        release.set()
        worker.join(5)
    assert results == [False] and path.read_bytes() == before
    assert reporter._sink.writer is reporter._sink.state is None


def test_explicit_context_nested_capture_and_stale_reporters(sctx, tmp_path):
    from copy import copy

    declaration = register()
    disabled = declaration.bind(sctx=sctx)
    other = copy(sctx)
    other.operation = OperationState()
    with api.capture_diagnostics(tmp_path / "outer.ndjson", sctx=sctx):
        reporter = declaration.bind(sctx=sctx)
        assert declaration.bind(sctx=other) is DISABLED
        with api.capture_diagnostics(tmp_path / "unused.ndjson", sctx=sctx):
            assert declaration.bind(sctx=sctx)._sink is reporter._sink
    with api.capture_diagnostics(tmp_path / "second.ndjson", sctx=sctx):
        assert not reporter.enabled and not disabled.enabled
        assert declaration.bind(sctx=sctx).enabled
    assert not (tmp_path / "unused.ndjson").exists()


def test_bind_from_feedback_without_ambient_thread_context(sctx, tmp_path):
    from sparkrun.feedback import get_feedback

    declaration = register()
    path = tmp_path / "callback.ndjson"
    with api.capture_diagnostics(path, sctx=sctx):
        feedback = get_feedback("example", sctx=sctx)
        with ThreadPoolExecutor(1) as pool:
            reporter = pool.submit(declaration.bind, feedback=feedback).result()
            assert pool.submit(reporter.snapshot, "state", lambda: {"from_callback": True}).result()
    entry = next(row for row in records(path) if row["_type"] == "plugin_diagnostic")
    assert entry["activity_id"] == feedback.activity_id and entry["operation_id"] == sctx.operation.id


def test_cancelled_or_full_storage_skips_builders(sctx, tmp_path):
    build = Mock(return_value={"value": 1})
    with api.capture_diagnostics(tmp_path / "cancel.ndjson", sctx=sctx):
        reporter = register().bind(sctx=sctx)
        reporter._sink.writer.error = "OSError"
        assert not reporter.snapshot("state", build)
        sctx.operation.cancel.set()
        assert not reporter.enabled and not reporter.snapshot("state", build)
    build.assert_not_called()


def test_recursion_and_control_exceptions(sctx, tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with api.capture_diagnostics(tmp_path / "interrupt.ndjson", sctx=sctx):
            reporter = register().bind(sctx=sctx)

            def build():
                assert not reporter.snapshot("state", lambda: pytest.fail("recursive builder"))
                return {"state": "known"}

            assert reporter.snapshot("state", build)
            reporter.snapshot("state", Mock(side_effect=KeyboardInterrupt()))
    assert not reporter.enabled


def test_cancellation_during_builder_reports_the_omitted_snapshot(sctx, tmp_path):
    path = tmp_path / "cancelled-builder.ndjson"
    with api.capture_diagnostics(path, sctx=sctx) as capture:
        reporter = register().bind(sctx=sctx)

        def build():
            sctx.operation.cancel.set()
            return {"collected": True}

        assert not reporter.snapshot("state", build)
    assert capture.result.partial
    summary = next(row for row in records(path) if row["_type"] == "plugin_diagnostics_summary")
    assert summary["reasons"] == {"cancelled": 1}
    assert summary["sections"]["state"] == {"attempts": 1, "dropped": 1}


@pytest.mark.parametrize(
    "text",
    [
        "--api-key top-secret",
        "prefix_api_key='top-secret'",
        '"password": "top-secret"',
        "?access_token=top-secret&x=1",
        "dbPassword: top-secret",
        "authorization=top-secret",
        "a" * 4096 + " --master-key top-secret",
    ],
)
def test_assignment_redaction_preserves_identifier_forms(text):
    from sparkrun.diagnostics.redaction import Redactor

    assert "top-secret" not in Redactor()(text)


def test_rejected_binding_does_not_lose_provenance_on_later_valid_binding(sctx, tmp_path):
    declaration = register()
    path = tmp_path / "binding.ndjson"
    with api.capture_diagnostics(path, sctx=sctx) as capture:
        assert declaration.bind(sctx=sctx, activity_id="x" * 513) is DISABLED
        assert declaration.bind(sctx=sctx).snapshot("state", lambda: {"valid": True})
    assert capture.result.partial
    assert sum(row["_type"] == "plugin_diagnostics_metadata" for row in records(path)) == 1
