"""Public feedback delivery, isolation, bounds, and legacy log compatibility."""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import threading
from types import SimpleNamespace

import pytest

from sparkrun.feedback import FeedbackEvent, get_feedback
from sparkrun.core.operation import OperationState, operation_scope, submit
from sparkrun.core.feedback_logging import forward_operation_logs


def test_structured_feedback_survives_quiet_logging_and_raw_thread(monkeypatch):
    seen = []
    logger = logging.getLogger("test.feedback")
    monkeypatch.setattr(logger, "level", logging.ERROR)
    with operation_scope(state=OperationState(id="one", events=seen.append)):
        reporter = get_feedback("example", logger=logger)
        receiver = reporter.child(host="host-a")
        thread = threading.Thread(
            target=lambda: receiver.progress(
                "Receiving", stage="transfer", current=2, total=5, total_kind="upper_bound", unit="bytes", metrics={"bytes_per_second": 3}
            )
        )
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert len(seen) == 1
    event = seen[0]
    assert event["type"] == "feedback" and event["feedback_version"] == 1
    assert event["operation_id"] == "one" and event["parent_id"] == reporter.activity_id
    assert event["host"] == "host-a" and event["total_kind"] == "upper_bound"
    assert json.loads(json.dumps(event)) == event


def test_nested_and_concurrent_operation_log_isolation(caplog):
    caplog.set_level(25)
    observed = {name: [] for name in ("a", "b")}
    barrier = threading.Barrier(2)

    def run(name):
        with operation_scope(state=OperationState(id=name, events=observed[name].append)):
            with operation_scope():
                barrier.wait(timeout=3)
                logging.getLogger("legacy").warning(name)
                get_feedback("plugin").progress(name)
                with ThreadPoolExecutor(1) as pool:
                    submit(pool, logging.getLogger("legacy").warning, name + "-child").result(timeout=3)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(run, observed))
    logging.getLogger("legacy").warning("outside")
    for name, events in observed.items():
        assert [event["message"] for event in events] == [name, name, name + "-child"]
        assert all(event["operation_id"] == name for event in events)
        assert [event["type"] for event in events] == ["log", "feedback", "log"]


def test_isolated_worker_forwards_legacy_background_logs_and_restores_level():
    root = logging.getLogger()
    original = root.level
    seen = []
    with forward_operation_logs(worker=OperationState(events=seen.append)):
        thread = threading.Thread(target=lambda: logging.getLogger("external").log(25, "preparing"))
        thread.start()
        thread.join(timeout=2)
    assert root.level == original
    assert [event["message"] for event in seen] == ["preparing"]


def test_feedback_callback_failure_and_recursive_logging_do_not_fail_work(caplog):
    caplog.set_level(25)
    calls = []

    def callback(event):
        calls.append(event)
        logging.getLogger("callback").warning("observer diagnostics")
        raise RuntimeError("UI unavailable")

    with operation_scope(state=OperationState(events=callback)):
        get_feedback("plugin").progress("transferring")
    assert len(calls) == 1
    assert "Operation feedback subscriber failed" in caplog.text


def test_sctx_events_alone_provides_launch_phases_without_duplicates(caplog):
    caplog.set_level(25)
    seen = []
    sctx = SimpleNamespace(events=seen.append, operation=None, progress=None, timing=None)
    with operation_scope(sctx):
        sctx.progress.phase(3)
        sctx.progress.step("transfer")
        sctx.progress.phase_end()
    assert [event["stage"] for event in seen] == ["phase", "step", "phase_end"]
    assert len(caplog.records) == 3


def test_explicit_sctx_and_bound_heartbeat():
    seen = []
    arrived = threading.Event()

    def receive(event):
        seen.append(event)
        arrived.set()

    sctx = SimpleNamespace(events=receive, operation=None)
    reporter = get_feedback("plugin", sctx=sctx)
    with reporter.heartbeat("waiting", interval=0.01):
        assert arrived.wait(2)
    count = len(seen)
    assert seen[0]["operation_id"] == sctx.operation.id
    assert seen[0]["stage"] == "heartbeat"
    assert count >= 1


@pytest.mark.parametrize(
    "fields",
    [
        {"current": -1},
        {"current": float("nan")},
        {"total": 10},
        {"total_kind": "exact"},
        {"metrics": {"nested": {}}},
        {"metrics": {"infinity": float("inf")}},
        {"metrics": {str(i): i for i in range(33)}},
        {"message": "x" * 16385},
        {"status": "complete"},
    ],
)
def test_invalid_feedback_is_rejected(fields):
    with pytest.raises(ValueError):
        FeedbackEvent(**{"source": "plugin", "message": "status", "activity_id": "activity", **fields}).to_dict()


def test_request_reporter_captures_operation_at_creation():
    from sparkrun.plugins import ImageCopyRequest, FEEDBACK_API_VERSION

    seen = []
    with operation_scope(state=OperationState(id="source", events=seen.append)):
        request = ImageCopyRequest("image", None, ("host-a",), ("host-a",))
    with operation_scope(state=OperationState(id="other")):
        request.feedback.progress("receiving")
    assert FEEDBACK_API_VERSION == 1 and seen[0]["operation_id"] == "source"


def test_launch_progress_observer_failure_is_advisory():
    from sparkrun.core.progress import LaunchProgress

    def unavailable(event):
        raise OSError("subscriber disconnected")

    with operation_scope(state=OperationState(events=unavailable)):
        LaunchProgress().phase(1)
