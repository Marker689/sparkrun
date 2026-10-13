"""Benchmark scheduling: per-task abstraction and the executor loop.

Frameworks build a nonempty list of
:class:`BenchTask` instances via
:meth:`sparkrun.benchmarking.base.BenchmarkingPlugin.build_task_list`. The
scheduler then dispatches each task as a single benchmark subprocess,
persists per-task JSON, and updates :class:`BenchmarkRunState` so the run
can be resumed after a crash.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from copy import deepcopy

from sparkrun.feedback import get_feedback
from sparkrun.benchmarking._credentials import BenchmarkCredentials
from sparkrun.benchmarking._process import run_benchmark_process
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sparkrun.benchmarking.aggregator import consolidate_results, gap_analysis, read_task_result
from sparkrun.benchmarking.run_state import BenchmarkRunState

if TYPE_CHECKING:
    from sparkrun.benchmarking.base import BenchmarkingPlugin
    from sparkrun.benchmarking.progress import BenchmarkProgress

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchTask:
    """One scheduled benchmark invocation.

    Attributes:
        index: 0-based position in the schedule. Stable across resumes.
        label: Short human description for the progress UI (e.g. ``"d=4096 c=2"``).
        run_args: Framework-specific args dict; opaque to the scheduler.
        schedule_entry: Raw per-task override dict from the YAML schedule
            (or the auto-generated default). Persisted in run state so the
            schedule can be reconstructed on resume.
    """

    index: int
    label: str
    run_args: dict[str, Any] = field(default_factory=dict)
    schedule_entry: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleRunResult:
    """Result returned by :func:`run_schedule`."""

    success: bool
    completed_count: int
    failed_count: int
    state: BenchmarkRunState
    consolidated: dict[str, Any]


def build_benchmark_tasks(fw: BenchmarkingPlugin, base_args: dict[str, Any], schedule: list[dict[str, Any]] | None) -> list[BenchTask]:
    """Validate the framework's executable task contract before launch or resume."""
    from sparkrun.core.benchmark_profiles import BenchmarkError

    tasks = fw.build_task_list(base_args, schedule)
    if not isinstance(tasks, list) or not tasks:
        raise BenchmarkError("framework %r must return a nonempty list of BenchTask records" % fw.framework_name)
    if any(not isinstance(task, BenchTask) or type(task.index) is not int or task.index != index for index, task in enumerate(tasks)):
        raise BenchmarkError("framework %r must return contiguous zero-based BenchTask indices" % fw.framework_name)
    return tasks


def _task_result_path(fw, task, state, cache_dir) -> Path:
    return state.runs_dir(cache_dir) / ("%03d%s.json" % (task.index, fw.result_filename_suffix(task)))


def _collect_completed_results(fw, tasks, state, cache_dir) -> dict[str, Any]:
    """Restore accepted artifacts, marking only missing/invalid successes pending.

    Shared by scheduling and result-processing recovery. The caller persists
    any state changes; this function never runs commands or scans stray files.
    """
    paths = []
    for task in tasks:
        if task.index not in state.completed_indices:
            continue
        path = _task_result_path(fw, task, state, cache_dir)
        if read_task_result(path) is None:
            state.mark_failed(task.index, "missing or invalid result artifact")
        else:
            paths.append(path)
    return consolidate_results(paths, fw)


class _AttemptProgress:
    """Isolate advisory parsers/renderers from required process and file work."""

    def __init__(self, fw, run_args, sink, benchmark_id, index, label="benchmark task"):
        self.parser = None
        self.sink, self.index, self.benchmark_id = sink, index, benchmark_id
        self.label = label
        self.reporter = get_feedback("benchmark." + fw.framework_name)
        try:
            self.parser = fw.progress_parser(run_args)
        except Exception:
            logger.debug("Benchmark progress parser initialization failed", exc_info=True)
        try:
            if self.parser is not None:
                self.reporter = get_feedback(self.parser.source)
                self._publish(self.parser.start())
            else:
                self._publish(self._task_observation())
        except Exception:
            self.parser = None
            logger.debug("Benchmark progress initialization failed", exc_info=True)

    def _task_observation(self, success=None):
        status = "running" if success is None else ("succeeded" if success else "failed")
        return dict(
            message=f"{self.label}: {status}",
            stage="execution",
            status=status,
            current=1 if success else 0,
            total=1,
            total_kind="exact",
            unit="tasks",
        )

    def _publish(self, fields):
        if fields is None:
            return
        fields = dict(fields)
        fields["metrics"] = {
            **fields.get("metrics", {}),
            "benchmark_id": self.benchmark_id,
            "task_index": self.index,
            "attempt_id": self.reporter.activity_id,
        }
        payload = self.reporter.report(**fields, log_message=False)
        observe = getattr(self.sink, "observe_task", None)
        if observe is not None:
            try:
                observe(self.index, deepcopy(payload))
            except Exception:
                logger.debug("Benchmark progress subscriber failed", exc_info=True)

    def feed(self, line):
        if self.parser is not None:
            try:
                self._publish(self.parser.feed(line))
            except Exception:
                logger.debug("Benchmark progress observation failed", exc_info=True)

    def finish(self, result=None, *, success=False):
        try:
            self._publish(self.parser.finish(result, success=success) if self.parser is not None else self._task_observation(success))
        except Exception:
            logger.debug("Benchmark progress finalization failed", exc_info=True)


def run_schedule(
    fw: "BenchmarkingPlugin",
    tasks: list[BenchTask],
    state: BenchmarkRunState,
    *,
    target_url: str,
    model: str,
    timeout: int | None,
    task_events: "BenchmarkProgress",
    cache_dir: str | None = None,
    exit_on_first_fail: bool = False,
    credentials: BenchmarkCredentials | None = None,
) -> ScheduleRunResult:
    """Iterate pending tasks. Returns when the schedule is complete or aborts.

    Args:
        fw: The benchmarking plugin that can construct subprocess commands.
        tasks: Ordered list of :class:`BenchTask` instances to execute.
        state: Resumable run state — mutated and saved throughout execution.
        target_url: Inference endpoint URL forwarded to the benchmark command.
        model: Model name forwarded to the benchmark command.
        timeout: Per-task subprocess timeout in seconds, or ``None`` for no limit.
        credentials: Ephemeral authentication injected only into command construction.
        task_events: Task event sink.
        cache_dir: Override for the sparkrun cache directory root.
        exit_on_first_fail: Stop immediately after the first task failure.
            Otherwise attempt the remaining tasks, skipping failures for this
            invocation. A later resume can retry those failed tasks.

    Returns:
        :class:`ScheduleRunResult` describing the outcome.
    """
    # Warmup is per measurement session, independent of inference ownership.
    credentials = credentials or BenchmarkCredentials()
    total = len(tasks)
    result_files = [_task_result_path(fw, task, state, cache_dir) for task in tasks]

    def _consolidate() -> dict[str, Any]:
        return consolidate_results((path for idx, path in enumerate(result_files) if idx in state.completed_indices), fw)

    consolidated = _collect_completed_results(fw, tasks, state, cache_dir)

    # Session bookkeeping — mark this execution session.
    state.mark_session_started()
    state.save(cache_dir)

    failed_this_run: set[int] = set()
    session_first_task = True

    def _do_loop() -> tuple[bool, bool]:
        """Inner loop over pending tasks.

        Returns:
            (aborted, exit_requested) — aborted=True means we should stop immediately.
        """
        nonlocal session_first_task, consolidated

        while True:
            idx = state.next_pending(total, exclude=failed_this_run)
            if idx is None:
                break

            task = tasks[idx]
            task_events.start_task(idx, task.label)

            # Build per-task args, applying framework's warmup/coherence rule.
            run_args = fw.apply_session_warmup_state(task.run_args, is_first_task=session_first_task)

            # Pin framework version to whatever was resolved on the first run
            # of this benchmark, so resumes don't drift onto a newer release.
            pinned_version = state.extras.get("framework_version")
            if pinned_version:
                run_args["framework_pinned_version"] = pinned_version

            result_file: Path = result_files[idx]
            log_file = result_file.with_suffix(".log")
            state.mark_started(idx)
            state.save(cache_dir)
            # Invalidate before command construction: builders may inspect or
            # produce the target file, and an interrupted retry must stay pending.
            result_file.unlink(missing_ok=True)
            cmd = fw.build_benchmark_command(target_url, model, credentials.arguments(run_args), result_file=str(result_file))

            observation = _AttemptProgress(fw, run_args, task_events, state.benchmark_id, idx, task.label)
            t_start = time.monotonic()
            try:
                log_fh = open(log_file, "w", encoding="utf-8")
                try:

                    def output(line, sink=log_fh):
                        sink.write(line)
                        sink.flush()

                    def stderr(line, write=output, progress=observation):
                        write(line)  # Required log failures still fail the task.
                        progress.feed(line)

                    streams = {"stderr": stderr} if observation.parser is not None else {}
                    rc = run_benchmark_process(cmd, timeout=timeout, credentials=credentials, stdout=output, **streams)
                except subprocess.TimeoutExpired:
                    observation.finish()
                    logger.warning("Task %d (%s) timed out after %s seconds", idx, task.label, timeout)
                    duration_s = time.monotonic() - t_start
                    failed_this_run.add(idx)
                    state.mark_failed(idx, "timeout after %ds" % timeout)
                    state.save(cache_dir)
                    task_events.end_task(idx, success=False, duration_s=duration_s)
                    if exit_on_first_fail:
                        state.mark_session_ended("partial")
                        state.save(cache_dir)
                        return True, False
                    continue
                finally:
                    log_fh.close()

                duration_s = time.monotonic() - t_start
                artifact = read_task_result(result_file) if rc == 0 else None
                observation.finish(artifact, success=artifact is not None)
                if rc == 0 and artifact is not None:
                    state.mark_completed(idx)
                    state.save(cache_dir)
                    task_events.end_task(idx, success=True, duration_s=duration_s)
                    consolidated = _consolidate()
                    task_events.update_results_table(consolidated)
                    session_first_task = False
                else:
                    failed_this_run.add(idx)
                    state.mark_failed(idx, "exit code %d" % rc if rc else "missing or invalid result artifact")
                    state.save(cache_dir)
                    task_events.end_task(idx, success=False, duration_s=duration_s)
                    if exit_on_first_fail:
                        state.mark_session_ended("partial")
                        state.save(cache_dir)
                        return True, False

            except KeyboardInterrupt:
                observation.finish()
                return False, True  # signal KeyboardInterrupt to caller
            except Exception:
                observation.finish()
                raise

        return False, False

    try:
        aborted, interrupted = _do_loop()

        if interrupted:
            state.mark_session_ended("interrupted")
            state.save(cache_dir)
            raise KeyboardInterrupt

        if aborted:
            # exit_on_first_fail already called mark_session_ended("partial") inside loop.
            return ScheduleRunResult(
                success=False,
                completed_count=len(state.completed_indices),
                failed_count=len(state.failed_indices),
                state=state,
                consolidated=consolidated,
            )

        # Post-loop gap analysis — done at most once.
        gaps = gap_analysis(tasks, consolidated, fw, completed_indices=state.completed_indices)
        if gaps:
            task_events.log("Found %d gap(s); re-queueing" % len(gaps))
            for gap_task in gaps:
                if gap_task.index in state.completed_indices:
                    state.mark_failed(gap_task.index, "missing measurement coverage")
            state.save(cache_dir)
            # Re-enter the loop for gap tasks.
            aborted, interrupted = _do_loop()
            if interrupted:
                state.mark_session_ended("interrupted")
                state.save(cache_dir)
                raise KeyboardInterrupt
            if aborted:
                return ScheduleRunResult(
                    success=False,
                    completed_count=len(state.completed_indices),
                    failed_count=len(state.failed_indices),
                    state=state,
                    consolidated=consolidated,
                )

        # A bounded gap pass may still leave successful commands without the
        # requested measurements. Persist those gaps as failures for resume.
        consolidated = _consolidate()
        for task in gap_analysis(tasks, consolidated, fw, completed_indices=state.completed_indices):
            if task.index in state.completed_indices:
                state.mark_failed(task.index, "missing measurement coverage")
        consolidated = _consolidate()
        task_events.update_results_table(consolidated)

        if state.is_complete(total):
            state.mark_session_ended("completed")
        else:
            state.mark_session_ended("partial")
        state.save(cache_dir)

        return ScheduleRunResult(
            success=state.is_complete(total),
            completed_count=len(state.completed_indices),
            failed_count=len(state.failed_indices),
            state=state,
            consolidated=consolidated,
        )

    except KeyboardInterrupt:
        # Already handled mark_session_ended("interrupted") above; just re-raise.
        raise

    except Exception:
        logger.exception("Unexpected error in run_schedule for benchmark %s", state.benchmark_id)
        state.mark_crash()
        state.mark_session_ended("crashed")
        state.save(cache_dir)
        return ScheduleRunResult(
            success=False,
            completed_count=len(state.completed_indices),
            failed_count=len(state.failed_indices),
            state=state,
            consolidated=consolidated,
        )
