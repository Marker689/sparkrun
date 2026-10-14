"""Shared, opt-in bundle lifecycle; frontends only select a destination."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import inspect
import logging

from sparkrun.diagnostics.ndjson_writer import NDJSONWriter
from sparkrun.diagnostics.redaction import Redactor
from sparkrun.diagnostics.run_collector import RunDiagnosticsCollector

ACTIVE: ContextVar[DiagnosticsCapture | None] = ContextVar("sparkrun_diagnostics", default=None)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiagnosticsResult:
    """Bundle collection status. Partial collection does not imply workload failure."""

    output_path: str
    schema_version: int = 1
    partial: bool = False
    truncated: bool = False
    sections: tuple[dict, ...] = ()
    hosts: dict[str, bool] = field(default_factory=dict)


class DiagnosticsCapture(RunDiagnosticsCollector):
    """Operation capture handle returned by ``api.capture_diagnostics``.

    ``result`` is populated when the outermost capture exits. Nested captures
    reuse this handle. Collection failures are recorded without failing work.
    """

    def __init__(self, output_path, *, sctx, max_bytes, log_tail, dry_run, secrets=()):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 16384 <= max_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_bytes must be between 16384 and 67108864")
        if isinstance(log_tail, bool) or not isinstance(log_tail, int) or not 0 <= log_tail <= 2000:
            raise ValueError("log_tail must be between 0 and 2000")
        super().__init__(str(output_path), [], {}, dry_run=dry_run)
        self.redact = Redactor(secrets)
        self._writer = NDJSONWriter(output_path, exclusive=True, redact=self.redact, max_bytes=max_bytes)
        self.sctx = sctx
        self.log_tail = log_tail
        self.sections = []
        self.host_status = {}
        self.result = None
        self._targets = set()
        self._workloads = set()
        self._logs = set()
        self._interrupted = False

    def section(self, name, action):
        """Run a best-effort section and retain its explicit outcome."""
        from sparkrun.core.operation import CURRENT

        try:
            state = CURRENT.get()
            if state is not None:
                state.check()
            value = action()
        except (KeyboardInterrupt, InterruptedError):
            self._interrupted = True
            self.status(name, "cancelled", "Collection interrupted or deadline exceeded")
            raise
        except Exception as exc:
            self.status(name, "error", str(exc) or type(exc).__name__)
            return None
        self.status(name, "ok")
        return value

    def status(self, name, status, detail=None):
        record = {"section": name, "status": status}
        if detail:
            record["detail"] = self.redact(str(detail))[:2000]
        self.sections.append(self.redact(record))
        self.writer.emit("bundle_section", record)

    def collect_config(self):
        from sparkrun.diagnostics.spark_collector import collect_config_diagnostics

        self.section("config", lambda: collect_config_diagnostics(self.writer, config=self.sctx.config))
        self.section("clusters", lambda: collect_config_diagnostics(self.writer, cluster_mgr=self.sctx.cluster_manager, strict=True))
        self.section(
            "registries", lambda: collect_config_diagnostics(self.writer, registry_mgr=self.sctx.config.get_registry_manager(), strict=True)
        )

    def collect_target(
        self, *, hosts=None, cluster=None, workload_id=None, sudo_password=None, dry_run=False, executor_hint=None, prepared=False
    ):
        from sparkrun.api._resolve import resolve_cluster, resolve_cluster_for_job, scope_operation
        from sparkrun.orchestration.job_metadata import load_job_metadata
        from sparkrun.diagnostics.spark_collector import collect_spark_diagnostics, collect_sudo_diagnostics
        from sparkrun.orchestration.executor import cluster_status_scope

        if workload_id and not (dry_run or self._dry_run):
            self._workloads.add(workload_id)

        def resolve():
            if workload_id:
                meta = load_job_metadata(workload_id, cache_dir=str(self.sctx.config.cache_dir))
                if not meta:
                    raise ValueError("No saved metadata for workload %s" % workload_id)
                saved_hosts = hosts or meta.get("hosts")
                if not saved_hosts:
                    raise ValueError("No saved hosts for workload %s" % workload_id)
                target = resolve_cluster_for_job(cluster, saved_hosts, meta=meta, sctx=self.sctx)
                return target, meta.get("executor") or target.executor
            else:
                target = resolve_cluster(cluster, hosts, sctx=self.sctx)
            return target, executor_hint or target.executor

        resolved = self.section("target", resolve)
        if resolved is None:
            return
        target, executor = resolved
        key = (target.name, tuple(target.hosts), target.user, executor, dry_run or self._dry_run, sudo_password is not None)
        if key in self._targets:
            return
        self._targets.add(key)
        self.writer.emit("bundle_target", {"workload_id": workload_id, "cluster": target.name, "hosts": target.hosts, "executor": executor})
        host_scope = self.section(
            "host_scope", lambda: cluster_status_scope(target, executor=executor, config=self.sctx.config, v=self.sctx.variables)
        )
        if host_scope is None:
            return
        if host_scope != "host":
            self.status("hosts", "skipped", "Host probes require SSH/local hosts; native workload logs remain available")
            return
        if self._dry_run or dry_run:
            self.status("hosts", "skipped", "dry_run")
            return
        scoped = self.section("transport", lambda: scope_operation(target, sctx=self.sctx, prepare=not prepared))
        if scoped is None:
            return
        _, ssh = scoped
        data = self.section("hosts", lambda: collect_spark_diagnostics(list(target.hosts), ssh, writer=self.writer))
        if data is not None:
            for host in target.hosts:
                ok = data.get(host, {}).get("DIAG_COMPLETE") == "1"
                self.host_status[host] = ok
                if not ok:
                    self.status("host:" + host, "error", "Host probe failed or returned incomplete data")
        if sudo_password is not None:
            self.redact.add(sudo_password)
            data = self.section("sudo", lambda: collect_sudo_diagnostics(list(target.hosts), ssh, sudo_password, writer=self.writer))
            if data is not None:
                for host in target.hosts:
                    if data.get(host, {}).get("DIAG_SUDO_COMPLETE") != "1":
                        self.status("sudo:" + host, "error", "Privileged probe failed or returned incomplete data")

    def collect_logs(self, workload_id, *, dry_run=False, refresh=False):
        from sparkrun.api._logs import open_log_target
        from dataclasses import asdict

        if workload_id in self._logs and not refresh:
            return
        if self._dry_run or dry_run or self.log_tail == 0:
            self.status("logs:" + workload_id, "skipped", "dry_run or log_tail=0")
            return
        self._logs.add(workload_id)
        target = self.section("logs_target:" + workload_id, lambda: open_log_target(workload_id, scope="all", sctx=self.sctx))
        if target is None:
            return
        for source in target.sources:

            def read(source=source):
                lines = target.page(source, skip=0, count=self.log_tail, timeout=30)
                from sparkrun.core.operation import CURRENT

                state = CURRENT.get()
                if state is not None:
                    state.check()
                for line in lines:
                    self.writer.emit("workload_log", {"workload_id": workload_id, **asdict(line)})

            self.section("logs:" + source.source_id, read)

    def record_plan(self, plan, *, dry_run=False):
        self.emit_recipe(plan.recipe)
        if not dry_run and plan.cluster_id:
            self._workloads.add(plan.cluster_id)
        target = plan.executor_target
        self.collect_target(
            hosts=list(plan.host_list),
            cluster=plan.cluster,
            dry_run=dry_run,
            executor_hint=target.executor if target else None,
            prepared=True,
        )

    def record_result(self, name, result, *, dry_run=False):
        if name == "plan":
            self.record_plan(result, dry_run=dry_run)
        elif name == "run":
            self.emit_launch_result(result)
            self.emit_serve_command(result.serve_command, result.container_image)
            if result.cluster_id and not dry_run:
                self._workloads.add(result.cluster_id)
        elif name in {"benchmark", "resume_benchmark"}:
            self.writer.emit(
                "benchmark_result",
                {
                    "benchmark_id": result.benchmark_id,
                    "workload_id": result.cluster_id,
                    "success": result.success,
                    "state_dir": result.state_dir,
                    "outputs": result.outputs,
                },
            )
            if not result.success:
                self.emit_error(name, "Benchmark measurement policy was not satisfied")
            if result.cluster_id and not dry_run:
                self._workloads.add(result.cluster_id)

    def finish(self):
        from sparkrun.core.operation import CURRENT

        state = CURRENT.get()
        cancelled = self._interrupted or (state is not None and state.cancel.is_set())
        if not self._success:
            for workload_id in sorted(self._workloads):
                if cancelled:
                    self.status("logs:" + workload_id, "cancelled", "No additional reads after cancellation")
                else:
                    self.collect_logs(workload_id)
        self.emit_timeline(self.sctx.timing)
        self.emit_summary()
        if self.writer.error:
            self.status("storage", "error", self.writer.error)
        partial = self.writer.truncated or any(s["status"] != "ok" for s in self.sections)
        self.writer.emit(
            "bundle_complete",
            {
                "schema_version": 1,
                "partial": partial,
                "truncated": self.writer.truncated,
                "operation_success": self._success,
            },
        )
        self.result = self.snapshot()

    def snapshot(self):
        return DiagnosticsResult(
            str(self.writer.path),
            partial=self.writer.truncated or bool(self.writer.error) or any(s["status"] != "ok" for s in self.sections),
            truncated=self.writer.truncated,
            sections=tuple(self.sections),
            hosts=dict(self.host_status),
        )


@contextmanager
def capture_before_cleanup(workload_id, *, sctx, error=None):
    """Snapshot logs before owned teardown, even when later finalization may fail.

    Cleanup still runs if collection is interrupted. An existing operation
    failure takes precedence; otherwise a new interruption propagates afterward.
    This is inert outside an explicitly enabled, matching capture.
    """
    from sparkrun.core.operation import CURRENT

    capture = ACTIVE.get()
    interrupted = None
    if capture is not None and workload_id and sctx.operation is capture.sctx.operation:
        if isinstance(error, (KeyboardInterrupt, InterruptedError)) or (isinstance(error, SystemExit) and error.code == 130):
            capture._interrupted = True
        state = CURRENT.get()
        if capture._interrupted or (state is not None and state.cancel.is_set()):
            capture.status("logs_before_cleanup:" + workload_id, "cancelled", "No additional reads after cancellation")
        else:
            try:
                # Earlier standalone collection in the same capture must not
                # prevent a fresh snapshot at the destructive boundary.
                capture.section("logs_before_cleanup:" + workload_id, lambda: capture.collect_logs(workload_id, refresh=True))
            except (KeyboardInterrupt, InterruptedError) as exc:
                interrupted = exc
    try:
        yield
    except BaseException as cleanup_error:
        if interrupted is not None and error is None:
            interrupted.add_note("Cleanup also failed: %s" % cleanup_error)
            raise interrupted from cleanup_error
        raise
    if interrupted is not None and error is None:
        raise interrupted


def capture_operation(function):
    """Observe shared API boundaries only when explicitly capturing diagnostics."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        capture = ACTIVE.get()
        if capture is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        name = function.__name__
        if bound.arguments.get("sctx") is None:
            bound.arguments["sctx"] = capture.sctx
        # A different session is a different operation, even inside this context.
        if bound.arguments["sctx"].operation is not capture.sctx.operation:
            return function(*bound.args, **bound.kwargs)
        options = bound.arguments.get("options")
        if options is not None:
            capture.writer.emit(
                "operation_options",
                {
                    "operation": name,
                    **{
                        key: getattr(options, key, None)
                        for key in ("overrides", "hosts", "dry_run", "executor", "scheduler", "bench_args", "skip_run", "no_stop")
                    },
                },
            )
        supplied_plan = bound.arguments.get("plan")
        dry_run = getattr(bound.arguments.get("options"), "dry_run", bound.arguments.get("dry_run", False))
        if supplied_plan is not None:
            capture.section("plan", lambda: capture.record_plan(supplied_plan, dry_run=dry_run))
        capture.phase_start(name)
        try:
            result = function(*bound.args, **bound.kwargs)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, InterruptedError)):
                capture._interrupted = True
            capture.phase_end(name, error=str(exc) or type(exc).__name__)
            capture.emit_error(name, exc if isinstance(exc, Exception) else str(exc) or type(exc).__name__)
            raise
        capture.phase_end(name)
        capture.section(name + "_result", lambda: capture.record_result(name, result, dry_run=dry_run))
        return result

    return wrapped


@contextmanager
def capture(output_path, *, sctx, max_bytes, log_tail, dry_run, secrets=()):
    from sparkrun.core.operation import operation_scope
    from sparkrun.core.timing import Timeline

    existing = ACTIVE.get()
    if existing is not None and (sctx is None or sctx.operation is existing.sctx.operation):
        for secret in secrets:
            existing.redact.add(secret)
        yield existing
        return
    from sparkrun.api._context import resolve_sctx

    sctx = resolve_sctx(sctx)
    session = DiagnosticsCapture(output_path, sctx=sctx, max_bytes=max_bytes, log_tail=log_tail, dry_run=dry_run, secrets=secrets)
    previous_timing = sctx.timing
    previous_progress = sctx.progress
    previous_progress_timing = previous_progress.timeline if previous_progress is not None else None
    if previous_timing is None:
        sctx.timing = Timeline()
    try:
        with operation_scope(sctx) as state:
            session.redact.operation_secrets = state.secrets
            session.open()
            token = ACTIVE.set(session)
            from sparkrun.diagnostics._plugin_sink import PluginSink

            plugin_sink = PluginSink(session.writer, state, max_bytes=max_bytes)
            previous_sink = state.diagnostics_sink
            state.diagnostics_sink = plugin_sink

            def observer(event):
                session.writer.emit("feedback", event)

            state.observers.append(observer)
            try:
                session.emit_header(schema_version=1, operation_id=state.id)
                session.collect_config()
                try:
                    yield session
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, InterruptedError)) or (isinstance(exc, SystemExit) and exc.code == 130):
                        session._interrupted = True
                    if not isinstance(exc, SystemExit) or exc.code not in (None, 0):
                        session.emit_error("operation", exc if isinstance(exc, Exception) else str(exc) or type(exc).__name__)
                    raise
            finally:
                try:
                    plugin_sink.close()
                    if plugin_sink.partial:
                        session.status(
                            "plugin_diagnostics",
                            "error",
                            "Plugin snapshots omitted or rejected (%d contributor bindings omitted); inspect plugin summaries"
                            % plugin_sink.omitted,
                        )
                    session.finish()
                except Exception:
                    # Bundle finalization must never replace an operation's outcome.
                    logger.warning("Diagnostic bundle finalization failed", exc_info=True)
                finally:
                    state.diagnostics_sink = previous_sink
                    state.observers.remove(observer)
                    ACTIVE.reset(token)
                    session.close()
    finally:
        sctx.timing = previous_timing
        sctx.progress = previous_progress
        if previous_progress is not None:
            previous_progress.timeline = previous_progress_timing
