"""Capture-local admission and synchronous bounded plugin output. No plugin discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
import threading
import time

from sparkrun.diagnostics._bounded import payload_copy
from sparkrun.diagnostics.plugin import DIAGNOSTICS_API_VERSION, DISABLED, DiagnosticsReporter

RECORD_BYTES = 16 * 1024
PLUGIN_BYTES = 256 * 1024
TOTAL_BYTES = 1024 * 1024
METADATA_BYTES = 2048
SUMMARY_BYTES = 4096


@dataclass
class _Budget:
    limit: int
    count_limit: int
    used: int = 0  # Includes in-flight and summary reservations.
    normal_used: int = 0
    attempts: int = 0
    normal_attempts: int = 0

    def available(self, terminal):
        if self.attempts >= self.count_limit or (not terminal and self.normal_attempts >= self.count_limit * 3 // 4):
            return 0
        free = self.limit - self.used
        return free if terminal else min(free, self.limit * 3 // 4 - self.normal_used)

    def reserve(self, size, terminal):
        self.used += size
        self.attempts += 1
        if not terminal:
            self.normal_used += size
            self.normal_attempts += 1

    def settle(self, reserved, written, terminal):
        self.used += written - reserved
        if not terminal:
            self.normal_used += written - reserved


@dataclass
class _Participant:
    declaration: object
    budget: _Budget
    sections: dict
    attempts: dict = field(default_factory=dict)
    dropped: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)
    pending: set = field(default_factory=set)
    metadata_attempted: bool = False


@dataclass(eq=False)
class _Attempt:
    section: str


def _scope_valid(value):
    return value is None or (type(value) is str and 0 < len(value) <= 512)


class PluginSink:
    def __init__(self, writer, state, *, max_bytes):
        self.writer = writer
        self.state = state
        self.operation_id = state.id
        self.budget = _Budget(min(TOTAL_BYTES, max(0, max_bytes - 8192) // 4), 512)
        self.participants = {}
        self._lock = threading.RLock()
        self._commit = threading.Lock()
        self._local = threading.local()
        self.omitted = 0
        self.partial = False
        self.truncated = False

    @property
    def enabled(self):
        state = self.state
        return (
            self.writer is not None
            and state is not None
            and not state.cancel.is_set()
            and (state.deadline is None or time.monotonic() < state.deadline)
        )

    def _drop(self, participant, section, reason, attempt=None):
        with self._lock:
            if self.writer is None:
                return
            if attempt is not None:
                participant.pending.discard(attempt)
            self.partial = True
            if reason in {"capacity", "section_count", "record_bytes", "bundle_bytes"}:
                self.truncated = True
            participant.reasons[reason] = min(1_000_000_000, participant.reasons.get(reason, 0) + 1)
            if type(section) is str and section in participant.sections:
                participant.dropped[section] = min(1_000_000_000, participant.dropped.get(section, 0) + 1)

    def bind(self, declaration, *, activity_id, workload_id):
        if not self.enabled:
            return DISABLED
        name = declaration.owner.plugin_id
        with self._lock:
            if not self.enabled:
                return DISABLED
            participant = self.participants.get(name)
            if participant is None:
                limit = min(PLUGIN_BYTES, self.budget.limit)
                reserved = METADATA_BYTES + SUMMARY_BYTES
                if len(self.participants) >= 32 or limit < reserved or self.budget.limit - self.budget.used < reserved:
                    self.omitted = min(1_000_000_000, self.omitted + 1)
                    self.partial = self.truncated = True
                    return DISABLED
                participant = _Participant(declaration, _Budget(limit, 128, used=reserved), {s.name: s for s in declaration.sections})
                self.participants[name] = participant
                self.budget.used += reserved
            if not _scope_valid(activity_id) or not _scope_valid(workload_id):
                self._drop(participant, None, "scope")
                return DISABLED
        with self._commit:
            writer = self.writer
            if not self.enabled or writer is None:
                return DISABLED
            if not participant.metadata_attempted:
                participant.metadata_attempted = True
                module = sys.modules.get(declaration.owner.module)
                version = vars(module).get("__version__") if module is not None else None
                metadata = {
                    "plugin": name,
                    "module": declaration.owner.module[:512],
                    "version": version[:128] if type(version) is str else None,
                    "source_kind": declaration.owner.source[0][:128],
                }
                try:
                    size, reason = writer.emit_bounded("plugin_diagnostics_metadata", metadata, max_bytes=METADATA_BYTES)
                except Exception:
                    size, reason = 0, "storage"
                with self._lock:
                    self.budget.used += size - METADATA_BYTES
                    participant.budget.used += size - METADATA_BYTES
                if reason:
                    self._drop(participant, None, reason)
        return DiagnosticsReporter(self, declaration, activity_id, workload_id)

    def snapshot(self, reporter, section, build, *, host, stage):
        if not self.enabled:
            return False
        participant = self.participants.get(reporter._plugin.owner.plugin_id)
        if participant is None:
            return False
        if getattr(self._local, "busy", False):
            self._drop(participant, section, "recursive")
            return False
        with self._lock:
            declared = participant.sections.get(section) if type(section) is str and len(section) <= 64 else None
            if declared is None:
                self._drop(participant, None, "section")
                return False
            if not _scope_valid(host) or not _scope_valid(stage):
                self._drop(participant, section, "scope")
                return False
            attempts = participant.attempts.get(section, 0)
            if attempts >= 32:
                self._drop(participant, section, "section_count")
                return False
            terminal = declared.priority == "terminal"
            writer = self.writer
            if writer is None:
                return False
            allowance = min(RECORD_BYTES, self.budget.available(terminal), participant.budget.available(terminal), writer.available_bytes)
            if allowance <= 0:
                self._drop(participant, section, "capacity")
                return False
            participant.attempts[section] = attempts + 1
            for budget in (self.budget, participant.budget):
                budget.reserve(allowance, terminal)
            attempt = _Attempt(section)
            participant.pending.add(attempt)
        size = 0
        self._local.busy = True
        try:
            try:
                data = payload_copy(build())
            except Exception:
                # Never stringify arbitrary producer exceptions/objects, or log
                # a failure that would recursively enter captured feedback.
                self._drop(participant, section, "payload", attempt)
                return False
            record = {
                "diagnostics_api_version": DIAGNOSTICS_API_VERSION,
                "operation_id": self.operation_id,
                "plugin": reporter._plugin.owner.plugin_id,
                "section": section,
                "schema_version": declared.schema_version,
                "data": data,
            }
            record.update(
                {
                    k: v
                    for k, v in {
                        "activity_id": reporter._activity_id,
                        "workload_id": reporter._workload_id,
                        "host": host,
                        "stage": stage,
                    }.items()
                    if v is not None
                }
            )
            with self._commit:
                writer = self.writer
                if not self.enabled or writer is None:
                    if writer is not None:
                        self._drop(participant, section, "cancelled", attempt)
                    return False
                try:
                    size, reason = writer.emit_bounded("plugin_diagnostic", record, max_bytes=allowance)
                except Exception:
                    reason = "storage"
                if reason:
                    self._drop(participant, section, reason, attempt)
                with self._lock:
                    participant.pending.discard(attempt)
                return reason is None
        except BaseException:
            self._drop(participant, section, "interrupted", attempt)
            raise
        finally:
            self._local.busy = False
            with self._lock:
                participant.pending.discard(attempt)
                for budget in (self.budget, participant.budget):
                    budget.settle(allowance, size, terminal)

    def close(self):
        """Detach before core finalization; late builders cannot write or retain the operation."""
        with self._commit:
            with self._lock:
                writer = self.writer
                if writer is None:
                    return
                self.writer = None
                self.state = None
                participants, self.participants = self.participants, {}
                for name, participant in participants.items():
                    if participant.pending:
                        self.partial = True
                        participant.reasons["closed"] = len(participant.pending)
                        for attempt in participant.pending:
                            section = attempt.section
                            participant.dropped[section] = participant.dropped.get(section, 0) + 1
                        participant.pending.clear()
                    record = {
                        "plugin": name,
                        "sections": {
                            name: {"attempts": participant.attempts.get(name, 0), "dropped": participant.dropped.get(name, 0)}
                            for name in participant.sections
                            if name in participant.attempts or name in participant.dropped
                        },
                        "reasons": dict(participant.reasons),
                    }
                    try:
                        _, reason = writer.emit_bounded("plugin_diagnostics_summary", record, max_bytes=SUMMARY_BYTES)
                    except Exception:
                        reason = "storage"
                    if reason:
                        self.partial = True
                        self.truncated = self.truncated or reason != "storage"
                if self.truncated:
                    writer.truncated = True
