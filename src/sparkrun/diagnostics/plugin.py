"""Public, lazy plugin snapshots. Registration is static; reporters are operation-bound."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re
from typing import Literal, TYPE_CHECKING

from sparkrun.core.plugin_identity import PluginIdentity, current_registration_owner
from sparkrun.core.operation import CURRENT
from sparkrun.core.registration import enlist_registry_state, register_unique

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext
    from sparkrun.diagnostics._plugin_sink import PluginSink
    from sparkrun.feedback import FeedbackReporter

DIAGNOSTICS_API_VERSION = 1
JSONValue = dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None


@dataclass(frozen=True)
class DiagnosticSection:
    name: str
    schema_version: int = 1
    priority: Literal["normal", "terminal"] = "normal"

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.name):
            raise ValueError("Diagnostic section needs a stable local identifier")
        if type(self.schema_version) is not int or not 1 <= self.schema_version <= 65535:
            raise ValueError("Diagnostic section schema_version must be between 1 and 65535")
        if self.priority not in {"normal", "terminal"}:
            raise ValueError("Diagnostic section priority must be normal or terminal")


@dataclass(frozen=True)
class PluginDiagnostics:
    """Loader-owned declarations. Safe to retain on a shared plugin provider."""

    owner: PluginIdentity
    sections: tuple[DiagnosticSection, ...]

    def bind(
        self, *, sctx: SparkrunContext | None = None, feedback: FeedbackReporter | None = None, activity_id=None, workload_id=None
    ) -> DiagnosticsReporter:
        state = (
            getattr(sctx, "operation", None)
            if sctx is not None
            else (getattr(feedback, "_state", None) if feedback is not None else CURRENT.get())
        )
        sink = getattr(state, "diagnostics_sink", None) if state is not None else None
        if sink is None or not sink.enabled:
            return DISABLED
        if _PLUGINS.get(self.owner.plugin_id) is not self:
            return DISABLED
        if feedback is not None:
            if feedback._state is not state:
                return DISABLED
            activity_id = activity_id or feedback.activity_id
        return sink.bind(self, activity_id=activity_id, workload_id=workload_id)


@dataclass(frozen=True)
class DiagnosticsReporter:
    """Pass to workers; never cache across operations. Builders do cheap local work only.

    ``snapshot`` is advisory and returns whether a record was flushed. Ordinary
    collection failures never fail work. Control-flow exceptions propagate.
    """

    _sink: PluginSink | None = None
    _plugin: PluginDiagnostics | None = None
    _activity_id: str | None = None
    _workload_id: str | None = None

    @property
    def enabled(self) -> bool:
        return self._sink is not None and self._sink.enabled

    def snapshot(
        self, section: str, build: Callable[[], dict[str, JSONValue]], *, host: str | None = None, stage: str | None = None
    ) -> bool:
        if self._sink is None:
            return False
        return self._sink.snapshot(self, section, build, host=host, stage=stage)


DISABLED = DiagnosticsReporter()
_PLUGINS: dict[str, PluginDiagnostics] = {}
enlist_registry_state(globals(), "_PLUGINS")


def register_plugin_diagnostics(sections: Sequence[DiagnosticSection]) -> PluginDiagnostics:
    """Declare up to 16 sections during the plugin's existing registration hook."""
    owner = current_registration_owner()
    if not isinstance(owner, PluginIdentity):
        raise ValueError("Plugin diagnostics require loader-owned plugin registration")
    if not 1 <= len(sections) <= 16 or any(type(section) is not DiagnosticSection for section in sections):
        raise ValueError("Plugin diagnostics require between 1 and 16 DiagnosticSection declarations")
    if len({section.name for section in sections}) != len(sections):
        raise ValueError("Duplicate plugin diagnostic section")
    declaration = PluginDiagnostics(owner, tuple(sections))
    previous = _PLUGINS.get(owner.plugin_id)
    if previous is not None and previous == declaration:
        return previous
    register_unique(_PLUGINS, owner.plugin_id, declaration, description="plugin diagnostics")
    return declaration
