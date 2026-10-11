"""Presentation-independent progress contract for benchmark schedulers."""

from typing import Any, Protocol


class BenchmarkProgress(Protocol):
    """Receives task events; implementations may render, forward, or ignore them."""

    def start_task(self, idx: int, label: str) -> None: ...

    def end_task(self, idx: int, success: bool, duration_s: float | None = None) -> None: ...

    def update_results_table(self, consolidated: dict[str, Any]) -> None: ...

    def log(self, message: str) -> None: ...


class ObservedBenchmarkProgress(BenchmarkProgress, Protocol):
    """Optional sink extension for advisory per-attempt feedback."""

    def observe_task(self, idx: int, feedback: dict[str, Any]) -> None: ...


class BenchmarkProgressParser(Protocol):
    """Per-attempt parser for redacted, decoded stderr lines.

    Methods return FeedbackReporter.report fields; observations cannot accept
    artifacts or complete scheduler tasks. Unknown output should return None.
    """

    source: str

    def start(self) -> dict[str, Any]: ...

    def feed(self, line: str) -> dict[str, Any] | None: ...

    def finish(self, result: dict[str, Any] | None, *, success: bool) -> dict[str, Any]: ...
