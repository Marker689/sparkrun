"""Regression tests for CLI cluster-id target shortcuts."""

from __future__ import annotations

from unittest import mock

from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.cli._common import _is_cluster_id

_INTENT = "9e23ec987b7e4fc8"
_PLACEMENT = "2699ff346a41"
_BARE_CANONICAL = f"{_INTENT}_{_PLACEMENT}"
_FULL_CANONICAL = f"sparkrun_{_BARE_CANONICAL}"


def test_is_cluster_id_normalizes_status_short_canonical_id() -> None:
    """Given the id printed by status, `_is_cluster_id` returns the canonical id."""
    assert _is_cluster_id(_BARE_CANONICAL) == _FULL_CANONICAL


def test_logs_accepts_status_short_canonical_id() -> None:
    """Given a status logs command, logs follows cluster-id metadata instead of recipe lookup."""
    meta = {
        "cluster_id": _FULL_CANONICAL,
        "recipe": "test-recipe",
        "model": "test/model",
        "runtime": "sglang",
        "hosts": ["10.24.11.13"],
    }

    with (
        mock.patch("sparkrun.orchestration.job_metadata.load_job_metadata", return_value=meta),
        mock.patch("sparkrun.api.logs", return_value=iter(())) as api_logs,
    ):
        result = CliRunner().invoke(main, ["logs", _BARE_CANONICAL])

    assert result.exit_code == 0, result.output
    api_logs.assert_called_once()
    assert api_logs.call_args.kwargs["workload_id"] == _FULL_CANONICAL
    assert api_logs.call_args.kwargs["hosts"] == ("10.24.11.13",)


def test_stop_accepts_status_short_canonical_id() -> None:
    """Given a status stop command, stop sends the normalized canonical id to the API."""
    meta = {
        "cluster_id": _FULL_CANONICAL,
        "recipe": "test-recipe",
        "model": "test/model",
        "runtime": "sglang",
        "hosts": ["10.24.11.13"],
    }
    stop_result = mock.Mock(errors=[], hosts_targeted=("10.24.11.13",))

    with (
        mock.patch("sparkrun.orchestration.job_metadata.load_job_metadata", return_value=meta),
        mock.patch("sparkrun.api.stop", return_value=stop_result) as api_stop,
    ):
        result = CliRunner().invoke(main, ["stop", _BARE_CANONICAL])

    assert result.exit_code == 0, result.output
    api_stop.assert_called_once()
    call_kwargs = api_stop.call_args.kwargs
    assert call_kwargs["workload_id"] == _FULL_CANONICAL
    assert call_kwargs["hosts"] == ("10.24.11.13",)
