"""model-cache CLI: cluster/host targeting and human-readable reports."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from sparkrun.cli import _model_cache as cli
from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.models.artifacts import CacheValidationReport, ModelFileFailure
from sparkrun.models.cache_reports import CacheEntry, CacheReport
from sparkrun.utils.cli_formatters import format_bytes, format_cache_validation_reports, format_model_cache_reports

SHA = "a" * 40
NOW = 1_000_000.0


def entry(**overrides):
    values = dict(
        repo="org/model",
        revision=SHA,
        snapshot="/c/hub/models--org--model/snapshots/" + SHA,
        tracked=True,
        present=True,
        last_used=NOW - 9 * 86400,
        logical_bytes=60 * 2**30,
        reclaimable_bytes=58 * 2**30,
        pinned=False,
        claims={},
        eligible=True,
        reason="eligible",
        reason_code="eligible",
    )
    return CacheEntry(**{**values, **overrides})


def report(entries, **overrides):
    values = dict(
        host="node1",
        cache_root="/models/hf",
        dry_run=True,
        entries=tuple(entries),
        free_bytes=900 * 2**30,
        reclaimed_bytes=0,
        planned_bytes=0,
        target_met=True,
        budget_scope="private_managed_snapshots",
        reclaimed_bytes_kind="allocated_bytes_estimate",
        free_bytes_after_kind="projected",
    )
    return CacheReport(**{**values, **overrides})


@pytest.fixture
def clusters(tmp_path):
    return ClusterManager(tmp_path / "config")


@pytest.fixture
def run(monkeypatch, clusters):
    """Invoke model-cache with a context over *clusters*; returns (result, inventory mock)."""
    config = SimpleNamespace(ssh_user=None, ssh_key=None, ssh_options=None, default_hosts=None, hf_cache_dir="/control/hf")
    monkeypatch.setattr(cli, "_get_context", lambda ctx: SimpleNamespace(config=config, cluster_manager=clusters))
    probe = Mock(return_value="/home/remote/.cache/huggingface")
    monkeypatch.setattr("sparkrun.orchestration.primitives.probe_remote_hf_cache", probe)
    monkeypatch.setattr("sparkrun.utils.is_local_host", lambda host: host == "localhost")
    inventory = Mock(return_value=())
    monkeypatch.setattr(cli.api, "inventory_cache", inventory)

    def invoke(*args):
        return CliRunner().invoke(cli.model_cache, ["list", *args]), inventory, probe

    return invoke


def test_no_cluster_targets_localhost_control_cache(run):
    result, inventory, probe = run()
    assert result.exit_code == 0, result.output
    assert inventory.call_args.kwargs["hosts"] == ("localhost",)
    assert inventory.call_args.kwargs["cache_dir"] == "/control/hf"
    assert "Hosts: localhost" in result.output
    probe.assert_not_called()


def test_default_cluster_supplies_hosts_and_cache_dir(run, clusters):
    clusters.create("lab", ["n1", "n2"], cache_dir="/mnt/models")
    clusters.set_default("lab")
    result, inventory, probe = run()
    assert result.exit_code == 0, result.output
    assert inventory.call_args.kwargs["hosts"] == ("n1", "n2")
    assert inventory.call_args.kwargs["cache_dir"] == "/mnt/models"
    assert "Cluster: lab (default)" in result.output
    probe.assert_not_called()


def test_named_cluster_without_cache_dir_probes_remote_home(run, clusters):
    clusters.create("lab", ["n1", "n2"])
    result, inventory, probe = run("--cluster", "lab")
    assert result.exit_code == 0, result.output
    assert inventory.call_args.kwargs["cache_dir"] == "/home/remote/.cache/huggingface"
    assert probe.call_args.args[0] == "n1"


def test_explicit_cache_dir_beats_cluster(run, clusters):
    clusters.create("lab", ["n1"], cache_dir="/mnt/models")
    result, inventory, _ = run("--cluster", "lab", "--cache-dir", "/other")
    assert result.exit_code == 0, result.output
    assert inventory.call_args.kwargs["cache_dir"] == "/other"


def test_legacy_host_flag_merges_with_hosts_and_skips_default_cluster(run, clusters):
    clusters.create("lab", ["n1", "n2"])
    clusters.set_default("lab")
    result, inventory, _ = run("--host", "a", "--host", "b", "--hosts", "c", "--cache-dir", "/x")
    assert result.exit_code == 0, result.output
    assert inventory.call_args.kwargs["hosts"] == ("a", "b", "c")


def test_unknown_cluster_is_an_error(run):
    result, inventory, _ = run("--cluster", "nope")
    assert result.exit_code != 0
    assert "not found" in result.output
    inventory.assert_not_called()


def test_json_output_has_no_banner(run, clusters):
    clusters.create("lab", ["n1"], cache_dir="/mnt/models")
    result, inventory, _ = run("--cluster", "lab", "--json")
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "[]"


def test_format_bytes():
    assert format_bytes(None) == "-"
    assert format_bytes(512) == "512 B"
    assert format_bytes(1536) == "1.5 KiB"
    assert format_bytes(58 * 2**30) == "58.0 GiB"


def test_inventory_table_lists_full_revision_and_claim_ids():
    claimed = entry(
        repo="org/claimed",
        revision="b" * 40,
        pinned=True,
        claims={"claim-1": {"workload": "wl", "hosts": ["n1", "n2"], "created": NOW - 7200}},
        eligible=False,
        reason="pinned",
        reason_code="pinned",
    )
    untracked = entry(repo="org/loose", revision="main", tracked=False, last_used=None, eligible=False, reason="untracked")
    text = format_model_cache_reports([report([entry(), claimed, untracked])], action="inventory", now=NOW, show_untracked=True)
    assert "node1  /models/hf  (free 900.0 GiB)" in text
    assert "org/loose" in text and "not shown" not in text
    assert SHA in text and "b" * 40 in text  # pin/reconcile need the full commit
    assert "9d ago" in text
    assert "org/claimed@bbbbbbbbbbbb  claim-1  workload=wl  hosts=n1,n2  created 2h ago" in text
    assert "3 revision(s), 2 tracked; 1 eligible for pruning (~58.0 GiB reclaimable)." in text
    assert "{" not in text


def test_prune_preview_and_apply_footers():
    preview = report([entry(action="would_remove")], planned_bytes=58 * 2**30, free_bytes_after=958 * 2**30)
    text = format_model_cache_reports([preview], action="prune", now=NOW)
    assert "would remove (eligible)" in text
    assert "Would reclaim ~58.0 GiB from 1 revision(s)." in text
    assert "Free space after: 958.0 GiB (projected)" in text
    assert "Re-run with --apply" in text

    applied = report(
        [entry(action="removed")], dry_run=False, reclaimed_bytes=58 * 2**30, target_met=False, free_bytes_after_kind="measured"
    )
    text = format_model_cache_reports([applied], action="prune", now=NOW)
    assert "Reclaimed ~58.0 GiB from 1 revision(s)." in text
    assert "Space target not met" in text
    assert "--apply" not in text


def test_untracked_snapshots_are_counted_not_listed_by_default():
    untracked = entry(repo="org/loose", revision="main", tracked=False, last_used=None, eligible=False, reason="untracked")
    text = format_model_cache_reports([report([entry(), untracked])], now=NOW)
    assert "org/loose" not in text
    assert "2 revision(s), 1 tracked" in text
    assert "1 untracked snapshot(s) not shown" in text and "--all" in text
    only = format_model_cache_reports([report([untracked])], now=NOW)
    assert "No managed revisions." in only and "1 untracked snapshot(s) not shown" in only


def test_list_all_flag_shows_untracked(run, monkeypatch):
    untracked = entry(repo="org/loose", revision="main", tracked=False, last_used=None, eligible=False, reason="untracked")
    result, inventory, _ = run()
    inventory.return_value = (report([untracked]),)
    assert "org/loose" not in run()[0].output
    assert "org/loose" in run("--all")[0].output


def test_empty_cache_report():
    assert "No cached revisions." in format_model_cache_reports([report([])], now=NOW)


def test_validation_table_truncates_failures():
    failures = tuple(ModelFileFailure("shard-%03d.safetensors" % i, "missing") for i in range(25))
    text = format_cache_validation_reports(
        [
            CacheValidationReport("node1", "m", "metadata", "incomplete", 30, failures),
            CacheValidationReport("node2", "", "metadata", "unverified", detail="no saved manifest"),
        ]
    )
    lines = text.splitlines()
    assert lines[0].split() == ["HOST", "STATUS", "LEVEL", "FILES", "DETAIL"]
    assert "node2  unverified  metadata  0      no saved manifest" in text
    assert "node1: 25 problem file(s)" in text
    assert "shard-019.safetensors: missing" in text and "shard-020" not in text
    assert "... and 5 more (use --json for the full list)" in text
