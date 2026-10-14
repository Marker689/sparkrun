"""Tests for Click ``shell_complete`` methods on sparkrun CLI parameter types.

Covers the kubectl-style completion added for ``logs``/``stop`` (and the
other ``TARGET``-taking commands): ``TargetType.shell_complete`` returns
running-workload cluster_ids from the local job metadata cache, falling
back to recipe-name completion when no jobs are cached.
"""

from __future__ import annotations

from _status_fixtures import host_snapshot

from pathlib import Path
from _status_fixtures import observed_status

import pytest
from unittest import mock
import yaml

from sparkrun.cli._common import (
    TARGET,
    _complete_targets,
    _describe_job,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_job_meta(jobs_dir: Path, digest: str, **fields) -> Path:
    """Write a YAML job metadata file with sane defaults (mirrors test_api_schedule_status_jobs)."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "cluster_id": f"sparkrun_{digest}",
        "recipe": "test-recipe",
        "runtime": "vllm",
        "executor": "docker",
        "executor_destination_key": "",
        "hosts": ["h1"],
        **fields,
    }
    path = jobs_dir / (f"{digest}.yaml")
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def jobs_cache(tmp_path: Path, monkeypatch) -> Path:
    """Redirect the sparkrun cache (and thus ``~/.cache/sparkrun/jobs/``) to tmp_path.

    Also disables completion's live status sweep.  Without that every
    completion test would attempt real SSH to its fixture hostnames — slow,
    non-hermetic, and dependent on how fast the resolver fails.  Tests that
    exercise the live path re-enable it and patch ``api.status`` (see
    :class:`TestLiveStatusCompletion`).
    """
    import sparkrun.core.config as _config_module

    monkeypatch.setattr(_config_module, "DEFAULT_CACHE_DIR", tmp_path, raising=False)
    monkeypatch.setattr("sparkrun.cli._common._completion_status_timeout", lambda: 0.0)
    return tmp_path / "jobs"


# ---------------------------------------------------------------------------
# _describe_job
# ---------------------------------------------------------------------------


def test_describe_job_full():
    """recipe + runtime + hosts all present → all three in the description."""
    from sparkrun.api._models import JobInfo

    job = JobInfo(workload_id="sparkrun_abc", recipe="my-recipe", runtime="vllm", hosts=("h1", "h2"))
    desc = _describe_job(job)
    assert "my-recipe" in desc
    assert "vllm" in desc
    assert "h1,h2" in desc


def test_describe_job_empty():
    """No recipe / runtime / hosts → empty string (no crash)."""
    from sparkrun.api._models import JobInfo

    job = JobInfo(workload_id="sparkrun_abc")
    assert _describe_job(job) == ""


def test_describe_job_partial():
    """Only recipe set → just the recipe."""
    from sparkrun.api._models import JobInfo

    job = JobInfo(workload_id="sparkrun_abc", recipe="my-recipe")
    assert _describe_job(job) == "my-recipe"


# ---------------------------------------------------------------------------
# _complete_targets
# ---------------------------------------------------------------------------


def test_complete_targets_empty_when_no_cache(jobs_cache: Path):
    """No jobs directory → empty list, never raises."""
    assert _complete_targets("") == []


def test_complete_targets_returns_all_on_empty_prefix(jobs_cache: Path):
    """Empty incomplete → all cached cluster_ids."""
    _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alpha")
    _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="beta")

    items = _complete_targets("")
    assert len(items) == 2
    # No cluster on the line and none configured in this sandbox, so there is
    # nothing to scope recipe names against — the ids are still offered.
    assert {i.value for i in items} == {"sparkrun_aaaaaaaaaaaa", "sparkrun_bbbbbbbbbbbb"}


def test_complete_targets_matches_full_form(jobs_cache: Path):
    """``sparkrun_abc…`` prefix matches the canonical cluster_id."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="alpha")
    _write_job_meta(jobs_cache, "bbbbb1234567", recipe="beta")

    items = _complete_targets("sparkrun_abc")
    assert len(items) == 1
    assert items[0].value == "sparkrun_abcdef123456"


def test_complete_targets_matches_bare_digest(jobs_cache: Path):
    """Bare hex digest prefix also matches (short-form CLI input)."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="alpha")
    _write_job_meta(jobs_cache, "bbbbb1234567", recipe="beta")

    items = _complete_targets("abcd")
    assert len(items) == 1
    # The returned value is always the canonical form.
    assert items[0].value == "sparkrun_abcdef123456"


def test_complete_targets_descriptions(jobs_cache: Path):
    """Completion items carry recipe + runtime + hosts in the description."""
    _write_job_meta(
        jobs_cache,
        "abcdef123456",
        recipe="@eugr/inkling-small-nvfp4",
        runtime="vllm",
        hosts=["127.0.0.1", "192.168.70.8"],
    )

    items = _complete_targets("")
    assert len(items) == 1
    desc = items[0].help or ""
    assert "@eugr/inkling-small-nvfp4" in desc
    assert "vllm" in desc
    assert "127.0.0.1,192.168.70.8" in desc


def test_complete_targets_no_match(jobs_cache: Path):
    """Prefix that matches nothing → empty list."""
    _write_job_meta(jobs_cache, "abcdef123456")
    assert _complete_targets("zzz") == []


def test_complete_targets_handles_exception(jobs_cache: Path, monkeypatch):
    """If api.list_jobs() raises, completion degrades to empty list (never crashes)."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="alpha")

    def _boom(*a, **kw):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr("sparkrun.api.list_jobs", _boom)
    assert _complete_targets("") == []


# ---------------------------------------------------------------------------
# TargetType.shell_complete
# ---------------------------------------------------------------------------


def test_target_shell_complete_offers_cached_workloads(jobs_cache: Path):
    """When jobs are cached, TargetType completes them rather than every recipe."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="my-recipe")

    values = {i.value for i in TARGET.shell_complete(ctx=None, param=None, incomplete="")}
    assert values == {"sparkrun_abcdef123456"}


def test_target_shell_complete_falls_back_to_recipes(jobs_cache: Path):
    """No jobs cached → recipe-name completion (inherited from RecipeNameType)."""
    # jobs_cache exists but is empty (no .yaml files)
    items = TARGET.shell_complete(ctx=None, param=None, incomplete="")
    # No jobs and no registries in the test sandbox → empty list, but must
    # not raise.
    assert items == []


def test_target_shell_complete_defers_for_paths(jobs_cache: Path):
    """Path-like incomplete (./) defers to recipe/file completion, not cluster_ids."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="my-recipe")

    # A path prefix should never match a cluster_id — it goes to the parent
    # RecipeNameType.shell_complete (file / recipe path completion).
    items = TARGET.shell_complete(ctx=None, param=None, incomplete="./")
    # File completion returns [] for a non-existent directory in the test sandbox.
    cluster_id_values = {i.value for i in items} & {"sparkrun_abcdef123456"}
    assert cluster_id_values == set()


@pytest.mark.parametrize("prefix", ["~/", "/abs/"])
def test_target_shell_complete_defers_for_other_paths(jobs_cache: Path, prefix: str):
    """``~`` and ``/`` path prefixes also defer to recipe/file completion."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="my-recipe")

    items = TARGET.shell_complete(ctx=None, param=None, incomplete=prefix)
    cluster_id_values = {i.value for i in items} & {"sparkrun_abcdef123456"}
    assert cluster_id_values == set()


def test_target_shell_complete_defers_for_at_registry(jobs_cache: Path):
    """``@registry`` incomplete defers to recipe completion, not cluster_ids."""
    _write_job_meta(jobs_cache, "abcdef123456", recipe="my-recipe")

    items = TARGET.shell_complete(ctx=None, param=None, incomplete="@")
    # @ prefix triggers registry-name completion; cluster_ids should not appear.
    cluster_id_values = {i.value for i in items} & {"sparkrun_abcdef123456"}
    assert cluster_id_values == set()


def test_target_shell_complete_bare_digest(jobs_cache: Path):
    """Typing a bare hex prefix completes to the full cluster_id."""
    _write_job_meta(jobs_cache, "628f56e0461d", recipe="inkling")

    items = TARGET.shell_complete(ctx=None, param=None, incomplete="628f")
    assert len(items) == 1
    assert items[0].value == "sparkrun_628f56e0461d"


# ---------------------------------------------------------------------------
# Recipe-name completion, default-cluster scoping, and the running filter
# ---------------------------------------------------------------------------


@pytest.fixture
def default_cluster(monkeypatch):
    """Make ``resolve_cluster()`` (no args) resolve to a known host set.

    That is what a bare ``logs <recipe>`` resolves against, so it decides
    which jobs may be offered by recipe name.
    """
    from sparkrun.core.cluster_manager import ClusterDefinition

    def _fake(*args, **kwargs):
        return ClusterDefinition(name="default", hosts=["h1", "h2"])

    monkeypatch.setattr("sparkrun.api._resolve.resolve_cluster", _fake)
    return {"h1", "h2"}


def test_recipe_name_offered_for_default_cluster_jobs(jobs_cache: Path, default_cluster):
    """On bash the value is all the user sees, so it must be the meaningful one.

    ``BashComplete.format_completion`` emits ``type,value`` and drops the help
    text, so a hex cluster_id completes to something unreadable.  A recipe name
    resolves through intent discovery and is what a human actually types.
    """
    _write_job_meta(jobs_cache, "abcdef123456", recipe="my-recipe", hosts=["h1"])

    values = {i.value for i in _complete_targets("")}
    assert values == {"my-recipe"}


def test_offcluster_job_is_offered_by_id_not_recipe(jobs_cache: Path, default_cluster):
    """A recipe name only resolves against the default cluster.

    Offering one whose deployment lives elsewhere would complete to something
    that then fails to resolve — worse than not offering it.  The cluster_id
    form carries its own hosts, so it stays available.
    """
    _write_job_meta(jobs_cache, "abcdef123456", recipe="elsewhere", hosts=["tnr-dead"])

    values = {i.value for i in _complete_targets("")}
    assert values == {"sparkrun_abcdef123456"}


def test_recipe_names_are_deduped(jobs_cache: Path, default_cluster):
    """Relaunches share a recipe; the name is worth offering exactly once."""
    _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="same", hosts=["h1"])
    _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="same", hosts=["h1"])

    values = [i.value for i in _complete_targets("")]
    assert values.count("same") == 1
    # The second deployment stays addressable by id.
    assert "sparkrun_bbbbbbbbbbbb" in values or "sparkrun_aaaaaaaaaaaa" in values


def test_running_snapshot_filters_dead_jobs(jobs_cache: Path, default_cluster):
    """A fresh snapshot hides what it saw *not* running."""
    from sparkrun.orchestration.job_metadata import save_running_snapshot

    _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alive", hosts=["h1"])
    _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="dead", hosts=["h1"])
    save_running_snapshot(host_snapshot({"sparkrun_aaaaaaaaaaaa"}, ["h1", "h2"]), cache_dir=str(jobs_cache.parent))

    values = {i.value for i in _complete_targets("")}
    assert values == {"alive"}


def test_stale_snapshot_hides_nothing(jobs_cache: Path, default_cluster):
    """Completion must not hide a workload on information it can't vouch for."""
    from sparkrun.orchestration.job_metadata import save_running_snapshot

    _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alive", hosts=["h1"])
    _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="dead", hosts=["h1"])
    save_running_snapshot(host_snapshot({"sparkrun_aaaaaaaaaaaa"}, ["h1"]), cache_dir=str(jobs_cache.parent))

    with mock.patch("sparkrun.orchestration.job_metadata.RUNNING_SNAPSHOT_MAX_AGE_S", -1):
        values = {i.value for i in _complete_targets("")}
    assert values == {"alive", "dead"}


def test_uncovered_hosts_are_unknown_not_dead(jobs_cache: Path, default_cluster):
    """A partial sweep must not read as "everything else is dead".

    Placement queries a candidate subset, not the whole cluster, so a snapshot
    routinely covers only some hosts.  A job outside it was never looked at.
    """
    from sparkrun.orchestration.job_metadata import save_running_snapshot

    _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="onh1", hosts=["h1"])
    _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="onh2", hosts=["h2"])
    # Only h1 was swept, and nothing was running there.
    save_running_snapshot(host_snapshot(set(), ["h1"]), cache_dir=str(jobs_cache.parent))

    values = {i.value for i in _complete_targets("")}
    assert values == {"onh2"}, "the unswept host's job must survive; the swept host's must not"


def test_completion_parses_at_most_the_limit(jobs_cache: Path, default_cluster, monkeypatch):
    """Cost is dominated by YAML parsing, so the cap must bound the parse."""
    from sparkrun.cli import _common

    for i in range(40):
        _write_job_meta(jobs_cache, "%012x" % i, recipe="r%d" % i, hosts=["h1"])

    monkeypatch.setattr(_common, "COMPLETION_JOB_LIMIT", 5)
    parsed: list = []

    import sparkrun.api._jobs as jobs_mod

    real_from_file = jobs_mod._job_info_from_file

    def _counting(path):
        parsed.append(path)
        return real_from_file(path)

    monkeypatch.setattr(jobs_mod, "_job_info_from_file", _counting)
    _complete_targets("")
    assert len(parsed) == 5


def test_cluster_option_on_the_line_scopes_recipe_names(jobs_cache: Path, monkeypatch):
    """``logs --cluster lab <TAB>`` scopes to lab, not to the default cluster.

    Completion runs after Click has parsed the options it has seen, so the
    target is knowable.  Without this a user who always passes ``--cluster``
    (and has no ``default_hosts``) is never offered a recipe name at all.
    """
    from sparkrun.core.cluster_manager import ClusterDefinition

    def _fake(cluster_input=None, hosts_input=None, **kwargs):
        if cluster_input == "lab":
            return ClusterDefinition(name="lab", hosts=["h9"])
        raise RuntimeError("no default cluster configured")

    monkeypatch.setattr("sparkrun.api._resolve.resolve_cluster", _fake)
    _write_job_meta(jobs_cache, "abcdef123456", recipe="lab-recipe", hosts=["h9"])

    class _Ctx:
        params = {"cluster_name": "lab", "hosts": None}

    assert {i.value for i in _complete_targets("", _Ctx())} == {"lab-recipe"}
    # With no cluster on the line and none configured, there is no target to
    # scope against — so the job is offered by id rather than by a name that
    # would not resolve.  The target is never *guessed*.
    assert {i.value for i in _complete_targets("")} == {"sparkrun_abcdef123456"}


def test_url_sourced_recipe_completes_by_id(jobs_cache: Path, default_cluster):
    """A URL "name" is valid input but useless as a completion value."""
    _write_job_meta(
        jobs_cache,
        "abcdef123456",
        recipe="https://spark-arena.com/api/recipes/cd00e976/raw",
        hosts=["h1"],
    )
    assert {i.value for i in _complete_targets("")} == {"sparkrun_abcdef123456"}


class TestLiveStatusCompletion:
    """Completion queries the cluster so the list reflects what is running.

    The sweep costs an SSH round-trip per TAB.  That is the accepted price: a
    cached snapshot is only as fresh as the last command that happened to
    sweep, and a stale "running" list is the useless-hex-digest problem
    restated.
    """

    @pytest.fixture
    def live(self, monkeypatch):
        """Re-enable the sweep (the shared fixture disables it) and stub status."""
        monkeypatch.setattr("sparkrun.cli._common._completion_status_timeout", lambda: 5.0)

        from sparkrun.core.cluster_manager import ClusterDefinition

        monkeypatch.setattr(
            "sparkrun.api._resolve.resolve_cluster",
            lambda *a, **k: ClusterDefinition(name="lab", hosts=["h1", "h2"]),
        )

        def _install(running, errors=None):
            from sparkrun.core.cluster_status import HostOccupancy, RunningWorkload

            errors = errors or {}
            hosts = tuple(
                HostOccupancy(host=h, workloads=tuple(RunningWorkload(cluster_id=c) for c in running.get(h, ())))
                for h in ("h1", "h2")
                if h not in errors
            )
            calls: list = []

            def _fake_status(hosts_arg, **kwargs):
                calls.append(kwargs)
                return observed_status(hosts=hosts, executor="docker", errors=dict(errors))

            monkeypatch.setattr("sparkrun.api.status", _fake_status)
            return calls

        return _install

    def test_dead_jobs_are_not_offered(self, jobs_cache: Path, live):
        _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alive", hosts=["h1"])
        _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="dead", hosts=["h1"])
        live({"h1": ["sparkrun_aaaaaaaaaaaa"]})

        assert {i.value for i in _complete_targets("")} == {"alive"}

    def test_running_workload_without_metadata_is_still_offered(self, jobs_cache: Path, live):
        """Launched from another machine, or its metadata pruned.

        It is exactly what the user is reaching for, and the cluster just told
        us it is there — so the id is offered even with nothing cached.
        """
        _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="known", hosts=["h1"])
        live({"h1": ["sparkrun_aaaaaaaaaaaa", "sparkrun_cccccccccccc"]})

        assert {i.value for i in _complete_targets("")} == {"known", "sparkrun_cccccccccccc"}

    def test_sweep_is_bounded_by_a_timeout(self, jobs_cache: Path, live):
        """The one hard limit on how long a TAB can take."""
        _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alive", hosts=["h1"])
        calls = live({"h1": ["sparkrun_aaaaaaaaaaaa"]})

        _complete_targets("")
        assert calls and calls[0]["ssh_kwargs"]["timeout"] == 5.0

    def test_unreachable_host_leaves_its_jobs_offered(self, jobs_cache: Path, live):
        """A host we failed to reach is unknown, not empty."""
        _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="onh1", hosts=["h1"])
        _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="onh2", hosts=["h2"])
        live({"h1": []}, errors={"h2": "unreachable"})

        # h1 was swept and had nothing → its job is gone.  h2 was not → kept.
        assert {i.value for i in _complete_targets("")} == {"onh2"}

    def test_query_failure_falls_back_to_the_cached_snapshot(self, jobs_cache: Path, monkeypatch):
        from sparkrun.core.cluster_manager import ClusterDefinition
        from sparkrun.orchestration.job_metadata import save_running_snapshot

        monkeypatch.setattr("sparkrun.cli._common._completion_status_timeout", lambda: 5.0)
        monkeypatch.setattr(
            "sparkrun.api._resolve.resolve_cluster",
            lambda *a, **k: ClusterDefinition(name="lab", hosts=["h1"]),
        )
        monkeypatch.setattr("sparkrun.api.status", mock.Mock(side_effect=RuntimeError("network down")))

        _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alive", hosts=["h1"])
        _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="dead", hosts=["h1"])
        save_running_snapshot(host_snapshot({"sparkrun_aaaaaaaaaaaa"}, ["h1"]), cache_dir=str(jobs_cache.parent))

        assert {i.value for i in _complete_targets("")} == {"alive"}

    def test_no_cluster_and_no_jobs_skips_the_sweep(self, jobs_cache: Path, monkeypatch):
        """Nothing to query and nothing to offer — don't open a connection."""
        monkeypatch.setattr("sparkrun.cli._common._completion_status_timeout", lambda: 5.0)
        monkeypatch.setattr(
            "sparkrun.api._resolve.resolve_cluster",
            mock.Mock(side_effect=RuntimeError("no default cluster")),
        )
        status = mock.Mock()
        monkeypatch.setattr("sparkrun.api.status", status)

        assert _complete_targets("") == []
        status.assert_not_called()

    def test_unresolvable_cluster_skips_the_sweep(self, jobs_cache: Path, monkeypatch):
        """No target means nothing to query — and nothing gets hidden either.

        The target is taken from the command line or the configured default,
        never guessed: pointing an SSH sweep at a cluster the user did not name
        is worse than offering an unfiltered list.
        """
        monkeypatch.setattr("sparkrun.cli._common._completion_status_timeout", lambda: 5.0)
        monkeypatch.setattr(
            "sparkrun.api._resolve.resolve_cluster",
            mock.Mock(side_effect=RuntimeError("no default cluster")),
        )
        status = mock.Mock()
        monkeypatch.setattr("sparkrun.api.status", status)

        _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="r", hosts=["h1"])
        assert {i.value for i in _complete_targets("")} == {"sparkrun_aaaaaaaaaaaa"}
        status.assert_not_called()


def test_completion_status_timeout_is_configurable(monkeypatch, tmp_path):
    """Zero disables the sweep — the escape hatch for a flaky link."""
    import sparkrun.core.config as _config_module
    from sparkrun.cli._common import COMPLETION_STATUS_TIMEOUT_S, _completion_status_timeout

    monkeypatch.setattr(_config_module, "DEFAULT_CONFIG_DIR", tmp_path, raising=False)
    assert _completion_status_timeout() == COMPLETION_STATUS_TIMEOUT_S

    (tmp_path / "config.yaml").write_text("completion:\n  status_timeout_s: 0\n")
    assert _completion_status_timeout() == 0.0


def test_offcluster_jobs_are_hidden_once_the_cluster_answers(jobs_cache: Path, monkeypatch):
    """A verified sweep retires other clusters' leftovers.

    They can't be verified — a torn-down cloud instance keeps its jobs in the
    cache forever and its hostnames stop resolving — so treating them as
    "unknown, therefore keep" is what leaves a completion list full of dead
    entries.  They are reachable by naming their cluster, which sweeps it.
    """
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.cluster_status import HostOccupancy, RunningWorkload

    monkeypatch.setattr("sparkrun.cli._common._completion_status_timeout", lambda: 5.0)
    monkeypatch.setattr(
        "sparkrun.api._resolve.resolve_cluster",
        lambda *a, **k: ClusterDefinition(name="lab", hosts=["h1"]),
    )
    monkeypatch.setattr(
        "sparkrun.api.status",
        lambda hosts, **kw: observed_status(
            hosts=(HostOccupancy(host="h1", workloads=(RunningWorkload(cluster_id="sparkrun_aaaaaaaaaaaa"),)),),
            executor="docker",
        ),
    )

    _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="live-here", hosts=["h1"])
    _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="dead-here", hosts=["h1"])
    _write_job_meta(jobs_cache, "cccccccccccc", recipe="gone-cloud", hosts=["tnr-dead"])

    assert {i.value for i in _complete_targets("")} == {"live-here"}


def test_jobs_survive_when_nothing_could_be_verified(jobs_cache: Path, default_cluster):
    """Without a snapshot, hiding anything would be guessing — so nothing is."""
    _write_job_meta(jobs_cache, "cccccccccccc", recipe="gone-cloud", hosts=["tnr-dead"])
    assert {i.value for i in _complete_targets("")} == {"sparkrun_cccccccccccc"}


class TestCompletionSnapshotReuse:
    """A burst of TABs should cost one sweep, not one per keystroke."""

    @pytest.fixture
    def cluster_lab(self, monkeypatch):
        from sparkrun.core.cluster_manager import ClusterDefinition

        monkeypatch.setattr("sparkrun.cli._common._completion_status_timeout", lambda: 5.0)
        monkeypatch.setattr(
            "sparkrun.api._resolve.resolve_cluster",
            lambda *a, **k: ClusterDefinition(name="lab", hosts=["h1", "h2"]),
        )

    def _spy_status(self, monkeypatch):
        from sparkrun.core.cluster_status import HostOccupancy, RunningWorkload

        calls: list = []

        def _fake(hosts, **kwargs):
            calls.append(hosts)
            return observed_status(
                hosts=(
                    HostOccupancy(host="h1", workloads=(RunningWorkload(cluster_id="sparkrun_aaaaaaaaaaaa"),)),
                    HostOccupancy(host="h2"),
                ),
                executor="docker",
            )

        monkeypatch.setattr("sparkrun.api.status", _fake)
        return calls

    def test_fresh_covering_snapshot_skips_the_sweep(self, jobs_cache: Path, cluster_lab, monkeypatch):
        from sparkrun.orchestration.job_metadata import save_running_snapshot

        _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alive", hosts=["h1"])
        _write_job_meta(jobs_cache, "bbbbbbbbbbbb", recipe="dead", hosts=["h1"])
        save_running_snapshot(host_snapshot({"sparkrun_aaaaaaaaaaaa"}, ["h1", "h2"]), cache_dir=str(jobs_cache.parent))
        calls = self._spy_status(monkeypatch)

        assert {i.value for i in _complete_targets("")} == {"alive"}
        assert calls == [], "a fresh snapshot covering the target must not trigger SSH"

    def test_snapshot_from_another_cluster_does_not_count(self, jobs_cache: Path, cluster_lab, monkeypatch):
        """It says nothing about these hosts.

        Accepting it would mark the target's hosts unobserved, which puts every
        dead job straight back into the list.
        """
        from sparkrun.orchestration.job_metadata import save_running_snapshot

        _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alive", hosts=["h1"])
        save_running_snapshot(host_snapshot({"sparkrun_zzzz"}, ["other-host"]), cache_dir=str(jobs_cache.parent))
        calls = self._spy_status(monkeypatch)

        _complete_targets("")
        assert calls, "a snapshot covering unrelated hosts must not be reused"

    def test_partial_coverage_does_not_count(self, jobs_cache: Path, cluster_lab, monkeypatch):
        """h2 was never swept, so the snapshot can't speak for the cluster."""
        from sparkrun.orchestration.job_metadata import save_running_snapshot

        _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alive", hosts=["h1"])
        save_running_snapshot(host_snapshot({"sparkrun_aaaaaaaaaaaa"}, ["h1"]), cache_dir=str(jobs_cache.parent))
        calls = self._spy_status(monkeypatch)

        _complete_targets("")
        assert calls

    def test_expired_snapshot_triggers_a_fresh_sweep(self, jobs_cache: Path, cluster_lab, monkeypatch):
        """Otherwise a workload stopped minutes ago keeps being offered."""
        from sparkrun.orchestration.job_metadata import save_running_snapshot

        _write_job_meta(jobs_cache, "aaaaaaaaaaaa", recipe="alive", hosts=["h1"])
        save_running_snapshot(host_snapshot({"sparkrun_aaaaaaaaaaaa"}, ["h1", "h2"]), cache_dir=str(jobs_cache.parent))
        monkeypatch.setattr("sparkrun.cli._common._completion_cache_ttl", lambda: 0.0)
        calls = self._spy_status(monkeypatch)

        _complete_targets("")
        assert calls, "ttl=0 must sweep every time"


def test_status_records_the_snapshot_for_completion(tmp_path, monkeypatch):
    """``api.status`` is the choke point every sweep passes through.

    Recording there is what makes completion's first sweep pay for the ones
    after it — and what lets any other command's sweep prime it too.
    """
    import sparkrun.core.config as _config_module
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.orchestration.job_metadata import load_running_snapshot
    from sparkrun.orchestration.ssh import RemoteResult

    monkeypatch.setattr(_config_module, "DEFAULT_CACHE_DIR", tmp_path, raising=False)

    import sparkrun.api as api

    ps = '{"Names":"sparkrun_aaaaaaaaaaaaaaaa_111111111111_solo","Status":"Up 1 min","Image":"img","ID":"x","Labels":""}'
    with mock.patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=[
            RemoteResult(host="h1", returncode=0, stdout=ps, stderr=""),
            RemoteResult(host="h2", returncode=1, stdout="", stderr="unreachable"),
        ],
    ):
        api.status(["h1", "h2"], cluster=ClusterDefinition(name="c", hosts=["h1", "h2"]))

    snapshot = load_running_snapshot(cache_dir=str(tmp_path))
    running, covered = snapshot.cluster_ids, snapshot.coverage[0].hosts
    assert "sparkrun_aaaaaaaaaaaaaaaa_111111111111" in running
    # h2 failed, so it is not claimed as observed — otherwise a reader would
    # conclude "nothing running there" about a host nobody could reach.
    assert covered == {"h1"}


@pytest.mark.parametrize("command, complete_var", [("sparkrun", "_SPARKRUN_COMPLETE"), ("jetson-run", "_JETSON_RUN_COMPLETE")])
def test_completion_silences_plugin_failures_without_silencing_normal_commands(tmp_path, monkeypatch, caplog, command, complete_var):
    import click
    from click.testing import CliRunner
    from sparkrun.cli import ext
    from sparkrun.core.external_plugins import load_external_plugins
    from sparkrun.core.bootstrap import get_variables

    module_name = "completion_incompatible_" + command.replace("-", "_")
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / (module_name + ".py")).write_text("# no API declaration\n")
    variables = get_variables()
    monkeypatch.setattr(ext, "ensure_cli_extensions", lambda *_args, **_kwargs: load_external_plugins(variables, paths=[tmp_path]))
    group = ext.PluggableGroup(command)
    group.add_command(click.Command("start"))
    runner = CliRunner()
    result = runner.invoke(
        group,
        [],
        prog_name=command,
        complete_var=complete_var,
        env={
            complete_var: "bash_complete",
            "COMP_WORDS": command + " st",
            "COMP_CWORD": "1",
        },
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == "plain,start\n"
    assert result.stderr == ""
    assert "Skipping plugin" not in caplog.text
    assert "Traceback" not in caplog.text

    group._cli_ext_loaded = False
    normal = runner.invoke(group, ["--help"], prog_name=command, complete_var=complete_var)
    assert normal.exit_code == 0, normal.output
    assert "Skipping plugin %s: missing API declaration" % module_name in caplog.text
