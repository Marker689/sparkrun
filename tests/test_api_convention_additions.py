"""Additive api conventions: ``sctx=`` everywhere config is built, optional hosts.

Each entry point here previously either constructed its own
``SparkrunConfig()`` / took raw ``ssh_kwargs`` with no way to share a context,
or required a positional host list even when a cluster had been named.  The
additions are strictly additive: existing call shapes keep their behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import sparkrun.api as api
from sparkrun.api._context import default_sctx
from sparkrun.core.cluster_status import empty_status


@pytest.fixture
def sctx():
    context = default_sctx()
    context.config.ssh_user = "ctx-user"
    return context


# --------------------------------------------------------------------------
# features / plugins: config from sctx, explicit config still wins
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name, target, calls",
    [
        ("features", "change_feature", ("enable", "disable", "reset")),
        ("plugins", "change_plugin", ("enable", "disable")),
    ],
)
def test_lifecycle_actions_take_config_from_sctx(monkeypatch, sctx, module_name, target, calls):
    import importlib

    module = importlib.import_module("sparkrun.api." + module_name)
    seen = []
    monkeypatch.setattr(module, target, lambda config, name, value, dry_run=False: seen.append(config))
    explicit = object()

    for call in calls:
        getattr(module, call)("x", sctx=sctx)
        getattr(module, call)("x", config=explicit, sctx=sctx)

    assert seen == [sctx.config, explicit] * len(calls)


# --------------------------------------------------------------------------
# model_cache: SSH arguments / config from sctx, explicit keys win
# --------------------------------------------------------------------------


def test_model_cache_management_uses_sctx_ssh_settings(monkeypatch, sctx):
    import sparkrun.models.cache as cache_mod

    seen = []
    monkeypatch.setattr(cache_mod, "cache_operation", lambda **kwargs: seen.append(kwargs["ssh_kwargs"]) or ())

    api.model_cache.inventory_cache(cache_dir="/c", sctx=sctx)
    api.model_cache.prune_cache(cache_dir="/c", sctx=sctx, ssh_kwargs={"ssh_user": "explicit"})
    api.model_cache.inventory_cache(cache_dir="/c", ssh_kwargs={"ssh_user": "bare"})

    assert seen[0]["ssh_user"] == "ctx-user"
    assert seen[1]["ssh_user"] == "explicit"
    assert seen[2] == {"ssh_user": "bare"}  # unchanged without sctx


def test_model_cache_repair_takes_config_and_ssh_from_sctx(monkeypatch, sctx):
    import sparkrun.api.model_cache as model_cache
    from sparkrun.models.artifacts import ModelArtifactRequest

    seen = {}
    monkeypatch.setattr(model_cache, "prepare_model", lambda model, hosts, **kwargs: seen.update(kwargs))

    model_cache.repair_model(ModelArtifactRequest("org/m"), cache_dir="/c", sctx=sctx)

    assert seen["config"] is sctx.config
    assert seen["ssh_kwargs"]["ssh_user"] == "ctx-user"


# --------------------------------------------------------------------------
# setup.run_setup_steps: a config-less context borrows the sctx config
# --------------------------------------------------------------------------


def test_run_setup_steps_fills_context_config_from_sctx(monkeypatch, sctx):
    import sparkrun.api.setup._runner as runner
    from sparkrun.core.setup_actions import SetupActionContext
    from sparkrun.core.setup_models import CheckContext

    seen = []
    monkeypatch.setattr(runner, "_run_setup_steps", lambda states, context, action, **kwargs: seen.append(context.config))
    action = SetupActionContext(user="u", dry_run=True)
    own = SimpleNamespace()

    runner.run_setup_steps({}, CheckContext(cluster_name=None, multi_host=False), action, sctx=sctx)
    runner.run_setup_steps({}, CheckContext(cluster_name=None, multi_host=False, config=own), action, sctx=sctx)
    runner.run_setup_steps({}, CheckContext(cluster_name=None, multi_host=False), action)

    assert seen == [sctx.config, own, None]


# --------------------------------------------------------------------------
# status / status_report / stop_all / open_telemetry: hosts from the cluster
# --------------------------------------------------------------------------


@pytest.fixture
def lab(sctx):
    sctx.cluster_manager.create("lab", ["h1", "h2"])
    return "lab"


@pytest.fixture
def swept(monkeypatch):
    seen: list[list[str]] = []

    def _query(cluster_def, hosts, **kwargs):
        seen.append(list(hosts))
        return empty_status(hosts)

    monkeypatch.setattr("sparkrun.orchestration.executor.query_status_for_cluster", _query)
    return seen


def test_status_without_hosts_uses_the_named_cluster(sctx, lab, swept):
    snapshot = api.status(cluster=lab, sctx=sctx)
    assert swept == [["h1", "h2"]]
    assert [h.host for h in snapshot.hosts] == ["h1", "h2"]


def test_status_without_hosts_or_cluster_uses_the_default_cluster(sctx, lab, swept):
    sctx.cluster_manager.set_default(lab)
    api.status(sctx=sctx)
    assert swept == [["h1", "h2"]]


def test_status_positional_hosts_still_win(sctx, lab, swept):
    api.status(["h2"], cluster=lab, sctx=sctx)
    assert swept == [["h2"]]


def test_status_report_without_hosts_uses_the_named_cluster(sctx, lab, swept, tmp_path):
    report = api.status_report(cluster=lab, cache_dir=str(tmp_path), sctx=sctx)
    assert swept == [["h1", "h2"]]
    assert sorted(report.idle_hosts) == ["h1", "h2"]


def test_stop_all_without_hosts_sweeps_the_named_cluster(sctx, lab, swept, tmp_path):
    result = api.stop_all(cluster=lab, cache_dir=str(tmp_path), dry_run=True, sctx=sctx)
    assert swept == [["h1", "h2"]]
    assert result.dry_run is True


def test_status_without_any_host_source_is_hosts_unreachable(sctx, swept):
    with pytest.raises(api.HostsUnreachable):
        api.status(sctx=sctx)


def test_open_telemetry_without_hosts_uses_the_named_cluster(monkeypatch, sctx, lab):
    opened = []

    class _Provider:
        def open(self, hosts, **kwargs):
            opened.append(list(hosts))
            return "session"

    monkeypatch.setattr("sparkrun.orchestration.telemetry.get_telemetry_provider", lambda scope, v=None: _Provider())

    assert api.open_telemetry(cluster=lab, sctx=sctx) == "session"
    assert opened == [["h1", "h2"]]


def test_stop_all_refuses_without_hosts_or_cluster(sctx):
    """A teardown never falls back to the default cluster or default_hosts."""
    from sparkrun import api

    with pytest.raises(api.SparkrunError, match="requires hosts or a cluster"):
        api.stop_all(sctx=sctx)
