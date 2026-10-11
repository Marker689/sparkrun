"""Identity audit integration through shared setup probes, plans and frontends."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from sparkrun.api.setup import SshRotationResult, probe_setup_hosts, build_setup_plan, evaluate_host
from sparkrun.api.setup._ssh_identity import identity_script
from sparkrun.api.setup._ssh_identity_remote import MARKER
from sparkrun.cli import main
from sparkrun.cli._setup._ssh_rotate import _offer_ssh_identity_rotation
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.setup_models import OK, WARN
from sparkrun.core.setup_steps import register_setup_constraint
from sparkrun.orchestration.ssh import RemoteResult
from test_setup_check import _facts_kv, _FACTS_ALL_GOOD
from test_ssh_host_identity import keys as keys
from test_wizard_ssh_gate import patched_cluster_mgr as patched_cluster_mgr


@pytest.fixture
def measured(monkeypatch, keys):
    hosts = ["10.0.0.1", "10.0.0.2"]
    data = {
        host: {"keys": {"key.pub": keys[index]}, "machine_id": str(index + 1) * 64, "boot_id": str(index + 3) * 64}
        for index, host in enumerate(hosts)
    }
    failures = {}
    calls = []
    script = identity_script()

    def remote(host, body, **kwargs):
        calls.append((host, body, kwargs))
        if failures.get(host) == "unreachable":
            return RemoteResult(host, 255, "", "Connection refused")
        if body == script:
            return RemoteResult(host, 0, MARKER + json.dumps(data[host]), "")
        return RemoteResult(host, 0, _facts_kv(dict(_FACTS_ALL_GOOD, CHECK_MESH_TOTAL="1", CHECK_MESH_OK="1")), "")

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", remote)
    monkeypatch.setattr("sparkrun.orchestration.networking.detect_cx7_for_hosts", Mock(return_value={}))
    monkeypatch.setattr("sparkrun.api.setup._rdma._run_probe", Mock(return_value={}))
    rotate = Mock(return_value=SshRotationResult(tuple(hosts)))
    mesh = Mock(return_value=True)
    monkeypatch.setattr("sparkrun.api.setup.rotate_ssh_host_keys", rotate)
    monkeypatch.setattr("sparkrun.cli._setup._ssh._run_ssh_mesh_native", mesh)
    kwargs = {"ssh_user": "tester", "ssh_key": "/test-key", "ssh_options": ["-p", "2222"]}
    return SimpleNamespace(
        hosts=hosts,
        data=data,
        failures=failures,
        calls=calls,
        script=script,
        config=SparkrunConfig(),
        kwargs=kwargs,
        rotate=rotate,
        mesh=mesh,
    )


def probe(env, **kwargs):
    return probe_setup_hosts(env.hosts, ssh_kwargs=env.kwargs, config=env.config, **kwargs)


def checks(states, ctx):
    return {
        host: next(item for item in evaluate_host(state, ctx) if item.key == "ssh_identity")
        for host, state in states.items()
        if state.reachable
    }


def duplicate(env, kind):
    if kind in {"key", "both"}:
        env.data[env.hosts[1]]["keys"] = env.data[env.hosts[0]]["keys"]
    if kind in {"machine", "both"}:
        env.data[env.hosts[1]]["machine_id"] = env.data[env.hosts[0]]["machine_id"]


@pytest.mark.parametrize("kind", ["key", "machine", "both"])
def test_api_shared_step_reports_duplicates_without_actions(measured, kind):
    duplicate(measured, kind)
    states, context = probe(measured, cluster_name="my lab")
    assert context.ssh_identity_report.complete
    for host, item in checks(states, context).items():
        assert item.status == WARN
        assert "duplicate" in item.detail and all(h in item.detail for h in measured.hosts)
        assert "--cluster 'my lab'" in item.guidance and "--user tester" in item.guidance
        assert ("--machine-id" in item.guidance) == (kind != "key")
        assert "1" * 64 not in item.detail  # machine-ID hashes stay out of user output
        entries = {e.step.key: e for e in build_setup_plan(states[host], context)}
        assert entries["ssh_identity"].step.apply is None
        assert not entries["ssh_identity"].needs_action
        assert entries["ssh_mesh"].satisfied  # duplicate identities do not cause an unnecessary re-mesh
    measured.rotate.assert_not_called()
    measured.mesh.assert_not_called()


@pytest.mark.parametrize("problem", ["none", "missing", "unreachable", "aliases"])
def test_shared_step_distinguishes_unique_incomplete_and_aliases(measured, problem):
    if problem == "missing":
        measured.data[measured.hosts[1]].pop("machine_id")
    elif problem == "unreachable":
        measured.failures[measured.hosts[1]] = "unreachable"
    elif problem == "aliases":
        measured.data[measured.hosts[1]] = measured.data[measured.hosts[0]]
    states, ctx = probe(measured)
    for item in checks(states, ctx).values():
        assert item.status == (WARN if problem in {"missing", "unreachable"} else OK)
        if item.status == WARN:
            assert "incomplete" in item.detail and "ssh-rotate-keys" not in item.guidance
        else:
            assert "no duplicates" in item.detail


@pytest.mark.parametrize("mode", ["disabled", "constraint", "single", "discovery"])
def test_step_selection_controls_identity_io(measured, monkeypatch, mode):
    if mode == "disabled":
        monkeypatch.setenv("SPARKRUN_FEATURE_SETUP_STEPS_SSH_IDENTITY", "0")
    elif mode == "constraint":
        register_setup_constraint(
            "no_identity", lambda key, state, ctx: "excluded" if key == "ssh_identity" and state.host == measured.hosts[1] else ""
        )
    elif mode == "single":
        measured.hosts[:] = measured.hosts[:1]
    states, ctx = probe(measured, discovery_only=(mode == "discovery"))
    audited = [host for host, script, _ in measured.calls if script == measured.script]
    assert audited == (measured.hosts[:1] if mode == "constraint" else [])
    if mode in {"disabled", "single"}:
        assert all("ssh_identity" not in {item.key for item in evaluate_host(state, ctx)} for state in states.values())


def test_identity_probe_failure_warns_without_hiding_other_readiness(measured, monkeypatch):
    monkeypatch.setattr("sparkrun.api.setup.check_ssh_host_identities", Mock(side_effect=OSError("probe unavailable")))
    states, ctx = probe(measured)
    assert all(item.status == WARN and "probe unavailable" in item.detail for item in checks(states, ctx).values())
    assert all(state.reachable for state in states.values())


@pytest.mark.parametrize("as_json", [False, True])
def test_setup_check_reports_identity_findings_read_only(measured, as_json):
    duplicate(measured, "both")
    result = CliRunner().invoke(main, ["setup", "check", "--hosts", ",".join(measured.hosts), *(["--json"] if as_json else [])])
    assert result.exit_code == 0, result.output
    if as_json:
        data = json.loads(result.output)
        assert data["advisories"] == 2 and data["critical_gaps"] == 0
        for row in data["results"].values():
            item = next(i for i in row["checks"] if i["key"] == "ssh_identity")
            assert item["status"] == WARN
            assert "--hosts 10.0.0.1,10.0.0.2" in item["guidance"]
            assert "--machine-id" in item["guidance"]
    else:
        assert "duplicate SSH host key" in result.output and "duplicate machine-id" in result.output
        assert "ssh-rotate-keys" in result.output
    assert "Rotate these host identities now?" not in result.output
    measured.rotate.assert_not_called()
    measured.mesh.assert_not_called()


@pytest.mark.parametrize("mode", ["accept-key", "accept-machine", "decline", "yes", "dry", "incomplete", "unique"])
def test_wizard_offer_requires_confirmed_complete_duplicates(measured, monkeypatch, mode):
    if mode != "unique":
        duplicate(measured, "key" if mode == "accept-key" else "both")
    if mode == "incomplete":
        measured.data[measured.hosts[1]].pop("boot_id")
    states, ctx = probe(measured, cluster_name="lab")
    confirm = Mock(return_value=mode.startswith("accept"))
    monkeypatch.setattr("click.confirm", confirm)
    # The real rotation backend would replace these records; the offer uses
    # the real confirmation and shared frontend, with only mutation mocked.
    done = _offer_ssh_identity_rotation(states, ctx, "tester", measured.kwargs, measured.config, yes=mode == "yes", dry_run=mode == "dry")
    if mode.startswith("accept"):
        assert done
        confirm.assert_called_once_with("Rotate these host identities now?", default=False)
        measured.rotate.assert_called_once()
        assert measured.rotate.call_args.args == (measured.hosts,)
        assert measured.rotate.call_args.kwargs["ssh_kwargs"] == measured.kwargs
        assert measured.rotate.call_args.kwargs["machine_id"] == (mode == "accept-machine")
        measured.mesh.assert_called_once()
    else:
        assert not done
        measured.rotate.assert_not_called()
        measured.mesh.assert_not_called()
        if mode != "decline":
            confirm.assert_not_called()


@pytest.mark.parametrize("accept", [True, False])
def test_wizard_rotates_only_after_confirmation_and_reprobes(measured, monkeypatch, v, patched_cluster_mgr, accept):
    from copy import deepcopy
    from sparkrun.api.setup import SshProbe
    from sparkrun.core.setup_manifest import ManifestManager

    original = deepcopy(measured.data)
    duplicate(measured, "both")
    monkeypatch.setattr("sparkrun.api.setup.probe_ssh_access", Mock(return_value=[SshProbe(h, ok=True) for h in measured.hosts]))
    monkeypatch.setattr("sparkrun.models.distribute.detect_shared_cache", Mock(return_value=False))
    monkeypatch.setattr("sparkrun.cli._setup._ssh._run_ssh_mesh", Mock(return_value=True))
    monkeypatch.setattr("sparkrun.utils.net.local_ip_for", lambda *a: None)
    monkeypatch.setattr("subprocess.run", Mock(return_value=SimpleNamespace(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")))
    confirm = Mock(side_effect=lambda message, **kw: accept if message == "Rotate these host identities now?" else False)
    monkeypatch.setattr("click.confirm", confirm)
    probes = []

    def watch_probe(*args, **kwargs):
        result = probe_setup_hosts(*args, **kwargs)
        probes.append(result)
        return result

    monkeypatch.setattr("sparkrun.core.setup_probe.probe_setup_hosts", watch_probe)

    def rotated(*args, **kwargs):
        measured.data.update(original)
        return SshRotationResult(tuple(measured.hosts), machine_id_rotated=True)

    measured.rotate.side_effect = rotated
    result = CliRunner().invoke(main, ["setup", "wizard", "--hosts", ",".join(measured.hosts), "--cluster", "lab", "--user", "tester"])
    assert result.exit_code == 0, result.output
    assert "duplicate machine-id" in result.output
    assert sum(call.args[0] == "Rotate these host identities now?" for call in confirm.call_args_list) == 1
    if accept:
        measured.rotate.assert_called_once()
        measured.mesh.assert_called_once()
        assert len(probes) >= 2
        assert probes[0][1].ssh_identity_report.duplicates
        assert not probes[-1][1].ssh_identity_report.duplicates
        assert "Readiness: ok" in result.output
        assert "Reboot each rotated host" in result.output
        manifest = ManifestManager(patched_cluster_mgr.clusters_dir).load("lab")
        assert set(manifest.phases["ssh_mesh"].hosts) == set(measured.hosts)
    else:
        measured.rotate.assert_not_called()
        measured.mesh.assert_not_called()
        assert "Rotation cancelled" in result.output
        assert "Readiness: needs attention" in result.output
