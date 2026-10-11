"""Rotation orchestration and CLI confirmation without contacting real hosts."""

import ast
import base64
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from click.testing import CliRunner
import pytest

from sparkrun.api.setup import SshIdentityReport, SshRotationResult
from sparkrun.api.setup import _ssh_rotation as rotation
from sparkrun.api.setup._ssh_identity import public_key
from sparkrun.api.setup._ssh_identity_remote import MARKER
from sparkrun.cli import main
from sparkrun.cli._setup._ssh import _check_ssh_host_identities
from sparkrun.orchestration.ssh import RemoteResult
from test_ssh_host_identity import keys as keys, identity


def _script_args(script):
    source = base64.b64decode(script.split()[2]).decode()
    call = ast.parse(source).body[-1].body[0].value
    return ast.literal_eval(call.args[0]), {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}


@pytest.fixture
def env(tmp_path, monkeypatch, keys):
    old = [identity("h1", keys[0]), identity("h2", keys[0], boot="c")]
    report = SshIdentityReport(tuple(old))
    monkeypatch.setattr(rotation, "check_ssh_host_identities", Mock(return_value=report))
    known = tmp_path / "ssh config" / "known_hosts"
    known.parent.mkdir()
    known.write_text("h1 %s\nh2 %s\nunrelated %s\n" % tuple([public_key(keys[0])[0]] * 3))
    monkeypatch.setattr(rotation, "_connection", lambda host, kw: rotation._Connection((host,), (known,), 22))
    stages = {}
    calls = []
    recovery = tmp_path / "recovery"
    fault = {}

    def response(host, data=None, error=""):
        return RemoteResult(host, 1 if error else 0, MARKER + json.dumps(data) if data else "", error)

    def data(host, machine=False):
        index = int(host[-1])
        return {
            "keys": {"ssh_host_ed25519_key.pub": public_key(keys[index])[0]},
            "machine_id": ("d" if index == 1 else "e") * 64 if machine else "a" * 64,
            "boot_id": ("b" if index == 1 else "c") * 64,
            "errors": [],
        }

    def sudo(host, script, password, *, timeout, ssh_options, **kw):
        action, args = _script_args(script)
        calls.append((host, action, args, ssh_options))
        assert ssh_options[:2] == ["-o", "StrictHostKeyChecking=yes"]
        if fault.get(action) == host:
            return response(host, error=action + " failed")
        if action == "prepare":
            stages[host] = "prepared"
            return response(host, data(host, args["machine_id"]))
        if action == "activate":
            saved = json.loads(next(recovery.glob("*/recovery.json")).read_text())
            assert saved["hosts"][host]["new_keys"] == data(host)["keys"]
            assert stages[host] == "prepared"
            stages[host] = "active"
            if fault.get("interrupt") == host:
                raise KeyboardInterrupt
        elif action == "rollback":
            stages[host] = "prepared"
        elif action == "finish":
            stages.pop(host)
        return response(host, {action: True})

    def run(host, script, *, ssh_options, **kw):
        action, args = _script_args(script)
        calls.append((host, action, args, ssh_options))
        if action == "probe":
            if fault.get("reload_pending"):
                fault["reload_pending"] -= 1
                return response(host, error="listener not ready yet")
            if fault.get("verify") == host:
                return response(host, error="host key verification failed")
            known_option = next(option for option in ssh_options if option.startswith("UserKnownHostsFile="))
            pinned = Path(known_option.split("=", 1)[1].strip('"')).read_text()
            assert public_key(keys[0])[0] not in pinned
            assert data(host)["keys"]["ssh_host_ed25519_key.pub"] in pinned
            assert "ControlPath=none" in ssh_options
            result = data(host, fault.get("machine", False))
            if fault.get("wrong_key") == host:
                result["keys"] = old[0].public_keys
            return response(host, result)
        assert action == "known-hosts"
        if fault.get("peer") == host:
            return response(host, error="peer refresh failed")
        return response(host, {"updated": True})

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_sudo_script", sudo)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", run)
    return SimpleNamespace(known=known, recovery=recovery, calls=calls, stages=stages, fault=fault, keys=keys, old=old)


def _rotate(env, **kwargs):
    return rotation.rotate_ssh_host_keys(
        ["h1", "h2"],
        ssh_kwargs={"ssh_user": "ubuntu", "ssh_options": ["-o", "StrictHostKeyChecking=no"]},
        recovery_dir=env.recovery,
        **kwargs,
    )


@pytest.mark.parametrize("machine", [False, True])
def test_rotation_updates_controller_and_peers_and_removes_backups(env, machine):
    env.fault["machine"] = machine
    result = _rotate(env, machine_id=machine)
    assert result.ok, result.failures
    assert result.rotated == ("h1", "h2") and result.machine_id_rotated is machine
    assert not env.stages
    text = env.known.read_text()
    assert "h1 " + public_key(env.keys[1])[0] in text
    assert "h2 " + public_key(env.keys[2])[0] in text
    assert "unrelated " + public_key(env.keys[0])[0] in text
    assert "h1 " + public_key(env.keys[0])[0] not in text
    for host, action, args, _ in env.calls:
        if action == "known-hosts":
            assert args["records"]["localhost"] == [public_key(env.keys[int(host[-1])])[0]]
            assert args["records"]["h1"] == [public_key(env.keys[1])[0]]
            assert args["records"]["h2"] == [public_key(env.keys[2])[0]]
    saved = json.loads(Path(result.recovery_file).read_text())
    assert all(value["phase"] == "complete" for value in saved["hosts"].values())


def test_failed_preparation_never_activates_and_cleans_prepared_host(env):
    env.fault["prepare"] = "h2"
    result = _rotate(env)
    assert not result.ok and "preparation failed" in result.failures["h2"]
    assert not env.stages
    assert not any(action == "activate" for _, action, _, _ in env.calls)
    assert "h1 " + public_key(env.keys[0])[0] in env.known.read_text()


@pytest.mark.parametrize("failure", ["activate", "verify", "wrong_key"])
def test_partial_failure_rolls_back_failed_host_and_keeps_successful_host(env, failure):
    env.fault[failure] = "h2"
    result = _rotate(env)
    assert result.rotated == ("h1",)
    assert not result.ok and "original identities restored" in result.failures["h2"]
    assert not env.stages
    assert "h1 " + public_key(env.keys[1])[0] in env.known.read_text()
    assert "h2 " + public_key(env.keys[0])[0] in env.known.read_text()


def test_failed_rollback_retains_remote_and_local_recovery(env):
    env.fault.update(verify="h1", rollback="h1")
    result = _rotate(env)
    assert "ROLLBACK FAILED" in result.failures["h1"]
    assert env.stages == {"h1": "active"}
    assert Path(result.recovery_file).is_file()
    assert list(env.recovery.glob("*/*-recovery.known_hosts"))


def test_interrupted_activation_never_deletes_recovery_keys(env):
    env.fault["interrupt"] = "h1"
    with pytest.raises(KeyboardInterrupt):
        _rotate(env)
    assert env.stages == {"h1": "active"}
    saved = json.loads(next(env.recovery.glob("*/recovery.json")).read_text())
    assert saved["hosts"]["h1"]["phase"] == "activating"


def test_unknown_host_prevents_any_rotation(env, monkeypatch):
    monkeypatch.setattr(
        rotation, "check_ssh_host_identities", lambda *a, **kw: SshIdentityReport((replace(env.old[0], errors=("changed host key",)),))
    )
    result = _rotate(env)
    assert not result.ok
    assert not env.calls


def test_multiple_addresses_of_one_machine_are_rejected_for_rotation(env, monkeypatch):
    monkeypatch.setattr(
        rotation, "check_ssh_host_identities", lambda *a, **kw: SshIdentityReport(tuple(replace(i, boot_id_hash="b" * 64) for i in env.old))
    )
    result = _rotate(env)
    assert "same running machine" in result.failures["h2"]
    assert not env.calls


def test_rotation_dry_run_never_probes_or_writes(env, monkeypatch):
    probe = Mock(side_effect=AssertionError("dry run connected"))
    monkeypatch.setattr(rotation, "check_ssh_host_identities", probe)
    _rotate(env, dry_run=True)
    assert not env.calls and not env.recovery.exists()
    probe.assert_not_called()


def test_effective_config_preserves_paths_with_spaces_and_custom_port(tmp_path, monkeypatch):
    first = tmp_path / "home with spaces" / "known_hosts"
    second = tmp_path / "another home" / "known_hosts2"
    output = "hostname 10.0.0.1\nport 2222\nhostkeyalias trusted-name\nuserknownhostsfile %s %s\n" % (first, second)
    monkeypatch.setattr(rotation.subprocess, "run", Mock(return_value=SimpleNamespace(stdout=output)))
    result = rotation._connection("spark", {})
    assert result.files == (first, second)
    assert {"[spark]:2222", "[10.0.0.1]:2222", "trusted-name"} <= set(result.names)


@pytest.fixture
def cli(monkeypatch, tmp_path, keys):
    resolve = Mock(return_value=(["h1", "h2"], "ubuntu", {"ssh_user": "ubuntu", "ssh_key": "/key", "ssh_options": ["-p", "2222"]}))
    rotate = Mock(return_value=SshRotationResult(("h1", "h2")))
    mesh = Mock(return_value=True)
    audit = Mock(return_value=SshIdentityReport((identity("h1", keys[1]), identity("h2", keys[2], "c", "d"))))
    monkeypatch.setattr("sparkrun.cli._setup._ssh_rotate._resolve_setup_context", resolve)
    monkeypatch.setattr("sparkrun.api.setup.rotate_ssh_host_keys", rotate)
    monkeypatch.setattr("sparkrun.cli._setup._ssh._run_ssh_mesh_native", mesh)
    monkeypatch.setattr("sparkrun.api.setup.check_ssh_host_identities", audit)
    monkeypatch.setattr("sparkrun.cli._setup._sudo._record_setup_phase", Mock())
    return SimpleNamespace(rotate=rotate, mesh=mesh, audit=audit, resolve=resolve)


@pytest.mark.parametrize("flags,answer", [([], "n\n"), (["--dry-run"], ""), ([], "")])
def test_decline_eof_and_dry_run_do_not_touch_hosts(cli, flags, answer):
    result = CliRunner().invoke(main, ["setup", "ssh-rotate-keys", "--cluster", "lab", *flags], input=answer)
    assert result.exit_code in (0, 1), result.output
    cli.rotate.assert_not_called()
    cli.mesh.assert_not_called()
    cli.audit.assert_not_called()


@pytest.mark.parametrize("flags,answer", [([], "y\n"), (["--yes"], "")])
def test_confirmation_rotates_and_refreshes_only_selected_targets(cli, flags, answer):
    result = CliRunner().invoke(main, ["setup", "ssh-rotate-keys", "--cluster", "lab", "--machine-id", *flags], input=answer)
    assert result.exit_code == 0, result.output
    assert cli.resolve.call_args.args[2] == "lab"
    assert cli.rotate.call_args.args[0] == ["h1", "h2"]
    assert cli.rotate.call_args.kwargs["machine_id"] is True
    assert cli.mesh.call_args.args == (["h1", "h2"], "ubuntu")
    assert cli.mesh.call_args.kwargs["ssh_key"] == "/key"
    assert cli.mesh.call_args.kwargs["ssh_options"] == ["-o", "StrictHostKeyChecking=yes", "-o", "ControlPath=none", "-p", "2222"]
    assert "Reboot each rotated host" in result.output


def test_failure_is_not_reported_as_success_or_followed_by_mesh(cli):
    cli.rotate.return_value = SshRotationResult(("h1",), {"h2": "reload failed"}, "/recovery.json")
    result = CliRunner().invoke(main, ["setup", "ssh-rotate-keys", "--cluster", "lab", "--yes"])
    assert result.exit_code == 1
    assert "reload failed" in result.output and "/recovery.json" in result.output
    assert "rotation complete" not in result.output
    cli.mesh.assert_not_called()


def test_audit_warns_with_cluster_scoped_remediation(cli, keys, capsys):
    from sparkrun.api.setup._ssh_identity import find_duplicate_identities

    items = [identity("h1", keys[0]), identity("h2", keys[0], boot="c")]
    cli.audit.return_value = SshIdentityReport(tuple(items), find_duplicate_identities(items))
    _check_ssh_host_identities(["h1", "h2"], {"ssh_user": "ubuntu"}, cluster_name="lab")
    out = capsys.readouterr()
    assert "duplicate SSH host key" in out.err and "duplicate machine-id" in out.err
    assert "--cluster" in out.err and "lab" in out.err and "--machine-id" in out.err
    assert "a" * 64 not in out.err
    assert "No duplicate" not in out.out


def test_async_sshd_reload_retries_only_with_staged_keys(env):
    env.fault["reload_pending"] = 2
    result = _rotate(env)
    assert result.ok, result.failures
    probes = [options for host, action, _, options in env.calls if host == "h1" and action == "probe"]
    assert len(probes) == 3
    assert all("StrictHostKeyChecking=yes" in options and "ControlPath=none" in options for options in probes)


@pytest.mark.parametrize("sudo", [False, True])
@pytest.mark.parametrize("name", ["plain", "path with spaces", "user's files", 'path with "quotes"'])
def test_pinned_connection_options_parse_in_real_openssh(tmp_path, monkeypatch, sudo, name):
    """Exercise both transports through subprocess argv and the real SSH parser.

    Inject -G and an empty configuration so no connection or remote operation
    happens. Transport-level mocks previously hid double-quoted argv errors.
    """
    import os
    import shutil
    import subprocess
    from sparkrun.orchestration.ssh import run_remote_script, run_remote_sudo_script

    if shutil.which("ssh") is None:
        pytest.skip("OpenSSH client unavailable")
    if os.name == "nt" and '"' in name:
        pytest.skip("Double quotes are not valid in Windows filenames")
    known = tmp_path / name / "new.known_hosts"
    config = tmp_path / "empty.conf"
    config.write_text("")
    real_run = subprocess.run

    def parse(argv, **kwargs):
        assert argv[0] == "ssh"
        return real_run([argv[0], "-G", "-F", str(config), *argv[1:]], **kwargs)

    monkeypatch.setattr("sparkrun.orchestration.ssh.subprocess.run", parse)
    pinned = rotation._pinned_kwargs({"ssh_options": ["-o", "StrictHostKeyChecking=no"]}, known)
    result = (
        run_remote_sudo_script("spark.invalid", "true", None, **pinned) if sudo else run_remote_script("spark.invalid", "true", **pinned)
    )
    assert result.success, result.stderr
    parsed = dict(line.split(None, 1) for line in result.stdout.splitlines() if " " in line)
    assert parsed["userknownhostsfile"] == known.as_posix()
    assert parsed["globalknownhostsfile"] == "none"
    assert parsed["stricthostkeychecking"] in {"true", "yes"}
    assert parsed["controlmaster"] in {"false", "no"}
    assert parsed.get("controlpath", "none") == "none"


def test_rotation_reuses_password_but_keeps_sudo_n_probe_for_cached_auth(env, monkeypatch):
    """NOPASSWD/cached auth must never receive a password as script input."""
    from sparkrun.orchestration.ssh import run_remote_sudo_script as fake_sudo
    from sparkrun.orchestration.sudo import SudoPasswords

    attempts = []
    shared = Mock(return_value="secret")
    per_host = Mock(return_value="own-secret")
    # h1 initially needs a password; its successful sudo then caches auth.
    cached = set()

    def sudo(host, script, password, **kwargs):
        action, _ = _script_args(script)
        attempts.append((host, action, password))
        assert kwargs["quiet"] is True
        if host in cached:
            assert password is None
        elif password is None:
            return RemoteResult(host, 1, "", "sudo: a password is required")
        elif password != ("secret" if host == "h1" else "own-secret"):
            return RemoteResult(host, 1, "", "sudo: incorrect password")
        if host == "h1":
            cached.add(host)
        return fake_sudo(host, script, password, **kwargs)

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_sudo_script", sudo)
    result = _rotate(env, passwords=SudoPasswords(prompt_shared=shared, prompt_host=per_host))
    assert result.ok, result.failures
    shared.assert_called_once()
    per_host.assert_called_once_with("h2")
    assert ("h1", "activate", None) in attempts
    assert ("h1", "activate", "secret") not in attempts
    assert ("h2", "activate", "own-secret") in attempts
    assert ("h2", "finish", "own-secret") in attempts
