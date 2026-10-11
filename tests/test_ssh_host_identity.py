"""Cloned-machine checks and the stdlib remote rotation payload, in temp roots."""

from dataclasses import replace
import base64
import hashlib
import hmac
import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.api.setup import SshHostIdentity, check_ssh_host_identities
from sparkrun.api.setup._ssh_identity import find_duplicate_identities, public_key, parse_identity, payload
from sparkrun.api.setup import _ssh_identity_remote as remote
from sparkrun.orchestration.ssh import RemoteResult


@pytest.fixture
def keys(tmp_path):
    result = []
    for index in range(3):
        path = tmp_path / str(index)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(path)], check=True, capture_output=True)
        result.append(path.with_suffix(".pub").read_text().strip())
    return result


def identity(host, key, machine="a", boot="b"):
    return SshHostIdentity(host, {"ssh_host_ed25519_key.pub": public_key(key)[0]}, machine * 64, boot * 64)


def test_duplicate_key_ignores_comments_and_filename(keys):
    a = identity("h1", keys[0] + " first")
    b = replace(identity("h2", keys[0] + " second", "c", "d"), public_keys={"other.pub": public_key(keys[0])[0]})
    duplicates = find_duplicate_identities([a, b])
    assert len(duplicates) == 1
    assert duplicates[0].kind == "SSH host key"
    assert duplicates[0].hosts == ("h1", "h2")
    assert duplicates[0].fingerprint.startswith("SHA256:")


def test_duplicate_machine_ids_are_independent_of_keys(keys):
    duplicates = find_duplicate_identities([identity("h1", keys[0]), identity("h2", keys[1], boot="c")])
    assert [d.kind for d in duplicates] == ["machine-id"]


def test_aliases_of_one_running_machine_are_not_duplicates(keys):
    assert not find_duplicate_identities([identity("h1", keys[0]), identity("10.0.0.1", keys[0])])


def test_missing_boot_id_does_not_hide_possible_duplicates(keys):
    a = replace(identity("h1", keys[0]), boot_id_hash=None)
    b = replace(identity("h2", keys[0]), boot_id_hash=None)
    assert {d.kind for d in find_duplicate_identities([a, b])} == {"machine-id", "SSH host key"}


def test_probe_retains_missing_failed_and_invalid_hosts(monkeypatch, keys):
    data = {"keys": {"key.pub": keys[0]}, "machine_id": "a" * 64, "boot_id": "b" * 64}
    run = Mock(
        return_value=[
            RemoteResult("good", 0, remote.MARKER + json.dumps(data), ""),
            RemoteResult("denied", 255, "", "Permission denied"),
            RemoteResult("bad", 0, "login banner only", ""),
        ]
    )
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", run)
    report = check_ssh_host_identities(["good", "denied", "bad", "missing"], ssh_kwargs={"ssh_user": "ubuntu"})
    assert not report.complete
    assert len(report.identities) == 4
    assert [bool(i.errors) for i in report.identities] == [False, True, True, True]
    assert run.call_args.kwargs["ssh_user"] == "ubuntu"


def test_probe_dry_run_never_connects(monkeypatch):
    run = Mock(side_effect=AssertionError("no connections"))
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", run)
    assert not check_ssh_host_identities(["h1"], dry_run=True).complete
    run.assert_not_called()


def test_bad_keys_and_ids_are_reported_as_incomplete():
    result = parse_identity("h", {"keys": {"key": "ssh-ed25519 $$$"}, "machine_id": "not-a-hash"})
    assert not result.public_keys and result.machine_id_hash is None and result.errors
    with pytest.raises(ValueError):
        payload(remote.MARKER + "{}\n" + remote.MARKER + "{}")


def test_known_hosts_update_preserves_unrelated_aliases_hashes_and_markers(keys):
    key, new = public_key(keys[0])[0], public_key(keys[1])[0]
    salt = b"s" * 20
    hashed = "|1|%s|%s" % (
        base64.b64encode(salt).decode(),
        base64.b64encode(hmac.new(salt, b"h1", hashlib.sha1).digest()).decode(),
    )
    text = "# comment\nh1,h2 %s\n%s %s\nuntouched %s\n@revoked h1 %s\n" % (key, hashed, key, key, key)
    updated = remote.known_hosts_text(text, {"h1": [new]})
    assert "h1,h2" not in updated and hashed not in updated
    assert "h2 " + key in updated and "untouched " + key in updated
    assert "@revoked h1 " + key in updated
    assert "h1 " + new in updated
    assert remote.known_hosts_text(updated, {"h1": [new]}) == updated


def test_matching_known_hosts_wildcard_is_not_silently_retained(keys):
    with pytest.raises(ValueError, match="wildcard"):
        remote.known_hosts_text("*.lab %s\n" % keys[0], {"h1.lab": [keys[1]]})


@pytest.fixture
def target(tmp_path, monkeypatch, keys):
    if os.name == "nt":
        pytest.skip("the remote payload targets Linux, independently of the controller OS")
    root = tmp_path / "target"
    ssh = root / "etc/ssh"
    ssh.mkdir(parents=True)
    (root / "etc/machine-id").write_text("a" * 32 + "\n")
    dbus = root / "var/lib/dbus"
    dbus.mkdir(parents=True)
    (dbus / "machine-id").write_text("a" * 32 + "\n")
    boot = root / "proc/sys/kernel/random"
    boot.mkdir(parents=True)
    (boot / "boot_id").write_text("b" * 32)
    for kind in remote.KEY_TYPES:
        (ssh / ("ssh_host_%s_key" % kind)).write_text("old private " + kind)
        (ssh / ("ssh_host_%s_key.pub" % kind)).write_text(keys[0])
    calls = []
    failures = []

    def run(args, *, check=True):
        calls.append(args)
        if args[0] == "ssh-keygen":
            return subprocess.run(args, check=True, capture_output=True, text=True)
        if args[0] == "ip":
            return SimpleNamespace(stdout="[]", returncode=0)
        if args[-1] == "-T":
            return SimpleNamespace(stdout="hostkey /etc/ssh/ssh_host_ed25519_key\n", returncode=0)
        if args[:2] == ["systemctl", "reload"] and failures:
            failures.pop()
            raise subprocess.CalledProcessError(1, args, stderr="reload failed")
        return SimpleNamespace(stdout="yes\n", returncode=0)

    monkeypatch.setattr(remote, "_run", run)
    return SimpleNamespace(root=root, calls=calls, failures=failures, token="c" * 32, ssh=ssh)


def test_remote_probe_never_exports_raw_machine_id_or_private_keys(target):
    data = remote.probe(target.root)
    assert data["machine_id"] == hashlib.sha256(("a" * 32).encode()).hexdigest()
    assert "a" * 32 not in json.dumps(data)
    assert "old private" not in json.dumps(data)
    assert not data["errors"]


def test_staged_rotation_reloads_and_keeps_backup_until_verified(target):
    original = (target.ssh / "ssh_host_ed25519_key").read_bytes()
    staged = remote.prepare(target.token, root=target.root)
    assert (target.ssh / "ssh_host_ed25519_key").read_bytes() == original
    assert set(staged["keys"]) == {"ssh_host_%s_key.pub" % k for k in remote.KEY_TYPES}
    remote.activate(target.token, root=target.root)
    assert remote.probe(target.root)["keys"] == staged["keys"]
    assert (target.root / "etc/machine-id").read_text().strip() == "a" * 32
    assert (remote._stage(target.root) / "old").is_dir()
    assert ["systemctl", "reload", "ssh.service"] in target.calls
    remote.finish(target.token, root=target.root)
    assert not remote._stage(target.root).exists()


def test_machine_id_rotation_changes_dbus_copy_without_reusing_it(target):
    staged = remote.prepare(target.token, machine_id=True, root=target.root)
    remote.activate(target.token, root=target.root)
    current = (target.root / "etc/machine-id").read_text()
    assert current.strip() != "a" * 32
    assert (target.root / "var/lib/dbus/machine-id").read_text() == current
    assert remote.probe(target.root)["machine_id"] == staged["machine_id"]
    remote.rollback(target.token, root=target.root)
    assert (target.root / "etc/machine-id").read_text().strip() == "a" * 32
    assert (target.root / "var/lib/dbus/machine-id").read_text().strip() == "a" * 32


def test_reload_failure_restores_all_original_identities(target):
    before = {path.name: path.read_bytes() for path in target.ssh.iterdir()}
    remote.prepare(target.token, machine_id=True, root=target.root)
    target.failures.append(True)
    with pytest.raises(subprocess.CalledProcessError):
        remote.activate(target.token, root=target.root)
    for name, original in before.items():
        assert (target.ssh / name).read_bytes() == original
    assert (target.root / "etc/machine-id").read_text().strip() == "a" * 32
    remote.finish(target.token, root=target.root)


def test_existing_rotation_and_wrong_token_are_refused(target):
    remote.prepare(target.token, root=target.root)
    with pytest.raises(FileExistsError):
        remote.prepare("d" * 32, root=target.root)
    with pytest.raises(ValueError, match="another"):
        remote.finish("d" * 32, root=target.root)
    assert remote._stage(target.root).exists()


def test_machine_id_symlink_is_refused_before_changing_any_key(target):
    path = target.root / "etc/machine-id"
    path.unlink()
    path.symlink_to(target.root / "var/lib/dbus/machine-id")
    with pytest.raises(ValueError, match="regular file"):
        remote.prepare(target.token, machine_id=True, root=target.root)
    assert (target.ssh / "ssh_host_ed25519_key").read_text() == "old private ed25519"
    assert not remote._stage(target.root).exists()


@pytest.mark.skipif(os.name == "nt", reason="remote Linux file locking")
def test_inflight_action_prevents_concurrent_cleanup(tmp_path, monkeypatch):
    import fcntl

    stage = tmp_path / "rotation"
    monkeypatch.setattr(remote, "STAGE", str(stage))
    monkeypatch.setattr(remote.os, "geteuid", lambda: 0)
    finish = Mock(side_effect=AssertionError("must not delete in-flight backups"))
    monkeypatch.setattr(remote, "finish", finish)
    with (tmp_path / "rotation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="still running"):
            remote.main("finish", token="a" * 32)
    finish.assert_not_called()
