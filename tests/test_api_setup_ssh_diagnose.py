"""api.setup.diagnose_ssh — probing and verdicts behind `setup ssh --diagnose`.

Every ``ssh`` call is answered by a fake keyed on the command shape; no host or
terminal is involved.
"""

from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from sparkrun.api.setup import (
    SSH_DIAG_CONNECT_FAILED,
    SSH_DIAG_HOST_DONE,
    SSH_DIAG_KEY_CHECK_DONE,
    SSH_DIAG_KEY_CHECK_SKIPPED,
    SSH_DIAG_LOCAL_KEY,
    diagnose_ssh,
)

FACTS = {
    "home_dir": "/home/u",
    "home_perms": "755",
    "ssh_perms": "700",
    "ak_perms": "600",
    "ak_lines": "2",
    "sshd_ak_file": "default",
    "sshd_pubkey": "default",
    "sshd_dropin_files": "none",
    "sshd_dropin_ak_file": "none",
    "sshd_dropin_pubkey": "none",
}

PUBKEY = "ssh-ed25519 AAAAKEYDATA me@laptop"
VERBOSE = "debug1: Offering public key: /home/me/.ssh/other_key ED25519 SHA256:x\ndebug1: Authentications that can continue: publickey\n"


class FakeSsh:
    """Answers each ssh invocation by what it is doing."""

    def __init__(self, *, pubkey_rc=0, connect_rc=0, facts=None, key_count="1"):
        self.pubkey_rc = pubkey_rc
        self.connect_rc = connect_rc
        self.facts = FACTS if facts is None else facts
        self.key_count = key_count
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        last = argv[-1]
        if argv[:2] == ["ssh", "-G"]:
            return subprocess.CompletedProcess(argv, 1, "", "")
        if "BatchMode=yes" in argv:
            return subprocess.CompletedProcess(argv, self.pubkey_rc, "", VERBOSE if self.pubkey_rc else "")
        if "-O" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if last == "true":
            return subprocess.CompletedProcess(argv, self.connect_rc)
        if last.startswith("grep -cF"):
            return subprocess.CompletedProcess(argv, 0, self.key_count + "\n", "")
        return subprocess.CompletedProcess(argv, 0, "".join("%s=%s\n" % kv for kv in self.facts.items()), "")


@pytest.fixture
def local_key(tmp_path, monkeypatch):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_ed25519.pub").write_text(PUBKEY + "\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    return str(ssh_dir / "id_ed25519.pub")


def _diagnose(fake, hosts=("h1",), user="u", local_user="u"):
    events = []
    with mock.patch("sparkrun.api.setup._ssh_diagnose.subprocess.run", side_effect=fake):
        report = diagnose_ssh(list(hosts), user, local_user=local_user, on_event=lambda kind, r, d: events.append(kind))
    return report, events


def test_passing_host(local_key):
    report, events = _diagnose(FakeSsh())
    assert report.local_pubkey == PUBKEY and report.local_pubkey_source == local_key
    assert report.all_passed
    (d,) = report.results
    assert d.pubkey_ok and d.connected and d.home_ok and d.standard_checks_pass
    assert events[0] == SSH_DIAG_LOCAL_KEY
    assert SSH_DIAG_KEY_CHECK_SKIPPED in events  # same user
    assert events[-1] == SSH_DIAG_HOST_DONE


def test_unreachable_host_is_abandoned_and_fails_the_report(local_key):
    report, events = _diagnose(FakeSsh(connect_rc=255))
    assert not report.all_passed
    assert events[-1] == SSH_DIAG_CONNECT_FAILED
    assert not report.results[0].facts


def test_cross_user_checks_the_local_key(local_key):
    fake = FakeSsh(key_count="0")
    report, events = _diagnose(fake, user="cluster", local_user="me")
    d = report.results[0]
    assert report.cross_user and d.key_checked and not d.key_installed
    assert SSH_DIAG_KEY_CHECK_DONE in events
    grep = next(c for c in fake.calls if c[-1].startswith("grep -cF"))
    assert "ssh-ed25519 AAAAKEYDATA" in grep[-1] and "me@laptop" not in grep[-1]
    assert not d.standard_checks_pass  # the missing key explains the rejection


def test_rejection_reports_the_offered_key_and_bad_permissions(local_key):
    facts = dict(FACTS, home_perms="777", sshd_dropin_pubkey="PubkeyAuthentication no")
    report, _ = _diagnose(FakeSsh(pubkey_rc=255, facts=facts))
    d = report.results[0]
    assert not d.pubkey_ok and not report.all_passed
    assert any("Offering public key" in line for line in d.ssh_debug_lines)
    assert d.offered_key_path == "/home/me/.ssh/other_key"
    assert d.key_mismatch
    assert not d.home_ok
    assert d.dropin("pubkey") == "PubkeyAuthentication no"
    assert d.dropin("ak_file") is None


def test_no_local_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    report, events = _diagnose(FakeSsh(), user="cluster", local_user="me")
    assert report.local_pubkey is None
    assert SSH_DIAG_KEY_CHECK_SKIPPED in events


def test_cli_renders_the_report(local_key, capsys):
    from sparkrun.cli._setup._commands import _run_ssh_diagnose

    facts = dict(FACTS, home_perms="777", sshd_dropin_files="/etc/ssh/sshd_config.d/50.conf", sshd_dropin_pubkey="PubkeyAuthentication no")
    with mock.patch("sparkrun.api.setup._ssh_diagnose.subprocess.run", side_effect=FakeSsh(pubkey_rc=255, facts=facts, key_count="0")):
        _run_ssh_diagnose(["h1"], "cluster", "me")
    out = capsys.readouterr().out
    for expected in (
        "Cross-user: yes",
        "Local public key (%s): ssh-ed25519 AAAAKEYDATA" % local_key,
        "--- Diagnosing cluster@h1 ---",
        "  [1/4] Testing pubkey authentication (BatchMode)...\n        FAIL\n        SSH debug (key-related):",
        "        Key present: NO",
        "FAIL — must not be group/world-writable",
        "    ↳ PubkeyAuthentication override: PubkeyAuthentication no",
        "    chmod go-w ~cluster",
        "    ** Drop-in override DISABLES pubkey auth: PubkeyAuthentication no",
        "    ** KEY MISMATCH: SSH is offering '/home/me/.ssh/other_key'",
        "Some hosts failed. See remediation suggestions above.",
    ):
        assert expected in out, expected
    assert "All standard checks passed" not in out
