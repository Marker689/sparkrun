"""Tests for the SSH access bootstrap (:mod:`sparkrun.api.setup`).

Covers the path a control machine takes when it has never talked to the
cluster: diagnose the failure, find or make a key, install it, re-verify.
No real hosts are contacted — ``ssh``/``ssh-keygen`` are mocked except where a
generated key is genuinely written to ``tmp_path``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from sparkrun.api.setup import (
    SPARKRUN_KEY_NAME,
    OpenSshUnavailable,
    SshKeyError,
    ensure_local_key,
    install_public_key_interactive,
    mesh_ssh_keys_native,
    probe_ssh_access,
)
from sparkrun.api.setup._mesh import build_collect_key_script, build_install_keys_script
from sparkrun.api.setup._ssh_access import build_authorized_key_script
from sparkrun.orchestration.ssh import RemoteResult


# ---------------------------------------------------------------------------
# probe_ssh_access — the diagnosis is the whole point
# ---------------------------------------------------------------------------


def _probe_with(results):
    with mock.patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=results,
    ):
        return probe_ssh_access([r.host for r in results], "ubuntu")


def test_probe_reports_success():
    probes = _probe_with([RemoteResult(host="h1", returncode=0, stdout="", stderr="")])
    assert probes[0].ok
    assert not probes[0].auth_failed
    assert probes[0].reachable


def test_probe_classifies_permission_denied_as_auth_failure():
    """The exact stderr from the reported Windows-control-machine session."""
    probes = _probe_with(
        [
            RemoteResult(
                host="10.24.11.13",
                returncode=255,
                stdout="",
                stderr="root@10.24.11.13: Permission denied (publickey,password).",
            )
        ]
    )
    assert not probes[0].ok
    assert probes[0].auth_failed
    # Reachable: the host answered, it just rejected us — that's what makes it
    # a bootstrap candidate rather than a networking problem.
    assert probes[0].reachable


def test_probe_classifies_changed_host_key_separately():
    probes = _probe_with(
        [
            RemoteResult(
                host="h1",
                returncode=255,
                stdout="",
                stderr="@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@",
            )
        ]
    )
    assert probes[0].host_key_failed
    # Must NOT be treated as an auth failure: we never auto-install a key on a
    # host whose identity changed underneath us.
    assert not probes[0].auth_failed


def test_probe_classifies_network_failure_as_unreachable():
    probes = _probe_with(
        [
            RemoteResult(
                host="h1",
                returncode=255,
                stdout="",
                stderr="ssh: connect to host h1 port 22: No route to host",
            )
        ]
    )
    assert not probes[0].reachable
    assert not probes[0].auth_failed


def test_probe_preserves_input_order_and_fills_gaps():
    results = [
        RemoteResult(host="h2", returncode=0, stdout="", stderr=""),
        RemoteResult(host="h1", returncode=0, stdout="", stderr=""),
    ]
    with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results):
        probes = probe_ssh_access(["h1", "h2", "h3"], "ubuntu")
    assert [p.host for p in probes] == ["h1", "h2", "h3"]
    assert probes[2].error == "no result"


def test_probe_uses_batch_mode_and_accept_new():
    """BatchMode comes from build_ssh_cmd; accept-new must be added here.

    Without accept-new, a never-before-seen host fails host key verification
    under BatchMode and would be misreported as unreachable.
    """
    with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[]) as m:
        probe_ssh_access(["h1"], "ubuntu", options=["-o", "Custom=1"])
    opts = m.call_args.kwargs["ssh_options"]
    assert "StrictHostKeyChecking=accept-new" in opts
    assert "Custom=1" in opts


def test_probe_empty_hosts_makes_no_calls():
    with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as m:
        assert probe_ssh_access([], "ubuntu") == []
    m.assert_not_called()


# ---------------------------------------------------------------------------
# ensure_local_key
# ---------------------------------------------------------------------------


def _write_identity(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    priv = directory / name
    priv.write_text("PRIVATE")
    priv.with_name(name + ".pub").write_text("ssh-ed25519 AAAAKEY %s\n" % name)
    return priv


def test_ensure_local_key_prefers_explicit_path(tmp_path):
    ssh_dir = tmp_path / ".ssh"
    _write_identity(ssh_dir, "id_ed25519")
    custom = _write_identity(tmp_path / "custom", "mykey")

    key = ensure_local_key(preferred=custom, ssh_dir=ssh_dir)
    assert key.path == custom
    assert key.public_key == "ssh-ed25519 AAAAKEY mykey"
    assert not key.generated


def test_ensure_local_key_falls_back_to_default_identity(tmp_path):
    ssh_dir = tmp_path / ".ssh"
    _write_identity(ssh_dir, "id_ed25519")
    key = ensure_local_key(ssh_dir=ssh_dir)
    assert key.path.name == "id_ed25519"
    assert not key.generated


def test_ensure_local_key_ignores_identity_missing_its_pub(tmp_path):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir(parents=True)
    (ssh_dir / "id_ed25519").write_text("PRIVATE")  # no .pub alongside
    _write_identity(ssh_dir, "id_rsa")

    key = ensure_local_key(ssh_dir=ssh_dir)
    assert key.path.name == "id_rsa"


@pytest.mark.skipif(not __import__("shutil").which("ssh-keygen"), reason="ssh-keygen not installed")
def test_ensure_local_key_generates_into_sparkrun_key_dir(tmp_path):
    ssh_dir = tmp_path / ".ssh"
    key_dir = tmp_path / "config" / "ssh"
    key = ensure_local_key(ssh_dir=ssh_dir, key_dir=key_dir)

    assert key.generated
    assert key.path.name == SPARKRUN_KEY_NAME
    assert key.path.exists() and key.public_path.exists()
    assert key.public_key.startswith("ssh-ed25519 ")
    # sparkrun's identity lives with its config, not among personal keys.
    assert key.path.parent == key_dir
    assert not (ssh_dir / SPARKRUN_KEY_NAME).exists()
    # A personal identity is never what we create.
    assert not (ssh_dir / "id_ed25519").exists()


@pytest.mark.skipif(not __import__("shutil").which("ssh-keygen"), reason="ssh-keygen not installed")
def test_ensure_local_key_reuses_legacy_key_from_ssh_dir(tmp_path):
    """Keys generated before sparkrun owned a key dir keep working."""
    ssh_dir = tmp_path / ".ssh"
    legacy = _write_identity(ssh_dir, SPARKRUN_KEY_NAME)

    key = ensure_local_key(ssh_dir=ssh_dir, key_dir=tmp_path / "config" / "ssh")
    assert key.path == legacy
    assert not key.generated


@pytest.mark.skipif(not __import__("shutil").which("ssh-keygen"), reason="ssh-keygen not installed")
def test_ensure_local_key_reuses_previously_generated_key(tmp_path):
    ssh_dir = tmp_path / ".ssh"
    key_dir = tmp_path / "config" / "ssh"
    first = ensure_local_key(ssh_dir=ssh_dir, key_dir=key_dir)
    second = ensure_local_key(ssh_dir=ssh_dir, key_dir=key_dir)

    assert first.generated
    assert not second.generated
    assert second.public_key == first.public_key


def test_ensure_local_key_respects_generate_false(tmp_path):
    with pytest.raises(SshKeyError):
        ensure_local_key(ssh_dir=tmp_path / ".ssh", key_dir=tmp_path / "config" / "ssh", generate=False)


def test_ensure_local_key_reports_missing_openssh(tmp_path):
    with mock.patch("shutil.which", return_value=None):
        with pytest.raises(OpenSshUnavailable):
            ensure_local_key(ssh_dir=tmp_path / ".ssh", key_dir=tmp_path / "config" / "ssh")


# ---------------------------------------------------------------------------
# Remote script builders
# ---------------------------------------------------------------------------


def test_authorized_key_script_embeds_key_and_is_idempotent_shaped():
    script = build_authorized_key_script("ssh-ed25519 AAAA me@box")
    assert "KEY='ssh-ed25519 AAAA me@box'" in script
    assert "grep -qxF" in script  # dedupe before append
    assert "chmod 600" in script


@pytest.mark.parametrize("bad", ["", "  ", "key-one\nkey-two", "has'quote"])
def test_authorized_key_script_rejects_unsafe_keys(bad):
    with pytest.raises(ValueError):
        build_authorized_key_script(bad)


def test_install_keys_script_carries_keys_in_a_quoted_heredoc():
    """Keys go through stdin, never argv — that's what makes this work on Windows."""
    script = build_install_keys_script(["ssh-ed25519 A a@b", "ssh-ed25519 B c@d"])
    assert "<<'SPARKRUN_MESH_KEYS'" in script  # quoted: no expansion of key text
    assert "ssh-ed25519 A a@b" in script
    assert "ssh-ed25519 B c@d" in script


def test_install_keys_script_rejects_heredoc_collision():
    with pytest.raises(ValueError):
        build_install_keys_script(["SPARKRUN_MESH_KEYS"])


def test_install_keys_script_requires_at_least_one_key():
    with pytest.raises(ValueError):
        build_install_keys_script(["", "   "])


def test_collect_key_script_generates_then_prints_marked_key():
    script = build_collect_key_script()
    assert "ssh-keygen -t ed25519" in script
    assert "SPARKRUN_PUBKEY" in script


@pytest.mark.parametrize(
    "script",
    [
        build_authorized_key_script("ssh-ed25519 AAAA me@box"),
        build_install_keys_script(["ssh-ed25519 A a@b"]),
        build_collect_key_script(),
    ],
)
def test_generated_scripts_are_valid_bash(script, tmp_path):
    path = tmp_path / "s.sh"
    path.write_text(script)
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0


def test_authorized_key_script_actually_dedupes(tmp_path):
    """Run it twice against a fake HOME; the key must appear exactly once."""
    home = tmp_path / "home"
    home.mkdir()
    path = tmp_path / "s.sh"
    path.write_text(build_authorized_key_script("ssh-ed25519 AAAA me@box"))
    for _ in range(2):
        assert subprocess.run(["bash", str(path)], env={"HOME": str(home), "PATH": "/usr/bin:/bin"}).returncode == 0
    lines = (home / ".ssh" / "authorized_keys").read_text().strip().splitlines()
    assert lines == ["ssh-ed25519 AAAA me@box"]


# ---------------------------------------------------------------------------
# install_public_key_interactive
# ---------------------------------------------------------------------------


def test_install_public_key_disables_pubkey_auth_and_pipes_the_script():
    with mock.patch("shutil.which", return_value="/usr/bin/ssh"):
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)) as m:
            assert install_public_key_interactive("h1", "ubuntu", "ssh-ed25519 AAAA me@box")

    cmd = m.call_args.args[0]
    # Public-key auth off, or a control machine with many identities can
    # exhaust MaxAuthTries before a password is ever offered.
    assert "PubkeyAuthentication=no" in cmd
    assert "PreferredAuthentications=password,keyboard-interactive" in cmd
    assert "ubuntu@h1" in cmd
    assert cmd[-2:] == ["bash", "-s"]
    # stdout/stderr are NOT captured: OpenSSH must reach the terminal to prompt.
    assert "capture_output" not in m.call_args.kwargs
    # Piped as bytes with LF endings only: text mode would CRLF-mangle the
    # script on a Windows control machine, and the remote bash would fail with
    # "invalid option" / "unexpected end of file".
    piped = m.call_args.kwargs["input"]
    assert isinstance(piped, bytes)
    assert b"ssh-ed25519 AAAA me@box" in piped
    assert b"\r" not in piped


def test_install_public_key_reports_failure_without_raising():
    with mock.patch("shutil.which", return_value="/usr/bin/ssh"):
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=255)):
            assert install_public_key_interactive("h1", "ubuntu", "ssh-ed25519 AAAA me@box") is False


def test_install_public_key_dry_run_makes_no_connection():
    with mock.patch("subprocess.run") as m:
        assert install_public_key_interactive("h1", "ubuntu", "ssh-ed25519 AAAA me@box", dry_run=True)
    m.assert_not_called()


def test_install_public_key_requires_ssh_binary():
    with mock.patch("shutil.which", return_value=None):
        with pytest.raises(OpenSshUnavailable):
            install_public_key_interactive("h1", "ubuntu", "ssh-ed25519 AAAA me@box")


def test_install_public_key_validates_before_connecting():
    with mock.patch("subprocess.run") as m:
        with pytest.raises(ValueError):
            install_public_key_interactive("h1", "ubuntu", "bad'key")
    m.assert_not_called()


# ---------------------------------------------------------------------------
# mesh_ssh_keys_native
# ---------------------------------------------------------------------------


def _collect_result(host, key):
    return RemoteResult(host=host, returncode=0, stdout="SPARKRUN_PUBKEY %s\n" % key, stderr="")


def test_native_mesh_collects_then_authorizes_every_key():
    calls = []

    def fake(hosts, script, **kw):
        calls.append((list(hosts), script))
        if "SPARKRUN_PUBKEY" in script:
            return [_collect_result(h, "ssh-ed25519 KEY-%s x@%s" % (h, h)) for h in hosts]
        return [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts]

    with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=fake):
        result = mesh_ssh_keys_native(["h1", "h2"], "ubuntu")

    assert result.ok
    assert set(result.public_keys) == {"h1", "h2"}
    install_script = calls[1][1]
    assert "ssh-ed25519 KEY-h1 x@h1" in install_script
    assert "ssh-ed25519 KEY-h2 x@h2" in install_script


def test_native_mesh_skips_hosts_whose_key_could_not_be_read():
    def fake(hosts, script, **kw):
        if "SPARKRUN_PUBKEY" in script:
            return [
                _collect_result("h1", "ssh-ed25519 KEY1 a@h1"),
                RemoteResult(host="h2", returncode=255, stdout="", stderr="Permission denied"),
            ]
        assert hosts == ["h1"]  # h2 is excluded from the install pass
        return [RemoteResult(host="h1", returncode=0, stdout="", stderr="")]

    with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=fake):
        result = mesh_ssh_keys_native(["h1", "h2"], "ubuntu")

    assert not result.ok
    assert "h2" in result.collect_failures
    assert set(result.public_keys) == {"h1"}


def test_native_mesh_dedupes_shared_home_directories():
    """Hosts on shared storage report the same key; install it once."""
    shared = "ssh-ed25519 SHARED a@b"

    def fake(hosts, script, **kw):
        if "SPARKRUN_PUBKEY" in script:
            return [_collect_result(h, shared) for h in hosts]
        assert script.count(shared) == 1
        return [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in hosts]

    with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=fake):
        result = mesh_ssh_keys_native(["h1", "h2"], "ubuntu")

    assert result.ok


def test_native_mesh_reports_install_failures():
    def fake(hosts, script, **kw):
        if "SPARKRUN_PUBKEY" in script:
            return [_collect_result(h, "ssh-ed25519 K-%s a@b" % h) for h in hosts]
        return [RemoteResult(host=h, returncode=1, stdout="", stderr="disk full") for h in hosts]

    with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=fake):
        result = mesh_ssh_keys_native(["h1"], "ubuntu")

    assert not result.ok
    assert result.install_failures["h1"] == "disk full"


def test_native_mesh_dry_run_touches_nothing():
    with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as m:
        result = mesh_ssh_keys_native(["h1", "h2"], "ubuntu", dry_run=True)
    m.assert_not_called()
    assert set(result.public_keys) == {"h1", "h2"}


def test_native_mesh_with_no_hosts_is_a_noop():
    with mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as m:
        assert mesh_ssh_keys_native([], "ubuntu").ok
    m.assert_not_called()
