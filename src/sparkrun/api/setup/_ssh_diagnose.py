"""SSH pubkey-auth diagnosis — console-free core of ``setup ssh --diagnose``.

For each host: try pubkey auth non-interactively (verbose, so a rejection says
why), then open an authenticated ControlMaster connection — **this step may
prompt for a password on the controlling terminal**, it is the point of the
diagnosis — and over it read the remote side's permissions and sshd settings,
and whether the local public key is in ``authorized_keys``. The verdicts
(``home_ok``, ``key_mismatch``, …) are computed here; wording and remediation
text are the renderer's.

Progress is reported through ``on_event`` so a renderer can print each step
before it runs, which matters because step 2 may block on a password prompt.

.. warning::
   **Experimental:** this module's function and result types may change in a
   minor release.
"""

from __future__ import annotations

import getpass
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

#: ``diagnose_ssh`` progress events. ``host`` is ``None`` for the first one.
SSH_DIAG_LOCAL_KEY = "local-key"  # local public key resolved (may be None)
SSH_DIAG_PUBKEY_START = "pubkey-start"
SSH_DIAG_PUBKEY_DONE = "pubkey-done"
SSH_DIAG_CONNECT_START = "connect-start"  # may prompt for a password next
SSH_DIAG_CONNECT_FAILED = "connect-failed"  # host abandoned
SSH_DIAG_COLLECT_START = "collect-start"
SSH_DIAG_KEY_CHECK_START = "key-check-start"
SSH_DIAG_KEY_CHECK_DONE = "key-check-done"
SSH_DIAG_KEY_CHECK_SKIPPED = "key-check-skipped"
SSH_DIAG_HOST_DONE = "host-done"

_SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3"]

#: ``ssh -v`` lines worth showing when pubkey auth fails.
_KEY_DEBUG_MARKERS = (
    "offering",
    "trying",
    "authentications that can continue",
    "no more authentication",
    "permission denied",
    "key_load",
    "identity file",
    "will attempt",
    "server accepts key",
    "authentication refused",
    "pubkey_prepare",
    "sign_and_send",
)

#: Collected over the authenticated connection as ``key=value`` lines.
_DIAG_SCRIPT = r"""set -eu
printf 'home_dir=%s\n' "$(eval echo ~)"
printf 'home_perms=%s\n' "$(stat -c '%a' ~ 2>/dev/null || echo unknown)"
printf 'home_owner=%s\n' "$(stat -c '%U' ~ 2>/dev/null || echo unknown)"
printf 'ssh_dir_exists=%s\n' "$(test -d ~/.ssh && echo yes || echo no)"
printf 'ssh_perms=%s\n' "$(stat -c '%a' ~/.ssh 2>/dev/null || echo missing)"
printf 'ak_exists=%s\n' "$(test -f ~/.ssh/authorized_keys && echo yes || echo no)"
printf 'ak_perms=%s\n' "$(stat -c '%a' ~/.ssh/authorized_keys 2>/dev/null || echo missing)"
printf 'ak_lines=%s\n' "$(wc -l < ~/.ssh/authorized_keys 2>/dev/null || echo 0)"
printf 'sshd_ak_file=%s\n' "$(grep -i '^AuthorizedKeysFile' /etc/ssh/sshd_config 2>/dev/null || echo default)"
printf 'sshd_pubkey=%s\n' "$(grep -i '^PubkeyAuthentication' /etc/ssh/sshd_config 2>/dev/null || echo default)"
printf 'sshd_strict=%s\n' "$(grep -i '^StrictModes' /etc/ssh/sshd_config 2>/dev/null || echo default)"
printf 'sshd_allow_users=%s\n' "$(grep -i '^AllowUsers' /etc/ssh/sshd_config 2>/dev/null || echo none)"
printf 'sshd_allow_groups=%s\n' "$(grep -i '^AllowGroups' /etc/ssh/sshd_config 2>/dev/null || echo none)"

# Check sshd drop-in config files (Ubuntu 22.04+ uses Include)
_dropin_dir="/etc/ssh/sshd_config.d"
if [ -d "$_dropin_dir" ]; then
    _dropin_files="$(ls -1 "$_dropin_dir"/*.conf 2>/dev/null | tr '\n' ',' || echo none)"
    printf 'sshd_dropin_files=%s\n' "${_dropin_files%,}"

    # Check for overrides in drop-in files that affect pubkey auth
    _dropin_ak="$(grep -rhi '^AuthorizedKeysFile' "$_dropin_dir"/ 2>/dev/null || echo none)"
    printf 'sshd_dropin_ak_file=%s\n' "$_dropin_ak"
    _dropin_pubkey="$(grep -rhi '^PubkeyAuthentication' "$_dropin_dir"/ 2>/dev/null || echo none)"
    printf 'sshd_dropin_pubkey=%s\n' "$_dropin_pubkey"
    _dropin_accepted="$(grep -rhi '^PubkeyAcceptedAlgorithms\|^PubkeyAcceptedKeyTypes' "$_dropin_dir"/ 2>/dev/null || echo none)"
    printf 'sshd_dropin_accepted_algs=%s\n' "$_dropin_accepted"
    _dropin_allow_users="$(grep -rhi '^AllowUsers' "$_dropin_dir"/ 2>/dev/null || echo none)"
    printf 'sshd_dropin_allow_users=%s\n' "$_dropin_allow_users"
    _dropin_allow_groups="$(grep -rhi '^AllowGroups' "$_dropin_dir"/ 2>/dev/null || echo none)"
    printf 'sshd_dropin_allow_groups=%s\n' "$_dropin_allow_groups"
else
    printf 'sshd_dropin_files=none\n'
    printf 'sshd_dropin_ak_file=none\n'
    printf 'sshd_dropin_pubkey=none\n'
    printf 'sshd_dropin_accepted_algs=none\n'
    printf 'sshd_dropin_allow_users=none\n'
    printf 'sshd_dropin_allow_groups=none\n'
fi

# PubkeyAcceptedAlgorithms in main sshd_config
printf 'sshd_accepted_algs=%s\n' "$(grep -i '^PubkeyAcceptedAlgorithms\|^PubkeyAcceptedKeyTypes' /etc/ssh/sshd_config 2>/dev/null || echo default)"

# Key types present in authorized_keys
printf 'ak_key_types=%s\n' "$(awk '{print $1}' ~/.ssh/authorized_keys 2>/dev/null | sort -u | tr '\n' ',' || echo none)"
"""


@dataclass
class SshHostDiagnosis:
    """What one host's diagnosis found. Verdicts are properties over ``facts``."""

    host: str
    user: str
    cross_user: bool = False
    #: Non-interactive (BatchMode) pubkey auth succeeded.
    pubkey_ok: bool = False
    #: Key-related ``ssh -v`` lines, collected when pubkey auth failed.
    ssh_debug_lines: list[str] = field(default_factory=list)
    #: An authenticated connection (possibly by password) was established.
    connected: bool = False
    #: ``key=value`` facts read from the host (empty when collection failed).
    facts: dict[str, str] = field(default_factory=dict)
    #: The local public key was looked up in ``authorized_keys``.
    key_checked: bool = False
    key_installed: bool = False
    #: The key the failed attempt offered, from the ``ssh -v`` output.
    offered_key_path: str | None = None
    #: That key differs from the local key sparkrun would install.
    key_mismatch: bool = False

    def fact(self, key: str, default: str) -> str:
        return self.facts.get(key, default)

    @property
    def home_ok(self) -> bool:
        return self.fact("home_perms", "unknown") in ("700", "750", "755")

    @property
    def ssh_dir_ok(self) -> bool:
        return self.fact("ssh_perms", "missing") == "700"

    @property
    def authorized_keys_ok(self) -> bool:
        return self.fact("ak_perms", "missing") == "600"

    @property
    def authorized_keys_file_ok(self) -> bool:
        return self.fact("sshd_ak_file", "default") == "default"

    @property
    def pubkey_setting_ok(self) -> bool:
        value = self.fact("sshd_pubkey", "default")
        return value == "default" or "yes" in value.lower()

    def dropin(self, name: str) -> str | None:
        """An sshd drop-in override (``ak_file``, ``pubkey``, ``accepted_algs``,
        ``allow_users``, ``allow_groups``), or ``None`` when none is set."""
        value = self.fact("sshd_dropin_%s" % name, "none")
        return None if value == "none" else value

    @property
    def standard_checks_pass(self) -> bool:
        """Every check that explains a typical rejection passes (so the cause is elsewhere)."""
        return (
            self.home_ok
            and self.ssh_dir_ok
            and self.authorized_keys_ok
            and self.authorized_keys_file_ok
            and self.pubkey_setting_ok
            and (not self.cross_user or self.key_installed)
            and self.dropin("ak_file") is None
            and self.dropin("pubkey") is None
        )


@dataclass
class SshDiagnosticReport:
    """Outcome of :func:`diagnose_ssh`."""

    user: str
    local_user: str
    hosts: list[str]
    local_pubkey: str | None = None
    local_pubkey_source: str | None = None
    #: One entry per host diagnosed, including hosts that could not be reached.
    results: list[SshHostDiagnosis] = field(default_factory=list)

    @property
    def cross_user(self) -> bool:
        return self.user != self.local_user

    @property
    def all_passed(self) -> bool:
        return all(r.pubkey_ok and r.connected for r in self.results)


def find_local_public_key(user: str, host: str) -> tuple[str | None, str | None]:
    """The public key ssh would offer to ``user@host``, and the file it came from.

    Asks ``ssh -G`` (so ``~/.ssh/config`` ``IdentityFile`` is honored), then
    falls back to the standard key filenames.
    """
    try:
        resolved = subprocess.run(["ssh", "-G", "%s@%s" % (user, host)], capture_output=True, text=True, timeout=5)
        if resolved.returncode == 0:
            for line in resolved.stdout.splitlines():
                if line.startswith("identityfile "):
                    pub = os.path.expanduser(line.split(None, 1)[1]) + ".pub"
                    if os.path.isfile(pub):
                        with open(pub) as f:
                            return f.read().strip(), pub
    except Exception:
        logger.debug("ssh -G failed for %s@%s; trying standard key names", user, host, exc_info=True)

    for name in ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"):
        path = os.path.join(os.path.expanduser("~"), ".ssh", name)
        if os.path.isfile(path):
            with open(path) as f:
                return f.read().strip(), path
    return None, None


def _offered_key_path(debug_lines: list[str]) -> str | None:
    """The key path in the first "Offering public key:" / "Will attempt key:" line."""
    for line in debug_lines:
        lowered = line.lower()
        if "offering public key:" in lowered or "will attempt key:" in lowered:
            # "debug1: Offering public key: /home/me/.ssh/id_ed25519_shared ..."
            parts = line.split(":", 2)
            if len(parts) >= 3:
                tokens = parts[2].strip().split()
                if tokens:
                    return tokens[0]
    return None


def diagnose_ssh(
    hosts: list[str],
    user: str,
    *,
    local_user: str | None = None,
    on_event: Callable[[str, SshDiagnosticReport, SshHostDiagnosis | None], None] | None = None,
) -> SshDiagnosticReport:
    """Diagnose why pubkey SSH as *user* fails (or confirm it works) on each host.

    EXPERIMENTAL. Interactive: establishing the authenticated connection may
    prompt for *user*'s password on the controlling terminal. Requires an
    OpenSSH client with ControlMaster support.
    """
    report = SshDiagnosticReport(user=user, local_user=local_user or getpass.getuser(), hosts=list(hosts))

    def emit(kind: str, diag: SshHostDiagnosis | None = None) -> None:
        if on_event is not None:
            on_event(kind, report, diag)

    if hosts:
        report.local_pubkey, report.local_pubkey_source = find_local_public_key(user, hosts[0])
    emit(SSH_DIAG_LOCAL_KEY)

    control_dir = tempfile.mkdtemp(prefix="sparkrun-diag-")
    os.chmod(control_dir, 0o700)
    try:
        for h in hosts:
            _diagnose_host(report, h, control_dir, emit)
    finally:
        shutil.rmtree(control_dir, ignore_errors=True)
    return report


def _diagnose_host(report: SshDiagnosticReport, h: str, control_dir: str, emit) -> None:
    user = report.user
    target = "%s@%s" % (user, h)
    diag = SshHostDiagnosis(host=h, user=user, cross_user=report.cross_user)
    report.results.append(diag)
    control_path = os.path.join(control_dir, "cm-%%r@%%h:%%p")

    # Step 1: pubkey auth alone (no password, no ControlMaster); -v shows WHY
    # a key was rejected.
    emit(SSH_DIAG_PUBKEY_START, diag)
    pubkey = subprocess.run(
        ["ssh", "-v"] + _SSH_OPTS + ["-o", "ControlPath=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", target, "true"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    diag.pubkey_ok = pubkey.returncode == 0
    if not diag.pubkey_ok and pubkey.stderr:
        diag.ssh_debug_lines = [
            line.strip() for line in pubkey.stderr.splitlines() if any(marker in line.lower() for marker in _KEY_DEBUG_MARKERS)
        ]
    emit(SSH_DIAG_PUBKEY_DONE, diag)

    # Step 2: authenticated ControlMaster — may prompt for a password.
    cm_opts = ["-o", "ControlMaster=auto", "-o", "ControlPersist=2m", "-o", "ControlPath=%s" % control_path]
    emit(SSH_DIAG_CONNECT_START, diag)
    connected = subprocess.run(["ssh"] + _SSH_OPTS + cm_opts + [target, "true"], timeout=60)
    if connected.returncode != 0:
        emit(SSH_DIAG_CONNECT_FAILED, diag)
        return
    diag.connected = True
    cm_ssh = ["ssh"] + _SSH_OPTS + cm_opts + [target]

    # Step 3: remote permissions and sshd settings.
    emit(SSH_DIAG_COLLECT_START, diag)
    collected = subprocess.run(cm_ssh + [_DIAG_SCRIPT], capture_output=True, text=True, timeout=15)
    if collected.returncode == 0:
        for line in collected.stdout.strip().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                diag.facts[k.strip()] = v.strip()

    # Step 4: is the local key authorized there? Only meaningful cross-user.
    if report.local_pubkey and report.cross_user:
        emit(SSH_DIAG_KEY_CHECK_START, diag)
        key_parts = report.local_pubkey.split()
        if len(key_parts) >= 2:
            # Key type + data only: the comment need not match.
            key_match = "%s %s" % (key_parts[0], key_parts[1])
            found = subprocess.run(
                cm_ssh + ["grep -cF '%s' ~/.ssh/authorized_keys 2>/dev/null || echo 0" % key_match],
                capture_output=True,
                text=True,
                timeout=10,
            )
            diag.key_checked = True
            diag.key_installed = found.stdout.strip() != "0"
            emit(SSH_DIAG_KEY_CHECK_DONE, diag)
    else:
        emit(SSH_DIAG_KEY_CHECK_SKIPPED, diag)

    if not diag.pubkey_ok and diag.ssh_debug_lines and report.local_pubkey_source:
        diag.offered_key_path = _offered_key_path(diag.ssh_debug_lines)
        expected_private = report.local_pubkey_source.removesuffix(".pub")
        diag.key_mismatch = bool(diag.offered_key_path and diag.offered_key_path != expected_private)

    emit(SSH_DIAG_HOST_DONE, diag)

    subprocess.run(["ssh"] + _SSH_OPTS + ["-o", "ControlPath=%s" % control_path, "-O", "exit", target], capture_output=True, timeout=5)


__all__ = [
    "SSH_DIAG_LOCAL_KEY",
    "SSH_DIAG_PUBKEY_START",
    "SSH_DIAG_PUBKEY_DONE",
    "SSH_DIAG_CONNECT_START",
    "SSH_DIAG_CONNECT_FAILED",
    "SSH_DIAG_COLLECT_START",
    "SSH_DIAG_KEY_CHECK_START",
    "SSH_DIAG_KEY_CHECK_DONE",
    "SSH_DIAG_KEY_CHECK_SKIPPED",
    "SSH_DIAG_HOST_DONE",
    "SshHostDiagnosis",
    "SshDiagnosticReport",
    "find_local_public_key",
    "diagnose_ssh",
]
