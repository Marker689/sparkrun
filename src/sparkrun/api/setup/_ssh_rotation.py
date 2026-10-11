"""Staged SSH server key rotation with authenticated trust handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from sparkrun.orchestration.sudo import SudoPasswords
from sparkrun.utils.fs import file_lock

from ._ssh_identity import SshHostIdentity, check_ssh_host_identities, find_duplicate_identities, identity_script, parse_identity, payload
from ._ssh_identity_remote import STAGE, _atomic_write, known_hosts_text

_STRICT = ["-o", "StrictHostKeyChecking=yes", "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "UpdateHostKeys=no"]


@dataclass(frozen=True)
class SshRotationResult:
    rotated: tuple[str, ...] = ()
    failures: dict[str, str] = field(default_factory=dict)
    recovery_file: str | None = None
    machine_id_rotated: bool = False

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class _Connection:
    names: tuple[str, ...]
    files: tuple[Path, ...]
    port: int


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_.:\[\]%-]+", value) or value.startswith("-"):
        raise ValueError("unsupported SSH host alias: %r" % value)
    return value.lower()


def _endpoint(host, port):
    host = _name(host)
    return host if port == 22 else "[%s]:%d" % (host.strip("[]"), port)


def _connection(host, ssh_kwargs):
    from sparkrun.orchestration.ssh import build_ssh_cmd

    cmd = build_ssh_cmd(host, **ssh_kwargs)
    result = subprocess.run([cmd[0], "-G", *cmd[1:]], capture_output=True, text=True, timeout=15, check=True)
    values = dict(line.split(None, 1) for line in result.stdout.splitlines() if " " in line)
    port = int(values.get("port", "22"))
    if not 1 <= port <= 65535:
        raise ValueError("invalid SSH port")
    names = {_endpoint(host, port), _endpoint(values.get("hostname", host), port)}
    alias = values.get("hostkeyalias", "none")
    if alias != "none":
        names.add(_name(alias))
        names.add(_endpoint(alias, port))
    # ssh -G expands tokens but does not quote filenames containing spaces.
    # Split at the next absolute path, not at each space (Windows home
    # directories commonly contain spaces). Refuse ambiguous relative paths.
    raw_files = values.get("userknownhostsfile", "")
    file_names = re.split(r"\s+(?=(?:/|[A-Za-z]:[\\/]|~[/\\]))", raw_files.strip())
    if not raw_files or any(p == "none" or not Path(p).expanduser().is_absolute() for p in file_names):
        raise ValueError("rotation requires absolute, persistent UserKnownHostsFile paths")
    files = tuple(Path(p).expanduser().resolve() for p in file_names)
    if not files or any(p == Path(os.devnull).resolve() for p in files):
        raise ValueError("rotation requires a persistent UserKnownHostsFile (not 'none' or the null device)")
    return _Connection(tuple(sorted(names)), files, port)


def _records(identities, connections):
    records = {}
    owners = {}
    addresses = {}
    for host, identity in identities.items():
        for address in identity.addresses:
            addresses.setdefault(address, set()).add(host)
    for host, identity in identities.items():
        connection = connections[host]
        names = set(connection.names)
        # Shared docker bridges/virtual addresses do not identify a machine.
        names.update(_endpoint(ip, connection.port) for ip in identity.addresses if addresses[ip] == {host})
        for name in names:
            if name in owners and owners[name] != host:
                raise ValueError("SSH alias %s refers to more than one selected machine" % name)
            owners[name] = host
            records[name] = list(identity.public_keys.values())
    return records


def _update_local(files, records):
    for path in sorted(set(files)):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with file_lock(path.with_name(path.name + ".sparkrun.lock")):
            text = path.read_text() if path.exists() else ""
            _atomic_write(path, known_hosts_text(text, records).encode())


def _pinned_kwargs(ssh_kwargs, path):
    # First-set OpenSSH options win over caller defaults. A fresh connection
    # must authenticate a staged key, never reuse an old multiplexed session.
    quoted_path = '"%s"' % path.as_posix().replace('"', '\\"')
    options = [*_STRICT, "-o", "UserKnownHostsFile=" + quoted_path, "-o", "GlobalKnownHostsFile=none"]
    return {**ssh_kwargs, "ssh_options": options + list(ssh_kwargs.get("ssh_options") or [])}


def rotate_ssh_host_keys(
    hosts: list[str],
    *,
    ssh_kwargs: dict | None = None,
    passwords: SudoPasswords | None = None,
    machine_id: bool = False,
    recovery_dir: Path,
    dry_run: bool = False,
    progress=None,
) -> SshRotationResult:
    """Rotate after the frontend has obtained explicit operator confirmation.

    Requires Linux/systemd targets and existing verified SSH access. New keys
    are staged and recorded locally before activation, then checked using a
    fresh strictly pinned SSH connection. The controller and peers receive
    authenticated public keys, without ssh-keyscan or disabled host checking.
    A partial failure names the affected host and retains recovery information.
    Client login keys and authorized_keys are handled separately by SSH mesh.
    """
    from sparkrun.orchestration.ssh import run_remote_script, run_remote_sudo_script

    hosts = list(dict.fromkeys(hosts))
    if dry_run:
        return SshRotationResult()
    if not hosts:
        return SshRotationResult(failures={"targets": "no hosts selected"})
    kw = dict(ssh_kwargs or {})
    kw["ssh_options"] = [*_STRICT, *list(kw.get("ssh_options") or [])]
    passwords = passwords or SudoPasswords()
    failures = {}
    rotated = []
    prepared: dict[str, SshHostIdentity | None] = {}
    activating: set[str] = set()
    connections = {}
    old = {}
    token = uuid.uuid4().hex
    directory = Path(recovery_dir) / token
    journal = directory / "recovery.json"
    state = {"operation": token, "remote_backup": STAGE, "hosts": {}, "rotated": rotated}

    def emit(host, message):
        if progress:
            progress(host, message)

    def sudo(host, action, params=None):
        script = identity_script(action, token=token, machine_id=machine_id)
        params = params or kw

        def attempt(password):
            return run_remote_sudo_script(host, script, password, timeout=180, quiet=True, **params)

        return passwords.run(host, attempt, try_nopasswd=True)

    def save():
        _atomic_write(journal, json.dumps(state, indent=2).encode())

    def error(result):
        return (result.stderr or result.stdout or "SSH operation failed").strip()[-1000:]

    try:
        report = check_ssh_host_identities(hosts, ssh_kwargs=kw)
        for identity in report.identities:
            if identity.errors:
                failures[identity.host] = "; ".join(identity.errors)
            old[identity.host] = identity
        boot_hosts = {}
        for host, identity in old.items():
            if identity.boot_id_hash in boot_hosts:
                failures[host] = "same running machine as %s; select only one address per machine" % boot_hosts[identity.boot_id_hash]
            boot_hosts[identity.boot_id_hash] = host
        if set(old) != set(hosts):
            failures["probe"] = "identity probe did not return every selected host"
        if failures:
            return SshRotationResult(failures=failures)
        for host in hosts:
            connections[host] = _connection(host, kw)
        # Refuse ambiguous aliases or uneditable trust files before changing a host.
        records = _records(old, connections)
        files = {p for connection in connections.values() for p in connection.files}
        for path in files:
            if path.exists() and not path.is_file():
                raise ValueError("known_hosts is not a regular file: %s" % path)
            known_hosts_text(path.read_text() if path.exists() else "", records)
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not os.access(path if path.exists() else path.parent, os.W_OK):
                raise ValueError("known_hosts file is not writable: %s" % path)
        directory.mkdir(parents=True, mode=0o700)
        save()
        emit("controller", "Recovery record: %s" % journal)
        for host in hosts:
            emit(host, "Preparing replacement SSH host keys")
            result = sudo(host, "prepare")
            if not result.success:
                failures[host] = "preparation failed: " + error(result)
                break
            # Remember ownership even when the response is malformed, so the
            # finally block can remove this operation's unactivated keys.
            prepared[host] = None
            identity = parse_identity(host, payload(result.stdout))
            if identity.errors or set(identity.public_keys.values()) & set(old[host].public_keys.values()):
                raise ValueError("%s did not return complete, new host identities" % host)
            if machine_id and identity.machine_id_hash == old[host].machine_id_hash:
                raise ValueError("%s did not generate a new machine-id" % host)
            prepared[host] = identity
            state["hosts"][host] = {
                "phase": "prepared",
                "old_keys": old[host].public_keys,
                "new_keys": identity.public_keys,
                "known_hosts_names": connections[host].names,
                "known_hosts_files": [str(path) for path in connections[host].files],
            }
            save()
        if failures:
            return SshRotationResult(failures=failures, recovery_file=str(journal))
        new = {host: identity for host, identity in prepared.items() if identity is not None}
        if any(d.kind == "SSH host key" for d in find_duplicate_identities(list(new.values()))):
            raise ValueError("replacement SSH host keys are not unique across hosts")
        new_records = _records(new, connections)
        for host in hosts:
            # New and old keys together are used only to reach a host for
            # rollback when an interrupted activation has an uncertain outcome.
            only_new = directory / (str(hosts.index(host)) + "-new.known_hosts")
            both = directory / (str(hosts.index(host)) + "-recovery.known_hosts")
            _atomic_write(only_new, known_hosts_text("", new_records).encode())
            combined = {name: list(keys) + records.get(name, []) for name, keys in new_records.items()}
            _atomic_write(both, known_hosts_text("", combined).encode())
            pinned = _pinned_kwargs(kw, only_new)
            emit(host, "Activating and verifying new SSH host keys")
            state["hosts"][host]["phase"] = "activating"
            save()
            activating.add(host)
            result = sudo(host, "activate")
            try:
                if not result.success:
                    raise ValueError(error(result))
                # systemctl reload may acknowledge SIGHUP before sshd has
                # finished reopening its listener/loading keys. Every retry
                # still pins only the staged keys and opens a fresh session.
                for attempt in range(3):
                    if attempt:
                        time.sleep(0.25)
                    verified = run_remote_script(host, identity_script(), quiet=True, timeout=30, **pinned)
                    if verified.success:
                        break
                if not verified.success:
                    raise ValueError(error(verified))
                actual = parse_identity(host, payload(verified.stdout))
                expected = new[host]
                if actual.errors or actual.public_keys != expected.public_keys or actual.machine_id_hash != expected.machine_id_hash:
                    raise ValueError("host identity after rotation does not match the staged keys/machine-id")
            except (ValueError, TypeError) as exc:
                rollback = sudo(host, "rollback", _pinned_kwargs(kw, both))
                failures[host] = "activation/verification failed: %s; %s" % (
                    exc,
                    "original identities restored" if rollback.success else "ROLLBACK FAILED: " + error(rollback),
                )
                if rollback.success:
                    activating.discard(host)
                    state["hosts"][host]["phase"] = "rolled_back"
                if not rollback.success:
                    prepared.pop(host, None)  # retain the only remote recovery copy
                break
            state["hosts"][host]["phase"] = "verified"
            rotated.append(host)
            old[host] = actual
            save()
            try:
                _update_local(connections[host].files, _records(old, connections))
            except (OSError, ValueError) as exc:
                failures[host] = "keys rotated; controller trust update failed: %s" % exc
                prepared.pop(host, None)  # recovery journal pins the active keys
                break
            # The old private keys must not accumulate after successful rotation.
            cleaned = sudo(host, "finish", pinned)
            prepared.pop(host, None)
            state["hosts"][host]["phase"] = "complete" if cleaned.success else "cleanup_failed"
            if not cleaned.success:
                failures[host] = "keys rotated; could not remove recovery keys at %s: %s" % (STAGE, error(cleaned))
        if rotated:
            emit("cluster", "Refreshing authenticated host keys on cluster peers")
            current_records = _records(old, connections)
            for host in hosts:
                peer_records = dict(current_records)
                for loopback in ("localhost", "127.0.0.1", "::1"):
                    peer_records[_endpoint(loopback, connections[host].port)] = list(old[host].public_keys.values())
                result = run_remote_script(host, identity_script("known-hosts", records=peer_records), quiet=True, timeout=30, **kw)
                if not result.success:
                    failures.setdefault(host, "peer known_hosts refresh failed: " + error(result))
        state["failures"] = failures
        save()
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        failures["rotation"] = str(exc)
    finally:
        # Unactivated preparations (and successful rollbacks) are safe to
        # discard. Failed rollback paths were removed above and are retained.
        for host in prepared:
            if host in activating:
                continue
            result = sudo(host, "finish")
            if not result.success:
                failures.setdefault(host, "could not clean up prepared keys at %s: %s" % (STAGE, error(result)))
    return SshRotationResult(tuple(rotated), failures, str(journal) if journal.exists() else None, machine_id and bool(rotated))
