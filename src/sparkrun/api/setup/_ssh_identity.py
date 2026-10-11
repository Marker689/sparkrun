"""Read-only, authenticated checks for cloned Linux host identities."""

from __future__ import annotations

import base64
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import re
from typing import Any

from . import _ssh_identity_remote as remote


@dataclass(frozen=True)
class SshHostIdentity:
    host: str
    public_keys: dict[str, str] = field(default_factory=dict)
    machine_id_hash: str | None = None
    boot_id_hash: str | None = None
    addresses: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DuplicateHostIdentity:
    kind: str
    fingerprint: str
    hosts: tuple[str, ...]


@dataclass(frozen=True)
class SshIdentityReport:
    identities: tuple[SshHostIdentity, ...] = ()
    duplicates: tuple[DuplicateHostIdentity, ...] = ()

    @property
    def complete(self) -> bool:
        return bool(self.identities) and all(
            identity.public_keys and identity.machine_id_hash and identity.boot_id_hash and not identity.errors
            for identity in self.identities
        )


def identity_script(action="probe", *, token="", machine_id=False, records=None) -> str:
    from sparkrun.utils.shell import b64_wrap_python

    source = inspect.getsource(remote)
    source += "\nimport sys\ntry:\n    main(%r, token=%r, machine_id=%r, records=%r)\n" % (action, token, machine_id, records)
    source += "except Exception as exc:\n    print(getattr(exc, 'stderr', '') or str(exc), file=sys.stderr)\n    sys.exit(1)\n"
    return b64_wrap_python(source, quoted=False)


def payload(stdout: str) -> dict[str, Any]:
    lines = [line[len(remote.MARKER) :] for line in stdout.splitlines() if line.startswith(remote.MARKER)]
    if len(lines) != 1:
        raise ValueError("missing or ambiguous host identity response")
    data = json.loads(lines[0])
    if not isinstance(data, dict):
        raise ValueError("invalid host identity response")
    return data


def public_key(key: str) -> tuple[str, str]:
    """Validate the SSH wire type and compare the key, independent of comments."""
    fields = key.split()
    if len(fields) < 2 or fields[0] not in {*remote.KEY_TYPES.values(), "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521"}:
        raise ValueError("unsupported or invalid SSH host public key")
    blob = base64.b64decode(fields[1], validate=True)
    size = int.from_bytes(blob[:4], "big")
    if size <= 0 or blob[4 : 4 + size].decode("ascii") != fields[0] or len(blob) <= 4 + size:
        raise ValueError("invalid SSH public key encoding")
    normalized = "%s %s" % (fields[0], base64.b64encode(blob).decode())
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    return normalized, fingerprint


def parse_identity(host: str, data: dict[str, Any]) -> SshHostIdentity:
    errors = [str(error) for error in data.get("errors", [])]
    keys = {}
    for name, value in data.get("keys", {}).items():
        try:
            keys[name] = public_key(value)[0]
        except (ValueError, TypeError, UnicodeError):
            errors.append("invalid host public key: %s" % name)
    if not keys and not any("public key" in error for error in errors):
        errors.append("no readable SSH host public keys")
    digests = {}
    for label in ("machine_id", "boot_id"):
        value = data.get(label)
        digests[label] = value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else None
        if digests[label] is None and not any(label.replace("_", "-") in error for error in errors):
            errors.append("%s unavailable" % label)
    import ipaddress

    addresses = []
    for value in data.get("addresses", []):
        try:
            ip = ipaddress.ip_address(value)
            if not (ip.is_loopback or ip.is_link_local or ip.is_unspecified):
                addresses.append(str(ip))
        except ValueError:
            continue
    return SshHostIdentity(host, keys, digests["machine_id"], digests["boot_id"], tuple(dict.fromkeys(addresses)), tuple(errors))


def find_duplicate_identities(identities: list[SshHostIdentity]) -> tuple[DuplicateHostIdentity, ...]:
    groups = defaultdict(list)
    for identity in identities:
        if identity.machine_id_hash:
            groups[("machine-id", identity.machine_id_hash)].append(identity)
        for key in set(identity.public_keys.values()):
            _, fingerprint = public_key(key)
            groups[("SSH host key", fingerprint)].append(identity)
    duplicates = []
    for (kind, fingerprint), members in sorted(groups.items()):
        # boot_id is generated on each boot. Unlike machine-id/hostname/keys,
        # it is not an identifier baked into a cloned disk image.
        machines = {member.boot_id_hash or member.host for member in members}
        if len(machines) > 1:
            duplicates.append(DuplicateHostIdentity(kind, fingerprint, tuple(dict.fromkeys(m.host for m in members))))
    return tuple(duplicates)


def check_ssh_host_identities(hosts: list[str], *, ssh_kwargs: dict | None = None, dry_run=False) -> SshIdentityReport:
    """Compare SSH server public keys and hashed machine IDs over existing SSH.

    No private keys or raw machine IDs leave the hosts. Missing/unreachable
    hosts remain explicit incomplete results; an incomplete audit is not proof
    of unique identities. Multiple addresses with one boot ID count once.
    """
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    hosts = list(dict.fromkeys(hosts))
    if dry_run or not hosts:
        return SshIdentityReport()
    results = run_remote_scripts_parallel(hosts, identity_script(), quiet=True, timeout=30, **(ssh_kwargs or {}))
    by_host = {result.host: result for result in results}
    identities = []
    for host in hosts:
        result = by_host.get(host)
        if result is None or not result.success:
            detail = (result.stderr.strip() or "SSH identity probe failed") if result else "no probe result"
            identities.append(SshHostIdentity(host, errors=(detail,)))
            continue
        try:
            identities.append(parse_identity(host, payload(result.stdout)))
        except (ValueError, TypeError, AttributeError):
            identities.append(SshHostIdentity(host, errors=("invalid SSH identity response",)))
    return SshIdentityReport(tuple(identities), find_duplicate_identities(identities))
