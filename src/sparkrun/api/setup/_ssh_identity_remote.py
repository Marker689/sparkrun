"""Standalone stdlib payload executed on Linux setup targets over SSH.

Keep this module independent of Sparkrun: its source is sent to python3 on
hosts that need not have Sparkrun installed. Importing it performs no actions.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import hmac
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

MARKER = "SPARKRUN_SSH_IDENTITY "
STAGE = "/etc/ssh/.sparkrun-key-rotation"
KEY_TYPES = {"rsa": "ssh-rsa", "ecdsa": "ecdsa-sha2-nistp256", "ed25519": "ssh-ed25519"}


def _run(args, *, check=True):
    return subprocess.run(args, check=check, capture_output=True, text=True, timeout=60)


def _atomic_write(path, data, mode=0o600):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".sparkrun-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.chmod(temporary, mode)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _identifier(path):
    value = path.read_text().strip().lower().replace("-", "")
    if not re.fullmatch(r"[0-9a-f]{32}", value) or int(value, 16) == 0:
        raise ValueError("invalid identifier")
    # /etc/machine-id is confidential; comparisons need only its digest.
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def probe(root=Path("/"), *, key_dir=None):
    result = {"keys": {}, "machine_id": None, "boot_id": None, "addresses": [], "errors": []}
    for label, path in (("machine_id", "etc/machine-id"), ("boot_id", "proc/sys/kernel/random/boot_id")):
        try:
            result[label] = _identifier(root / path)
        except (OSError, ValueError):
            result["errors"].append("cannot read a valid %s" % path)
    for path in sorted((key_dir or root / "etc/ssh").glob("ssh_host_*_key.pub")):
        try:
            fields = path.read_text().split()
            if len(fields) < 2:
                raise ValueError("invalid public key")
            result["keys"][path.name] = " ".join(fields[:2])
        except (OSError, ValueError):
            result["errors"].append("cannot read %s" % path.name)
    if not result["keys"]:
        result["errors"].append("no readable SSH host public keys")
    try:
        for interface in json.loads(_run(["ip", "-j", "address", "show"]).stdout):
            for address in interface.get("addr_info", []):
                ip = ipaddress.ip_address(address["local"])
                if not (ip.is_loopback or ip.is_link_local or ip.is_unspecified):
                    result["addresses"].append(str(ip))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        pass  # Extra interface aliases are optional, unlike the identity facts.
    return result


def _sshd():
    return shutil.which("sshd") or "/usr/sbin/sshd"


def _config():
    lines = _run([_sshd(), "-T"]).stdout.splitlines()
    keys = [line.split(None, 1)[1] for line in lines if line.startswith("hostkey ")]
    allowed = {"/etc/ssh/ssh_host_%s_key" % kind for kind in KEY_TYPES}
    if not keys or not set(keys) <= allowed:
        raise ValueError("rotation requires standard /etc/ssh/ssh_host_{rsa,ecdsa,ed25519}_key paths")
    for line in lines:
        if line.startswith(("hostcertificate ", "hostkeyagent ")) and line.split(None, 1)[1] != "none":
            raise ValueError("host certificates/HostKeyAgent require administrator-managed rotation")
    _run([_sshd(), "-t"])
    return sorted(set(keys))


def _service():
    for unit in ("ssh.service", "sshd.service"):
        if _run(["systemctl", "is-active", "--quiet", unit], check=False).returncode == 0:
            if _run(["systemctl", "show", "-p", "CanReload", "--value", unit]).stdout.strip() == "yes":
                return unit
    raise ValueError("an active, reloadable systemd SSH service is required")


def _stage(root):
    return root / STAGE.lstrip("/")


def _load(root, token):
    stage = _stage(root)
    if stage.is_symlink():
        raise ValueError("unsafe rotation staging directory")
    meta = json.loads((stage / "state.json").read_text())
    if meta["token"] != token:
        raise ValueError("another SSH key rotation owns the staging directory")
    return stage, meta


def prepare(token, machine_id=False, root=Path("/")):
    configured, service = _config(), _service()
    keys = sorted("/etc/ssh/ssh_host_%s_key" % kind for kind in KEY_TYPES)
    stage = _stage(root)
    # An existing directory is an interrupted/concurrent rotation, never
    # permission to overwrite its only recovery copy.
    stage.mkdir(mode=0o700)
    try:
        (stage / "new").mkdir(mode=0o700)
        (stage / "old").mkdir(mode=0o700)
        paths = [p + suffix for p in keys for suffix in ("", ".pub")]
        if machine_id:
            path = root / "etc/machine-id"
            if path.is_symlink() or os.path.ismount(path) or not path.is_file():
                raise ValueError("machine-id must be a writable regular file, not a symlink or mount")
            paths.append("/etc/machine-id")
            dbus = root / "var/lib/dbus/machine-id"
            if dbus.is_symlink():
                if dbus.resolve() != path.resolve():
                    raise ValueError("unexpected /var/lib/dbus/machine-id symlink")
            elif dbus.exists():
                if not dbus.is_file() or os.path.ismount(dbus):
                    raise ValueError("unexpected /var/lib/dbus/machine-id file")
                paths.append("/var/lib/dbus/machine-id")
        previous = {}
        for index, name in enumerate(paths):
            path = root / name.lstrip("/")
            if path.is_symlink():
                raise ValueError("refusing to rotate symlink %s" % name)
            previous[name] = None
            if path.exists():
                if not path.is_file():
                    raise ValueError("not a regular file: %s" % name)
                backup = str(index)
                shutil.copy2(path, stage / "old" / backup)
                previous[name] = backup
        meta = {"token": token, "keys": keys, "configured": configured, "service": service, "previous": previous, "phase": "prepared"}
        if machine_id:
            meta["machine_id"] = uuid.uuid4().hex
        for name in keys:
            filename = Path(name).name
            kind = filename.removeprefix("ssh_host_").removesuffix("_key")
            _run(["ssh-keygen", "-q", "-t", kind, "-N", "", "-f", str(stage / "new" / filename)])
        _atomic_write(stage / "state.json", json.dumps(meta).encode())
        result = probe(root, key_dir=stage / "new")
        if machine_id:
            result["machine_id"] = hashlib.sha256(meta["machine_id"].encode()).hexdigest()
        return result
    except BaseException:
        shutil.rmtree(stage)
        raise


def rollback(token, root=Path("/")):
    stage, meta = _load(root, token)
    if meta["phase"] != "prepared":
        for name, backup in meta["previous"].items():
            path = root / name.lstrip("/")
            if backup is None:
                path.unlink(missing_ok=True)
            else:
                original = stage / "old" / backup
                _atomic_write(path, original.read_bytes(), original.stat().st_mode & 0o777)
        _run([_sshd(), "-t"])
        _run(["systemctl", "reload", meta["service"]])
    meta["phase"] = "prepared"
    _atomic_write(stage / "state.json", json.dumps(meta).encode())
    return {"rolled_back": True}


def activate(token, root=Path("/")):
    stage, meta = _load(root, token)
    if meta["phase"] != "prepared" or _config() != meta["configured"]:
        raise ValueError("SSH configuration or rotation state changed since preparation")
    meta["phase"] = "activating"
    _atomic_write(stage / "state.json", json.dumps(meta).encode())
    try:
        for name in meta["keys"]:
            for suffix, mode in (("", 0o600), (".pub", 0o644)):
                _atomic_write(root / (name + suffix).lstrip("/"), (stage / "new" / (Path(name).name + suffix)).read_bytes(), mode)
        _run([_sshd(), "-t"])
        _run(["systemctl", "reload", meta["service"]])
        if "machine_id" in meta:
            for name in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
                if name in meta["previous"]:
                    _atomic_write(root / name.lstrip("/"), (meta["machine_id"] + "\n").encode(), 0o444)
        meta["phase"] = "active"
        _atomic_write(stage / "state.json", json.dumps(meta).encode())
    except BaseException:
        rollback(token, root)
        raise
    return {"activated": True}


def finish(token, root=Path("/")):
    stage, meta = _load(root, token)
    if meta["phase"] == "activating":
        raise ValueError("rotation outcome uncertain; restore/verify before deleting recovery keys")
    shutil.rmtree(stage)
    return {"cleaned": True}


def _matches(pattern, host):
    if pattern.startswith("|1|"):
        _, _, salt, digest = pattern.split("|")
        return hmac.compare_digest(
            base64.b64encode(hmac.new(base64.b64decode(salt, validate=True), host.encode(), hashlib.sha1).digest()).decode(), digest
        )
    return pattern.lower() == host.lower()


def known_hosts_text(text, records):
    """Replace only named hosts, including hashes; retain other hosts/markers."""
    output = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or line.lstrip().startswith(("#", "@")) or len(fields) < 3:
            output.append(line)
            continue
        patterns = fields[0].split(",")
        for pattern in patterns:
            if any(c in pattern for c in "*?!") and any(fnmatch.fnmatchcase(host, pattern.lstrip("!")) for host in records):
                raise ValueError("matching wildcard/negated known_hosts entry requires manual review")
        remaining = [p for p in patterns if not any(_matches(p, host) for host in records)]
        if remaining:
            output.append(line if remaining == patterns else " ".join([",".join(remaining), *fields[1:]]))
    for host, keys in sorted(records.items()):
        output.extend("%s %s" % (host, key) for key in keys)
    return "\n".join(output) + "\n"


def update_known_hosts(records):
    import fcntl

    directory = Path.home() / ".ssh"
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / "known_hosts"
    with (directory / ".sparkrun-known-hosts.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Respect an intentional symlink while atomically replacing its target.
        path = path.resolve()
        text = path.read_text() if path.exists() else ""
        _atomic_write(path, known_hosts_text(text, records).encode())
    return {"updated": True}


def main(action, token="", machine_id=False, records=None):
    if action == "probe":
        result = probe()
    elif action == "known-hosts":
        result = update_known_hosts(records or {})
    else:
        if os.geteuid() != 0 or not re.fullmatch(r"[0-9a-f]{32}", token):
            raise ValueError("rotation needs root and a valid operation token")
        import fcntl

        # A timed-out SSH caller may retry while its original remote process
        # is still running. Never let activation race rollback or cleanup.
        fd = os.open(STAGE + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("another host-key action is still running; recovery keys retained") from exc
            if action == "prepare":
                result = prepare(token, machine_id)
            else:
                result = {"activate": activate, "rollback": rollback, "finish": finish}[action](token)
    print(MARKER + json.dumps(result), flush=True)
