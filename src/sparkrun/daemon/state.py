"""Controller binding, protected state, and runtime discovery without bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.application_profile import get_application_profile
from sparkrun.utils.fs import atomic_private_write


def private_directory(path):
    if os.name == "nt":
        from sparkrun.utils.windows import private_directory as windows_directory

        return windows_directory(path)
    path = Path(path)
    # Validate ancestors before creating children: a private leaf is not safe
    # beneath another user's writable or symlink-controlled directory.
    uid = os.getuid() if hasattr(os, "getuid") else None
    protected = False
    for parent in reversed(path.absolute().parents):
        if not parent.exists():
            parent.mkdir(mode=0o700)
        info = parent.lstat()
        sticky_root = bool(info.st_mode & stat.S_ISVTX) and info.st_uid == 0
        if (
            not stat.S_ISDIR(info.st_mode)
            or (uid is not None and info.st_uid not in {0, uid})
            or (uid is not None and info.st_mode & 0o022 and not sticky_root and not protected)
        ):
            raise ValueError(f"Unsafe daemon directory ancestor: {parent}")
        if info.st_uid == uid and not info.st_mode & 0o077:
            protected = True  # Other users cannot traverse this subtree.
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
        raise ValueError(f"Unsafe daemon directory: {path}")
    path.chmod(0o700)
    return path


@dataclass(frozen=True)
class Binding:
    config_path: Path
    profile: str
    key: str
    directory: Path
    runtime: Path

    @classmethod
    def from_config(cls, config=None):
        config = config or SparkrunConfig()
        path = config.config_path.expanduser().resolve()
        profile = config.profile.id
        from sparkrun.core.process_identity import user_identity

        uid = user_identity()
        key = hashlib.sha256(f"{uid}\0{profile}\0{path}".encode()).hexdigest()[:20]
        directory = path.parent / "daemon" / key
        runtime_root = Path(os.environ.get("XDG_RUNTIME_DIR", str(path.parent / "run")))
        runtime = runtime_root / get_application_profile().id / "daemon" / key
        if os.name != "nt" and len(str(runtime / ("0" * 32 + ".sock")).encode()) > 100:
            import tempfile

            runtime = Path(tempfile.gettempdir()).resolve() / f"sparkrun-{uid}-{key}"
        return cls(path, profile, key, directory, runtime)

    @property
    def socket(self):
        return self.runtime / "api.sock"

    @property
    def connection(self):
        return self.runtime / "connection.json"

    def prepare(self):
        private_directory(self.directory.parent)
        private_directory(self.directory)
        private_directory(self.runtime)
        return self

    def info(self):
        return {"profile": self.profile, "binding": self.key, "config_path": str(self.config_path)}

    def read_state(self):
        try:
            return json.loads((self.directory / "server.json").read_text())
        except FileNotFoundError:
            return None

    def write_state(self, value):
        self.prepare()
        atomic_private_write(self.directory / "server.json", json.dumps(value))


def read_connection(path):
    """Read local credentials only after verifying ownership and permissions."""
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Connection file must be a regular file")
    if os.name == "nt":
        from sparkrun.utils.windows import check_private

        check_private(path)
        for parent in path.absolute().parents:
            check_private(parent, ancestor=True)
    elif info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Connection file must be owned by the current user and private")
    value = json.loads(path.read_text())
    if value.get("protocol") != 1 or not value.get("process") or not value.get("token"):
        raise ValueError("Invalid connection file")
    return value
