"""Filesystem helpers that behave the same on POSIX and Windows."""

from __future__ import annotations

import os
from contextlib import contextmanager

__all__ = ("open_private_write", "atomic_private_write", "file_lock")


def open_private_write(path: str | os.PathLike) -> int:
    """Open *path* for writing, owner-only, without following a symlink.

    ``O_NOFOLLOW`` refuses to write *through* a symlink, so another local user
    can't pre-create one at a predictable path and capture what we write. It is
    POSIX-only: on Windows ``os.O_NOFOLLOW`` does not exist, and naming it
    directly raises ``AttributeError`` — which callers that wrap the write in a
    broad ``except`` will swallow, leaving the file silently unwritten. (That is
    exactly how a Windows control machine ended up with no job metadata for jobs
    it had launched, so ``logs`` and ``stop`` could not find their hosts.)

    ``O_BINARY`` matters on Windows for the same reason it does elsewhere in
    sparkrun: text mode would rewrite ``\\n`` on write.

    Returns:
        A file descriptor opened ``O_WRONLY|O_CREAT|O_TRUNC`` with mode 0600.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    # Absent on Windows; the symlink hardening simply doesn't apply there.
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    return os.open(path, flags, 0o600)


def atomic_private_write(path: str | os.PathLike, content: str, *, overwrite: bool = True) -> bool:
    """Publish a complete owner-only text file atomically.

    With ``overwrite=False``, concurrent creators agree on the first complete
    file; return False when another creator already published it. The parent
    directory must exist. Replacing a symlink replaces the link itself.
    """
    from pathlib import Path
    import tempfile

    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".%s." % path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            if os.name == "nt":
                from sparkrun.utils.windows import secure_path

                secure_path(temporary)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            # On Windows a concurrent reader can briefly deny replacement.
            # Preserve atomic publication; never fall back to truncating in place.
            import time

            deadline = time.monotonic() + 1
            while True:
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError as exc:
                    if os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32} or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
        return True
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def file_lock(path, *, blocking=True):
    """Hold an exclusive process lock; retain the lock file after release.

    Callers handle reentrancy. The separate inode remains stable when the data
    file is replaced or deleted. Opening refuses symlinks where supported.
    """
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            import time

            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    import errno

                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    if not blocking:
                        raise BlockingIOError("Lock is already held") from exc
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
