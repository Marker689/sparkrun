"""Native Windows ownership, private storage, and worker containment.

Imported only on Windows. pywin32 is a platform-specific runtime dependency.
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false
from __future__ import annotations

from pathlib import Path
import stat
from typing import Any, cast

import win32api
import win32con
import win32job
import win32security


def user_sid():
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        return win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    finally:
        win32api.CloseHandle(token)


def user_id():
    return win32security.ConvertSidToStringSid(user_sid())


def account_home():
    """Read the OS account profile, not the process's USERPROFILE environment."""
    import win32profile

    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        return Path(win32profile.GetUserProfileDirectory(token))
    finally:
        win32api.CloseHandle(token)


def _trusted_sids():
    return {user_id(), "S-1-5-18", "S-1-5-32-544"}  # current user, SYSTEM, administrators


#: OWNER RIGHTS: an ACE for it grants access to the object's owner only, which
#: check_private has already required to be trusted. Runner and some profile
#: temp directories carry one with full control.
_OWNER_RIGHTS_SID = "S-1-3-4"


def _descriptor(path):
    if cast(Any, Path(path).lstat()).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"Reparse points are not allowed in private daemon paths: {path}")
    return win32security.GetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION,
    )


def check_private(path, *, ancestor=False):
    """Fail closed on public credentials or replaceable ancestor directories."""
    sd = _descriptor(path)
    trusted = _trusted_sids()
    if win32security.ConvertSidToStringSid(sd.GetSecurityDescriptorOwner()) not in trusted:
        # Windows system directories may be owned by TrustedInstaller.
        if not ancestor or win32security.ConvertSidToStringSid(sd.GetSecurityDescriptorOwner()) != (
            "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"
        ):
            raise ValueError(f"Foreign owner of daemon path: {path}")
    acl = sd.GetSecurityDescriptorDacl()
    if acl is None:
        raise ValueError(f"Unrestricted daemon path: {path}")
    # Ancestors may be readable by everyone, but not replaceable by them.
    dangerous = 0x00010000 | 0x00040000 | 0x00080000 | 0x40 | 0x10000000 | 0x40000000
    for index in range(acl.GetAceCount()):
        ace = acl.GetAce(index)
        (kind, flags), mask, sid = ace
        if flags & win32con.INHERIT_ONLY_ACE:
            continue
        if kind == win32security.ACCESS_ALLOWED_ACE_TYPE:
            sid_string = win32security.ConvertSidToStringSid(sid)
            if sid_string not in trusted and sid_string != _OWNER_RIGHTS_SID and (not ancestor or mask & dangerous):
                raise ValueError(f"Unsafe permissions on daemon path: {path} (SID {sid_string}, mask {mask:#x}, flags {flags:#x})")
        elif kind != win32security.ACCESS_DENIED_ACE_TYPE:
            raise ValueError(f"Unsupported ACL on daemon path: {path}")


def secure_path(path):
    """Replace inherited access with a protected owner-only DACL."""
    sd = _descriptor(path)
    if win32security.ConvertSidToStringSid(sd.GetSecurityDescriptorOwner()) not in _trusted_sids():
        raise ValueError(f"Foreign owner of daemon path: {path}")
    acl = win32security.ACL()
    flags = win32con.OBJECT_INHERIT_ACE | win32con.CONTAINER_INHERIT_ACE if Path(path).is_dir() else 0
    acl.AddAccessAllowedAceEx(win32security.ACL_REVISION, flags, win32con.GENERIC_ALL, user_sid())
    win32security.SetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        win32security.DACL_SECURITY_INFORMATION | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        None,
        None,
        acl,
        None,
    )
    check_private(path)


def private_directory(path):
    path = Path(path).absolute()
    for parent in reversed(path.parents):
        if not parent.exists():
            parent.mkdir()
            secure_path(parent)
        check_private(parent, ancestor=True)
    path.mkdir(exist_ok=True)
    secure_path(path)
    return path


def worker_job_name(identity):
    import hashlib
    import json

    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return "Local\\sparkrun-worker-" + key


class WorkerJob:
    """Owned by the worker, never the API; last-handle close cleans up crashes.

    Successful jobs release containment so intentional persistent local services
    can outlive the workflow. Failed/cancelled jobs keep crash cleanup enabled.
    """

    def __init__(self, identity):
        win32api.SetLastError(0)
        self.handle = cast(Any, win32job.CreateJobObject(None, worker_job_name(identity)))
        if win32api.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            self.handle.Close()
            raise RuntimeError("Worker containment name is already in use")
        self._kill_on_close(True)
        try:
            win32job.AssignProcessToJobObject(self.handle, win32api.GetCurrentProcess())
        except BaseException:
            self.handle.Close()
            raise

    def _kill_on_close(self, enabled):
        _kill_on_close(self.handle, enabled)

    def release_success(self):
        self._kill_on_close(False)
        self.handle.Close()


def terminate_worker_job(identity):
    """Named containment survives API restarts and covers grandchildren too."""
    from sparkrun.core.process_identity import alive

    if not alive(identity):
        return
    job = cast(Any, win32job.OpenJobObject(win32job.JOB_OBJECT_TERMINATE, False, worker_job_name(identity)))
    try:
        if alive(identity):
            win32job.TerminateJobObject(job, 1)
    finally:
        win32api.CloseHandle(job)


def _kill_on_close(handle, enabled):
    info = win32job.QueryInformationJobObject(handle, win32job.JobObjectExtendedLimitInformation)
    info["BasicLimitInformation"]["LimitFlags"] = win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE if enabled else 0
    win32job.SetInformationJobObject(handle, win32job.JobObjectExtendedLimitInformation, info)


class ChildJob:
    """Parent-owned containment for a gated child, including its descendants."""

    def __init__(self, pid):
        # pywin32 rejects None for the name; an empty name is an unnamed job.
        self.handle = cast(Any, win32job.CreateJobObject(None, ""))
        try:
            _kill_on_close(self.handle, True)
            process = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, pid)
            try:
                win32job.AssignProcessToJobObject(self.handle, process)
            finally:
                win32api.CloseHandle(process)
        except BaseException:
            self.handle.Close()
            raise

    def close(self):
        self.handle.Close()
