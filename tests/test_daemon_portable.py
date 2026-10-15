"""Native OS matrix: real TCP listeners, workers, storage and process lifetime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from test_daemon import binding as binding, controlled_plugin as controlled_plugin, real_server, wait_state
from sparkrun.core.process_identity import alive, process_identity
from sparkrun.daemon.client import Client, DaemonError
from sparkrun.daemon.jobs import JobStore
from sparkrun.daemon.state import read_connection


TCP = ("--no-unix", "--tcp", "127.0.0.1:0")


def test_tcp_shutdown_rotation_and_worker_adoption(controlled_plugin):
    binding = controlled_plugin
    with real_server(binding, extra=TCP) as (process, client):
        assert not binding.socket.exists()
        connection = read_connection(binding.connection)
        with pytest.raises(DaemonError):
            Client(url=connection["url"]).info()
        from sparkrun.daemon.auth import TokenStore

        token_store = TokenStore(binding)
        identifier, read_token = token_store.create("read-only", scope="read")
        with pytest.raises(DaemonError):
            Client(url=connection["url"], token=read_token).request("POST", "/v1/shutdown", {})
        token_store.revoke(identifier)
        assert client.info()["protocol"] == 1
        job = client.call("plugins.daemon-fixture.wait", key="portable-once")["job_id"]
        row = wait_state(client, job, {"waiting_input"})
        worker = JobStore(binding.directory).get(job)["worker"]
        assert worker["endpoint"]["transport"] == "tcp"
        with pytest.raises(ValueError, match="credential"):
            from sparkrun.daemon.control import request

            request(worker, "wrong-nonce", "cancel")
        client.request("POST", "/v1/shutdown", {})
        process.wait(timeout=20)
        assert alive(worker["process"])
        assert not binding.connection.exists()
    with real_server(binding, extra=TCP) as (_, replacement):
        assert read_connection(binding.connection)["token"] != connection["token"]
        assert client.call("plugins.daemon-fixture.wait", key="portable-once")["job_id"] == job
        assert replacement.request("GET", "/v1/jobs/" + job)["prompt"] == row["prompt"]
        replacement.request("POST", f"/v1/jobs/{job}/input", {"prompt_id": row["prompt"]["prompt_id"], "value": "portable-secret"})
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            row = replacement.request("GET", "/v1/jobs/" + job)
            if row["prompt"] and row["prompt"]["kind"] == "approval":
                break
            time.sleep(0.05)
        replacement.request("POST", f"/v1/jobs/{job}/input", {"prompt_id": row["prompt"]["prompt_id"], "value": True})
        assert replacement.wait(job, timeout=15) == {"accepted": True, "echo": "[redacted]"}
        assert (binding.config_path.parent / "executions.txt").read_text().splitlines() == [job]
    for path in binding.directory.iterdir():
        if path.is_file():
            assert b"portable-secret" not in path.read_bytes()


@pytest.mark.parametrize("crash", [False, True])
def test_server_helper_controls_daemon_behind_launcher(binding, monkeypatch, crash):
    """Exercise a Windows-style Python launcher on every test platform."""
    import psutil
    from sparkrun.utils.fs import file_lock

    popen = subprocess.Popen
    launchers = []

    def launch(argv, **kwargs):
        child = popen(
            [sys.executable, "-c", "import subprocess, sys; sys.exit(subprocess.Popen(sys.argv[1:]).wait())", *argv],
            **kwargs,
        )
        launchers.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    identity = None
    try:
        with real_server(binding, extra=TCP) as (process, _):
            identity = binding.read_state()["process"]
            assert process.pid == identity["pid"]
            assert process.pid != launchers[0].pid
            if crash:
                process.kill()
                process.wait(timeout=10)
                assert not alive(identity)
        assert not alive(identity)
        assert launchers[0].poll() is not None
        with file_lock(binding.directory / "server.lock", blocking=False):
            pass
    finally:
        if alive(identity):
            psutil.Process(identity["pid"]).kill()


def test_tcp_api_crash_preserves_worker_and_cooperative_cancel(controlled_plugin):
    binding = controlled_plugin
    with real_server(binding, extra=TCP) as (process, client):
        job = client.call("plugins.daemon-fixture.wait")["job_id"]
        wait_state(client, job, {"waiting_input"})
        worker = JobStore(binding.directory).get(job)["worker"]
        process.kill()
        process.wait()
        assert alive(worker["process"])
    with real_server(binding, extra=TCP) as (_, client):
        client.request("DELETE", "/v1/jobs/" + job)
        wait_state(client, job, {"cancelled"})


def test_tcp_forced_cancel_contains_descendants(controlled_plugin):
    binding = controlled_plugin
    plugin = binding.directory / "plugins/daemon_fixture.py"
    plugin.write_text(
        plugin.read_text().replace(
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n        signal.signal(signal.SIGINT, signal.SIG_IGN)",
        )
    )
    with binding.config_path.open("a") as stream:
        stream.write("daemon:\n  cancel_grace: 1\n")
    with real_server(binding, extra=TCP) as (_, client):
        job = client.call("plugins.daemon-fixture.wait", {"mode": "tree"})["job_id"]
        deadline = time.monotonic() + 20
        children = []
        while not children and time.monotonic() < deadline:
            children = [e for e in JobStore(binding.directory).events(job) if e["type"] == "child"]
            time.sleep(0.05)
        assert children
        child = process_identity(children[0]["pid"])
        assert alive(child)
        client.request("DELETE", "/v1/jobs/" + job)
        wait_state(client, job, {"lost"}, timeout=20)
        deadline = time.monotonic() + 5
        while alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not alive(child)


def test_instance_lock_and_cli_shutdown(binding):
    with real_server(binding, extra=TCP) as (process, client):
        result = subprocess.run(
            [sys.executable, "-m", "sparkrun.daemon", "--config", str(binding.config_path), "serve", *TCP],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode != 0 and "already running" in result.stderr
        assert client.info()["protocol"] == 1
        result = subprocess.run(
            [sys.executable, "-m", "sparkrun.daemon", "--config", str(binding.config_path), "stop"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        process.wait(timeout=10)


def test_stale_identity_and_connection_helper(binding):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        identity = process_identity(child.pid)
        assert alive(identity)
        assert not alive({**identity, "created": "another-process"})
    finally:
        child.kill()
        child.wait()
    assert not alive(identity)
    with real_server(binding, extra=TCP):
        result = subprocess.run(
            [sys.executable, "-m", "sparkrun.daemon", "connection-status", str(binding.connection)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["alive"] is True
        assert "token" not in result.stdout


@pytest.mark.skipif(os.name != "nt", reason="Native Windows ACLs")
def test_windows_credentials_have_protected_acl(binding):
    import win32security
    import win32con
    from sparkrun.utils.windows import check_private

    with real_server(binding, extra=TCP):
        check_private(binding.connection)
        sd = win32security.GetNamedSecurityInfo(
            str(binding.connection), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION
        )
        assert sd.GetSecurityDescriptorControl()[0] & win32security.SE_DACL_PROTECTED
        acl = sd.GetSecurityDescriptorDacl()
        acl.AddAccessAllowedAce(
            win32security.ACL_REVISION, win32con.GENERIC_READ, win32security.CreateWellKnownSid(win32security.WinWorldSid, None)
        )
        win32security.SetNamedSecurityInfo(
            str(binding.connection), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION, None, None, acl, None
        )
        with pytest.raises(ValueError, match="permissions"):
            read_connection(binding.connection)


@pytest.mark.skipif(os.name != "nt", reason="Native Windows ACLs")
@pytest.mark.parametrize("ancestor", [True, False])
def test_windows_owner_rights_ace_is_not_foreign_access(tmp_path, ancestor):
    import win32con
    import win32security
    from sparkrun.utils.windows import check_private, secure_path

    directory = tmp_path / "owned"
    directory.mkdir()
    secure_path(directory)
    sd = win32security.GetNamedSecurityInfo(str(directory), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION)
    acl = sd.GetSecurityDescriptorDacl()
    owner_rights = win32security.ConvertStringSidToSid("S-1-3-4")
    acl.AddAccessAllowedAce(win32security.ACL_REVISION, win32con.GENERIC_ALL, owner_rights)
    win32security.SetNamedSecurityInfo(
        str(directory), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION, None, None, acl, None
    )

    check_private(directory, ancestor=ancestor)


@pytest.mark.parametrize("extra", [TCP, ("--no-unix", "--tcp", "0.0.0.0:0", "--allow-insecure")])
def test_tcp_restart_cli_preserves_transport_and_rotates_token(binding, extra):
    with real_server(binding, extra=extra) as (_, client):
        before = read_connection(binding.connection)
        result = subprocess.run(
            [sys.executable, "-m", "sparkrun.daemon", "--config", str(binding.config_path), "restart"],
            capture_output=True,
            text=True,
            timeout=40,
        )
        try:
            assert result.returncode == 0, result.stderr
            assert client.info()["protocol"] == 1
            after = read_connection(binding.connection)
            assert after["process"] != before["process"]
            assert after["token"] != before["token"]
            assert after["socket"] is None
            assert after["launch"] == before["launch"]
        finally:
            client.request("POST", "/v1/shutdown", {})
            deadline = time.monotonic() + 15
            while binding.read_state() and time.monotonic() < deadline:
                time.sleep(0.05)


@pytest.mark.skipif(os.name != "nt", reason="Native Windows crash containment")
def test_windows_worker_crash_kills_contained_child(controlled_plugin):
    import psutil

    binding = controlled_plugin
    with real_server(binding, extra=TCP) as (_, client):
        job = client.call("plugins.daemon-fixture.wait", {"mode": "tree"})["job_id"]
        store = JobStore(binding.directory)
        deadline = time.monotonic() + 20
        children = []
        while not children and time.monotonic() < deadline:
            children = [e for e in store.events(job) if e["type"] == "child"]
            time.sleep(0.05)
        assert children
        child = process_identity(children[0]["pid"])
        worker = store.get(job)["worker"]["process"]
        assert alive(worker) and alive(child)
        psutil.Process(worker["pid"]).kill()
        wait_state(client, job, {"lost"})
        deadline = time.monotonic() + 5
        while alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not alive(child)


def test_parent_watchdog_stops_api_only(controlled_plugin):
    binding = controlled_plugin
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(90)"])
    try:
        with real_server(binding, extra=(*TCP, "--parent-pid", str(parent.pid))) as (process, client):
            job = client.call("plugins.daemon-fixture.wait")["job_id"]
            wait_state(client, job, {"waiting_input"})
            worker = JobStore(binding.directory).get(job)["worker"]["process"]
            parent.kill()
            parent.wait()
            process.wait(timeout=20)
            assert alive(worker)
        with real_server(binding, extra=TCP) as (_, client):
            client.request("DELETE", "/v1/jobs/" + job)
            wait_state(client, job, {"cancelled"})
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


@pytest.mark.skipif(os.name != "nt", reason="Native Windows reparse points")
def test_windows_private_storage_rejects_junctions(binding):
    from sparkrun.daemon.state import private_directory

    target = binding.directory / "real"
    target.mkdir()
    link = binding.directory / "junction"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True, timeout=10)
    try:
        with pytest.raises(ValueError, match="Reparse"):
            private_directory(link / "credentials")
    finally:
        link.rmdir()


def test_uninspectable_process_is_not_treated_as_dead(monkeypatch):
    from types import SimpleNamespace
    from sparkrun.core import process_identity as identity
    import psutil

    monkeypatch.setattr(identity, "sys", SimpleNamespace(platform="darwin"))

    def inaccessible(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", inaccessible)
    with pytest.raises(PermissionError, match="Cannot inspect"):
        identity.alive({"pid": 42, "platform": "darwin", "created": 1})


@pytest.mark.skipif(os.name != "nt", reason="Native Windows file sharing")
def test_windows_atomic_publication_waits_for_reader(binding):
    from concurrent.futures import ThreadPoolExecutor
    from sparkrun.utils.fs import atomic_private_write
    import win32file
    import win32con

    path = binding.directory / "sharing.json"
    atomic_private_write(path, '{"old":true}')
    reader = win32file.CreateFile(str(path), win32con.GENERIC_READ, win32con.FILE_SHARE_READ, None, win32con.OPEN_EXISTING, 0, None)
    try:
        with ThreadPoolExecutor(1) as pool:
            writer = pool.submit(atomic_private_write, path, '{"new":true}')
            time.sleep(0.1)
            assert not writer.done()
            reader.Close()
            assert writer.result(timeout=5)
    finally:
        reader.Close()
    assert json.loads(path.read_text()) == {"new": True}


@pytest.mark.parametrize("host,connect_host", [("0.0.0.0", "127.0.0.1"), ("::", "[::1]")])
def test_remote_plaintext_requires_opt_in_and_keeps_auth(binding, host, connect_host):
    import socket
    from urllib.parse import urlsplit
    from sparkrun.core.application_profile import child_environment

    if host == "::":
        if not socket.has_ipv6:
            pytest.skip("IPv6 unavailable")
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
                probe.bind((host, 0))
        except OSError:
            pytest.skip("IPv6 bind unavailable")
    address = f"[{host}]:0" if ":" in host else f"{host}:0"
    extra = ("--no-unix", "--tcp", address)
    refused = subprocess.run(
        [sys.executable, "-m", "sparkrun.daemon", "--config", str(binding.config_path), "serve", *extra],
        env={**os.environ, **child_environment(config_path=binding.config_path)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert refused.returncode != 0 and "explicit --allow-insecure" in refused.stderr
    assert binding.read_state() is None
    with real_server(binding, extra=(*extra, "--allow-insecure")) as (_, local):
        state = binding.read_state()
        connection = read_connection(binding.connection)
        assert state["url"].startswith("http://")
        assert state["launch"]["allow_insecure"] is True
        # Local CLI discovery must not try to connect to an unspecified bind address.
        assert connection["url"].startswith("http://127.0.0.1:")
        url = f"http://{connect_host}:{urlsplit(state['url']).port}"
        for token in (None, "invalid-token"):
            with pytest.raises(DaemonError) as denied:
                Client(url=url, token=token).info()
            assert denied.value.status == 401
        remote = Client(url=url, token=connection["token"])
        assert remote.info()["binding"] == local.info()["binding"] == binding.key


@pytest.mark.parametrize("option", ["--tls-cert", "--tls-key"])
def test_allow_insecure_does_not_accept_partial_tls(binding, option):
    from sparkrun.core.application_profile import child_environment

    partial = binding.directory / "partial.pem"
    partial.write_text("incomplete TLS configuration")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sparkrun.daemon",
            "--config",
            str(binding.config_path),
            "serve",
            "--no-unix",
            "--tcp",
            "0.0.0.0:0",
            "--allow-insecure",
            option,
            str(partial),
        ],
        env={**os.environ, **child_environment(config_path=binding.config_path)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0 and "TLS requires both certificate and private key" in result.stderr
    assert binding.read_state() is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX OS account directory aliases")
def test_runtime_lease_canonicalizes_account_home(tmp_path, monkeypatch, real_runtime_state):
    from sparkrun.core.runtime_lease import environment_state

    target = tmp_path / "real-temp"
    target.mkdir()
    alias = tmp_path / "system-temp"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr("sparkrun.core.runtime_lease._account_home", lambda: alias)
    directory = environment_state()
    assert directory.parent == target.resolve() / ".sparkrun-runtime"
    assert directory.is_dir() and directory.stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(os.name != "nt", reason="Native Windows ACLs")
def test_windows_private_storage_rejects_replaceable_ancestor(binding):
    import win32con
    import win32security
    from sparkrun.utils.windows import private_directory, secure_path, user_sid

    parent = binding.directory / "public-parent"
    parent.mkdir()
    acl = win32security.ACL()
    acl.AddAccessAllowedAce(win32security.ACL_REVISION, win32con.GENERIC_ALL, user_sid())
    acl.AddAccessAllowedAce(win32security.ACL_REVISION, win32con.GENERIC_ALL, win32security.ConvertStringSidToSid("S-1-1-0"))
    win32security.SetNamedSecurityInfo(
        str(parent),
        win32security.SE_FILE_OBJECT,
        win32security.DACL_SECURITY_INFORMATION | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        None,
        None,
        acl,
        None,
    )
    with pytest.raises(ValueError, match="SID S-1-1-0, mask"):
        private_directory(parent / "credentials")
    assert not (parent / "credentials").exists()
    secure_path(parent)
    assert private_directory(parent / "credentials").is_dir()
