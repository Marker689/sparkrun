"""Deploy, start, stop, and scrape nv-monitor on remote hosts.

nv-monitor is a lightweight C binary that reads NVML directly and exposes
a Prometheus ``/metrics`` endpoint.  This module handles:

- Deploying the binary to remote hosts via rsync
- Starting/stopping the nv-monitor process over SSH
- Scraping the Prometheus endpoint via SSH port forwarding
"""

from __future__ import annotations

from sparkrun.core.application_profile import remote_cache_path, get_application_profile

import logging
import subprocess
from pathlib import Path
from sparkrun.utils.text import excerpt

from sparkrun.orchestration.ssh import (
    build_ssh_cmd,
    run_remote_command,
    run_remote_scripts_parallel,
    run_rsync_parallel,
)

logger = logging.getLogger(__name__)


def nv_monitor_remote_dir():
    return ".cache/" + get_application_profile().cache_namespace + "/bin"


def nv_monitor_remote_path():
    return remote_cache_path("bin/nv-monitor")


NV_MONITOR_DEFAULT_PORT = 29110
NV_MONITOR_VERSION = "1.0.0"


def _checksum_script(remote_path: str) -> str:
    """Return a bash script that outputs the SHA-256 of a remote file."""
    return ('if [ -x "%(path)s" ]; then sha256sum "%(path)s" | cut -d" " -f1; else echo MISSING; fi') % {"path": remote_path}


def require_nv_monitor_binaries():
    """Fail synchronously when this distribution lacks the optional binary bundle."""
    try:
        from sparkrun.bin import get_binary_checksum, get_binary_resource
    except ImportError as error:
        raise RuntimeError(
            "The nv-monitor backend is unavailable: this distribution does not include its binary assets. "
            "Select the bash monitoring backend instead."
        ) from error
    return get_binary_checksum, get_binary_resource


def ensure_nv_monitor(
    hosts: list[str],
    ssh_kwargs: dict,
    dry_run: bool = False,
) -> dict[str, bool]:
    """Ensure nv-monitor binary is deployed and up-to-date on all hosts.

    For each host, checks if the binary exists and its SHA-256 matches
    the bundled binary. Deploys (via rsync) if missing or mismatched.

    Args:
        hosts: List of remote hostnames.
        ssh_kwargs: SSH connection kwargs (ssh_user, ssh_key, ssh_options).
        dry_run: If True, report what would be done without acting.

    Returns:
        Dict mapping hostname to True if binary is ready, False if deploy failed.
    """
    get_binary_checksum, get_binary_resource = require_nv_monitor_binaries()

    local_checksum = get_binary_checksum("nv-monitor")
    logger.info("Local nv-monitor checksum: %s", local_checksum[:12])

    if dry_run:
        logger.info("[dry-run] Would ensure nv-monitor on %d host(s)", len(hosts))
        return {h: True for h in hosts}

    # Check both binaries exist on remote — if either is missing, redeploy
    prom2json_remote = "$HOME/%s/prom2json" % nv_monitor_remote_dir()
    check_script = ('NV=$(%s); P2J=$(%s); if [ "$NV" = "MISSING" ] || [ "$P2J" = "MISSING" ]; then echo MISSING; else echo "$NV"; fi') % (
        _checksum_script(nv_monitor_remote_path()),
        _checksum_script(prom2json_remote),
    )
    results = run_remote_scripts_parallel(
        hosts,
        check_script,
        ssh_user=ssh_kwargs.get("ssh_user"),
        ssh_key=ssh_kwargs.get("ssh_key"),
        ssh_options=ssh_kwargs.get("ssh_options"),
        timeout=15,
    )

    needs_deploy: list[str] = []
    status: dict[str, bool] = {}

    for r in results:
        remote_checksum = r.stdout.strip()
        if r.success and remote_checksum == local_checksum:
            logger.debug("nv-monitor on %s is up-to-date", r.host)
            status[r.host] = True
        else:
            reason = "missing" if remote_checksum == "MISSING" else "checksum mismatch"
            logger.info("nv-monitor on %s needs deploy (%s)", r.host, reason)
            needs_deploy.append(r.host)

    if not needs_deploy:
        return status

    # Create remote directory via bash.  On some DGX systems ~/.cache may
    # be owned by root (created by a system service), so we fix ownership
    # of each parent if we can, or fall back to creating our own tree.
    mkdir_script = (
        'dir="$HOME/%(d)s"; '
        'for p in "$HOME/.cache" "$HOME/.cache/%(namespace)s" "$dir"; do '
        '  if [ -d "$p" ] && [ ! -w "$p" ]; then '
        '    echo "Fixing permissions on $p" >&2; '
        '    sudo chown "$(id -u):$(id -g)" "$p" 2>/dev/null || true; '
        "  fi; "
        '  mkdir -p "$p" 2>/dev/null || true; '
        "done; "
        '[ -d "$dir" ] && [ -w "$dir" ] && echo OK || echo FAIL'
    ) % {"d": nv_monitor_remote_dir(), "namespace": get_application_profile().cache_namespace}

    mkdir_results = run_remote_scripts_parallel(
        needs_deploy,
        mkdir_script,
        ssh_user=ssh_kwargs.get("ssh_user"),
        ssh_key=ssh_kwargs.get("ssh_key"),
        ssh_options=ssh_kwargs.get("ssh_options"),
        timeout=15,
    )

    # Only rsync to hosts where the directory was created successfully.
    rsync_hosts: list[str] = []
    for r in mkdir_results:
        if r.success and "OK" in r.stdout:
            rsync_hosts.append(r.host)
        else:
            logger.warning(
                "Cannot create deploy directory on %s: %s",
                r.host,
                excerpt(r.stderr, 200) or excerpt(r.stdout, 200),
            )
            status[r.host] = False

    if not rsync_hosts:
        return status

    with get_binary_resource("nv-monitor") as nv_path, get_binary_resource("prom2json") as p2j_path:
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            for src, name in [(nv_path, "nv-monitor"), (p2j_path, "prom2json")]:
                staged = Path(tmpdir) / name
                shutil.copy2(src, staged)
                staged.chmod(0o755)

            # Use ~ for rsync dest (rsync expands ~ natively, unlike $HOME
            # which would be treated as a literal directory name).
            # --chmod=F755 ensures executable bit regardless of remote umask.
            deploy_results = run_rsync_parallel(
                tmpdir,
                rsync_hosts,
                "~/%s" % nv_monitor_remote_dir(),
                ssh_user=ssh_kwargs.get("ssh_user"),
                ssh_key=ssh_kwargs.get("ssh_key"),
                ssh_options=ssh_kwargs.get("ssh_options"),
                rsync_options=["-az", "--mkpath", "--partial", "--links", "--chmod=F755"],
                timeout=60,
            )

    for r in deploy_results:
        if r.success:
            logger.info("Deployed nv-monitor to %s", r.host)
            status[r.host] = True
        else:
            logger.warning("Failed to deploy nv-monitor to %s: %s", r.host, excerpt(r.stderr, 200))
            status[r.host] = False

    return status


def start_nv_monitor_ssh(
    host: str,
    ssh_kwargs: dict,
    port: int = NV_MONITOR_DEFAULT_PORT,
    local_forward_port: int | None = None,
) -> subprocess.Popen | None:
    """Start nv-monitor on a remote host and set up SSH port forwarding.

    Opens an SSH connection with local port forwarding (-L) that also
    starts nv-monitor on the remote side. The returned Popen handle
    keeps both the tunnel and the remote process alive.

    Args:
        host: Remote hostname.
        ssh_kwargs: SSH connection kwargs.
        port: Port for nv-monitor to listen on remotely.
        local_forward_port: Local port to forward to remote port.
            If None, uses the same as *port*.

    Returns:
        Popen handle for the SSH tunnel process, or None on failure.
    """
    local_port = local_forward_port or port

    cmd = build_ssh_cmd(
        host,
        ssh_user=ssh_kwargs.get("ssh_user"),
        ssh_key=ssh_kwargs.get("ssh_key"),
        ssh_options=ssh_kwargs.get("ssh_options"),
    )
    # Add port forwarding
    cmd.extend(["-L", "%d:localhost:%d" % (local_port, port)])
    # Wrap nv-monitor in a bash trap so it auto-cleans on SSH disconnect.
    # When the SSH connection drops, bash receives SIGHUP and the EXIT trap
    # kills the nv-monitor child — preventing orphaned processes.
    cmd.extend(
        [
            "bash",
            "-c",
            "trap 'kill %%1 2>/dev/null' EXIT HUP TERM INT; %s -n -p %d & wait" % (nv_monitor_remote_path(), port),
        ]
    )

    try:
        # stdin=/dev/null: this is a long-lived `ssh` tunnel, and `ssh` reads and
        # forwards stdin.  Inheriting a caller's terminal would silently consume
        # its keystrokes for the entire monitoring session, so the child gets none.
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info("Started nv-monitor on %s (port %d, forwarded to localhost:%d)", host, port, local_port)
        return proc
    except OSError as e:
        logger.warning("Failed to start nv-monitor SSH tunnel to %s: %s", host, e)
        return None


def stop_nv_monitor_ssh(proc: subprocess.Popen | None) -> None:
    """Terminate an SSH tunnel + nv-monitor process.

    Args:
        proc: Popen handle from :func:`start_nv_monitor_ssh`, or None.
    """
    if proc is None:
        return
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


def stop_nv_monitor_remote(
    host: str,
    ssh_kwargs: dict,
    port: int = NV_MONITOR_DEFAULT_PORT,
) -> None:
    """Kill any nv-monitor process on a remote host.

    Uses pkill to find and terminate nv-monitor processes matching the port.

    Args:
        host: Remote hostname.
        ssh_kwargs: SSH connection kwargs.
        port: Port the nv-monitor was started with (for identification).
    """
    kill_script = 'pkill -f "$HOME/.cache/%s/bin/nv-monitor.*-p %d" 2>/dev/null; true' % (get_application_profile().cache_namespace, port)
    run_remote_command(
        host,
        kill_script,
        ssh_user=ssh_kwargs.get("ssh_user"),
        ssh_key=ssh_kwargs.get("ssh_key"),
        ssh_options=ssh_kwargs.get("ssh_options"),
        timeout=10,
        quiet=True,
    )


def scrape_metrics(url: str, timeout: float = 5.0) -> dict[str, float]:
    """Scrape Prometheus metrics from an HTTP endpoint.

    Uses a raw TCP socket + HTTP/1.0 GET instead of ``urllib`` to minimize
    GIL overhead.  ``urllib.request.urlopen`` builds handler chains and
    response wrappers that are all GIL-bound Python work — too heavy for
    a tight polling loop that runs every 2 seconds per host.

    Args:
        url: Full URL to the metrics endpoint (e.g. ``"http://localhost:29110/metrics"``).
        timeout: HTTP request timeout in seconds.

    Returns:
        Parsed metrics dict from :func:`~sparkrun.utils.prometheus.parse_prometheus_text`.
    """
    import socket

    from sparkrun.utils.prometheus import parse_prometheus_text

    # Parse host:port from url (expects http://host:port/path)
    try:
        # Strip scheme
        rest = url.split("://", 1)[1] if "://" in url else url
        hostport, path = rest.split("/", 1) if "/" in rest else (rest, "metrics")
        if ":" in hostport:
            host, port_str = hostport.rsplit(":", 1)
            port = int(port_str)
        else:
            host, port = hostport, 80
    except (ValueError, IndexError):
        return {}

    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        try:
            sock.sendall(("GET /%s HTTP/1.0\r\nHost: %s\r\n\r\n" % (path, host)).encode())
            chunks = []
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            sock.close()

        raw = b"".join(chunks).decode("utf-8", errors="replace")
        # Split off HTTP headers
        header_end = raw.find("\r\n\r\n")
        body = raw[header_end + 4 :] if header_end >= 0 else raw
        return parse_prometheus_text(body)
    except Exception as e:
        logger.debug("Failed to scrape %s: %s", url, e)
        return {}
