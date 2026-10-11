"""``sparkrun setup fe-system-update`` — apt + firmware update across Founders Edition hosts."""

from __future__ import annotations

from sparkrun.core.operation import submit as _operation_submit

import re
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import click

from .._common import _get_cluster_manager, _get_context, _resolve_setup_context, dry_run_option, host_options
from . import setup

#: The update sequence, in order.  Each entry is ``(description, script)`` and
#: runs as root.  Ordering matters: the package lists must be refreshed before
#: the upgrade, and the firmware metadata before the firmware.
_FE_UPDATE_STEPS = [
    ("Updating package lists", "apt update"),
    ("Upgrading packages", "DEBIAN_FRONTEND=noninteractive apt dist-upgrade -y"),
    ("Refreshing firmware metadata", "fwupdmgr refresh --force"),
    ("Upgrading firmware", "fwupdmgr upgrade -y --no-reboot-check"),
]

#: Per-host, per-step cap.  `dist-upgrade` over a slow mirror is the long pole.
_STEP_TIMEOUT_S = 600

#: Reboot is fire-and-forget — the host drops the connection on its way down.
_REBOOT_TIMEOUT_S = 10

#: Backgrounded so the SSH call returns before the host goes away.
_REBOOT_SCRIPT = "nohup bash -c 'sleep 2 && reboot' &>/dev/null &"

#: How often a step that is still running says so.  apt and fwupdmgr are
#: captured, not streamed, so without this a `dist-upgrade` looks identical to
#: a hung SSH for minutes at a time.
_HEARTBEAT_S = 15

#: Lines worth surfacing from a step's output, most specific first.  Matched
#: against the whole of stdout rather than keyed to a step index, so reordering
#: or adding a step doesn't silently mislabel the summary.
_SUMMARY_PATTERNS = (
    r"^\d+ upgraded, \d+ newly installed.*",
    r"^\d+ packages can be upgraded.*",
    r"^All packages are up to date.*",
    r"^Successfully installed firmware.*",
    r"^No updatable devices.*",
    r"^Devices with no available firmware updates.*",
    r"^Successfully downloaded new metadata.*",
    r"^Metadata is up to date.*",
)


def _summarize(output: str) -> str:
    """Pick the one line from *output* that tells the operator what happened.

    apt and fwupdmgr both bury their verdict in a wall of progress noise, and
    the tail is usually the least informative part of it — `apt update` ends on
    a "Reading state information" line while the count of upgradable packages
    sits several lines above.  Falls back to the last non-empty line.
    """
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    for pattern in _SUMMARY_PATTERNS:
        for line in lines:
            if re.match(pattern, line):
                return line
    return lines[-1]


def _elapsed(seconds: float) -> str:
    """Render a duration the way an operator reads it: `45s`, `3m12s`."""
    seconds = int(seconds)
    if seconds < 60:
        return "%ds" % seconds
    return "%dm%02ds" % divmod(seconds, 60)


class _Heartbeat:
    """Report the hosts a step is still waiting on, every :data:`_HEARTBEAT_S`.

    Plain appended lines rather than a redrawn status line: this output is
    routinely piped to a file or a CI log, and it interleaves with completion
    lines printed from the main thread, so a carriage-return redraw would be
    both fragile and unreadable after the fact.

    Shares the caller's print lock so a heartbeat can't land in the middle of
    a completion block.
    """

    def __init__(self, hosts, lock: threading.Lock, *, enabled: bool = True, interval: float | None = None):
        self._pending = set(hosts)
        self._total = len(self._pending)
        self._lock = lock
        # Resolved at call time, not bound as a default: a default argument
        # freezes the module constant at import and silently ignores anyone
        # who reassigns it.
        self._interval = _HEARTBEAT_S if interval is None else interval
        self._enabled = enabled and self._total > 0
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _Heartbeat:
        if self._enabled:
            # Daemon: an interrupt must not be held up by the ticker.
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def done(self, host: str) -> None:
        with self._lock:
            self._pending.discard(host)

    def _tick(self) -> None:
        while not self._stop.wait(self._interval):
            with self._lock:
                if not self._pending:
                    return
                waiting = sorted(self._pending)
                shown = ", ".join(waiting[:4])
                if len(waiting) > 4:
                    shown += ", +%d more" % (len(waiting) - 4)
                click.echo(
                    "  ... %s elapsed — %d/%d done, still running: %s"
                    % (_elapsed(time.monotonic() - self._t0), self._total - len(waiting), self._total, shown)
                )


def _partition_control_node(host_list: list[str]) -> tuple[list[str], list[str]]:
    """Split *host_list* into ``(remote_hosts, control_node_aliases)``.

    A cluster's control node is frequently a member of its own cluster, and
    frequently appears in the host list twice — once by hostname and once by
    LAN IP (``spark-01`` *is* ``192.168.1.41``).  Both label the same
    machine, so every alias is collapsed into one target: updating it twice
    is wasted work and rebooting it twice is a race.

    Identity is decided by :func:`~sparkrun.utils.is_local_host` rather than by
    string equality (which is what let the two spellings through), and
    deliberately *not* by ``should_run_locally``: under indirect sudo the
    control node has to be reached over SSH like any other host, but it is
    still the machine whose reboot kills this process, so it still goes last.
    """
    from sparkrun.utils import is_local_host

    remote: list[str] = []
    aliases: list[str] = []
    for h in host_list:
        if is_local_host(h):
            aliases.append(h)
        else:
            remote.append(h)
    return remote, aliases


def _run_on_hosts(
    hosts: list[str],
    script: str,
    *,
    password: str | None,
    ssh_kwargs: dict,
    timeout: int,
    dry_run: bool,
    max_workers: int,
):
    """Run *script* as root on every host concurrently, yielding results as they land.

    Yields ``(result, elapsed_seconds)`` in completion order, so a fast host
    reports without waiting on a slow one.  Each host is independent here —
    the caller imposes whatever ordering the steps need.
    """
    from sparkrun.orchestration.sudo import run_sudo_script_on_host

    def _one(host: str):
        started = time.monotonic()
        r = run_sudo_script_on_host(
            host,
            script,
            password,
            ssh_kwargs=ssh_kwargs,
            timeout=timeout,
            dry_run=dry_run,
        )
        return r, time.monotonic() - started

    if len(hosts) == 1:
        # Skip the pool entirely so a single host (notably the control node)
        # keeps a plain, synchronous call stack.
        yield _one(hosts[0])
        return

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [_operation_submit(pool, _one, h) for h in hosts]
        for fut in as_completed(futures):
            yield fut.result()


def _report(result, elapsed: float) -> bool:
    """Print one host's outcome; return True when it succeeded."""
    if result.success:
        click.echo("  %-24s OK      %6s   %s" % (result.host, _elapsed(elapsed), _summarize(result.stdout)))
        return True

    click.echo("  %-24s FAILED  %6s   rc=%d" % (result.host, _elapsed(elapsed), result.returncode))
    # Both streams, and more of them than the old 200-char stderr slice: apt
    # reports plenty of real failures (a held lock, an unreachable mirror) on
    # stdout, so truncating to stderr alone routinely showed nothing at all.
    for label, stream in (("stderr", result.stderr), ("stdout", result.stdout)):
        text = stream.strip()
        if not text:
            continue
        for line in text.splitlines()[-5:]:
            click.echo("    %s | %s" % (label, line[:300]), err=True)
    return False


def _run_update_steps(
    hosts: list[str],
    *,
    password: str | None,
    ssh_kwargs: dict,
    dry_run: bool,
    max_workers: int,
    label: str = "",
) -> set[str]:
    """Run every update step over *hosts*, returning the hosts that failed.

    Steps are barriered: all hosts finish step N before any starts step N+1.
    Hosts run *within* a step concurrently, which is where the time goes —
    these are identical machines doing identical work, so per-step variance is
    small and the barrier costs little.  What it buys is worth more than that:
    the cluster stays in lockstep, so a failure leaves every host at the same
    point, and a step that fails everywhere aborts the run before the next one
    starts rather than after a `dist-upgrade` has already landed.

    A host that fails one step is dropped from the rest.
    """
    failed: set[str] = set()
    lock = threading.Lock()

    for idx, (desc, cmd) in enumerate(_FE_UPDATE_STEPS, start=1):
        active = [h for h in hosts if h not in failed]
        if not active:
            click.echo()
            click.echo("All hosts failed — skipping remaining steps %d-%d." % (idx, len(_FE_UPDATE_STEPS)))
            break

        click.echo()
        click.echo("[%d/%d] %s%s" % (idx, len(_FE_UPDATE_STEPS), desc, label))
        click.echo("  $ %s" % cmd)
        click.echo("  dispatched to %d host(s): %s" % (len(active), ", ".join(active)))

        ok = 0
        step_t0 = time.monotonic()
        with _Heartbeat(active, lock, enabled=not dry_run) as beat:
            for result, elapsed in _run_on_hosts(
                active,
                cmd,
                password=password,
                ssh_kwargs=ssh_kwargs,
                timeout=_STEP_TIMEOUT_S,
                dry_run=dry_run,
                max_workers=max_workers,
            ):
                beat.done(result.host)
                with lock:
                    if _report(result, elapsed):
                        ok += 1
                    else:
                        failed.add(result.host)

        click.echo("  %d/%d OK in %s" % (ok, len(active), _elapsed(time.monotonic() - step_t0)))

    return failed


def _reboot_hosts(
    hosts: list[str],
    *,
    password: str | None,
    ssh_kwargs: dict,
    dry_run: bool,
    max_workers: int,
) -> None:
    """Fire the reboot on every host concurrently."""
    for result, _elapsed in _run_on_hosts(
        hosts,
        _REBOOT_SCRIPT,
        password=password,
        ssh_kwargs=ssh_kwargs,
        timeout=_REBOOT_TIMEOUT_S,
        dry_run=dry_run,
        max_workers=max_workers,
    ):
        click.echo("  %-24s %s" % (result.host, "rebooting" if result.success else "reboot FAILED"))


@setup.command("fe-system-update", hidden=True)
@host_options
@click.option("--user", default=None, help="SSH user (default: cluster user or $USER)")
@dry_run_option
@click.pass_context
def setup_fe_system_update(ctx, hosts, hosts_file, cluster_name, user, dry_run):
    """Run a full system update on DGX Spark Founders Edition hosts.

    Updates system packages (apt), firmware (fwupdmgr), and reboots.
    Can target the local machine, cluster hosts, or both.

    Hosts are updated in parallel, one step at a time.  If this machine is
    itself a cluster member it is updated and rebooted last, so its restart
    can't cut the run short for everything else.

    \b
    Steps performed (as root):
      1. apt update
      2. apt dist-upgrade
      3. fwupdmgr refresh
      4. fwupdmgr upgrade
      5. reboot
    """
    from ._sudo import ensure_sudo_password

    sctx = _get_context(ctx)
    config = sctx.config

    # --- Step 1: Determine target hosts ---
    # If explicit hosts/cluster provided, use those directly
    explicit_hosts = hosts or hosts_file or cluster_name
    if explicit_hosts:
        host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)
    else:
        # Interactive: ask local vs cluster
        click.echo("Where would you like to run the system update?")
        click.echo()
        click.echo("  1) Local machine only")

        # Try to list cluster hosts
        mgr = _get_cluster_manager(sctx=sctx)
        default_cluster = mgr.get_default()
        cluster_hosts = []
        if default_cluster:
            try:
                cdata = mgr.get(default_cluster)
                cluster_hosts = cdata.hosts
            except Exception:
                pass

        if cluster_hosts:
            click.echo("  2) Cluster hosts (%s): %s" % (default_cluster, ", ".join(cluster_hosts)))
            click.echo("  3) All (local + cluster hosts)")
            choice = click.prompt("Selection", type=click.IntRange(1, 3), default=2)
        else:
            click.echo("  (No default cluster configured — cluster option unavailable)")
            choice = click.prompt("Selection", type=click.IntRange(1, 1), default=1)

        click.echo()

        if choice == 1:
            host_list = [socket.gethostname()]
        elif choice == 2:
            host_list = list(cluster_hosts)
        else:
            # Duplicate labels for this machine are collapsed below, by
            # identity rather than by the string compare that used to let
            # `spark-01` and its own LAN IP both through.
            host_list = [socket.gethostname()] + list(cluster_hosts)

        import os

        if user is None:
            user = config.ssh_user or os.environ.get("USER", "root")
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        ssh_kwargs = build_ssh_kwargs(config)
        if user:
            ssh_kwargs["ssh_user"] = user

    # TODO: guard to detect non-founders edition hosts and block them

    remote_hosts, control_aliases = _partition_control_node(host_list)
    # Every alias is the same machine; the first one is how we address it.
    control_node = control_aliases[0] if control_aliases else None
    ordered_hosts = remote_hosts + ([control_node] if control_node else [])

    # --- Step 2: Confirm the activity ---
    click.echo("Founders Edition System Update")
    click.echo("=" * 40)
    click.echo("Target hosts: %s" % ", ".join(ordered_hosts))
    if control_node:
        extra = [a for a in control_aliases if a != control_node]
        click.echo(
            "  %s is this machine%s — updated and rebooted last."
            % (control_node, " (also listed as %s)" % ", ".join(extra) if extra else "")
        )
    click.echo()
    click.echo("The following will be executed as root:")
    for desc, cmd in _FE_UPDATE_STEPS:
        click.echo("  - %s  (%s)" % (desc, cmd))
    click.echo("  - Reboot")
    click.echo()

    if not dry_run:
        if not click.confirm("Proceed?", default=False):
            click.echo("Aborted.")
            return

    # --- Step 3: Get sudo access ---
    sudo_password, indirect_user = ensure_sudo_password(
        ordered_hosts,
        user,
        ssh_kwargs,
        dry_run=dry_run,
        allow_indirect=True,
        default_user=user,
    )
    sudo_ssh_kwargs = dict(ssh_kwargs)
    if indirect_user:
        sudo_ssh_kwargs["ssh_user"] = indirect_user

    from sparkrun.orchestration.ssh import resolve_parallel_cap

    max_workers = resolve_parallel_cap(len(remote_hosts) or 1, config.max_parallel_ssh)

    # --- Step 4: Update the remote hosts, then this machine ---
    run_t0 = time.monotonic()
    failed_hosts: set[str] = set()
    if remote_hosts:
        failed_hosts |= _run_update_steps(
            remote_hosts,
            password=sudo_password,
            ssh_kwargs=sudo_ssh_kwargs,
            dry_run=dry_run,
            max_workers=max_workers,
            # No host count here — it goes stale the moment one drops out,
            # and the dispatch line below already carries the live one.
            label=" — cluster hosts" if control_node else "",
        )

    if control_node:
        click.echo()
        click.echo("Cluster hosts done. Updating this machine (%s) last." % control_node)
        failed_hosts |= _run_update_steps(
            [control_node],
            password=sudo_password,
            ssh_kwargs=sudo_ssh_kwargs,
            dry_run=dry_run,
            max_workers=1,
            label=" — this machine",
        )

    # --- Step 5: Reboot ---
    reboot_remote = [h for h in remote_hosts if h not in failed_hosts]
    reboot_control = control_node if control_node and control_node not in failed_hosts else None
    if reboot_remote or reboot_control:
        total = len(reboot_remote) + (1 if reboot_control else 0)
        click.echo()
        click.echo("[Reboot]")
        if dry_run:
            click.echo("  [dry-run] Would reboot: %s" % ", ".join(reboot_remote + ([reboot_control] if reboot_control else [])))
        elif not click.confirm("Updates complete. Reboot %d host(s) now?" % total, default=True):
            click.echo("Skipping reboot. Remember to reboot manually for updates to take effect.")
        else:
            if reboot_remote:
                _reboot_hosts(
                    reboot_remote,
                    password=sudo_password,
                    ssh_kwargs=sudo_ssh_kwargs,
                    dry_run=dry_run,
                    max_workers=max_workers,
                )
            if reboot_control:
                # Dead last: this one takes the CLI down with it.
                click.echo("  rebooting this machine (%s) — this session ends here." % reboot_control)
                _reboot_hosts(
                    [reboot_control],
                    password=sudo_password,
                    ssh_kwargs=sudo_ssh_kwargs,
                    dry_run=dry_run,
                    max_workers=1,
                )
            click.echo()
            click.echo("Reboot initiated on %d host(s)." % total)

    # --- Summary ---
    click.echo()
    click.echo("Results (%s total)" % _elapsed(time.monotonic() - run_t0))
    click.echo("-" * 40)
    for h in ordered_hosts:
        note = " (this machine)" if h == control_node else ""
        click.echo("  %-24s %s%s" % (h, "FAILED" if h in failed_hosts else "updated", note))

    ok = len([h for h in ordered_hosts if h not in failed_hosts])
    fail = len(failed_hosts)
    click.echo()
    if fail:
        click.echo("%d updated, %d failed (%s)" % (ok, fail, ", ".join(sorted(failed_hosts))))
        sys.exit(1)
    else:
        click.echo("%d host(s) updated successfully." % ok)
