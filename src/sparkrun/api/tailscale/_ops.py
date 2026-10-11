"""Implementation for :mod:`sparkrun.api.tailscale` — console-free.

Drives the orchestration layer (REST key minting + SSH fan-out) and returns
dataclasses. Never prints; the CLI renders. Host resolution is the caller's
job — these functions take an already-resolved ``host_list`` + ``ssh_kwargs``
(the setup-command convention), not a cluster selector.
"""

from __future__ import annotations

from sparkrun.core.coordination import operation_boundary

import logging
from dataclasses import dataclass

from sparkrun.core.context import SparkrunContext
from sparkrun.orchestration import tailscale as ts

from ._errors import (
    TailscaleAuthFailed,
    TailscaleExposeError,
    TailscaleNotConfigured,
    TailscaleSetupError,
)

logger = logging.getLogger(__name__)

# Default inference serve port when exposing a head node without an explicit
# ``--port`` (vLLM / SGLang / llama.cpp default to 8000).
DEFAULT_ENDPOINT_PORT = 8000


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostJoinState:
    """Per-host outcome of a ``join``."""

    host: str
    ok: bool
    install: str | None
    ip: str | None
    message: str | None = None


@dataclass(frozen=True)
class JoinResult:
    tag: str
    ephemeral: bool
    hosts: tuple[HostJoinState, ...]
    dry_run: bool = False

    @property
    def ok_count(self) -> int:
        return sum(1 for h in self.hosts if h.ok)


@dataclass(frozen=True)
class HostTailscaleStatus:
    host: str
    state: str
    ip: str | None
    hostname: str | None

    @property
    def joined(self) -> bool:
        return bool(self.ip) and self.state.lower() == "running"


@dataclass(frozen=True)
class StatusResult:
    hosts: tuple[HostTailscaleStatus, ...]


@dataclass(frozen=True)
class ExposeResult:
    target: str
    endpoint: str | None
    port: int
    url: str | None
    warnings: tuple[str, ...] = ()
    proxy_host_updated: bool = False


@dataclass(frozen=True)
class HostDownState:
    host: str
    state: str
    removed: bool = False


@dataclass(frozen=True)
class DownResult:
    hosts: tuple[HostDownState, ...]
    removed_devices: tuple[str, ...] = ()
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_settings(sctx: SparkrunContext, *, tag: str | None, ephemeral: bool | None) -> ts.TailscaleSettings:
    """Load Tailscale settings, applying per-invocation overrides.

    Translates the orchestration-level not-configured error into the api one.
    """
    try:
        settings = ts.load_settings(sctx.config)
    except ts.TailscaleNotConfigured as e:
        raise TailscaleNotConfigured(str(e)) from e
    except ts.TailscaleError as e:  # bad config/env tag or non-https base_url
        raise TailscaleSetupError(str(e)) from e
    if tag is not None:
        try:
            tag = ts.validate_tag(tag)
        except ts.TailscaleError as e:
            raise TailscaleSetupError(str(e)) from e
        settings = ts.TailscaleSettings(
            client_id=settings.client_id,
            client_secret=settings.client_secret,
            base_url=settings.base_url,
            tag=tag,
            tailnet=settings.tailnet,
            ephemeral=settings.ephemeral if ephemeral is None else ephemeral,
        )
    elif ephemeral is not None:
        settings = ts.TailscaleSettings(
            client_id=settings.client_id,
            client_secret=settings.client_secret,
            base_url=settings.base_url,
            tag=settings.tag,
            tailnet=settings.tailnet,
            ephemeral=ephemeral,
        )
    return settings


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------


@operation_boundary(mutation=True)
def join(
    sctx: SparkrunContext,
    host_list: list[str],
    ssh_kwargs: dict,
    *,
    tag: str | None = None,
    ephemeral: bool | None = None,
    enable_ssh: bool = False,
    hostname: str | None = None,
    sudo_password: str | None = None,
    dry_run: bool = False,
) -> JoinResult:
    """Install Tailscale + join *host_list* to the tailnet.

    Mints one short-lived, pre-authorized, tagged auth key for the whole batch,
    then fans the join script out over SSH with a sudo fallback. *hostname* sets
    the tailnet device name via ``tailscale up --hostname``; when it targets more
    than one host, Tailscale de-duplicates by appending ``-1``, ``-2``, ….
    """
    settings = _resolve_settings(sctx, tag=tag, ephemeral=ephemeral)

    if dry_run:
        dry_run_hosts = tuple(HostJoinState(host=h, ok=False, install=None, ip=None, message="dry-run") for h in host_list)
        return JoinResult(tag=settings.tag, ephemeral=settings.ephemeral, hosts=dry_run_hosts, dry_run=True)

    try:
        token = ts.fetch_access_token(settings)
        authkey = ts.mint_auth_key(settings, token)
    except ts.TailscaleAuthError as e:
        raise TailscaleAuthFailed(str(e)) from e
    except ts.TailscaleError as e:
        raise TailscaleSetupError(str(e)) from e

    primary, fallback = ts.build_join_scripts(authkey, settings.tag, hostname=hostname, enable_ssh=enable_ssh)

    from sparkrun.orchestration.sudo import run_with_sudo_fallback

    result_map, _still_failed = run_with_sudo_fallback(
        host_list,
        primary,
        fallback,
        ssh_kwargs,
        sudo_password=sudo_password,
    )

    hosts: list[HostJoinState] = []
    for h in host_list:
        r = result_map.get(h)
        if r is None:
            hosts.append(HostJoinState(host=h, ok=False, install=None, ip=None, message="no result"))
            continue
        markers = ts.parse_join_result(r.stdout or "")
        ok = bool(r.success) and markers.get("TS_OK") == "1"
        message = None
        if not ok:
            # Surface the tailscale CLI's own error (merged into stdout via 2>&1)
            # alongside the TS_ERROR marker, so failures are actionable rather
            # than a bare "up_failed".
            detail = markers.get("TS_ERROR") or ""
            noise = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip() and not ln.strip().startswith("TS_")]
            if noise:
                detail = "%s: %s" % (detail, noise[-1]) if detail else noise[-1]
            message = detail or (r.stderr or "").strip()[:200] or "join failed"
        hosts.append(
            HostJoinState(
                host=h,
                ok=ok,
                install=markers.get("TS_INSTALL"),
                ip=markers.get("TS_IP") or None,
                message=message,
            )
        )
    return JoinResult(tag=settings.tag, ephemeral=settings.ephemeral, hosts=tuple(hosts))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def status(sctx: SparkrunContext, host_list: list[str], ssh_kwargs: dict) -> StatusResult:
    """Probe each host's local Tailscale state (read-only, no sudo)."""
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    results = run_remote_scripts_parallel(host_list, ts.STATUS_SCRIPT, timeout=15, quiet=True, **ssh_kwargs)
    by_host = {r.host: r for r in results}
    hosts: list[HostTailscaleStatus] = []
    for h in host_list:
        r = by_host.get(h)
        if r is None or not r.success:
            hosts.append(HostTailscaleStatus(host=h, state="unreachable", ip=None, hostname=None))
            continue
        markers = ts.parse_join_result(r.stdout or "")
        hosts.append(
            HostTailscaleStatus(
                host=h,
                state=markers.get("TS_STATE") or "unknown",
                ip=markers.get("TS_IP") or None,
                hostname=markers.get("TS_HOSTNAME") or None,
            )
        )
    return StatusResult(hosts=tuple(hosts))


# ---------------------------------------------------------------------------
# expose
# ---------------------------------------------------------------------------


@operation_boundary(mutation=True)
def expose(
    sctx: SparkrunContext,
    *,
    proxy: bool = False,
    head_host: str | None = None,
    ssh_kwargs: dict | None = None,
    port: int | None = None,
    set_proxy_host: bool = False,
    sudo_password: str | None = None,
) -> ExposeResult:
    """Publish the inference endpoint on the tailnet.

    Exactly one of *proxy* / *head_host* must be given.

    - ``proxy``: local compute — report the control machine's tailnet URL for the
      sparkrun proxy (the control host runs kernel-mode Tailscale, so the raw
      port is reachable directly).
    - ``head_host``: an **action** — configure a ``tailscale serve --tcp`` forward
      on the head host (the inbound mechanism that works in userspace-networking
      mode) and report ``http://<tailnet-ip>:<port>/v1``.
    """
    if proxy == bool(head_host):
        raise TailscaleExposeError("expose requires exactly one of proxy=True or head_host=<host>.")

    if proxy:
        return _expose_proxy(sctx, port=port, set_proxy_host=set_proxy_host)
    assert head_host is not None  # exactly one target was required above
    return _expose_head(sctx, head_host, ssh_kwargs or {}, port=port, sudo_password=sudo_password)


def _expose_proxy(sctx: SparkrunContext, *, port: int | None, set_proxy_host: bool) -> ExposeResult:
    ip = ts.local_tailscale_ipv4()
    if not ip:
        raise TailscaleExposeError(
            "This machine is not on a tailnet (no tailscale IPv4). Join it first "
            "(`sparkrun setup tailscale join --hosts <this-host>` or `tailscale up`)."
        )
    dns = ts.local_tailscale_dnsname()
    proxy_cfg = sctx.proxy_config
    eff_port = port or proxy_cfg.port
    endpoint = dns or ip

    warnings: list[str] = []
    proxy_host_updated = False
    bind_host = proxy_cfg.host
    if set_proxy_host:
        proxy_cfg.set_proxy(host=ip)
        proxy_cfg.save()
        proxy_host_updated = True
    else:
        if bind_host in ("127.0.0.1", "localhost", "::1"):
            warnings.append(
                "Proxy is bound to %s and will NOT be reachable over the tailnet. "
                "Re-run with --set-proxy-host (binds the proxy to %s) or set proxy.host." % (bind_host, ip)
            )
        elif bind_host == "0.0.0.0":
            warnings.append("Proxy binds 0.0.0.0 — reachable on the tailnet but also on the LAN / public NIC unless firewalled.")

    url = "http://%s:%d/v1" % (endpoint, eff_port)
    return ExposeResult(
        target="proxy",
        endpoint=endpoint,
        port=eff_port,
        url=url,
        warnings=tuple(warnings),
        proxy_host_updated=proxy_host_updated,
    )


def _expose_head(sctx: SparkrunContext, head_host: str, ssh_kwargs: dict, *, port: int | None, sudo_password: str | None) -> ExposeResult:
    st = status(sctx, [head_host], ssh_kwargs)
    hs = st.hosts[0] if st.hosts else None
    if hs is None or hs.state == "unreachable":
        raise TailscaleExposeError("Could not reach %s over SSH to read its tailnet state." % head_host)
    if not hs.ip:
        raise TailscaleExposeError(
            "%s is not on a tailnet (state=%s). Join it first with `sparkrun setup tailscale join`." % (head_host, hs.state)
        )
    eff_port = port or DEFAULT_ENDPOINT_PORT

    # Configure a TCP serve forward on the host. Required for inbound in
    # userspace-networking mode (containers like Thunder); harmless elsewhere.
    primary, fallback = ts.build_serve_scripts(eff_port)

    from sparkrun.orchestration.sudo import run_with_sudo_fallback

    result_map, _still_failed = run_with_sudo_fallback([head_host], primary, fallback, ssh_kwargs, sudo_password=sudo_password)
    r = result_map.get(head_host)
    markers = ts.parse_join_result(r.stdout or "") if r else {}
    if not (r and r.success and markers.get("TS_SERVE_OK") == "1"):
        detail = markers.get("TS_ERROR") or ""
        noise = [ln.strip() for ln in ((r.stdout if r else "") or "").splitlines() if ln.strip() and not ln.strip().startswith("TS_")]
        if noise:
            detail = "%s: %s" % (detail, noise[-1]) if detail else noise[-1]
        raise TailscaleExposeError("`tailscale serve` failed on %s: %s" % (head_host, detail or "unknown error"))

    endpoint = markers.get("TS_IP") or hs.ip  # the stable tailnet IP
    warnings = [
        "Plain HTTP over the tailnet (TCP serve). The endpoint is reachable by anyone on the tailnet "
        "(subject to ACLs) — scope an ACL grant to tag:<port> and consider an --api-key on the server.",
    ]
    if hs.hostname:
        warnings.append("MagicDNS form (if enabled): http://%s:%d/v1" % (hs.hostname, eff_port))
    url = "http://%s:%d/v1" % (endpoint, eff_port)
    return ExposeResult(target=head_host, endpoint=endpoint, port=eff_port, url=url, warnings=tuple(warnings))


# ---------------------------------------------------------------------------
# down
# ---------------------------------------------------------------------------


@operation_boundary(mutation=True)
def down(
    sctx: SparkrunContext,
    host_list: list[str],
    ssh_kwargs: dict,
    *,
    remove: bool = False,
    sudo_password: str | None = None,
    dry_run: bool = False,
) -> DownResult:
    """Log hosts out of the tailnet; with *remove*, also delete their devices."""
    if dry_run:
        dry_run_hosts = tuple(HostDownState(host=h, state="dry-run") for h in host_list)
        return DownResult(hosts=dry_run_hosts, dry_run=True)

    # For --remove, resolve the joined hostnames AND the OAuth token/devices
    # *before* logging anyone out: a logged-out node no longer reports its
    # identity, and we must fail fast on missing creds rather than log hosts out
    # and then abort with nothing removed.
    pre_status = None
    remove_ctx = None
    if remove:
        pre_status = status(sctx, host_list, ssh_kwargs)
        remove_ctx = _remove_preflight(sctx)

    from sparkrun.orchestration.sudo import run_with_sudo_fallback

    result_map, _still_failed = run_with_sudo_fallback(
        host_list,
        ts.LOGOUT_SCRIPT,
        ts.LOGOUT_FALLBACK_SCRIPT,
        ssh_kwargs,
        sudo_password=sudo_password,
    )

    removed_ids: list[str] = []
    removed_hosts: set[str] = set()
    if remove:
        removed_ids, removed_hosts = _apply_removals(remove_ctx, host_list, pre_status)

    hosts: list[HostDownState] = []
    for h in host_list:
        r = result_map.get(h)
        markers = ts.parse_join_result(r.stdout or "") if r else {}
        state = markers.get("TS_DOWN") or ("failed" if not (r and r.success) else "unknown")
        hosts.append(HostDownState(host=h, state=state, removed=h in removed_hosts))
    return DownResult(hosts=tuple(hosts), removed_devices=tuple(removed_ids))


def _remove_preflight(sctx: SparkrunContext):
    """Resolve creds + token + device list for ``down --remove`` (fail-fast).

    Returns ``(settings, token, devices)``. Raised BEFORE any logout so a
    missing OAuth client can't leave hosts logged out with nothing removed.
    """
    settings = _resolve_settings(sctx, tag=None, ephemeral=None)
    try:
        token = ts.fetch_access_token(settings)
        devices = ts.list_devices(settings, token)
    except ts.TailscaleAuthError as e:
        raise TailscaleAuthFailed(str(e)) from e
    except ts.TailscaleError as e:
        raise TailscaleSetupError(str(e)) from e
    return settings, token, devices


def _apply_removals(remove_ctx, host_list: list[str], pre_status: StatusResult | None):
    """Delete tailnet devices matching the joined hosts. Returns (ids, hosts)."""
    settings, token, devices = remove_ctx

    # Build a lookup from each host to the tailnet identity it reported.
    reported = {hs.host: (hs.hostname, hs.ip) for hs in (pre_status.hosts if pre_status else ())}

    removed_ids: list[str] = []
    removed_hosts: set[str] = set()
    for h in host_list:
        rep_host, rep_ip = reported.get(h, (None, None))
        for dev in devices:
            if (rep_host and dev.hostname == rep_host) or (rep_ip and rep_ip in dev.addresses):
                try:
                    ts.delete_device(settings, token, dev.id)
                    removed_ids.append(dev.id)
                    removed_hosts.add(h)
                except ts.TailscaleError as e:  # noqa: PERF203 - best-effort per device
                    logger.debug("device delete failed for %s: %s", dev.id, e)
    return removed_ids, removed_hosts
