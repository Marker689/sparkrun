"""Render the shared setup plan as a read-only readiness report."""

from __future__ import annotations
import logging
import sys
import click
from sparkrun.core.application_profile import render_identity_text
from sparkrun.core.setup_models import CheckContext as CheckContext, CheckItem, HostState as HostState, OK, WARN, FAIL, SKIP
from sparkrun.core.setup_steps import evaluate_host
from .._common import host_options, json_option

logger = logging.getLogger(__name__)
_STATUS_LABEL = {OK: "OK", WARN: "WARN", FAIL: "FAIL", SKIP: "SKIP"}
_STATUS_MARK = {OK: "[OK]  ", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[--]  "}


def _render_host(host: str, items: list[CheckItem]) -> None:
    click.echo(host)
    for item in items:
        line = "  %s %s" % (_STATUS_MARK[item.status], item.label)
        if item.detail:
            line += " — %s" % item.detail
        click.echo(line)
        if item.guidance and item.status in (WARN, FAIL):
            click.echo("         → %s" % item.guidance)
    click.echo()


def register(setup_group) -> None:
    """Attach the ``check`` command to the ``setup`` group.

    Called from ``cli/_setup/__init__.py`` to avoid an import cycle (the
    group is defined there).
    """

    @setup_group.command("check")
    @host_options
    @json_option(help="Emit the full results as JSON")
    @click.pass_context
    def setup_check(ctx, hosts, hosts_file, cluster_name, output_json):
        """Check cluster hosts against the setup steps, without changing anything.

        Probes each host for the things ``{app_command} setup wizard`` configures
        (Docker access, NVIDIA CDI, host IPC reaping, earlyoom, sudoers, SSH
        host identities, mesh, CX7) and reports gaps with the command that fixes each. Uses the default
        cluster unless ``--cluster``/``--hosts`` is given. Read-only.

        Examples:

          {app_command} setup check

          {app_command} setup check --cluster mylab

          {app_command} setup check --hosts 10.0.0.1,10.0.0.2 --json
        """
        from sparkrun.core.config import SparkrunConfig

        from .._common import _resolve_setup_context, print_json

        config = SparkrunConfig()
        host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user=None)

        from sparkrun.core.setup_probe import probe_setup_hosts
        from sparkrun.core.setup_steps import build_setup_plan
        from .._common import _get_cluster_manager

        manager = _get_cluster_manager()
        resolved_name = cluster_name or (manager.get_default() if not (hosts or hosts_file) else None)
        cluster = manager.get(resolved_name) if resolved_name else None
        states, check_ctx = probe_setup_hosts(host_list, ssh_kwargs=ssh_kwargs, config=config, cluster=cluster, cluster_name=resolved_name)
        if not output_json:
            click.echo("Setup check for %d host(s)" % len(host_list))
            click.echo("=" * 56)
            click.echo()
        results_by_host = {}
        unreachable = []
        json_hosts = {}

        # Render in a deterministic host order.
        for host in host_list:
            state = states[host]
            if not state.reachable:
                unreachable.append(host)
                if not output_json:
                    click.echo("%s: SSH connectivity — %s" % (host, state.error))
                json_hosts[host] = {"reachable": False, "checks": [], "error": state.error}
                continue

            items = evaluate_host(state, check_ctx)
            results_by_host[host] = items
            if not output_json:
                _render_host(host, items)
            json_hosts[host] = {
                "reachable": True,
                "steps": [
                    {
                        "key": p.step.key,
                        "selected": p.selected,
                        "platform": p.platform,
                        "reason": p.reason,
                        "blocked_by": list(p.blocked_by),
                    }
                    for p in build_setup_plan(state, check_ctx)
                ],
                "checks": [{"key": i.key, "label": i.label, "status": i.status, "detail": i.detail, "guidance": i.guidance} for i in items],
            }

        # Aggregate.
        fail_count = sum(1 for items in results_by_host.values() for i in items if i.status == FAIL)
        warn_count = sum(1 for items in results_by_host.values() for i in items if i.status == WARN)
        gaps = fail_count + warn_count + len(unreachable)

        if not output_json:
            click.echo("=" * 56)
            if gaps == 0:
                click.echo("All checks passed across %d host(s). No setup gaps found." % len(host_list))
            else:
                parts = []
                if unreachable:
                    parts.append("%d unreachable host(s)" % len(unreachable))
                if fail_count:
                    parts.append("%d critical gap(s)" % fail_count)
                if warn_count:
                    parts.append("%d advisory(s)" % warn_count)
                click.echo("Found %s." % ", ".join(parts))
                click.echo(
                    render_identity_text("Fix per the '→' guidance above, or run '{app_command} setup wizard%s'." % check_ctx.cluster_flag)
                )

        if output_json:
            print_json(
                {
                    "cluster": resolved_name,
                    "hosts": host_list,
                    "unreachable": unreachable,
                    "critical_gaps": fail_count,
                    "advisories": warn_count,
                    "results": json_hosts,
                }
            )

        # Critical gaps (or unreachable hosts) fail the command; advisories don't.
        if fail_count or unreachable:
            sys.exit(1)
