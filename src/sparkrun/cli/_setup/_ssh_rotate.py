"""Confirmed SSH host identity rotation for selected cluster machines."""

from __future__ import annotations

import click

from .._common import _resolve_setup_context, dry_run_option, host_options
from . import setup


@setup.command("ssh-rotate-keys")
@host_options
@click.option("--user", "-u", default=None, help="SSH username (default: cluster/config user).")
@click.option("--machine-id", is_flag=True, help="Also regenerate /etc/machine-id and its D-Bus copy. Reboot each target afterward.")
@click.option("--yes", "-y", is_flag=True, help="Confirm rotation for all selected hosts without the confirmation prompt.")
@dry_run_option
@click.pass_context
def setup_ssh_rotate_keys(ctx, hosts, hosts_file, cluster_name, user, machine_id, yes, dry_run):
    """Rotate SSH server host keys, verify access, and refresh the SSH mesh.

    Uses existing verified SSH access and sudo on the selected Linux/systemd
    hosts. Stages new server keys before activation and validates a fresh SSH
    connection before discarding old keys. Updates known_hosts on the control
    machine and cluster peers, then refreshes passwordless SSH setup for the
    same targets. The control machine is rotated only if explicitly selected
    as a target. SSH login identities are preserved.

    Use --machine-id to fix duplicated OS identities too. Plan to reboot those
    hosts afterward; running services retain the old machine ID until reboot.
    This command does not reboot hosts. Other SSH clients must independently
    update their trust records. Host certificates/custom host-key paths need
    administrator-managed rotation.

    \b
    Examples:
      {app_command} setup ssh-rotate-keys --cluster mylab
      {app_command} setup ssh-rotate-keys --cluster mylab --machine-id
      {app_command} setup ssh-rotate-keys --hosts 10.0.0.1,10.0.0.2 --dry-run
    """
    from sparkrun.core.config import SparkrunConfig

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)
    selected_cluster = cluster_name
    if not selected_cluster and not (hosts or hosts_file):
        from .._common import _get_cluster_manager

        selected_cluster = _get_cluster_manager().get_default()
    _rotate_ssh_identities(
        host_list, user, ssh_kwargs, config, selected_cluster=selected_cluster, machine_id=machine_id, yes=yes, dry_run=dry_run
    )


def _rotate_ssh_identities(
    host_list, user, ssh_kwargs, config, *, selected_cluster=None, machine_id=False, yes=False, dry_run=False, record_phase=True
):
    """Shared confirmed rotation frontend; return whether rotation completed."""
    from sparkrun.api.setup import rotate_ssh_host_keys
    from ._ssh import _check_ssh_host_identities, _run_ssh_mesh_native
    from ._sudo import interactive_sudo_passwords, _record_setup_phase

    host_list = list(dict.fromkeys(host_list))
    if not host_list:
        raise click.UsageError("Select hosts with --hosts, --hosts-file, or --cluster.")
    click.echo("Rotate SSH server host keys on %d host(s) as '%s':" % (len(host_list), user))
    for host in host_list:
        click.echo("  %s" % host)
    if machine_id:
        click.echo("Also regenerate machine IDs. Reboot these hosts afterward to apply their new OS identities.")
    click.echo("Other SSH clients will need to trust the new host keys.")
    click.echo("SSH reconnects may fail briefly while cluster trust records are refreshed.")
    if dry_run:
        click.echo("[dry-run] Would stage keys, validate/reload SSH, verify reconnection, and refresh SSH setup for these hosts.")
        return False
    if not yes and not click.confirm("Rotate these host identities now?", default=False):
        click.echo("Rotation cancelled.")
        return False

    result = rotate_ssh_host_keys(
        host_list,
        ssh_kwargs=ssh_kwargs,
        passwords=interactive_sudo_passwords(user),
        machine_id=machine_id,
        recovery_dir=config.cache_dir / "ssh-rotations",
        progress=lambda host, message: click.echo("  %s: %s" % (host, message)),
    )
    for host in result.rotated:
        click.echo("  Rotated: %s" % host)
    if result.failures:
        for host, error in result.failures.items():
            click.echo("  %s: %s" % (host, error), err=True)
        if result.recovery_file:
            click.echo("Recovery details and authenticated public keys: %s" % result.recovery_file, err=True)
        raise click.ClickException("SSH identity rotation is incomplete; review the hosts above before retrying.")

    click.echo("Refreshing SSH setup for %s..." % ("cluster '%s'" % selected_cluster if selected_cluster else "the selected hosts"))
    mesh_options = ["-o", "StrictHostKeyChecking=yes", "-o", "ControlPath=none", *list(ssh_kwargs.get("ssh_options") or [])]
    if not _run_ssh_mesh_native(host_list, user, ssh_key=ssh_kwargs.get("ssh_key"), ssh_options=mesh_options):
        raise click.ClickException("Host keys rotated, but SSH mesh refresh failed. Re-run setup ssh for these hosts.")
    if selected_cluster and record_phase:
        _record_setup_phase(selected_cluster, user, host_list, "ssh_mesh", mesh_hosts=host_list)
    _check_ssh_host_identities(host_list, ssh_kwargs, cluster_name=selected_cluster)
    click.echo("SSH host-key rotation complete.")
    if machine_id:
        click.echo("Reboot each rotated host when convenient to apply the new machine ID to running services.")
    return True


def _offer_ssh_identity_rotation(
    states, context, user, ssh_kwargs, config, *, yes=False, dry_run=False, manifest_mgr=None, refresh_plan=None
):
    """Offer repair only for measured duplicate identities in the shared plan.

    A wizard's --yes accepts defaults; rotating identities defaults to no.
    Keep that mode non-interactive and leave the explicit rotation command as
    remediation. The standalone command has its own deliberate --yes opt-in.
    """
    from sparkrun.core.setup_steps import build_setup_plan
    from sparkrun.core.setup_models import WARN
    from ._check import _render_host

    if dry_run:
        return False
    selected = set()
    for host, state in states.items():
        entries = [entry for entry in build_setup_plan(state, context) if entry.selected and entry.step.key == "ssh_identity"]
        if entries:
            selected.add(host)
        findings = [item for entry in entries for item in entry.checks if item.status == WARN]
        if findings:
            _render_host(host, findings)
    report = context.ssh_identity_report
    if report is None or not report.duplicates or not report.complete:
        return False
    hosts = [identity.host for identity in report.identities]
    if not set(hosts).issubset(selected):
        return False
    if yes:
        click.echo(
            "Duplicate identities require explicit confirmation. Use the ssh-rotate-keys command above, or re-run the wizard without --yes."
        )
        return False
    click.echo("The duplicate host identities can be regenerated now, followed by an SSH mesh refresh.")
    rotated = _rotate_ssh_identities(
        hosts,
        user,
        ssh_kwargs,
        config,
        selected_cluster=context.cluster_name,
        machine_id=any(d.kind == "machine-id" for d in report.duplicates),
        record_phase=False,
    )
    if rotated and manifest_mgr is not None and context.cluster_name:
        # Reuse the wizard's held manifest lock, not another command's manager.
        manifest_mgr.record_phase(context.cluster_name, user, hosts, "ssh_mesh", mesh_hosts=hosts)
    if rotated and refresh_plan is not None:
        refresh_plan()
    return rotated
