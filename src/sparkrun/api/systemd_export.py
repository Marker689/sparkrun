"""Console-free ``export systemd``: run one inference workload as a systemd unit.

**Experimental** (may change in a minor release).

The generated unit runs ``<app> run <recipe.yaml> --cluster <slug>-systemd``
on the cluster's head node, so the deployment is three files there — the baked
recipe, a cluster definition, and the unit — plus a sudo step to install and
enable the unit.  The steps are separate calls so a caller can report progress
and collect the sudo password between them:

1. :func:`plan` — resolve the recipe source, cluster, SSH user and the
   generated recipe / cluster YAML (no SSH beyond transport preparation).
2. :func:`locate_host` — find the executable and ``$HOME`` on the head node;
   :func:`bootstrap_host` installs a fresh uv tool when it is absent.
3. :func:`render` — the unit file and the two install scripts.
4. :func:`install_files` → :func:`install_unit` → optionally :func:`start`.

:func:`render_uninstall` / :func:`uninstall` remove it again.

**No placement is baked in.**  The generated cluster carries the source
cluster's whole host list and its definition (user, executor, transport,
scheduler, …), so the service's ``run`` plans at every start exactly as
``run --cluster <source>`` would: with live occupancy and the cluster's own
scheduler.  Narrowing the hosts here would decide placement once at export
time (with no status) and again at every start inside that narrowed set —
the "decide once" trap ``api.plan`` exists to avoid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from sparkrun.api._errors import HostsUnreachable, JobNotFound, SparkrunError

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.systemd_service import RemoteBootstrapResult

# A unit name and a path component, interpolated into single-quoted shell.
_SERVICE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class SystemdExportError(SparkrunError):
    """A systemd export step failed; :attr:`detail` carries remote stderr."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


@dataclass(frozen=True)
class SystemdExportOptions:
    """Inputs for :func:`plan`.

    Exactly one of :attr:`recipe` / :attr:`job` names the workload.
    """

    #: Recipe name/path, or a loaded :class:`~sparkrun.core.recipe.Recipe`.
    recipe: "str | Recipe | None" = None
    #: Override mapping baked into the exported recipe (recipe source only).
    overrides: dict[str, Any] = field(default_factory=dict)
    #: A recorded workload's cluster_id: export the recipe, overrides and
    #: image it was launched with.
    job: str | None = None
    #: Source cluster name or definition (``None``: the default chain).
    cluster: "str | ClusterDefinition | None" = None
    #: Explicit hosts; replace the cluster's host list when both are given.
    hosts: tuple[str, ...] | None = None
    #: Service slug (default: the recipe's slug).
    service_name: str | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class SystemdExportPlan:
    recipe: "Recipe"
    overrides: dict[str, Any]
    #: The cluster the service is derived from.
    source_cluster: "ClusterDefinition"
    #: The definition installed on the head node as :attr:`cluster_name`.
    service_cluster: "ClusterDefinition"
    slug: str
    #: ``<namespace>-<slug>``, the systemd unit name.
    service: str
    unit_path: str
    cluster_name: str
    head_host: str
    ssh_user: str
    ssh_kwargs: dict[str, Any]
    recipe_yaml: str
    cluster_yaml: str
    #: Self-update channel a bootstrap install would use.
    channel: str
    dry_run: bool = False


@dataclass(frozen=True)
class SystemdHost:
    """Where the application lives on the head node."""

    executable: str
    user_home: str


@dataclass(frozen=True)
class SystemdExportArtifacts:
    host: SystemdHost
    unit_text: str
    #: User-level script: writes the recipe and cluster files.
    install_script: str
    #: Root script: writes the unit, reloads systemd, enables it.
    sudo_install_script: str


def plan(options: SystemdExportOptions, *, sctx: "SparkrunContext | None" = None) -> SystemdExportPlan:
    """Resolve what to deploy and where, without touching the head node.

    Raises:
        JobNotFound: :attr:`SystemdExportOptions.job` has no metadata.
        HostsUnreachable: No host source could be determined.
        SparkrunError: Invalid inputs, an unresolvable recipe or cluster, or an
            SSH user that cannot be determined.
    """
    import yaml

    from sparkrun.api._context import resolve_sctx
    from sparkrun.api._resolve import resolve_cluster, resolve_cluster_for_job, scope_operation
    from sparkrun.orchestration.systemd_service import unit_path
    from sparkrun.core.application_profile import resource_name
    from sparkrun.utils.env import resolve_ssh_user
    from sparkrun.utils.shell import validate_unix_username

    sctx = resolve_sctx(sctx)
    hosts = list(options.hosts) if options.hosts is not None else None

    if options.job is not None and options.recipe is None:
        recipe, overrides, meta = _recipe_from_job(options.job, sctx)
        if options.cluster is None and hosts is None and meta.get("hosts"):
            # The job's own hosts, reached as the cluster it recorded.
            cluster = resolve_cluster_for_job(None, list(meta["hosts"]), meta=meta, sctx=sctx)
        else:
            cluster = resolve_cluster(options.cluster, hosts, sctx=sctx)
    elif options.recipe is not None and options.job is None:
        from sparkrun.api._resolve import resolve_recipe

        recipe = resolve_recipe(options.recipe, sctx=sctx) if isinstance(options.recipe, str) else options.recipe
        overrides = dict(options.overrides)
        cluster = resolve_cluster(options.cluster, hosts, sctx=sctx)
    else:
        raise SparkrunError("Name exactly one of a recipe or a job to export")

    if not cluster.hosts:
        raise HostsUnreachable("No hosts provided, no default cluster, and no default hosts configured")

    slug = options.service_name or recipe.slug
    if not _SERVICE_NAME_RE.fullmatch(slug):
        raise SparkrunError("Invalid service name %r: use letters, digits, '.', '_' and '-'" % slug)

    scoped, ssh_kwargs = scope_operation(cluster, sctx=sctx, dry_run=options.dry_run)
    ssh_user = resolve_ssh_user(cluster.user, scoped.config, fallback="")
    if not ssh_user:
        raise SparkrunError("Cannot determine the SSH user; set the cluster's user or ssh.user in config.yaml")
    try:
        validate_unix_username(ssh_user)
    except ValueError as exc:
        raise SparkrunError("SSH user %r cannot be used as a systemd User=" % ssh_user) from exc
    ssh_kwargs["ssh_user"] = ssh_user

    cluster_name = "%s-systemd" % slug
    service_cluster = _service_cluster(cluster, cluster_name, ssh_user)
    return SystemdExportPlan(
        recipe=recipe,
        overrides=overrides,
        source_cluster=cluster,
        service_cluster=service_cluster,
        slug=slug,
        service=resource_name("-%s" % slug),
        unit_path=unit_path(slug),
        cluster_name=cluster_name,
        head_host=service_cluster.hosts[0],
        ssh_user=ssh_user,
        ssh_kwargs=ssh_kwargs,
        recipe_yaml=str(recipe.export(overrides=overrides)),
        cluster_yaml=yaml.safe_dump(_cluster_document(service_cluster), default_flow_style=False),
        channel=scoped.config.self_update_channel,
        dry_run=options.dry_run,
    )


def locate_host(plan: SystemdExportPlan) -> SystemdHost | None:
    """Find the executable and ``$HOME`` on the head node (placeholders on dry run)."""
    from sparkrun.orchestration.systemd_service import detect_remote_command

    executable, user_home = detect_remote_command(plan.head_host, plan.ssh_kwargs, dry_run=plan.dry_run)
    if executable is None or user_home is None:
        return None
    return SystemdHost(executable=executable, user_home=user_home)


def bootstrap_host(plan: SystemdExportPlan) -> "RemoteBootstrapResult":
    """Install a fresh uv tool on the head node; never replaces an existing one."""
    from sparkrun.orchestration.systemd_service import bootstrap_remote_command

    try:
        return bootstrap_remote_command(plan.head_host, plan.ssh_kwargs, channel=plan.channel, dry_run=plan.dry_run)
    except ValueError as exc:  # the profile has no source for this channel
        raise SystemdExportError(str(exc)) from exc


def render(plan: SystemdExportPlan, host: SystemdHost) -> SystemdExportArtifacts:
    """Render the unit file and both install scripts for *host*."""
    from sparkrun.orchestration.systemd_service import render_install_script, render_sudo_install_script, render_systemd_unit

    unit_text = render_systemd_unit(plan.slug, plan.recipe, plan.cluster_name, plan.ssh_user, host.executable, host.user_home)
    return SystemdExportArtifacts(
        host=host,
        unit_text=unit_text,
        install_script=render_install_script(plan.slug, plan.recipe_yaml, plan.cluster_yaml, plan.cluster_name, host.user_home),
        sudo_install_script=render_sudo_install_script(plan.slug, unit_text, user_home=host.user_home, cluster_name=plan.cluster_name),
    )


def install_files(plan: SystemdExportPlan, artifacts: SystemdExportArtifacts) -> str:
    """Write the recipe and cluster files on the head node; returns its stdout."""
    from sparkrun.orchestration.ssh import run_remote_script

    r = run_remote_script(plan.head_host, artifacts.install_script, **plan.ssh_kwargs, dry_run=plan.dry_run)
    if not r.success:
        raise SystemdExportError("Failed to install service files on %s." % plan.head_host, r.stderr.strip())
    return r.stdout.strip()


def install_unit(plan: SystemdExportPlan, artifacts: SystemdExportArtifacts, *, sudo_password: str | None) -> str:
    """Install and enable the unit (sudo); returns the remote stdout."""
    return _sudo(plan, artifacts.sudo_install_script, sudo_password, "Failed to install systemd unit on %s." % plan.head_host)


def start(plan: SystemdExportPlan, *, sudo_password: str | None) -> None:
    """Start the installed unit (sudo)."""
    _sudo(plan, "systemctl start '%s'" % plan.service, sudo_password, "Failed to start %s." % plan.service)


def render_uninstall(plan: SystemdExportPlan) -> str:
    """Render the uninstall script, reading ``$HOME`` on the head node.

    Dry run assumes ``/home/<ssh user>`` and does not connect.
    """
    from sparkrun.orchestration.ssh import run_remote_command
    from sparkrun.orchestration.systemd_service import render_uninstall_script

    if plan.dry_run:
        user_home = "/home/%s" % plan.ssh_user
    else:
        r = run_remote_command(plan.head_host, "echo $HOME", **plan.ssh_kwargs, timeout=10)
        if not r.success:
            raise SystemdExportError("Could not detect home directory on %s." % plan.head_host, r.stderr.strip())
        user_home = r.stdout.strip()
    return render_uninstall_script(plan.slug, plan.cluster_name, user_home)


def uninstall(plan: SystemdExportPlan, script: str, *, sudo_password: str | None) -> str:
    """Stop, disable and remove the unit and its files (sudo); returns stdout."""
    return _sudo(plan, script, sudo_password, "Failed to uninstall %s on %s." % (plan.service, plan.head_host))


# ---------------------------------------------------------------------------


def _sudo(plan: SystemdExportPlan, script: str, sudo_password: str | None, failure: str) -> str:
    from sparkrun.orchestration.sudo import run_sudo_script_on_host

    r = run_sudo_script_on_host(plan.head_host, script, sudo_password, ssh_kwargs=plan.ssh_kwargs, dry_run=plan.dry_run)
    if not r.success:
        raise SystemdExportError(failure, r.stderr.strip())
    return r.stdout.strip()


def _recipe_from_job(cluster_id: str, sctx: "SparkrunContext") -> "tuple[Recipe, dict[str, Any], dict[str, Any]]":
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import load_job_metadata

    meta = load_job_metadata(cluster_id, cache_dir=str(sctx.config.cache_dir))
    if not meta:
        raise JobNotFound("No job metadata found for '%s'." % cluster_id)

    recipe_state = meta.get("recipe_state")
    if recipe_state:
        recipe = Recipe._deserialize(recipe_state)
    else:
        recipe_name = meta.get("recipe")
        if not recipe_name:
            raise SparkrunError("Job metadata has no recipe name or state.")
        from sparkrun.api._resolve import resolve_recipe

        recipe = resolve_recipe(recipe_name, sctx=sctx)

    if meta.get("effective_container_image"):
        recipe.container = meta["effective_container_image"]
    return recipe, dict(meta.get("overrides") or {}), meta


def _service_cluster(source: "ClusterDefinition", name: str, ssh_user: str) -> "ClusterDefinition":
    """The source cluster, renamed and made self-contained for the head node.

    ``env_file`` is a control-machine path the head node does not have, so the
    cluster env is resolved here and the file reference dropped; ``sync_source``
    is import provenance, which a copy is not.
    """
    from sparkrun.core.cluster_manager import ClusterError

    try:
        env = source.resolve_env()
    except ClusterError as exc:
        raise SparkrunError(str(exc)) from exc
    return replace(source, name=name, description="", user=ssh_user, env=env, env_file=None, sync_source=None)


def _cluster_document(cluster: "ClusterDefinition") -> dict[str, Any]:
    data = cluster.to_dict()
    if not data.get("description"):
        data.pop("description", None)
    if cluster.distribution is not None and not cluster.distribution.is_default():
        data["distribution"] = cluster.distribution.to_dict()
    return data


__all__ = [
    "SystemdExportArtifacts",
    "SystemdExportError",
    "SystemdExportOptions",
    "SystemdExportPlan",
    "SystemdHost",
    "bootstrap_host",
    "install_files",
    "install_unit",
    "locate_host",
    "plan",
    "render",
    "render_uninstall",
    "start",
    "uninstall",
]
