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

**The baked recipe is trusted on the head node** (it is a local file there),
so installing an untrusted recipe's hooks or isolation overrides needs
:attr:`SystemdExportOptions.trust`, the same authorization ``run`` asks for.
:func:`install_files` enforces it (:attr:`SystemdExportPlan.trust_required`);
previewing and uninstalling never need it.

**Pinning is opt-in** (:attr:`SystemdExportOptions.pin`): a mutable image tag
or model revision becomes a digest / commit once, at export, so the service
serves the same build after every restart.  References already immutable are
kept.  Unlike ``export recipe --realize`` nothing else is decided here: an
offline pin requires every candidate host to agree, because any of them may
run the service.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from sparkrun.api._errors import HostsUnreachable, JobNotFound, SparkrunError

if TYPE_CHECKING:
    from sparkrun.api._pin import Pin
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
    #: Executor keys (``entrypoint``, ``shm_size``, …) go to its
    #: ``executor_config``, the rest to ``defaults``.
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
    #: Authorize an untrusted recipe's hooks and isolation overrides.
    trust: bool = False
    #: Pin a mutable container tag and model revision in the baked recipe.
    pin: bool = False
    #: With :attr:`pin`: read pins off the hosts instead of the registry and
    #: the Hub.  ``None`` defers to the cluster's ``offline:`` setting.
    offline: bool | None = None
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
    #: What :attr:`SystemdExportOptions.pin` changed (empty without it).
    pins: "tuple[Pin, ...]" = ()
    #: Why installing needs :attr:`SystemdExportOptions.trust`; ``None`` when it does not.
    trust_required: str | None = None


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
    from sparkrun.core.resolve import split_executor_overrides
    from sparkrun.utils.env import resolve_ssh_user
    from sparkrun.utils.shell import validate_unix_username
    from sparkrun.utils.yaml_helpers import LiteralBlockDumper

    if options.offline is not None and not options.pin:
        raise SparkrunError("offline chooses where pins come from; it only applies with pin")

    sctx = resolve_sctx(sctx)
    hosts = list(options.hosts) if options.hosts is not None else None
    executor_overrides: dict[str, Any] = {}

    if options.job is not None and options.recipe is None:
        recipe, overrides, meta = _recipe_from_job(options.job, sctx)
        executor_overrides = dict(meta.get("executor_overrides") or {})
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

    # ``-o entrypoint=''`` configures the container, not the serve command.
    overrides, lifted = split_executor_overrides(overrides)
    executor_overrides.update(lifted)
    trust_required = _untrusted_reason(recipe, options.trust, sctx)

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

    data = recipe.to_dict(overrides=overrides or None, executor_overrides=executor_overrides or None)
    pins: tuple[Pin, ...] = ()
    if options.pin:
        pins = _pin_service_recipe(data, recipe, cluster, ssh_kwargs, offline=options.offline, sctx=scoped)
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
        recipe_yaml=yaml.dump(data, Dumper=LiteralBlockDumper, indent=2, sort_keys=False, default_flow_style=False),
        cluster_yaml=yaml.safe_dump(_cluster_document(service_cluster), default_flow_style=False),
        channel=scoped.config.self_update_channel,
        dry_run=options.dry_run,
        pins=pins,
        trust_required=trust_required,
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
    """Write the recipe and cluster files on the head node; returns its stdout.

    Raises:
        TrustRejected: :attr:`SystemdExportPlan.trust_required` is set.
    """
    from sparkrun.api._errors import TrustRejected
    from sparkrun.orchestration.ssh import run_remote_script

    if plan.trust_required:
        raise TrustRejected(plan.trust_required)

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


def _untrusted_reason(recipe: "Recipe", trust: bool, sctx: "SparkrunContext") -> str | None:
    """Why baking *recipe* into the head node's local (trusted) recipe file needs trust, or ``None``.

    On the head node the service recipe is a local path, which ``run`` trusts
    without asking, so installing is where the authorization has to happen.
    """
    from sparkrun.core.launcher import recipe_hook_commands, recipe_isolation_overrides, resolve_recipe_trust

    hooks = recipe_hook_commands(recipe)
    isolation = recipe_isolation_overrides(recipe)
    if not (hooks or isolation) or resolve_recipe_trust(recipe, trust, sctx=sctx):
        return None
    what = ["hooks (%s)" % ", ".join(hooks)] if hooks else []
    if isolation:
        what.append("container isolation overrides")
    return (
        "Recipe '%s' is not from a trusted source and sets %s. The service runs it as a local recipe on the head "
        "node, where it would be trusted without asking. Audit the recipe, then install with trust (--trust)."
        % (getattr(recipe, "qualified_name", None) or recipe.name, " and ".join(what))
    )


def _pin_service_recipe(
    data: dict[str, Any], recipe: "Recipe", cluster: "ClusterDefinition", ssh_kwargs: dict[str, Any], *, offline: bool | None, sctx
) -> "tuple[Pin, ...]":
    """Pin *data* against every candidate host: the service may start on any of them."""
    from sparkrun.api._pin import PinScope, pin_recipe_data
    from sparkrun.core.bootstrap import get_runtime
    from sparkrun.core.cluster_manager import ResolvedClusterConfig
    from sparkrun.core.offline import resolve_offline

    default_image = None
    if not data.get("container") and not data.get("containers"):
        # Written in only when every host agrees: the service is not placed yet.
        runtime = get_runtime(recipe.runtime, sctx.variables)
        try:
            images = {runtime.resolve_container(recipe, host_hardware=cluster.hardware_for(host)) for host in cluster.hosts}
        except ValueError as exc:  # unprobed hosts under a require-metadata application profile
            raise SparkrunError("Cannot pin the platform default image: %s" % exc) from exc
        images.discard(None)
        if len(images) > 1:
            raise SparkrunError(
                "Cannot pin: the recipe names no container and the cluster's hosts default to different images (%s). "
                "Name the container in the recipe or with --image." % ", ".join(sorted(images))
            )
        default_image = next(iter(images), None)
    local_cache_dir, remote_cache_dir, _mode, _iface = ResolvedClusterConfig.from_definition(cluster).resolve_transfer_config(sctx.config)
    scope = PinScope(
        recipe=recipe,
        hosts=tuple(cluster.hosts),
        ssh_kwargs=dict(ssh_kwargs),
        default_image=default_image,
        cache_dir=remote_cache_dir,
        local_cache_dir=local_cache_dir,
    )
    return pin_recipe_data(data, scope, offline=resolve_offline(offline, cluster).offline, sctx=sctx)


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
