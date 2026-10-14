"""Remote systemd service artifacts for an inference workload (``export systemd``).

Pure script/unit renderers plus the two remote probes the export needs (locate
the head node's executable; bootstrap a fresh uv tool install).  Console-free:
callers decide what to print.  :mod:`sparkrun.api.systemd_export` composes
these into the deployment steps.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass

from sparkrun.core.application_profile import child_environment, get_application_profile, resource_name

_UNIT_TEMPLATE = textwrap.dedent("""\
    # sparkrun.distribution={owner}
    [Unit]
    Description={app_command} inference: {recipe_name} ({model})
    After=network-online.target docker.service
    Wants=network-online.target
    Requires=docker.service

    [Service]
    Type=simple
    User={ssh_user}
    Group={ssh_user}
    ExecStart={sparkrun_path} run {service_dir}/recipe.yaml \\
        --cluster {cluster_name} --foreground --no-follow
    Restart=on-failure
    RestartSec=30
    TimeoutStartSec=600
    TimeoutStopSec=120
    Environment=SPARKRUN_APPLICATION_PROFILE={profile_ref}
    Environment=HOME={user_home}
    Environment=PATH=/usr/local/bin:/usr/bin:/bin:{extra_path}
    WorkingDirectory={user_home}
    StandardOutput=journal
    StandardError=journal
    SyslogIdentifier={namespace}-{slug}

    [Install]
    WantedBy=multi-user.target
""")


def unit_path(slug: str) -> str:
    """Absolute path of the system unit installed for *slug*."""
    return "/etc/systemd/system/%s.service" % resource_name("-%s" % slug)


def service_dir(slug: str, user_home: str) -> str:
    return "%s/.config/%s/services/%s" % (user_home, get_application_profile().config_namespace, slug)


def render_systemd_unit(slug, recipe, cluster_name, ssh_user, sparkrun_path, user_home):
    """Render a systemd unit file from template values."""
    extra_path = os.path.dirname(sparkrun_path) if sparkrun_path else ""
    return _UNIT_TEMPLATE.format(
        app_command=get_application_profile().command,
        namespace=get_application_profile().resource_namespace,
        owner=get_application_profile().id,
        profile_ref=child_environment()["SPARKRUN_APPLICATION_PROFILE"],
        recipe_name=recipe.name,
        model=recipe.model,
        slug=slug,
        ssh_user=ssh_user,
        sparkrun_path=sparkrun_path,
        service_dir=service_dir(slug, user_home),
        cluster_name=cluster_name,
        user_home=user_home,
        extra_path=extra_path,
    )


def service_owner_guard(path, *, unit=False):
    """Reject foreign artifacts before installing or removing service files."""
    from sparkrun.utils.shell import quote

    legacy_check = 'grep -q "^Description=sparkrun inference:" "$_artifact"' if unit else "true"
    return """_artifact={path}
if [ -e "$_artifact" ]; then
    _owner=$(sed -n 's/^# sparkrun.distribution=//p; s/^distribution: *//p' "$_artifact" | sort -u)
    if [ "$_owner" != {owner} ]; then
        if [ -n "$_owner" ] || [ {owner} != sparkrun ] || ! {legacy}; then
            echo "Refusing to replace service artifact owned by another application: $_artifact" >&2
            exit 1
        fi
    fi
fi""".format(path=quote(path), owner=quote(get_application_profile().id), legacy=legacy_check)


def service_artifacts(slug, cluster_name, user_home):
    root = "%s/.config/%s" % (user_home, get_application_profile().config_namespace)
    return (
        "%s/services/%s/recipe.yaml" % (root, slug),
        "%s/services/%s/cluster.yaml" % (root, slug),
        "%s/clusters/%s.yaml" % (root, cluster_name),
    )


def service_guards(slug, cluster_name, user_home):
    return "\n".join(
        [
            service_owner_guard(unit_path(slug), unit=True),
            *(service_owner_guard(p) for p in service_artifacts(slug, cluster_name, user_home)),
        ]
    )


def service_lock(user_home, *, unit_path=None):
    """Lock the persistent home directory before the privileged unit lock.

    Both phases can lock this inode, and service removal cannot unlink it.
    Sharing it also serializes services referencing the same cluster file.
    """
    from sparkrun.utils.shell import quote

    script = "exec 9<%s\nflock -x 9" % quote(user_home)
    if unit_path is not None:
        script += "\nexec 8>%s\nflock -x 8" % quote(unit_path + ".lock")
    return script


def render_install_script(slug, recipe_yaml, cluster_yaml, cluster_name, user_home):
    """Render user-level install script (no sudo needed)."""
    clusters_dir = "%s/.config/%s/clusters" % (user_home, get_application_profile().config_namespace)
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail

        {lock}
        {guards}
        mkdir -p '{service_dir}' '{clusters_dir}'
        _recipe_tmp=$(mktemp '{service_dir}/.recipe.XXXXXX')
        _cluster_tmp=$(mktemp '{service_dir}/.cluster.XXXXXX')
        _definition_tmp=$(mktemp '{clusters_dir}/.cluster.XXXXXX')
        trap 'rm -f "$_recipe_tmp" "$_cluster_tmp" "$_definition_tmp"' EXIT

        # Write baked recipe
        cat > "$_recipe_tmp" << 'SPARKRUN_RECIPE_EOF'
        # sparkrun.distribution={owner}
        {recipe_yaml}
        SPARKRUN_RECIPE_EOF

        # Write cluster reference
        cat > "$_cluster_tmp" << 'SPARKRUN_CLUSTER_EOF'
        # sparkrun.distribution={owner}
        {cluster_yaml}
        SPARKRUN_CLUSTER_EOF

        # Write cluster definition for sparkrun run --cluster
        cp "$_cluster_tmp" "$_definition_tmp"
        mv -f "$_recipe_tmp" '{service_dir}/recipe.yaml'
        mv -f "$_cluster_tmp" '{service_dir}/cluster.yaml'
        mv -f "$_definition_tmp" '{clusters_dir}/{cluster_name}.yaml'

        echo "Service files installed to {service_dir}"
    """).format(
        lock=service_lock(user_home),
        guards=service_guards(slug, cluster_name, user_home),
        owner=get_application_profile().id,
        service_dir=service_dir(slug, user_home),
        clusters_dir=clusters_dir,
        cluster_name=cluster_name,
        recipe_yaml=recipe_yaml,
        cluster_yaml=cluster_yaml,
    )


def render_sudo_install_script(slug, unit_contents, *, user_home, cluster_name):
    """Render sudo install script (writes unit file, enables service)."""
    path = unit_path(slug)
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail

        {lock}
        {guards}
        {required}
        _unit_tmp=$(mktemp '{unit_path}.XXXXXX')
        trap 'rm -f "$_unit_tmp"' EXIT
        cat > "$_unit_tmp" << 'SPARKRUN_UNIT_EOF'
        {unit_contents}
        SPARKRUN_UNIT_EOF
        chmod 644 "$_unit_tmp"
        mv -f "$_unit_tmp" '{unit_path}'

        # Reload systemd and enable service
        systemctl daemon-reload
        systemctl enable '{namespace}-{slug}'

        echo "Service {namespace}-{slug} installed and enabled"
    """).format(
        namespace=get_application_profile().resource_namespace,
        unit_path=path,
        lock=service_lock(user_home, unit_path=path),
        guards=service_guards(slug, cluster_name, user_home),
        required="\n".join(
            "test -f '%s' || { echo 'Service files missing; retry installation' >&2; exit 1; }" % p
            for p in service_artifacts(slug, cluster_name, user_home)
        ),
        slug=slug,
        unit_contents=unit_contents,
    )


def render_uninstall_script(slug, cluster_name, user_home):
    """Render uninstall script (sudo: stop, disable, remove unit; user: remove service dir)."""
    cluster_file = "%s/.config/%s/clusters/%s.yaml" % (user_home, get_application_profile().config_namespace, cluster_name)
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail

        SERVICE_NAME='{namespace}-{slug}'
        UNIT_PATH='/etc/systemd/system/{namespace}-{slug}.service'

        {lock}
        {guards}

        # Stop and disable if active
        if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
            systemctl stop "$SERVICE_NAME"
            echo "Stopped $SERVICE_NAME"
        fi
        if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
            systemctl disable "$SERVICE_NAME"
            echo "Disabled $SERVICE_NAME"
        fi

        # Remove unit file
        if [ -f "$UNIT_PATH" ]; then
            rm -f "$UNIT_PATH"
            systemctl daemon-reload
            echo "Removed $UNIT_PATH"
        fi

        # Remove service directory and cluster definition
        rm -rf '{service_dir}'
        rm -f '{cluster_file}'
        echo "Removed service files for {slug}"
    """).format(
        namespace=get_application_profile().resource_namespace,
        lock=service_lock(user_home, unit_path=unit_path(slug)),
        guards=service_guards(slug, cluster_name, user_home),
        slug=slug,
        service_dir=service_dir(slug, user_home),
        cluster_file=cluster_file,
    )


def detect_remote_command(host, ssh_kwargs, dry_run=False):
    """Detect the application executable and user home on a remote host.

    Returns (executable_path, user_home) on success, or (None, None) if not found.
    Under *dry_run* returns placeholder paths without connecting.
    """
    from sparkrun.orchestration.ssh import run_remote_script

    script = textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
        SPARKRUN_PATH=$(which @APP_COMMAND@ 2>/dev/null || true)
        if [ -z "$SPARKRUN_PATH" ]; then
            echo "ERROR: sparkrun not found on PATH" >&2
            exit 1
        fi
        echo "$SPARKRUN_PATH"
        echo "$HOME"
    """).replace("@APP_COMMAND@", get_application_profile().command)

    if dry_run:
        return "/usr/local/bin/" + get_application_profile().command, "/home/user"

    result = run_remote_script(host, script, **ssh_kwargs, timeout=15)
    if not result.success:
        return None, None

    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        return None, None

    return lines[0].strip(), lines[1].strip()


@dataclass(frozen=True)
class RemoteBootstrapResult:
    """Outcome of :func:`bootstrap_remote_command`."""

    success: bool
    #: The uv requirement installed (or that would be, under dry run).
    requirement: str
    #: Remote stderr (diagnostics or the refusal reason); empty on dry run.
    output: str = ""
    dry_run: bool = False


def bootstrap_remote_command(host, ssh_kwargs, *, channel, dry_run=False) -> RemoteBootstrapResult:
    """Bootstrap a fresh remote uv tool installation of the application.

    Never repair/replace an existing environment here: a missing PATH shim does
    not mean its daemons have stopped. Replacements belong to setup install's
    destination-aware runtime guard.  Dry run executes nothing.
    """
    import re

    from sparkrun.core.channels import channel_requirement
    from sparkrun.orchestration.ssh import run_remote_script
    from sparkrun.utils.shell import quote

    requirement = channel_requirement(channel)
    package = re.sub(r"[-_.]+", "-", get_application_profile().package).lower()
    script = (
        textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
        UV_PATH=$(which uv 2>/dev/null || true)
        if [ -z "$UV_PATH" ]; then
            echo "ERROR: uv not found — cannot auto-install sparkrun" >&2
            echo "Install uv first: https://docs.astral.sh/uv/" >&2
            exit 1
        fi
        TOOL_DIR=$("$UV_PATH" tool dir)
        case "$TOOL_DIR" in
            /*) ;;
            *) echo "ERROR: Cannot locate the destination uv tools directory" >&2; exit 1 ;;
        esac
        if [ -e "$TOOL_DIR"/@PACKAGE@ ] || [ -L "$TOOL_DIR"/@PACKAGE@ ]; then
            echo "ERROR: Existing uv environment found; automatic replacement is disabled." >&2
            echo "Stop/drain its daemons and workers, then run the application's setup install command remotely." >&2
            exit 1
        fi
        "$UV_PATH" tool install @REQUIREMENT@
    """)
        .replace("@PACKAGE@", quote(package))
        .replace("@REQUIREMENT@", quote(requirement))
    )

    if dry_run:
        return RemoteBootstrapResult(success=True, requirement=requirement, dry_run=True)

    result = run_remote_script(host, script, **ssh_kwargs, timeout=120)
    return RemoteBootstrapResult(success=result.success, requirement=requirement, output=result.stderr.strip())
