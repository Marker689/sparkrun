"""``sparkrun setup k8s`` — Kubernetes setup subcommands.

Thin CLI layer over :mod:`sparkrun.plugins.k8s.api`:

- ``setup k8s kubectl`` — resolve / download / list the ``kubectl`` binary.
- ``setup k8s info`` — probe the target cluster.
- ``setup k8s sa`` — configure the sparkrun service account + RBAC.

The api layer holds all logic; these commands only parse flags, call the
api, and render results to the TTY.
"""

from __future__ import annotations


def build_setup_command():
    """Construct the setup group only when the CLI attaches enabled plugins."""
    from sparkrun.core.application_profile import render_identity_text

    from contextlib import closing
    import functools
    import shlex

    import click

    from sparkrun.cli._common import _get_context

    def kube_options(func):
        """Shared ``--kubeconfig / --context / --namespace`` decorator."""

        @click.option("--kubeconfig", default=None, help="Path to a kubeconfig file (overrides config / $KUBECONFIG).")
        @click.option("--context", "kube_context", default=None, help="kubeconfig context to target.")
        @click.option("--namespace", "-n", default=None, help="Namespace to target.")
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def follow_logs(sctx, *, name, kind, namespace, kubeconfig, context):
        from . import api
        from sparkrun.api import SparkrunError, LogSourceChanged

        try:
            with closing(
                api.logs(sctx, name=name, kind=kind, namespace=namespace, kubeconfig=kubeconfig, context=context, follow=True)
            ) as lines:
                for line in lines:
                    click.echo("[%s] %s" % (line.label, line.text) if line.resource is not None else line.text)
        except KeyboardInterrupt:
            pass  # Stop the local reader; the submitted workload keeps running.
        except (SparkrunError, LogSourceChanged) as exc:
            raise click.ClickException(str(exc)) from exc

    SETUP_K8S_FEATURE = "cli.setup.k8s"

    def _setup_k8s_enabled_at_import() -> bool:
        """Best-effort flag resolution for help-visibility (hidden when off/unknown)."""
        try:
            from sparkrun.core.config import SparkrunConfig

            return SparkrunConfig().is_feature_enabled(SETUP_K8S_FEATURE)
        except Exception:  # noqa: BLE001 — never let a config read break CLI import
            return False

    @click.group("k8s", hidden=not _setup_k8s_enabled_at_import())
    @click.pass_context
    def setup_k8s(ctx):
        """Kubernetes setup: kubectl acquisition, cluster info, service account.

        {app_command} manages its own ``kubectl`` under ``~/.cache/{app_command}/kubectl/``
        and can configure a scoped service account for driving workloads. These
        commands are the foundation for the (experimental) k8s executor.

        Experimental — gated behind the ``cli.setup.k8s`` feature flag.
        """
        if not _get_context(ctx).config.is_feature_enabled(SETUP_K8S_FEATURE):
            raise click.ClickException(
                render_identity_text(
                    "The 'setup k8s' commands are experimental and disabled. Enable them with: {app_command} setup features enable cli.setup.k8s"
                )
            )

    # ---------------------------------------------------------------------------
    # kubectl
    # ---------------------------------------------------------------------------

    @setup_k8s.command("kubectl")
    @click.option("--version", default=None, help="Download / resolve a specific version (e.g. v1.31.0).")
    @click.option("--list", "list_", is_flag=True, help="List cached kubectl binaries and exit.")
    @click.option("--path", "show_path", is_flag=True, help="Print only the resolved binary path.")
    @click.option("--no-download", is_flag=True, help="Do not download; fail if nothing is cached / on PATH.")
    @click.pass_context
    def setup_k8s_kubectl(ctx, version, list_, show_path, no_download):
        """Resolve, download, or list the managed kubectl binary."""
        from . import api
        from sparkrun.plugins.k8s.orchestration import list_cached

        sctx = _get_context(ctx)

        if list_:
            cached = list_cached(sctx.config.cache_dir)
            if not cached:
                click.echo("No cached kubectl binaries.")
                return
            for binary in cached:
                click.echo("%-12s %-14s %s" % (binary.version, "%s-%s" % (binary.os_name, binary.arch), binary.path))
            return

        try:
            binary = api.ensure_kubectl(sctx, version=version, download=not no_download)
        except api.KubectlUnavailable as exc:
            raise click.ClickException(str(exc)) from exc

        if show_path:
            click.echo(str(binary.path))
            return
        click.echo("kubectl %s (%s) [%s]" % (binary.version or "unknown", "%s-%s" % (binary.os_name, binary.arch), binary.source))
        click.echo("  path: %s" % binary.path)

    # ---------------------------------------------------------------------------
    # info
    # ---------------------------------------------------------------------------

    @setup_k8s.command("info")
    @kube_options
    @click.option("--no-pin", is_flag=True, help="Do not pin the server version for this context.")
    @click.pass_context
    def setup_k8s_info(ctx, kubeconfig, kube_context, namespace, no_pin):
        """Probe the target cluster and show client / server versions."""
        from . import api

        sctx = _get_context(ctx)
        try:
            info = api.cluster_info(
                sctx,
                kubeconfig=kubeconfig,
                context=kube_context,
                namespace=namespace,
                pin=not no_pin,
            )
        except api.KubectlUnavailable as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo("Context:        %s" % (info.current_context or "(default)"))
        click.echo("Namespace:      %s" % (info.namespace or "(default)"))
        click.echo("Client version: %s" % (info.client_version or "unknown"))
        if info.reachable:
            click.echo("Server version: %s" % info.server_version)
            click.secho("Cluster reachable.", fg="green")
        else:
            click.echo("Server version: unreachable")
            click.secho("Cluster unreachable: %s" % (info.message or ""), fg="red")

    # ---------------------------------------------------------------------------
    # nodes
    # ---------------------------------------------------------------------------

    @setup_k8s.command("nodes")
    @kube_options
    @click.option("--selector", "-l", default=None, help="kubectl label selector to filter nodes.")
    @click.option("--gpu-only", is_flag=True, help="Show only nodes with detected accelerators.")
    @click.pass_context
    def setup_k8s_nodes(ctx, kubeconfig, kube_context, namespace, selector, gpu_only):
        """Show cluster nodes with GPU hardware detected from labels.

        Synthesizes {app_command} hardware metadata from GPU Feature Discovery /
        Node Feature Discovery labels — the k8s-native analog of an SSH probe,
        and the basis for hybrid-cluster scheduling.
        """
        from . import api
        from sparkrun.platforms import resolve_platform

        sctx = _get_context(ctx)
        try:
            nodes = api.list_nodes(sctx, kubeconfig=kubeconfig, context=kube_context, selector=selector, gpu_only=gpu_only)
        except (api.ClusterUnreachable, api.KubectlUnavailable) as exc:
            raise click.ClickException(str(exc)) from exc

        if not nodes:
            click.echo("No nodes found.")
            return

        for node in nodes:
            hw = node.hardware
            platform = resolve_platform(hw)
            cordon = "" if node.schedulable else "  [cordoned]"
            click.secho("%s%s" % (node.name, cordon), bold=True)
            if hw.accelerators:
                for accel in hw.accelerators:
                    mem = " %.0fGB" % accel.memory_gb if accel.memory_gb else ""
                    caps = ", ".join(sorted(accel.capabilities))
                    click.echo("  %d× %s%s  (%s)" % (accel.count, accel.model, mem, caps))
            else:
                click.echo("  (no accelerators detected)")
            click.echo("  allocatable: %d/%d nvidia.com/gpu" % (node.allocatable_gpus, node.capacity_gpus))
            click.echo("  platform:    %s" % (platform.display_name if platform else "unknown"))

    # ---------------------------------------------------------------------------
    # sa
    # ---------------------------------------------------------------------------

    @setup_k8s.command("sa")
    @click.argument("name", default="sparkrun")
    @kube_options
    @click.option("--no-create-namespace", is_flag=True, help="Do not create the namespace (assume it exists).")
    @click.option("--token-duration", default=None, help="Token lifetime for kubectl create token (e.g. 8760h).")
    @click.option("--no-kubeconfig", is_flag=True, help="Do not write a derived kubeconfig.")
    @click.option("--dry-run", is_flag=True, help="Print the manifests without applying.")
    @click.pass_context
    def setup_k8s_sa(ctx, name, kubeconfig, kube_context, namespace, no_create_namespace, token_duration, no_kubeconfig, dry_run):
        """Configure the {app_command} service account, RBAC, and a scoped kubeconfig.

        Creates a cluster-wide ClusterRole scoped to only the verbs {app_command}
        needs (pods, pods/log, pods/exec, batch/jobs, services, configmaps,
        secrets, nodes) — not cluster-admin.
        """
        from . import api

        sctx = _get_context(ctx)
        try:
            result = api.configure_service_account(
                sctx,
                name=name,
                namespace=namespace,
                kubeconfig=kubeconfig,
                context=kube_context,
                create_namespace=not no_create_namespace,
                token_duration=token_duration,
                write_kubeconfig=not no_kubeconfig,
                dry_run=dry_run,
            )
        except (api.ClusterUnreachable, api.ServiceAccountError, api.KubectlUnavailable) as exc:
            raise click.ClickException(str(exc)) from exc

        if dry_run:
            click.echo(result.manifests_yaml)
            click.secho("(dry-run — nothing applied)", fg="yellow")
            return

        click.secho("Service account configured.", fg="green")
        click.echo("  service account: %s/%s" % (result.namespace, result.name))
        click.echo("  cluster role:    %s" % result.cluster_role)
        click.echo("  binding:         %s" % result.binding)
        if result.server:
            click.echo("  server:          %s" % result.server)
        if result.kubeconfig_path:
            click.echo("  kubeconfig:      %s (0600)" % result.kubeconfig_path)
            click.echo("")
            click.echo("Point the k8s executor at it via config.yaml:")
            click.echo("  executor_config:")
            click.echo("    kubeconfig: %s" % result.kubeconfig_path)

    # ---------------------------------------------------------------------------
    # kueue
    # ---------------------------------------------------------------------------

    @setup_k8s.command("kueue")
    @kube_options
    @click.option("--install", is_flag=True, help="Install Kueue + JobSet if missing (applies pinned release manifests).")
    @click.option("--kueue-version", default=None, help="Kueue release to install (overrides k8s.kueue.version).")
    @click.option("--jobset-version", default=None, help="JobSet release to install (overrides k8s.jobset.version).")
    @click.option("--yes", "-y", is_flag=True, help="Skip the install confirmation prompt.")
    @click.option("--dry-run", is_flag=True, help="Render the provisioning manifests without installing/applying.")
    @click.pass_context
    def setup_k8s_kueue(ctx, kubeconfig, kube_context, namespace, install, kueue_version, jobset_version, yes, dry_run):
        """Install Kueue + JobSet (gang scheduling) and provision {app_command} queues.

        Derives one ResourceFlavor per detected GPU node-class, a ClusterQueue
        with per-flavor quota, and a LocalQueue. Runs under the admin context.
        Kueue is required for all k8s-mode launches.
        """
        from . import api

        sctx = _get_context(ctx)

        if install and not yes and not dry_run:
            status = api.kueue_status(sctx, kubeconfig=kubeconfig, context=kube_context)
            missing = [n for n, ok in (("JobSet", status.jobset_installed), ("Kueue", status.kueue_installed)) if not ok]
            if missing:
                click.confirm(
                    "Install %s into this cluster (applies upstream release manifests)?" % " + ".join(missing),
                    abort=True,
                )

        try:
            result = api.setup_kueue(
                sctx,
                install=install,
                kueue_version=kueue_version,
                jobset_version=jobset_version,
                namespace=namespace,
                kubeconfig=kubeconfig,
                context=kube_context,
                dry_run=dry_run,
            )
        except (api.KueueSetupError, api.ClusterUnreachable, api.KubectlUnavailable) as exc:
            raise click.ClickException(str(exc)) from exc

        if dry_run:
            click.echo(result.manifests_yaml)
            click.secho("(dry-run — nothing installed or applied)", fg="yellow")
            return

        if result.installed_jobset:
            click.secho("Installed JobSet %s." % result.jobset_version, fg="green")
        if result.installed_kueue:
            click.secho("Installed Kueue %s." % result.kueue_version, fg="green")
        click.secho("Kueue queues provisioned.", fg="green")
        click.echo("  namespace:     %s" % result.namespace)
        click.echo("  cluster queue: %s" % result.cluster_queue)
        click.echo("  local queue:   %s" % result.local_queue)
        for flavor in result.flavors:
            click.echo("  flavor:        %s  (%s, %d GPU quota)" % (flavor.name, flavor.model, flavor.gpu_quota))

    # ---------------------------------------------------------------------------
    # launch (hidden — smoke-tests the JobSet launch path)
    # ---------------------------------------------------------------------------

    @setup_k8s.command("launch", hidden=True)
    @click.option("--name", required=True, help="JobSet / cluster name.")
    @click.option("--image", required=True, help="Workload container image.")
    @click.option("--ranks", required=True, help="Comma-separated per-rank GPU models (e.g. gb10,gb10,rtx-pro-6000-blackwell).")
    @click.option("--serve", "serve_command", required=True, help="Serve command run in each pod.")
    @click.option("--gpus-per-pod", type=int, default=1, help="GPUs requested per pod.")
    @click.option("--transport", type=click.Choice(["tcp", "rdma"]), default="tcp", help="NCCL transport tier.")
    @kube_options
    @click.option("--no-precheck", is_flag=True, help="Submit even if the feasibility check fails.")
    @click.option("--follow", is_flag=True, help="Stream JobSet logs after submit.")
    @click.option("--dry-run", is_flag=True, help="Render the JobSet + feasibility without submitting.")
    @click.pass_context
    def setup_k8s_launch(
        ctx, name, image, ranks, serve_command, gpus_per_pod, transport, kubeconfig, kube_context, namespace, no_precheck, follow, dry_run
    ):
        """Submit a Kueue-admitted JobSet launch (foundation for k8s-mode run)."""
        from . import api

        sctx = _get_context(ctx)
        rank_models = [r.strip() for r in ranks.split(",") if r.strip()]
        try:
            result = api.launch_jobset(
                sctx,
                name=name,
                rank_models=rank_models,
                image=image,
                serve_command=serve_command,
                gpus_per_pod=gpus_per_pod,
                transport=transport,
                namespace=namespace,
                kubeconfig=kubeconfig,
                context=kube_context,
                precheck=not no_precheck,
                dry_run=dry_run,
            )
        except (api.JobSetLaunchError, api.ClusterUnreachable, api.KubectlUnavailable) as exc:
            raise click.ClickException(str(exc)) from exc

        if dry_run:
            click.echo(result.manifests_yaml)
            click.echo("")
            click.secho("Feasibility:", bold=True)
            click.echo(result.feasibility_summary)
            click.secho("(dry-run — nothing submitted)", fg="yellow")
            return
        click.secho("JobSet submitted.", fg="green")
        click.echo("  jobset: %s/%s" % (result.namespace, result.name))
        click.echo("Reattach with: kubectl -n %s logs -f -l jobset.sigs.k8s.io/jobset-name=%s" % (result.namespace, result.name))
        if follow:
            follow_logs(sctx, name=result.name, kind="jobset", namespace=result.namespace, kubeconfig=kubeconfig, context=kube_context)

    # ---------------------------------------------------------------------------
    # run-job (hidden — smoke-tests the launcher-Job transport)
    # ---------------------------------------------------------------------------

    @setup_k8s.command("run-job", hidden=True)
    @click.option("--name", required=True, help="Job name.")
    @click.option("--image", default=None, help="Launcher image (defaults to k8s.launcher_image).")
    @click.option("--command", "command", default=None, help="argv (shell-split) to run in the image.")
    @click.option("--script", "script", default=None, help="Bash script to run via a mounted ConfigMap.")
    @kube_options
    @click.option("--follow", is_flag=True, help="Stream launcher logs until interrupted (Job keeps running).")
    @click.option("--dry-run", is_flag=True, help="Print the manifests without applying.")
    @click.pass_context
    def setup_k8s_run_job(ctx, name, image, command, script, kubeconfig, kube_context, namespace, follow, dry_run):
        """Apply an in-cluster launcher Job (foundation for job-driven launch)."""
        from . import api

        sctx = _get_context(ctx)
        argv = shlex.split(command) if command else None
        try:
            result = api.run_launcher_job(
                sctx,
                name=name,
                image=image,
                command=argv,
                script=script,
                namespace=namespace,
                kubeconfig=kubeconfig,
                context=kube_context,
                dry_run=dry_run,
            )
        except api.LauncherJobError as exc:
            raise click.ClickException(str(exc)) from exc

        if dry_run:
            click.echo(result.manifests_yaml)
            click.secho("(dry-run — nothing applied)", fg="yellow")
            return
        click.secho("Launcher Job applied.", fg="green")
        click.echo("  job:   %s/%s" % (result.namespace, result.job_name))
        click.echo("  image: %s" % result.image)
        click.echo("Reattach with: kubectl -n %s logs -f job/%s" % (result.namespace, result.job_name))
        if follow:
            follow_logs(sctx, name=result.job_name, kind="job", namespace=result.namespace, kubeconfig=kubeconfig, context=kube_context)

    return setup_k8s
