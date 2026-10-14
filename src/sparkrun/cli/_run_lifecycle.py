"""CLI ownership of readiness and log attachment after detached submission."""

from __future__ import annotations

import click

from sparkrun.core.launcher import LaunchResult, ReadinessWatcher
from sparkrun.orchestration.logs import LogFollower
from sparkrun.orchestration.primitives import build_ssh_kwargs


def wait_for_run(
    result: LaunchResult, *, show_logs, keep_following, on_ready, on_failure, trust=False, progress=None, run_hooks=True
) -> int:
    """Wait for readiness independently of logs; retain detached workloads on exit."""
    follower = None
    watcher = ReadinessWatcher(result, ssh_kwargs=build_ssh_kwargs(result.config), timeline=result.timeline)
    try:
        if show_logs:
            sources = result.runtime.log_sources(result.cluster_id, result.host_list, is_solo=result.is_solo)
            follower = LogFollower(
                result.runtime.executor,
                sources,
                on_line=lambda line: click.echo(line.text),
                ssh_kwargs=build_ssh_kwargs(result.config),
            ).start()
        click.echo("Waiting for server readiness...")
        watcher.start()
        reported_log_end = False
        while not watcher.done.wait(0.1):
            if follower is not None and follower.done.is_set() and not reported_log_end:
                detail = str(follower.error) if follower.error else "stream ended"
                click.echo("Warning: log following stopped (%s); still waiting for readiness." % detail, err=True)
                reported_log_end = True
        if watcher.error is not None:
            click.echo("Error: readiness could not be confirmed: %s" % watcher.error, err=True)
            return 1
        readiness = watcher.readiness
        if readiness is None:
            click.echo("Error: readiness observer ended without a result.", err=True)
            return 1
        if not readiness.ready:
            if readiness.reason == "cancelled":
                return 130
            on_failure(readiness)
            return 1
        if follower is not None and follower.done.is_set() and follower.error is not None and not reported_log_end:
            click.echo("Warning: log following stopped (%s)." % follower.error, err=True)
        on_ready(readiness)
        if run_hooks and (result.recipe.post_exec or result.recipe.post_commands):
            # Share the authoritative result: hooks must not send another probe.
            from sparkrun.core.launcher import post_launch_lifecycle

            try:
                post_launch_lifecycle(
                    result,
                    remote_cache_dir=result.effective_cache_dir,
                    trust=trust,
                    progress=progress,
                    readiness=readiness,
                )
            except SystemExit as error:
                # Includes successful stop_after_post. Finalize readers/timings.
                return int(error.code or 0)
        elif progress:
            progress.phase_skip(6)
        if keep_following and follower is not None:
            click.echo("Continuing log following; Ctrl-C detaches without stopping the workload.")
            while not follower.done.wait(0.1):
                pass
            detail = str(follower.error) if follower.error else "stream ended"
            click.echo("Error: log following stopped (%s); current workload health is unconfirmed." % detail, err=True)
            return 1
        return 0
    except (KeyboardInterrupt, click.Abort):
        click.echo("\nStopped waiting/following; detached workload was left running.", err=True)
        return 130
    except Exception as error:
        click.echo("Error while monitoring startup: %s" % error, err=True)
        return 1
    finally:
        # Stop log output before returning to the timing/diagnostic renderer.
        if follower is not None:
            follower.stop()
        watcher.stop()


def existing_launch(match, *, recipe, overrides, cluster, sctx) -> LaunchResult:
    """Recover original placement/connection for --ensure, never survivor order."""
    from sparkrun.api._resolve import resolve_cluster_for_job, resolve_runtime, scope_operation
    from sparkrun.core._executor_destination import metadata_executor_overrides
    from sparkrun.orchestration.executor import resolve_executor
    from sparkrun.orchestration.job_metadata import load_job_metadata

    meta = load_job_metadata(match.workload_id, sctx=sctx)
    if not meta or not meta.get("hosts"):
        raise click.ClickException("Cannot recover the existing job's head/placement; use --no-ready-wait to check presence only.")
    hosts = list(meta["hosts"])
    if set(hosts) - set(match.hosts):
        raise click.ClickException(
            "Existing job was not observed on all recorded hosts; readiness is unconfirmed. Inspect its status/logs."
        )
    cluster = resolve_cluster_for_job(cluster, hosts, meta=meta, sctx=sctx)
    scoped, _ = scope_operation(cluster, sctx=sctx)
    runtime = resolve_runtime(recipe, sctx=scoped)
    runtime.executor = resolve_executor(
        recipe=recipe,
        runtime=runtime,
        cluster=cluster,
        config=scoped.config,
        cli_overrides=metadata_executor_overrides(meta),
        v=scoped.variables,
        rootless=False,
        auto_user=False,
    )
    if not runtime.executor.supports_host_endpoint:
        raise click.UsageError("This executor has no supported readiness endpoint; use --no-ready-wait.")
    effective = {**overrides}
    for key in ("port", "served_model_name"):
        if key in meta:
            effective[key] = meta[key]
    return LaunchResult(
        rc=0,
        cluster_id=match.workload_id,
        host_list=hosts,
        is_solo=len(hosts) == 1,
        runtime=runtime,
        recipe=recipe,
        overrides=effective,
        container_image=recipe.container or "",
        effective_cache_dir=str(scoped.config.cache_dir),
        serve_port=int(effective.get("port") or recipe.defaults.get("port") or 8000),
        config=scoped.config,
        timeline=sctx.timing,
    )
