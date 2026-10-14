"""sparkrun run command — thin Click wrapper around :func:`sparkrun.api.run`.

The CLI handles presentation concerns (banner, VRAM display, diagnostic
bundle destination, pre-launch summary, post-launch echoing) and delegates the
actual launch orchestration to :func:`sparkrun.api.run`.  All
``--option`` flags map onto :class:`sparkrun.api.RunOptions` fields.
"""

from __future__ import annotations

from sparkrun.cli._diagnostics import with_run_diagnostics

from sparkrun.core.application_profile import render_identity_text

import dataclasses
import logging
import sys
from typing import Any

import click

import sparkrun.api as api
from sparkrun.core.validation import FAIL_ON_CHOICES, validate_for_launch
from sparkrun.models.hub import disable_hub_metadata
from sparkrun.orchestration.transfer import TransferError
from sparkrun.runtimes.compatibility import IncompatibleHardwareError

from ._common import (
    RECIPE_NAME,
    _apply_recipe_overrides,
    _expand_recipe_shortcut,
    _get_context,
    _is_recipe_url,
    _enter_offline_mode,
    _load_recipe,
    _simplify_recipe_ref,
    authorize_recipe_hooks,
    dry_run_option,
    host_options,
    _render_capacity_diagnostics,
    recipe_override_options,
    report_launch_validation,
    resolve_cluster_config,
    with_host_context,
    HIDE_ADVANCED_OPTIONS,
)

from ._run_presentation import describe_host_source, echo_run_identity, planning_output, render_run_plan

logger = logging.getLogger(__name__)


def _timing_tree_depth(ctx: click.Context) -> int | None:
    """Choose the ordinary ``run`` timing detail from global verbosity."""
    from sparkrun.utils.cli_formatters import timing_tree_depth_for_verbosity

    verbose_count = (ctx.find_root().obj or {}).get("verbose", 0)
    return timing_tree_depth_for_verbosity(verbose_count)


def _echo_endpoint_ready(readiness) -> None:
    """Announce a now-serving endpoint while the log attachment is active.

    Called while ``docker logs -f`` is writing to the same terminal, so it
    must be **one short line in one write**: a multi-line block would be
    interleaved with log output mid-render.  The full breakdown waits for
    the finalize step, once the stream has stopped.

    Goes to stderr so a caller piping the log stream keeps it uncontaminated.
    """
    from sparkrun.utils.cli_formatters import startup_readiness_durations
    from sparkrun.utils.text import format_duration

    observation = getattr(readiness, "startup_observation", None)
    if observation:
        # Same projection the end-of-launch block renders from, so the two
        # reports of the same launch cannot print different numbers.
        durations = startup_readiness_durations(observation)
        click.secho(
            render_identity_text(
                "\n[{app_command}] %s ready at http://%s:%d/v1; container-start TTR port-open %s, HTTP-ready %s, TTFT %s (%s, rank 0)\n"
                % (
                    "Endpoint" if observation.get("inference_requested") is False else "Inference",
                    readiness.head_ip,
                    readiness.port,
                    durations["port_open"],
                    durations["http_ready"],
                    durations["ttft"],
                    observation["measurement"],
                )
            ),
            fg="green",
            err=True,
        )
        return

    click.secho(
        render_identity_text(
            "\n[{app_command}] Endpoint ready at http://%s:%d/v1 after %s (engine init %s, model load %s)\n"
            % (
                readiness.head_ip,
                readiness.port,
                format_duration(readiness.total_wait_s),
                format_duration(readiness.port_wait_s),
                format_duration(readiness.health_wait_s),
            )
        ),
        fg="green",
        err=True,
    )


def _report_readiness_outcome(readiness) -> None:
    """Describe failure without confusing an expired wait with confirmed death."""
    if readiness is None or readiness.ready or readiness.reason == "cancelled":
        return
    if readiness.reason == "port":
        detail = "the head container stopped or port %d never opened" % readiness.port
    elif readiness.reason == "inference":
        detail = "the startup probe did not verify a first non-empty inference token"
    else:
        detail = "%s never returned HTTP 200" % readiness.health_url
    click.secho(
        render_identity_text("[{app_command}] ERROR: endpoint did not become ready — %s." % detail),
        fg="red",
        err=True,
    )


@click.command()
@click.argument("recipe_name", type=RECIPE_NAME)
@host_options
@recipe_override_options
@click.option(
    "--container-name",
    "cluster_id_override",
    default=None,
    hidden=HIDE_ADVANCED_OPTIONS,
    help="Override deterministic cluster ID (static container name)",
)
@click.option("--solo", is_flag=True, help="Force single-node mode", hidden=True)
@click.option("--port", type=int, default=None, help="Override serve port")
@click.option("--served-model-name", default=None, help="Override served model name")
@click.option("--ray-port", type=int, default=46379, help="Ray GCS port (vllm-ray)", hidden=HIDE_ADVANCED_OPTIONS)
@click.option("--init-port", type=int, default=25000, help="vllm/SGLang distributed init port", hidden=HIDE_ADVANCED_OPTIONS)
@click.option(
    "--dashboard/--no-dashboard",
    "dashboard",
    default=None,
    help="Enable/disable the Ray dashboard on the head node (Ray runtimes only; binds 0.0.0.0 when on). "
    "Overrides the recipe's runtime_config.dashboard; defaults to on.",
    hidden=HIDE_ADVANCED_OPTIONS,
)
@click.option("--dashboard-port", type=int, default=8265, help="Ray dashboard port", hidden=HIDE_ADVANCED_OPTIONS)
@dry_run_option
@click.option("--foreground", is_flag=True, help="Run in foreground (don't detach)")
@click.option("--ensure", is_flag=True, default=False, help="Reuse an existing job when present; still wait for readiness")
@click.option("--no-follow", is_flag=True, help="Suppress model logs; still wait for readiness")
@click.option("--follow", is_flag=True, help="Keep following model logs after readiness (Ctrl-C detaches)")
@click.option("--no-ready-wait", is_flag=True, help="Return after launch without waiting for readiness or following logs")
@click.option(
    "--no-auto-detect",
    is_flag=True,
    help="Skip HuggingFace Hub metadata lookups (VRAM estimate falls back to recipe metadata)",
)
@click.option("--no-sync-tuning", is_flag=True, help="Skip syncing tuning configs from registries")
@click.option("--no-rm", is_flag=True, help="Don't auto-remove containers on exit (keeps containers after stop)")
@click.option("--memory-limit", "memory", default=None, help="Container memory limit (e.g. 32G)")
@click.option("--rootful", is_flag=True, help="Run with --privileged as root inside container (legacy behavior)")
@click.option(
    "--restart",
    "restart_policy",
    default=None,
    help="Docker restart policy (no, always, unless-stopped, on-failure[:N])",
    hidden=HIDE_ADVANCED_OPTIONS,
)
@click.option(
    "--transfer-mode",
    default=None,
    type=click.Choice(["auto", "local", "push", "delegated", "pull"], case_sensitive=False),
    help=(
        "Resource transfer mode (overrides cluster setting). 'pull' has every node fetch from "
        "origin itself in parallel — needs registry/HF credentials on each node, costs N x egress, "
        "and does not fall back to the control machine (use 'auto' for that). Note a re-pushed "
        "image tag is not re-pulled unless you also pass --rebuild."
    ),
    hidden=True,
)
@click.option(
    "--collect-diagnostics",
    "diagnostics_path",
    default=None,
    type=click.Path(),
    hidden=HIDE_ADVANCED_OPTIONS,
    help="Collect diagnostics, including DEBUG logs, to an NDJSON file without increasing console verbosity",
)
@click.option(
    "--timings/--no-timings",
    "show_timings",
    default=True,
    hidden=HIDE_ADVANCED_OPTIONS,
    help="Print a per-stage timing breakdown when the run finishes (on by default; --no-timings suppresses it)",
)
@click.option(
    "--trust",
    is_flag=True,
    default=False,
    help="Run the recipe's hooks (pre_exec, mods, post_exec, post_commands) and host-path overrides without "
    "confirmation, even when the recipe comes from an untrusted registry or URL",
)
@click.option(
    "--runtime-cache/--no-runtime-cache",
    "runtime_cache",
    default=None,
    hidden=HIDE_ADVANCED_OPTIONS,
    help=(
        "Persist compilation/autotune caches (torch.compile, Triton, FlashInfer, TRT-LLM "
        "autotuner) on the target hosts across launches. Defaults to the recipe/cluster/config setting."
    ),
)
@click.option(
    "--scheduler",
    "scheduler_name",
    default=None,
    help="Registered scheduler name (e.g. 'greedy', 'occupancy-sparse', 'occupancy-dense'). Defaults to the recipe's scheduler field, then 'greedy'.",
    hidden=HIDE_ADVANCED_OPTIONS,
)
@click.option(
    "--offline/--online",
    "offline",
    default=None,
    help="Launch from images, models and recipes that already exist, with no internet access (copies inside the "
    "cluster are allowed). --online overrides a cluster set to offline.",
)
@click.option(
    "--rebuild/--no-rebuild",
    "rebuild",
    default=None,
    help="Force a fresh image: an unconditional 'docker pull' for registry images (the default docker-pull path), "
    "a from-scratch rebuild for eugr. Use when a local copy is stale or incomplete. "
    "Overrides the recipe's builder_config.rebuild setting.",
    hidden=HIDE_ADVANCED_OPTIONS,
)
@click.option(
    "--env",
    "-e",
    "env_overrides",
    multiple=True,
    hidden=HIDE_ADVANCED_OPTIONS,
    help="Set a container environment variable: -e KEY=VALUE (repeatable). Value is used verbatim, unlike -o env.KEY=VALUE.",
)
@click.option("--label", "labels_override", multiple=True, help="Set meta data on a container (e.g., --label com.example.key=value)")
@click.option(
    "--executor-args",
    multiple=True,
    hidden=HIDE_ADVANCED_OPTIONS,
    help="Arguments passed directly to the container executor (e.g. docker run)",
)
@click.option(
    "--fail-on",
    type=click.Choice(FAIL_ON_CHOICES),
    default=None,
    hidden=True,
    help="Least-severe recipe-validation finding that refuses the launch (default: error)",
)
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
@with_host_context
@with_run_diagnostics
def run(
    ctx,
    recipe_name,
    hosts,
    hosts_file,
    cluster_name,
    cluster_id_override,
    solo,
    port,
    tensor_parallel,
    pipeline_parallel,
    data_parallel,
    gpu_mem,
    served_model_name,
    max_model_len,
    image,
    ray_port,
    init_port,
    dashboard,
    dashboard_port,
    dry_run,
    ensure,
    foreground,
    no_follow,
    follow,
    no_ready_wait,
    no_auto_detect,
    no_sync_tuning,
    no_rm,
    memory,
    rootful,
    restart_policy,
    transfer_mode,
    diagnostics_path,
    show_timings,
    trust,
    runtime_cache,
    scheduler_name,
    rebuild,
    offline,
    env_overrides,
    labels_override,
    options,
    executor_args,
    fail_on,
    extra_args,
    *,
    host_list: list[str],
    cluster_mgr=None,
):
    """Run an inference recipe.

    RECIPE_NAME can be a recipe file path or a name to search for.

    Examples:

      {app_command} run glm-4.7-flash-awq --solo

      {app_command} run glm-4.7-flash-awq --hosts 192.168.11.13,192.168.11.14

      {app_command} run glm-4.7-flash-awq --cluster mylab

      {app_command} run my-recipe.yaml --port 9000 --gpu-mem 0.8

      {app_command} run my-recipe.yaml -o attention_backend=triton -o max_model_len=4096

      {app_command} run my-recipe.yaml -e VLLM_USE_V1=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    """
    from sparkrun.core.bootstrap import get_runtime

    if follow and (no_follow or no_ready_wait):
        raise click.UsageError("--follow cannot be combined with --no-follow or --no-ready-wait")
    if foreground and no_ready_wait:
        raise click.UsageError("--foreground cannot be combined with --no-ready-wait")

    sctx = _get_context(ctx)
    v = sctx.variables
    config = sctx.config

    # Set before anything can reach the Hub.  Process-wide rather than threaded
    # down as a parameter: ``estimate_vram`` is called from host resolution, the
    # banner, the scheduling pass inside ``api.run`` and telemetry, and a flag
    # that reached three of those four would still hang on the fourth.
    if no_auto_detect:
        disable_hub_metadata()

    # warn that --solo flag is not recommended if solo==True at this point
    if solo:
        click.echo("Notice: --solo flag is not recommended; it is better to explicitly specify parallelism via e.g. --tp 1", err=True)

    # Resolve the named cluster definition when one is in play.  Carries
    # per-host hardware metadata so downstream code can compute placement,
    # fit, and per-host backend selection. Explicit hosts can restrict a
    # named cluster without discarding its identity or settings.
    cluster_def = None
    if cluster_mgr is not None and (cluster_name or (not hosts and not hosts_file)):
        _name = cluster_name or cluster_mgr.get_default()
        if _name:
            try:
                cluster_def = cluster_mgr.get(_name)
            except Exception:
                cluster_def = None

    # Find and load recipe (defer resolution until overrides are built).
    # Retry after a registry refresh when the recipe isn't found, so that
    # copy-pasted recipe names from newly-published sources just work.
    # Offline is decided in api.plan, but the recipe loads before planning, so
    # the same resolver runs here once to keep that step local too.
    offline_early = _enter_offline_mode(offline, cluster_def)
    recipe, _recipe_path, registry_mgr = _load_recipe(config, recipe_name, resolve=False, retry_after_update=True, offline=offline_early)

    # If recipe was loaded from a URL, simplify for display
    _resolved_name = _expand_recipe_shortcut(recipe_name)
    recipe_ref = _simplify_recipe_ref(_resolved_name) if _is_recipe_url(_resolved_name) else None

    # Build overrides and resolve runtime (overrides may influence resolution)
    recipe, overrides = _apply_recipe_overrides(
        options,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        data_parallel=data_parallel,
        gpu_mem=gpu_mem,
        max_model_len=max_model_len,
        image=image,
        recipe=recipe,
        # custom overrides
        port=port,
        served_model_name=served_model_name,
    )

    # -e/--env lands in the same place as -o env.KEY=VALUE (recipe.env, the top
    # container-env tier) but keeps the value verbatim. Applied after, so it
    # wins if a key is given both ways.
    if env_overrides:
        from sparkrun.core.resolve import apply_env_overrides

        try:
            apply_env_overrides(recipe, env_overrides)
        except ValueError as e:
            raise click.UsageError(str(e)) from e

    # --rebuild/--no-rebuild is a builder-agnostic override carried in
    # builder_config so any builder (present or future) can honor it. Only
    # override the recipe's own builder_config.rebuild when the flag was given
    # explicitly (tri-state default None leaves the recipe value untouched).
    if rebuild is not None:
        recipe.builder_config["rebuild"] = rebuild

    # Get assigned runtime
    try:
        runtime = get_runtime(recipe.runtime, v)
    except ValueError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Validate recipe (after resolve so runtime is populated).  Errors abort
    # before any side effect: they name something sparkrun cannot honor (an
    # unresolvable builder or executor, a runtime that rejects the recipe), and
    # launching anyway produces a deployment that isn't the one described.
    # Warnings print and continue — each is a supported escape hatch.
    #
    # The unmapped-key half is left off here: ``launch_inference`` runs it
    # after the platform default tier and with ``-o`` already split into serve
    # vs executor keys, so running it now would both duplicate the output and
    # report executor overrides (``-o entrypoint=''``) as having no effect.
    issues, validation_failed = validate_for_launch(
        recipe,
        fail_on=fail_on,
        runtime=runtime,
        cluster=cluster_def,
        config=config,
        v=v,
        include_unmapped_keys=False,
        # Echo back the reference the user typed: it is what the `recipe
        # validate` hint has to be re-typable as, and for a URL recipe it beats
        # the raw URL. Threaded into validation as well as into the report,
        # because the collapsed deprecation line names the same command.
        recipe_ref=recipe_name,
    )
    report_launch_validation(recipe_name, issues, validation_failed)
    if validation_failed:
        sys.exit(1)

    host_source = describe_host_source(
        hosts=hosts, hosts_file=hosts_file, cluster_name=cluster_name, cluster_mgr=cluster_mgr, config=config
    )

    # Resolve cache dir, transfer mode, and transfer interface from cluster
    # config.  Independent of placement, so it can precede the plan.
    cluster_cfg = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr)
    local_cache_dir, remote_cache_dir, effective_transfer_mode, effective_transfer_interface = cluster_cfg.resolve_transfer_config(
        config, transfer_mode_override=transfer_mode
    )

    # Build executor config from CLI flags.  Executor keys given with -o
    # (``-o entrypoint=''``) are lifted out of ``overrides`` by RunOptions and
    # layered on top of these, so -o still wins over the flag.
    cli_executor_opts: dict[str, Any] = {}
    if no_rm:
        cli_executor_opts["auto_remove"] = False
    if memory:
        cli_executor_opts["memory_limit"] = memory
    if restart_policy:
        cli_executor_opts["restart_policy"] = restart_policy
    if labels_override:
        cli_executor_opts["labels"] = list(labels_override)

    # Build the typed RunOptions for the library API.  The CLI already
    # resolved the recipe, host list, cluster_def, and overrides above (so the
    # banner / VRAM block can render those before launch); passing the loaded
    # objects through avoids re-resolution inside the API and preserves the
    # cwd-recipe discovery the CLI does through ``_load_recipe``.
    #
    # ``hosts`` is the **candidate** list — every host placement may choose
    # from — not a pre-trimmed selection.  Narrowing it here would make this
    # command's own placement authoritative over the launch's, leaving
    # ``api.run`` unable to reach any host we discarded.
    run_options = api.RunOptions(
        recipe=recipe,
        hosts=tuple(host_list),
        cluster=cluster_def,
        overrides=dict(overrides),
        scheduler=scheduler_name,
        solo=solo,
        dry_run=dry_run,
        follow=not no_follow,
        detached=not foreground,
        trust=bool(trust),
        transfer_mode=effective_transfer_mode,
        transfer_interface=effective_transfer_interface,
        cache_dir=remote_cache_dir,
        runtime_cache=runtime_cache,
        local_cache_dir=local_cache_dir,
        ray_port=ray_port,
        dashboard_port=dashboard_port,
        dashboard=dashboard,
        init_port=init_port,
        executor_config=cli_executor_opts or None,
        rootful=rootful,
        workload_id_override=cluster_id_override,
        sync_tuning=not no_sync_tuning,
        extra_docker_opts=tuple(executor_args) if executor_args else None,
        topology=cluster_cfg.topology,
        recipe_ref=recipe_ref,
        offline=offline,
    )
    # The serve overrides, with executor keys already lifted out.
    overrides = dict(run_options.overrides)

    has_post_hooks = bool(recipe.post_exec or recipe.post_commands)
    if no_ready_wait and has_post_hooks:
        raise click.UsageError("--no-ready-wait cannot be used with post-launch hooks that require readiness")
    if not foreground and not no_ready_wait and not dry_run:
        from sparkrun.orchestration.executor import resolve_executor

        wait_executor = resolve_executor(
            recipe=recipe,
            runtime=runtime,
            cluster=cluster_def,
            config=config,
            cli_overrides=run_options.executor_overrides(),
            v=v,
            rootless=False,
            auto_user=False,
        )
        if not wait_executor.supports_host_endpoint:
            raise click.UsageError("This executor has no supported readiness endpoint; use --no-ready-wait for submission only")

    # --ensure: if this workload is already serving, don't launch — and don't
    # schedule either, which is why this runs before the plan.
    #
    # Keyed on the *intent* (recipe + parallelism + port), never on a
    # host-derived cluster_id: a cluster_id also encodes placement, so the old
    # lookup could not match a job placed by a status-aware scheduler (random
    # placement token) and would launch a duplicate every time.  The intent is
    # placement-independent, so the answer no longer depends on which scheduler
    # is configured.  Hosts are the full candidate list — a deployment that
    # landed on hosts this launch wouldn't pick still counts as running.
    if ensure:
        from dataclasses import replace
        from sparkrun.api._resolve import resolve_operation_target, resolve_cluster
        from sparkrun.orchestration.job_metadata import generate_intent_id

        try:
            cluster_def, target = resolve_operation_target(
                run_options,
                recipe=recipe,
                runtime=runtime,
                cluster=run_options.cluster or resolve_cluster(None, run_options.hosts, sctx=sctx, config=config),
                sctx=sctx,
            )
        except (ValueError, api.SparkrunError) as error:
            raise click.ClickException(str(error)) from error
        run_options = replace(run_options, cluster=cluster_def, executor_config={**run_options.executor_overrides(), **target.overrides})

        _match = api.find_running_intent(
            generate_intent_id(recipe, overrides),
            host_list,
            cluster=cluster_def,
            sctx=sctx,
        )
        if _match is not None:
            click.echo("Job already running (cluster_id: %s)" % _match.workload_id)
            click.echo("  Recipe: %s" % (_match.recipe or recipe.qualified_name or "unknown"))
            click.echo("  Hosts:  %s" % ", ".join(_match.hosts))
            if _match.other_cluster_ids:
                click.echo(
                    "  Warning: %d other deployment(s) of this workload are also running: %s"
                    % (len(_match.other_cluster_ids), ", ".join(_match.other_cluster_ids)),
                    err=True,
                )
            if no_ready_wait or dry_run or foreground:
                click.echo("Existing job found; readiness not checked.")
                sys.exit(0)
            from sparkrun.cli._run_lifecycle import existing_launch, wait_for_run

            existing = existing_launch(_match, recipe=recipe, overrides=overrides, cluster=cluster_def, sctx=sctx)
            sys.exit(
                wait_for_run(
                    existing,
                    show_logs=not no_follow,
                    keep_following=follow,
                    on_ready=_echo_endpoint_ready,
                    on_failure=_report_readiness_outcome,
                    run_hooks=False,
                )
            )

    echo_run_identity(recipe, runtime)

    # Display and launch the same plan, preserving the full candidate set.
    try:
        with planning_output():
            run_plan = api.plan(run_options, sctx=sctx)
    except api.InsufficientCapacity as e:
        click.echo("Error: %s" % e, err=True)
        _render_capacity_diagnostics(getattr(e, "status", None), list(getattr(e, "host_list", ()) or host_list), e.rejections)
        sys.exit(1)
    except api.SparkrunError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    host_list = list(run_plan.host_list)
    if render_run_plan(run_plan, run_options, host_source=host_source, verbose=(ctx.find_root().obj or {}).get("verbose", 0) >= 1):
        run_plan = dataclasses.replace(run_plan, hardware_reported=True)

    # Decide hook trust here, after the summary and before anything runs:
    # ``api.run`` refuses untrusted hooks rather than prompting.
    trust = authorize_recipe_hooks(run_plan.recipe, trust=bool(trust), dry_run=dry_run, sctx=sctx)
    run_options = dataclasses.replace(run_options, trust=trust)

    # Own the timeline here rather than letting ``launch_inference`` create one
    # internally: on a failed launch there is no LaunchResult to read it off,
    # and a launch that failed is exactly when the per-stage breakdown is worth
    # having.  ``launch_inference`` picks this up via ``sctx``.
    #
    # ``--no-timings`` suppresses the *table*, not the readiness watch below —
    # "the endpoint is up now" is worth having while logs scroll whether or
    # not you want a breakdown afterwards.  Diagnostics needs the timeline for
    # its own record regardless.
    if show_timings and sctx.timing is None:
        from sparkrun.core.timing import Timeline

        sctx.timing = Timeline()

    # Launch via the library API; the API call internally drives
    # ``launch_inference`` (which calls ``runtime.run``).  Tests that
    # mock ``runtime.run`` still observe the call because the runtime
    # layer is unchanged.
    #
    # ``plan=run_plan`` launches exactly what was displayed above — no
    # second resolution, no second transport prepare, no second occupancy
    # sweep, and no chance of the launch landing somewhere other than the
    # hosts named in the banner.
    # Spans the whole API call, not just ``launch_inference``: the difference
    # between this and the ``launch`` span nested inside it is ``api.run``'s
    # own preamble (transport prepare, eviction setup), which is otherwise
    # unattributed time the reader can see in the total but not account for.
    # Left open on the error paths below, which is what ``status="open"``
    # means — the run did not finish.
    _run_span = sctx.timing.begin("run") if sctx.timing is not None else None
    try:
        run_result = api.run(run_options, sctx=sctx, plan=run_plan)
    except KeyboardInterrupt:
        click.echo("\nLaunch interrupted; inspect workload status before retrying.", err=True)
        sys.exit(130)
    except TransferError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except IncompatibleHardwareError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    except api.SparkrunError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    if _run_span is not None and sctx.timing is not None:
        sctx.timing.end(_run_span, rc=int(run_result.rc))

    # Native handlers may return only the public result. Rendering and exit
    # status use that contract; legacy lifecycle needs its private launch handle.
    launch_result = run_result.launch_result
    result = launch_result if launch_result is not None else run_result

    # region USER FACING STDOUT INFORMATION

    click.echo("Cluster:   %s" % run_result.workload_id)
    click.echo()
    click.echo("Serve command:")
    for line in result.serve_command.strip().splitlines():
        click.echo("  %s" % line)
    click.echo()

    if result.runtime_info:
        click.echo("Runtime versions:")
        for k, v in sorted(result.runtime_info.items()):
            click.echo("  %-10s %s" % (k + ":", v))
        click.echo()

    # endregion

    exit_code = result.rc
    if result.rc == 0 and not foreground:
        if dry_run:
            if launch_result is not None and has_post_hooks:
                from sparkrun.core.launcher import post_launch_lifecycle

                post_launch_lifecycle(
                    launch_result,
                    remote_cache_dir=launch_result.effective_cache_dir,
                    trust=trust,
                    dry_run=True,
                    progress=sctx.progress,
                )
            elif sctx.progress:
                sctx.progress.phase_skip(6)
        elif no_ready_wait:
            if sctx.progress:
                sctx.progress.phase_skip(6)
            click.echo("Launch completed; readiness not checked (--no-ready-wait).")
        elif launch_result is None:
            click.echo("Error: launch returned no supported readiness handle; readiness is unconfirmed.", err=True)
            exit_code = 1
        else:
            from sparkrun.cli._run_lifecycle import wait_for_run

            exit_code = wait_for_run(
                launch_result,
                show_logs=not no_follow,
                keep_following=follow,
                on_ready=_echo_endpoint_ready,
                on_failure=_report_readiness_outcome,
                trust=trust,
                progress=sctx.progress,
            )
    elif sctx.progress:
        sctx.progress.phase_skip(6)

    # Printed last, and only here.  The tables are multi-line, so they cannot
    # be emitted while `docker logs -f` is writing to the same terminal
    # without being shredded mid-render — the live half of the report is the
    # single line `_echo_endpoint_ready` injects.  By this point the stream
    # has stopped and nothing else is writing.
    #
    # The startup block repeats that line's three numbers deliberately: it was
    # printed the instant the endpoint answered, which on a long launch is
    # thousands of log lines above where the user is now looking.
    #
    # A watcher still polling when the user interrupts leaves `serve.*` open,
    # which `format_launch_timings` renders "did not finish" — the honest
    # reading of "we stopped watching", not a claim that the stage failed.
    if show_timings:
        from sparkrun.utils.cli_formatters import STARTUP_SPAN_NAMES, format_launch_timings, format_startup_readiness

        # The observer stores the shared result here, including hook launches.
        # Empty for a legacy endpoint wait, whose stages remain in the tree.
        _startup = format_startup_readiness(
            getattr(result, "startup_observation", None),
            host=host_list[0] if host_list else None,
        )
        # The startup spans are dropped from the tree exactly when the block
        # above rendered them: the tree's rows sum to its total and those
        # three do not belong to that sum, so leaving them in shows the same
        # figures twice and breaks the one property the tree has.  They stay
        # in the export, which is what diagnostics and benchmark metadata read.
        _timings = (
            format_launch_timings(
                sctx.timing.export(),
                max_depth=_timing_tree_depth(ctx),
                omit=STARTUP_SPAN_NAMES if _startup else frozenset(),
            )
            if sctx.timing is not None
            else ""
        )
        for _block in (_startup, _timings):
            if _block:
                click.echo()
                click.echo(_block)
        if _startup or _timings:
            click.echo()

    sys.exit(exit_code)
