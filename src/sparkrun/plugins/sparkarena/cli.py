"""Lazy Click presentation for the in-tree Spark Arena integration.

This adapter ships with core and intentionally shares its private CLI helpers.
It is not an example of a stable external CLI API: installed plugins should use
CliOptionSpec and the public benchmark/resume APIs instead.
"""

from __future__ import annotations


def build_benchmark_options():
    import click
    from sparkrun.cli._common import HIDE_ADVANCED_OPTIONS

    return [
        click.Option(
            ["--arena", "arena_flag"],
            is_flag=True,
            default=False,
            help="Submit results to Spark Arena (requires '{app_command} arena login').",
        ),
        click.Option(
            ["--local-test"],
            is_flag=True,
            default=None,
            hidden=HIDE_ADVANCED_OPTIONS,
            help="Arena rehearsal: skip authentication and upload (requires --arena).",
        ),
    ]


def build_arena_command():
    import sys
    import click
    from sparkrun.core.application_profile import render_identity_text
    from sparkrun.cli.ext import ExtensibleCommand
    from sparkrun.cli._benchmark import _shared_run_options, _invoke_benchmark, _resume_benchmark_run
    from sparkrun.cli._common import HIDE_ADVANCED_OPTIONS, with_host_context, dry_run_option

    ASCII_ART = r"""
    !       _____                  __      ___
    !      / ___/____  ____ ______/ /__   /   |  ________  ____  ____ _
    !      \__ \/ __ \/ __ `/ ___/ //_/  / /| | / ___/ _ \/ __ \/ __ `/
    !     ___/ / /_/ / /_/ / /  / ,<    / ___ |/ /  /  __/ / / / /_/ /
    !    /____/ .___/\__,_/_/  /_/|_|  /_/  |_/_/   \___/_/ /_/\__,_/
    !        /_/
    """

    @click.group()
    @click.pass_context
    def arena(ctx):
        """Manage Spark Arena authentication.

        Run and submit benchmarks with `{app_command} benchmark <recipe> --arena`.
        """
        pass

    @arena.command()
    @click.option("--browser", "force_browser", is_flag=True, hidden=True, help="Force browser-based login")
    @click.option("--device", "force_device", is_flag=True, hidden=True, help="Force device code login")
    @click.pass_context
    def login(ctx, force_browser, force_device):
        """Authenticate with Spark Arena via OAuth."""
        from sparkrun.plugins.sparkarena.auth import run_login_flow

        if force_browser and force_device:
            click.echo("Error: --browser and --device are mutually exclusive.", err=True)
            sys.exit(1)

        success = run_login_flow(force_browser=force_browser, force_device=force_device)
        if success:
            click.echo(ASCII_ART)
        if not success:
            sys.exit(1)

    @arena.command()
    def logout():
        """Remove stored Spark Arena credentials."""
        from sparkrun.plugins.sparkarena.auth import clear_refresh_token, load_refresh_token

        if not load_refresh_token():
            click.echo("Not logged in.")
            return

        clear_refresh_token()
        click.echo("Logged out.")

    @arena.command()
    def status():
        """Show Spark Arena login status."""
        from sparkrun.plugins.sparkarena.auth import load_refresh_token, exchange_token

        token = load_refresh_token()
        if not token:
            click.echo("Not logged in.")
            click.echo(render_identity_text("Run '{app_command} arena login' to authenticate."))
            return

        try:
            result = exchange_token(token)
            if result.email:
                user_fmt = result.email
                if result.provider:
                    user_fmt += " (via %s)" % result.provider
                click.echo("Logged in to spark-arena as %s" % user_fmt)
            else:
                click.echo("Logged in to spark-arena (user id: %s)" % result.user_id)
        except RuntimeError as e:
            click.echo("Token invalid or expired: %s" % e)
            click.echo(render_identity_text("Run '{app_command} arena login' to re-authenticate."))

    class _ArenaBenchmarkGroup(click.Group):
        def parse_args(self, ctx, args):
            if args and args[0] not in self.commands and args[0] not in ("--help", "-h"):
                args = ["run", *args]
            return super().parse_args(ctx, args)

    @arena.group("benchmark", cls=_ArenaBenchmarkGroup, hidden=HIDE_ADVANCED_OPTIONS)
    def arena_benchmark():
        """Benchmark a recipe and submit results to Spark Arena.

        Prefer `{app_command} benchmark <recipe> --arena`.
        """

    @arena_benchmark.command("run", cls=ExtensibleCommand, extension_target="benchmark.run")
    @_shared_run_options
    @click.pass_context
    @with_host_context
    def arena_benchmark_run(ctx, **kwargs):
        """Benchmark a recipe and submit results to Spark Arena."""
        click.echo(ASCII_ART)
        kwargs["arena_flag"] = True
        return _invoke_benchmark(ctx, category="performance", **kwargs)

    @arena_benchmark.command("resume", cls=ExtensibleCommand, extension_target="benchmark.resume")
    @click.argument("benchmark_id")
    @dry_run_option
    @click.pass_context
    def arena_benchmark_resume(ctx, benchmark_id, dry_run, **kwargs):
        """Resume an Arena benchmark or retry its upload using the saved submission id."""
        kwargs["arena_flag"] = True
        return _resume_benchmark_run(ctx, benchmark_id, dry_run, integrations=ctx.command.pop_extension_values(ctx, kwargs))

    return arena
