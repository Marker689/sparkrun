"""CLI destination/rendering adapter for the shared capture API."""

from functools import wraps
import inspect

import click


def with_run_diagnostics(function):
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        path = values.get("diagnostics_path")
        if not path:
            return function(*args, **kwargs)
        from sparkrun import api
        from sparkrun.cli._common import _get_context

        session = None
        try:
            with api.capture_diagnostics(path, sctx=_get_context(values["ctx"]), dry_run=values.get("dry_run", False)) as session:
                return function(*args, **kwargs)
        except OSError as exc:
            if session is None:
                raise click.ClickException("Cannot open diagnostic bundle: %s" % exc) from exc
            raise
        finally:
            if session is not None:
                partial = session.result is None or session.result.partial
                click.echo("Diagnostics written to: %s%s" % (path, " (partial)" if partial else ""))

    return wrapped
