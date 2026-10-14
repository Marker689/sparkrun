"""Managed diagnostic artifacts, addressed only by an authorized job identity."""

from contextlib import contextmanager
from dataclasses import asdict
from functools import wraps
import inspect
import re

from sparkrun.daemon.state import Binding, private_directory
from sparkrun.api._wire import to_wire


def diagnostics_path(directory, job_id):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise ValueError("Invalid job identity")
    return directory / "artifacts" / (job_id + ".ndjson")


def descriptor(path, job_id):
    return {
        "name": "diagnostics",
        "media_type": "application/x-ndjson",
        "size_bytes": path.stat().st_size,
        "url": f"/v1/jobs/{job_id}/artifacts/diagnostics",
    }


def collect(arguments, sctx):
    from sparkrun import api

    binding = Binding.from_config(sctx.config)
    private_directory(binding.directory / "artifacts")
    path = diagnostics_path(binding.directory, sctx.operation.id)
    result = api.collect_diagnostics(path, sctx=sctx, **arguments)
    return to_wire({key: value for key, value in asdict(result).items() if key != "output_path"}) | {
        "artifact": descriptor(path, sctx.operation.id),
    }


@contextmanager
def capture(sctx, *, dry_run=False):
    from sparkrun import api

    binding = Binding.from_config(sctx.config)
    private_directory(binding.directory / "artifacts")
    path = diagnostics_path(binding.directory, sctx.operation.id)
    session = None
    try:
        with api.capture_diagnostics(path, sctx=sctx, dry_run=dry_run) as session:
            yield
    finally:
        if session is not None:
            try:
                # Advisory notification must also work after cancellation.
                sctx.operation.feedback({"type": "artifact", **descriptor(path, sctx.operation.id)})
            except OSError:
                pass  # Artifact failure must not mask the original outcome.


def with_diagnostics(function):
    """Optional capture around the whole adapter, including readiness/hooks."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        arguments = dict(bound.arguments["arguments"])
        enabled = arguments.pop("collect_diagnostics", False)
        bound.arguments["arguments"] = arguments
        if not enabled:
            return function(*bound.args, **bound.kwargs)
        with capture(bound.arguments["sctx"], dry_run=arguments.get("dry_run", False)):
            return function(*bound.args, **bound.kwargs)

    return wrapped
