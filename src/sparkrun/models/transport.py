"""Builtin manifest-scoped copies with atomic file replacement and bounded retry."""

from __future__ import annotations

import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
from functools import partial
import threading
import logging
import math
from pathlib import PurePosixPath
import time

from sparkrun.core.model_distribution import ModelCopyRequest, ModelTransferResult, transfer_timeouts
from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models._copy_progress import CopyProgress
from sparkrun.models._stream import ModelTransferStalled, ProgressWatch
from sparkrun.core.operation import CURRENT, submit
from sparkrun.orchestration.ssh import RemoteResult, build_ssh_opts_string, rsync_retry_disabled, should_run_locally
from sparkrun.orchestration.transfer import (
    classify_rsync_failure,
    rsync_attribute_errors_only,
    rsync_had_vanished_files,
    rsync_has_attribute_permission_error,
)
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)


def copy_script(request: ModelCopyRequest, target, ssh_kwargs: dict, *, relaxed: bool = False) -> str:
    """Copy only selected logical files; never traverse unrelated live blobs.

    Dereferencing the selected snapshot writes a self-contained snapshot with
    regular files. This avoids copying both blob and snapshot payloads or
    exporting links into a different host's cache-wide blob store.
    """
    source = str(PurePosixPath(request.source_cache_root) / request.manifest.relative_snapshot) + "/"
    destination = str(PurePosixPath(target.cache_root) / request.manifest.relative_snapshot) + "/"
    local_destination = (
        request.source_host is None and should_run_locally(target.host, ssh_kwargs.get("ssh_user"))
    ) or request.source_host == target.host
    args = ["rsync", "-r", "--copy-links", "--mkpath", "--protect-args", "--ignore-times", "--from0", "--partial-dir=.sparkrun-partial"]
    args += ["--info=progress2", "--outbuf=L", "--no-human-readable", "--omit-dir-times"]
    if request.idle_timeout is not None:
        # rsync's own I/O-idle timeout: the remote end exits by itself too,
        # even if this controller's watchdog never gets to cancel it.
        args.append("--timeout=%d" % math.ceil(request.idle_timeout))
    if not relaxed:
        args.append("--times")
    if not local_destination:
        opts = build_ssh_opts_string(**{key: ssh_kwargs[key] for key in ("ssh_user", "ssh_key", "ssh_options") if key in ssh_kwargs})
        # A user's ControlMaster must not collapse these bulk streams onto
        # one TCP connection/encryption process. Keep their identity/host trust.
        args += ["-e", "ssh -o ControlMaster=no -o ControlPath=none -o Compression=no " + opts + " -S none"]
        user = ssh_kwargs.get("ssh_user")
        destination = (user + "@" if user else "") + target.transfer_host + ":" + destination
    data = base64.b64encode(b"\0".join(path.encode() for path in target.required_files) + b"\0").decode()
    command = " ".join(str(quote(arg)) for arg in args)
    return (
        "set -euo pipefail\nexport LC_ALL=C\n"
        "list=$(mktemp)\ntrap 'rm -f -- \"$list\"' EXIT\n"
        f'printf %s {quote(data)} | base64 -d > "$list"\n'
        + command
        + ' --files-from="$list" -- '
        + str(quote(source))
        + " "
        + str(quote(destination))
        + "\n"
    )


# Large shards run separately; smaller files amortize SSH startup in batches.
_BATCH_BYTES = 64 * 1024**2


# rsync's exit status when its --timeout fired: no data moved for that long.
_RSYNC_TIMEOUT_RC = 30


def _stall_reason(target, sizes, idle, transferred) -> str:
    files = target.required_files
    subject = files[0] if len(files) == 1 else "%d files" % len(files)
    return "model transfer stalled: no progress for %ds (%s, %.2f of %.2f GiB transferred)" % (
        idle,
        subject,
        transferred / 2**30,
        sum(sizes[path] for path in files) / 2**30,
    )


def copy_limits(config) -> tuple[int, int]:
    get = getattr(config, "get", None)
    values = []
    for key, default in (("model_transfer_streams_per_host", 2), ("model_transfer_max_streams", 16)):
        value = get(key, default) if callable(get) else default
        if type(value) is not int or value < 1:
            raise ModelArtifactError(key + " must be a positive integer")
        values.append(value)
    return values[0], values[1]


def _batches(target, sizes):
    small, size = [], 0
    for path in sorted(target.required_files, key=lambda p: (-sizes[p], p)):
        if sizes[path] >= _BATCH_BYTES:
            yield replace(target, required_files=(path,))
        else:
            if small and size + sizes[path] > _BATCH_BYTES:
                yield replace(target, required_files=tuple(sorted(small)))
                small, size = [], 0
            small.append(path)
            size += sizes[path]
    if small:
        yield replace(target, required_files=tuple(sorted(small)))


def _remaining(deadline):
    return None if deadline is None else deadline - time.monotonic()


def _copy_job(request, target, io, ssh_kwargs, deadline, cancelled, parser, *, sizes, stall_retries):
    """Run one batch's rsync: bounded by progress, retried within limits.

    One retry each for vanished source files or attribute errors (the
    original policy), plus *stall_retries* resumes after a stall; the partial
    directory keeps what a stalled attempt already moved.
    """
    relaxed = False
    retried = False
    stalls = 0
    while True:
        remaining = _remaining(deadline)
        if remaining is not None and remaining <= 0:
            return "model transfer deadline exceeded"
        if cancelled():
            return "model copy cancelled"
        # A resume first checksums its partial file, which moves no bytes.
        watch = ProgressWatch(expected=sum(sizes[path] for path in target.required_files), armed=stalls == 0)
        try:
            result = io.execute(
                request.source_host,
                copy_script(request, target, ssh_kwargs, relaxed=relaxed),
                timeout=remaining,
                cancelled=cancelled,
                stdout=parser(watch),
                idle_timeout=request.idle_timeout,
                watch=watch,
            )
        except ModelTransferStalled as exc:
            stalled = (exc.idle, exc.transferred)
        except Exception as exc:
            return str(exc) or type(exc).__name__
        else:
            transfer = RemoteResult(
                target.host, result.returncode, result.stdout.decode(errors="replace"), result.stderr.decode(errors="replace")
            )
            if transfer.success or rsync_attribute_errors_only(transfer):
                return None
            if transfer.returncode == _RSYNC_TIMEOUT_RC and request.idle_timeout is not None:
                stalled = (request.idle_timeout, watch.transferred)
            else:
                logger.warning(
                    "Model copy %s -> %s attempt failed: %s",
                    request.source_host or "control",
                    target.host,
                    transfer.stderr[-1500:],
                )
                if not retried and not rsync_retry_disabled() and not cancelled():
                    retried = True
                    remaining = _remaining(deadline)
                    if remaining is not None and remaining <= 0:
                        return "model transfer deadline exceeded"
                    if rsync_had_vanished_files(transfer):
                        report = io.observe(
                            request.source_host,
                            request.manifest,
                            request.source_cache_root,
                            timeout=60 if remaining is None else min(60, remaining),
                            cancelled=cancelled,
                        )
                        if report.complete:
                            continue
                    elif rsync_has_attribute_permission_error(transfer):
                        relaxed = True
                        continue
                return classify_rsync_failure(transfer) + ": " + transfer.stderr[-1500:]
        reason = _stall_reason(target, sizes, *stalled)
        if stalls < stall_retries and not rsync_retry_disabled() and not cancelled():
            stalls += 1
            logger.warning(
                "Model copy %s -> %s: %s; resuming (%d/%d)", request.source_host or "control", target.host, reason, stalls, stall_retries
            )
            continue
        return reason


def builtin_copy(request: ModelCopyRequest, io: ModelHostIO, ssh_kwargs: dict) -> ModelTransferResult:
    """Fair bounded fan-out, with disjoint file batches per destination.

    One invocation has one source. Each rsync is bounded by its progress
    (``request.idle_timeout``); a total deadline applies only when one is
    configured (``request.timeout``) or the operation carries one. The
    coordinator alone publishes results/callbacks.
    Core has already acquired leases and collapsed shared-cache aliases.
    """
    per_host, maximum = copy_limits(request.config)
    stall_retries = transfer_timeouts(request.config).stall_retries
    state = CURRENT.get()
    deadlines = [time.monotonic() + request.timeout] if request.timeout is not None else []
    if state is not None and state.deadline is not None:
        deadlines.append(state.deadline)
    deadline = min(deadlines, default=None)
    stopped = threading.Event()

    def cancelled():
        return stopped.is_set() or request.cancelled() or (state is not None and state.cancel.is_set())

    sizes = {f.path: f.size for f in request.manifest.files}
    jobs, pending, outcomes, errors = {}, {}, {}, {}
    for target in request.targets:
        if not target.required_files:
            outcomes[target.host] = "already_present"
        pending[target.host] = deque()
        for batch in _batches(target, sizes):
            key = len(jobs)
            jobs[key] = (batch, sum(sizes[p] for p in batch.required_files))
            pending[target.host].append(key)
    progress = CopyProgress(request, jobs)
    active = {}
    active_hosts = {host: 0 for host in pending}
    # Round-robin admission stops a large first target monopolizing the pool.
    order = deque(pending)
    pool = ThreadPoolExecutor(max_workers=min(maximum, max(1, len(jobs))), thread_name_prefix="model-rsync")
    terminal_status = "failed"
    try:
        progress.emit(force=True, outcomes=outcomes)
        while active or any(pending.values()):
            if cancelled():
                raise ModelArtifactError("model copy cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                for host in pending:
                    if host not in outcomes:
                        outcomes[host], errors[host] = "failed", "model transfer deadline exceeded"
                break
            for _ in range(len(jobs) + len(order)):
                if len(active) >= maximum or not order:
                    break
                admitted = False
                for _ in range(len(order)):
                    host = order[0]
                    order.rotate(-1)
                    if host in errors:
                        pending[host].clear()
                    if pending[host] and active_hosts[host] < per_host:
                        key = pending[host].popleft()
                        target, _ = jobs[key]
                        progress.start(key)
                        future = submit(
                            pool,
                            _copy_job,
                            request,
                            target,
                            io,
                            ssh_kwargs,
                            deadline,
                            cancelled,
                            partial(progress.parser, key),
                            sizes=sizes,
                            stall_retries=stall_retries,
                        )
                        active[future] = key
                        active_hosts[host] += 1
                        admitted = True
                        break
                if not admitted:
                    break
            if not active:
                break
            poll = 0.1 if deadline is None else min(0.1, max(0, deadline - time.monotonic()))
            completed, _ = wait(active, timeout=poll, return_when=FIRST_COMPLETED)
            for future in completed:
                key = active.pop(future)
                target, _ = jobs[key]
                active_hosts[target.host] -= 1
                error = future.result()
                progress.finish(key, error is None)
                if error is not None:
                    outcomes[target.host], errors[target.host] = "failed", error
                    pending[target.host].clear()
                elif not pending[target.host] and not active_hosts[target.host] and target.host not in errors:
                    outcomes[target.host] = "complete"
                    if request.progress is not None:
                        original = next(t for t in request.targets if t.host == target.host)
                        request.progress(target.host, sum(sizes[p] for p in original.required_files))
            progress.emit(outcomes=outcomes)
        if cancelled():
            raise ModelArtifactError("model copy cancelled")
        terminal_status = "failed" if errors else "succeeded"
    except BaseException:
        for target in request.targets:
            outcomes.setdefault(target.host, "failed")
        raise
    finally:
        stopped.set()
        pool.shutdown(wait=True, cancel_futures=True)
        for key in jobs:
            progress.finish(key, False)
        progress.emit(outcomes=outcomes, status=terminal_status, force=True)
    return ModelTransferResult(request.manifest.identity, outcomes, errors)
