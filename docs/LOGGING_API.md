# Workload logging API

`sparkrun.api.open_log_target()` resolves a workload into a reusable `LogTarget`.
`sparkrun.api.logs()` is the convenience wrapper for one read. Both use the
runtime's `LogSource` descriptions, the executor's existing `read_logs_cmd()`, and
the shared reader in `sparkrun.orchestration.logs`.

## Resolve a target

```python
import threading
from sparkrun import api

cancel = threading.Event()
target = api.open_log_target(
    cluster_id,
    cluster="lab",
    scope="all",
    sctx=sctx,
    cancel=cancel,
)

print(target.cluster_id, target.runtime)
for source in target.sources:
    print(source.host, source.container, source.role, source.rank)
```

`LogTarget` is frozen. Its public data is `cluster_id`, the canonical `runtime`
name, and `sources: tuple[LogSource, ...]`. Executor and connection details are
private. Opening a target resolves transport, preserves the workload's recorded
SSH identity, and checks liveness. It does not spawn log readers. The target has
no `close()` method; each read owns its own processes. Reopen the target when
placement changes or when a fresh liveness check is needed.

Resolution accepts the same workload selectors as `api.logs`: a `cluster_id`,
or a `recipe` plus a host source (`hosts`, `cluster`, or configured defaults).
`cluster_id` takes precedence. Recipe overrides participate in intent discovery.

`hosts` describes the **full workload topology**. To inspect one worker, use
`host=` or `container=` instead of shortening `hosts`; rank and solo/cluster
container naming depend on that topology.

| Selector | Behavior |
| --- | --- |
| `scope="head"` | Default: primary source only. |
| `scope="all"` | All sources described by the runtime. |
| `host="worker"` | Filter all workload sources. Exact host identity wins; otherwise match hostname case-insensitively, ignoring a `user@` prefix. Ambiguous normalized identities require an exact selector. Does not change the connection user. |
| `container="..."` | Filter all workload sources by exact container/source name. |
| Both filters | Intersection. Explicit filters select from all sources even with the default head scope. |

Unmatched filters raise `JobNotFound`. Filtered liveness checks inspect only the
selected sources. Without filters, a head-only request still checks all nodes
and can recommend `--all-sources` when workers survive a failed head. No read or
failed precheck deletes job metadata or releases model-cache claims.

Native Kubernetes controller workloads continue using their plugin's log API;
this target API does not add native JobSet support.

## Read, follow, and cancel

```python
from contextlib import closing

# Finite read, grouped by source/rank. Stream to a renderer or export writer.
with closing(target.read(tail=None, cancel=cancel)) as lines:
    for line in lines:
        render(line)

# Initially read the last 200 lines per source, then stream arrivals.
with closing(target.follow(tail=200, cancel=cancel)) as lines:
    for line in lines:
        render(line)
```

`read(*, follow=False, tail=None, cancel=None, check=True)` returns a lazy
iterator of `LogLine`. `follow(*, tail=200, cancel=None, check=True)` delegates to
`read(follow=True, ...)`. `api.logs()` accepts the same read options, alongside
the resolution arguments and filters.

- `tail=None` reads all available lines; `tail=0` reads no initial history.
  Values must be nonnegative integers.
- Finite reads group sources using `core.log_source.source_sort_key`. Follow
  merges arrival order and adds local arrival timestamps; these are not remote
  event timestamps. Per-source ordering is preserved.
- Executor-backed reads keep transport/reader diagnostics outside `LogLine`
  records. Diagnostics are logged separately, drained concurrently, and retained
  as a bounded tail in subprocess failures' `stderr`. They never consume paging
  offsets. Docker workload stderr is combined with workload stdout inside the
  executor command, before the SSH boundary, so application errors remain in
  the log history.
- Nonzero reader exits raise `subprocess.CalledProcessError` by default. Spawn
  failures also propagate. `check=False` opts out of exit-status checking and
  permits best-effort following when one source fails.
- An explicit cancellation event overrides the target's captured default, which
  is the event supplied to `open_log_target`, or otherwise `sctx.operation.cancel`.
  This applies equally to direct target calls and the `api.logs` wrapper.
- Cancellation ends iteration normally, including when the reader is silent.
  Consumers must check the event before treating a finite read as complete.
  A file exporter must not publish a cancelled or failed read as a complete dump.
- Close partially consumed iterators in their consuming thread. To interrupt a
  read running on another thread, set its cancellation event and let that thread
  finish/close the iterator. Teardown stops reader processes, never the workload.

## Page older history

```python
source = target.sources[0]
page = target.page(source, skip=200, count=500, cancel=cancel, timeout=30.0)
if not cancel.is_set():
    prepend_history(page)
```

`page(source, *, skip, count=500, cancel=None, timeout=30.0)` returns at most
`count` lines immediately preceding the last `skip` lines, oldest first. The
source must belong to this target, counts must be nonnegative integers, and the
timeout must be positive and finite. Zero count returns an empty page.

Paging uses `tail=skip+count` and discards the last `skip` records. Short logs
return only the available preceding history: for ten lines, `skip=8, count=5`
returns the first two lines. The reader retains at most `count` records, though
transfer volume grows with `skip+count`. No executor window API is required.

Offsets are relative to the log's end at read time. Appends, rotation, and
truncation can shift pages. A TUI may compensate using delivered-line counts,
but paging remains best-effort rather than a stable snapshot cursor. Clear or
refresh history when switching targets and ignore results from obsolete reads.

Reader failures raise; the timeout raises `subprocess.TimeoutExpired` and reaps
the reader. Cancellation discards partial results and returns `[]`; check the
cancellation event before marking history exhausted.

## Daemon

`GET /v1/streams/logs` accepts `host` and `container` with the same semantics.
`hosts` remains a comma-separated full host list. Example query:

```text
/v1/streams/logs?cluster_id=WORKLOAD_ID&host=worker-2&tail=200&follow=true
```

The daemon validates filters, `scope=head|all`, a nonnegative integer `tail`,
and `follow=true|false` before opening the stream. Unknown parameters are rejected.
Log records, queue-gap notices, and read failures use the existing SSE formats.
Disconnect cancels both finite and following readers. Targets are resolved per
request; there is no server-side target ID or paging route. See [DAEMON_API.md](DAEMON_API.md).

## Guidance for frontend integrations

Use `target.runtime` for colorizer selection, `target.sources` for source
selection, and `source_sort_key` for consistent rank ordering. Follow, finite
export, and paging all reuse this target and reader stack.

File paths, export headers/prefixes, overwrite policy, partial-file handling,
and atomic publication belong in the frontend. Keep export contents streaming;
do not buffer the entire log to call a whole-file write helper. Atomic
no-overwrite publication must never fall back to a check followed by replacement.
There is no `LogTarget.dump` or `windowing_supported` property.
