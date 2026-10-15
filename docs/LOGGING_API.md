# Workload logging API

`sparkrun.api.open_log_target()` resolves a workload into a reusable `LogTarget`.
`sparkrun.api.logs()` is the convenience wrapper for one read. Both use the
runtime's `LogSource` descriptions (or native executor discovery), the executor's
`read_logs_cmd()`, and the shared reader in `sparkrun.orchestration.logs`.

## Resolve a target

```python
import threading
from sparkrun import api

cancel = threading.Event()
target = api.open_log_target(
    workload_id,
    cluster="lab",
    scope="all",
    sctx=sctx,
    cancel=cancel,
)

print(target.workload_id, target.runtime)
for source in target.sources:
    print(source.source_id, source.label, source.role, source.rank)
```

`LogTarget` is frozen. Its public data is `workload_id`, the canonical `runtime`
name, and `sources: tuple[LogSource, ...]`. Executor and connection details are
private. Opening a target resolves transport, preserves the workload's recorded
destination, and checks availability. It does not spawn log readers. The target has
no `close()` method; each read owns its own processes. Reopen the target when
placement changes or when a fresh liveness check is needed.

Resolution accepts the same workload selectors as `api.logs`: a `workload_id`,
or a `recipe` plus a host source (`hosts`, `cluster`, or configured defaults).
`workload_id` takes precedence. Recipe overrides participate in intent discovery.
The identifier is `RunResult.workload_id`; the name distinguishes a workload from
its host cluster.
The Python keyword, `LogTarget` field, and daemon query all use `workload_id`.

`hosts` describes the **full workload topology**. To inspect one worker, use
`host=` or `container=` instead of shortening `hosts`; rank and solo/cluster
container naming depend on that topology.

| Selector | Behavior |
| --- | --- |
| `scope="head"` | Default: primary source only. |
| `scope="all"` | All sources described by the runtime. |
| `host="worker"` | Filter all workload sources. Exact host identity wins; otherwise match hostname case-insensitively, ignoring a `user@` prefix. Ambiguous normalized identities require an exact selector. Does not change the connection user. |
| `container="..."` | Filter all workload sources by exact container/source name. |
| `source_id="log_..."` | Select one opaque source identity, including native resource and container instance. |
| Combined filters | Intersection. Explicit filters select from all sources even with the default head scope. |

Unmatched filters raise `JobNotFound`. Filtered liveness checks inspect only the
selected sources. Without filters, a head-only request still checks all nodes
and can recommend `--all-sources` when workers survive a failed head. No read or
failed precheck deletes job metadata or releases model-cache claims.

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

`GET /v1/streams/logs` accepts `host`, `container`, and `source_id` with the same semantics.
`hosts` remains a comma-separated full host list. Example query:

```text
/v1/streams/logs?workload_id=WORKLOAD_ID&host=worker-2&tail=200&follow=true
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
`LogTarget.dump(path, *, overwrite=False, cancel=None, prefix=None)` streams a
complete finite read to `path`: it writes `<path>.partial` with `O_EXCL` and mode
`0600`, finalizes by `link`/`replace`, refuses an existing target without
`overwrite`, and removes the partial on cancel or failure. There is no
`windowing_supported` property — paging is always available for a target with
sources (see *Page older history*).

## Native Kubernetes workloads

Saved native JobSet/Job workloads use the same API and daemon endpoint. The K8s
executor recovers the saved kubeconfig, context, and namespace. It checks application
ownership and the workload annotation, then follows controller UID ownership from
JobSet to Jobs to Pods. Matching labels alone do not establish membership.
`kubectl` runs locally on the CLI/daemon host; it never SSHes to a scheduled node.
Daemon clients cannot supply kubeconfig paths or arbitrary Kubernetes resources.

Each regular container is a source. `host` is the scheduled node, `container` is
the actual container name, and `resource` is a `LogResource` containing `kind`,
`namespace`, pod `name`, pod `uid`, and an opaque container `instance` token.
The instance includes restart count and runtime container ID. These fields and
`source_id` accompany every `LogLine`, including daemon SSE records. Container
names can repeat across pods; use `source_id` to select exactly one stream.

`scope="head"` selects the first regular container of rank zero; `scope="all"`
includes readable regular containers and sidecars across the snapshot. Rank comes
from the JobSet's replicated-job layout and controller labels/annotations, not pod
name parsing. Launcher Jobs use rank zero. Completed containers remain readable;
queued workloads and containers that have not started report that logs are not
available. Init containers and previous container instances are outside this cut.

Reads check pod UID and container instance before starting and after normal EOF.
A detected replacement/restart raises `LogSourceChanged`; reopen the target and
clear the old paging offsets. A page that detects such a change is not returned.
These checks cannot make Kubernetes resource lookup and log opening atomic: finite
exports must still discard partial output on failure. Follow never reconnects to
replacement pods automatically. Reopen explicitly to discover newly created pods.

The existing `plugins.k8s.api.logs(name=..., kind="jobset"|"job")` uses the same
resolver/readers and supports cancellation and source filters. It defaults to all
sources. Use the core target API for reusable targets/paging of saved workloads.

Kubernetes application stdout/stderr arrive through the kubectl log payload;
kubectl's own stderr remains diagnostic output.

Kubernetes serves only the latest retained log file through `kubectl logs`; paging
does not retrieve older rotated files or provide an archive. See the
[Kubernetes logging reference](https://kubernetes.io/docs/concepts/cluster-administration/logging/).
Full-history reads explicitly set `--tail=-1`; the selector-dependent kubectl
default is not used ([command reference](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_logs/)).
