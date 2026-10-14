# Diagnostic bundles

Diagnostic bundles are an opt-in library service shared by the CLI and daemon.
They collect configuration summaries, host probes, operation feedback, and
bounded workload log tails into a versioned NDJSON file. They do not upload data
or change host configuration.

## Standalone collection

```python
from sparkrun import api
from sparkrun.application import initialize

sctx = initialize()
result = api.collect_diagnostics(
    "diagnostics.ndjson",
    cluster="lab",
    workload_id="sparkrun_<intent>_<placement>",  # optional
    sctx=sctx,
)
print(result.output_path, result.partial, result.sections)
```

`hosts` and `cluster` follow shared API resolution. With `workload_id`, collection
uses saved placement and connection settings; missing metadata becomes an explicit
section error. It does not fall back to probing unrelated default hosts. Without
a workload, collection includes configuration and host probes only.

The existing `sparkrun setup diagnose` command uses this API. Its `--sudo` option
prompts for the optional `sudo_password` used for privileged DMI probes. Python
callers can supply that credential directly; it is registered for redaction and
never included in bundle metadata. Daemon collection has no privileged probe option.

## Capture around operations

```python
with api.capture_diagnostics("run.ndjson", sctx=sctx) as capture:
    result = api.run(api.RunOptions(recipe="my-recipe.yaml", cluster="lab"), sctx=sctx)
    readiness = api.wait_ready(result, sctx=sctx)
    if not readiness.ready:
        raise RuntimeError("Workload did not become ready")

print(capture.result)
```

Capture records shared planning/run/benchmark boundaries, selected operation
options, recipe and launch results, timing spans, Python logs, structured feedback,
and exceptions. Host probes use the planned placement; nested planning/run calls
reuse the bundle and avoid repeated host probes. Failed operations also attempt
to collect workload log tails. Readiness or other API calls inside the context
contribute their logs, feedback, and exceptions. A caller interpreting a returned
failure value must raise or call `capture.emit_error(...)` to mark that outcome.

`sparkrun run --collect-diagnostics FILE` wraps the CLI lifecycle with this API,
including planning and readiness. Python callers can similarly wrap `api.benchmark`
or `api.resume_benchmark`. Nested capture scopes reuse the outer bundle, destination,
and limits; `capture.result` becomes available when that outer scope closes.
Standalone collection called inside a capture adds its sections to that same
bundle and returns a status snapshot.

When a benchmark owns and stops its inference workload, capture snapshots its log
tails **before cleanup**. This also happens after successful measurements, because
cleanup, publication, or final notifications can still fail. The snapshot stays
in the bundle on success as well; retention is unchanged. Final capture does not
read those sources again after cleanup. Read failures become partial sections
without preventing cleanup or replacing the benchmark error. Cancellation skips
new reads; an interruption during collection still attempts cleanup.
`skip_run`, `no_stop`, reused workloads, and resume keep their existing ownership
behavior. When no cleanup is required, capture can collect failure tails at exit
for workloads recorded by planning or launch.

Capture preserves the application's logging levels. It records logs attributed
to its operation; it does not turn on global DEBUG logging or capture unrelated
requests. Background work needs the usual operation context propagation (shared
fan-out uses `sparkrun.core.operation.submit`). Structured feedback uses the public
[plugin feedback interface](FEEDBACK.md).

## Format, limits, and failure behavior

Records have `_type`, `_seq`, and `_ts` envelope fields. `diag_header` declares
`schema_version: 1`; `bundle_section` records report `ok`, `error`, `skipped`, or
`cancelled` outcomes. `bundle_complete` records `partial`, `truncated`, and
`operation_success`. Missing completion means the process did not finish the
bundle. Sequence gaps can occur when records exceed the storage budget.

- Default output limit: **16 MiB**, configurable with `max_bytes` from 16 KiB
  through 64 MiB. Individual records over 64 KiB are dropped and marked as
  truncation. Space is reserved for the completion marker.
- Default logs: **200 lines per source**, configurable with `log_tail` from 0
  through 2,000. Zero skips logs. Each source read has a 30-second timeout; host
  probes have a 60-second timeout. These are per-probe limits, not a total deadline.
- Files are created exclusively with private permissions. Existing files are
  never overwritten or appended to. An output-open error fails before work starts.
- Section failures are retained and collection continues where possible. `partial`
  includes skipped sections and truncation. Collection failure does not mean the
  workload failed, and a diagnostic failure does not replace a workload exception.
- Cancellation stops further probes and still attempts to finalize the file.
  `dry_run=True` skips remote probes and log reads explicitly.

Every record passes through the same redactor, including logs and errors. It
removes sensitive-key fields, known secret environment values, operation prompt
secrets, credential assignments, bearer tokens, URL credentials, and private-key
blocks. Arbitrary application text can still contain unrecognized secrets or
personal data; review a bundle before sharing it. Bundles retain hostnames,
hardware identity, paths, and configuration summaries useful for troubleshooting.

## Daemon collection and downloads

Submit `diagnostics.collect` through `POST /v1/ops/diagnostics.collect` or
`POST /v1/diagnostics/collect`. Arguments are `hosts`, `cluster`, `workload_id`,
`log_tail`, `max_bytes`, and `dry_run`; all are optional. This is a **read** job,
not an inline query. Clients cannot select a server filesystem path or submit
credentials in its arguments.

The result contains the section/host status and an `artifact` descriptor with
`name`, `media_type`, `size_bytes`, and `url`. Download the NDJSON bytes at:

```text
GET /v1/jobs/{job_id}/artifacts/diagnostics
```

Downloads require the same ownership and operation permissions as the job. They
are available after it finishes, including failed/cancelled jobs with a captured
bundle, and expire with daemon job history. A lost worker can leave an incomplete
file; inspect its completion marker. There is no arbitrary-path download endpoint.

`workloads.run`, `benchmarks.run`, and `benchmarks.resume` also accept
`collect_diagnostics: true` (default false). The whole adapter is captured, including
run readiness/hooks. An `artifact` event announces the download URL; the job's
normal result/error schema is preserved. The URL can also be derived from the job
ID if that event is no longer retained. The reference CLI client prints the job
result/events; download the artifact through an HTTP client.

## Scope and remaining extensions

Collection remains independent and opt-in. There is no default collection on every
run, automatic upload, retain-only-on-failure mode, plugin diagnostic collector
hook, or dedicated benchmark CLI flag. Python/daemon benchmark capture is available.
Plugins can submit in-memory snapshots through the reporter below; read-only
collector callbacks that perform additional inspection remain deferred.
Log tails remain bounded, retrospective snapshots. Sources already removed by an
executor, plugin, or external process may be unavailable; those collection errors
are reported as partial sections.

## Plugin diagnostic sections

`DIAGNOSTICS_API_VERSION = 1` is public through `sparkrun.diagnostics` and
`sparkrun.plugins`. Declare local section names during the existing plugin
registration callback. Sparkrun supplies the loader-owned namespace and applies
the same redaction, artifact permissions, and retention as the rest of the bundle.

```python
from sparkrun.plugins import DiagnosticSection, register_plugin_diagnostics

def register(v):
    provider.diagnostics = register_plugin_diagnostics([
        DiagnosticSection("route_selection"),
        DiagnosticSection("transfer_result", priority="terminal"),
    ])
    # Register the provider through its normal extension point as well.

def copy(request):
    diag = provider.diagnostics.bind(feedback=request.feedback)
    # Normal work computes route/outcome independently of diagnostics.
    if diag.enabled:
        diag.snapshot("route_selection", lambda: {
            "transport": route.transport,
            "fallback_reasons": reasons[:8],
        }, host=route.host, stage="preflight")
```

Store the declaration factory on a provider, but bind a new reporter for each
operation. `bind(sctx=sctx)` uses that context exclusively; `bind(feedback=...)`
uses an existing feedback reporter's operation and activity ID, including on
callback threads without ambient context. Otherwise binding uses the current
operation. Optional `activity_id` and `workload_id` supply existing identities.
Pass bound reporters to worker threads; they never retarget a subsequent capture.
Nested capture reuses the same sink and budgets. Closed reporters are inert.

When capture is disabled, binding returns a shared disabled reporter. No payload
builder, diagnostic file, or background worker runs. The `enabled` guard also
avoids creating a lambda and its arguments. `snapshot(section, build, host=...,
stage=...)` admits work before calling `build`, once and synchronously; it returns
whether the record was flushed. This return value must not control plugin work.
Builders select inexpensive existing data only. Do not perform new probes, file
scans, subprocess calls, or expensive computation in them. Use phase boundaries
and final outcomes, not polling, heartbeat, or transfer-chunk loops.

Snapshot payloads are strict built-in JSON objects. Limits include depth 8,
1,024 visited values, 64 entries per container, 128 characters per key, 2,048 per
string, and 8,192 total key/string characters. Integers are signed 64-bit and
floats must be finite. Arbitrary objects, iterators, and custom serialization are
rejected without stringifying them. Encoded records, including the envelope, are
limited to 16 KiB. Section names are stable lowercase identifiers, maximum 64
characters; optional scope fields are limited to 512 characters.

Each plugin can declare 16 sections and submit up to 128 attempts / 256 KiB per
capture, with 32 attempts per section across hosts/activities. At most 32 plugins
participate. Aggregate capacity is 512 attempts and the smaller of 1 MiB or one
quarter of the bundle's ordinary byte budget. One quarter of count/byte capacity
is protected from normal snapshots for terminal evidence and metadata/summaries;
metadata and summary space is reserved before admitting a contributor. Tiny
bundles may omit plugin sections entirely and report partial collection. Invalid
attempts consume count capacity too. Dropped calls do not construct payloads
after admission capacity is exhausted.

Records have type `plugin_diagnostic`, with core-owned `plugin`, `section`,
`schema_version`, `diagnostics_api_version`, and `operation_id` fields, optional
scope, and nested `data`. Plugins own each section's payload schema and version.
`plugin_diagnostics_metadata` records loaded provenance once, without rediscovery;
`plugin_diagnostics_summary` records bounded attempt/drop counts and rejection
reasons. Consumers should ignore unknown types/sections. Records do not also
enter the daemon event journal or Python logging.

Ordinary producer, validation, or write errors make diagnostics partial without
failing the operation. Process-control exceptions propagate. Cancellation stops
new snapshots; a producer still building during capture close cannot write after
`bundle_complete`, and its missing snapshot is reported. Saturated budgets retain
aggregate omission counts rather than emitting a warning per rejected record.
Collection failure and a reported workload failure are different: a successfully
captured failure snapshot does not itself mark the bundle partial or change the
operation's result.

Writes are synchronous and flushed once per accepted record, without `fsync`.
Payload processing and retained bookkeeping are bounded; builder execution and
slow filesystem latency have no hard time guarantee. Join producer threads as
part of their operation, and never hold a transfer/pipe-reader lock while writing
diagnostics. Use the offline `scripts/benchmark-plugin-diagnostics.py` harness
to measure incremental reporter overhead separately from probes and transfers.

OCI Relay contributes source/route decisions, bounded failure tails before
receiver cleanup, and transfer/cleanup outcomes. ColdSnap contributes artifact
compatibility decisions and controller results before receipt cleanup. Existing
structured feedback continues to supply their progress history. Plugins can
feature-detect `DIAGNOSTICS_API_VERSION` at registration to support older hosts.

## Native Kubernetes

Native Kubernetes workload logs use the same [logging API](LOGGING_API.md), with
resource/source identity preserved. Native control-plane executors skip the
SSH/Linux host probes explicitly; Kubernetes events, manifests, and node-level
diagnostic collectors are not part of this bundle version.
