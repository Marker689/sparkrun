# Plugin and operation feedback

`FEEDBACK_API_VERSION = 1` defines a public interface shared by CLI, library,
plugin, and daemon callers. Import `get_feedback`, `FeedbackReporter`, and
`FeedbackEvent` from `sparkrun.feedback` or `sparkrun.plugins`.

## Producing feedback

Create a reporter inside the operation, before spawning background work. Pass
`sctx` explicitly for custom pipelines that run outside a core API boundary.
Image and model copy/pull requests also carry a bound `request.feedback` reporter.
Creating a child preserves the operation and assigns a separate activity ID.

```python
from sparkrun.plugins import get_feedback

feedback = get_feedback("my-plugin", sctx=sctx)
receiver = feedback.child(host="spark-1")
receiver.progress(
    "Receiving missing layers",
    stage="transfer",
    status="running",
    current=1048576,
    total=4194304,
    total_kind="upper_bound",
    unit="bytes",
    metrics={"bytes_per_second": 524288, "reused_layers": 2},
)
with receiver.heartbeat("Verifying image"):
    verify_image()
receiver.progress("Image verified", stage="verify", status="succeeded")
```

Use a stable plugin name as `source` and stable identifiers as `stage`; don't
use changing display labels as identifiers. `child(activity_id=...)` supports
foreign span IDs; scope those IDs to the reporter's activity to avoid collisions.
`warning()` and `error()` use the corresponding severity. `report(level="info",
...)` emits details that clients can filter separately from normal progress.

Every report emits an event independently of Python logging verbosity and, by
default, renders one Python log record. `report()` returns the normalized envelope.
A frontend that renders it directly can use `report(..., log_message=False)` to
suppress duplicate text logging; subscriber delivery still occurs. Supply `logger=` to `get_feedback()` to retain a plugin's
existing logger. CLI output follows its normal verbosity/quiet settings. Without
an operation/subscriber, the reporter still logs. Plugins supporting older hosts
can feature-detect `get_feedback` and keep their existing logging fallback.

## Event contract

Structured events have `type: "feedback"`, `feedback_version: 1`, `operation_id`,
Unix-seconds `timestamp`, `source`, `message`, `level`, and `activity_id`.
Optional `parent_id`, `host`, `stage`, and `status` describe an activity.
Status is `running`, `succeeded`, `failed`, or `skipped`; activity completion is
not whole-job completion. The operation's result/error remains authoritative.

`current`, `total`, and `unit` describe counters. `total_kind` must be `exact` or
`upper_bound` when `total` is present, and `unknown` otherwise. Only exact totals
are suitable for a percentage. Relay's expected transfer size is an upper bound;
full byte counts do not prove import or verification success. `metrics` holds up
to 32 named JSON scalar values, e.g. elapsed seconds, rate, or cache reuse.

Messages are limited to 16,384 characters, identifiers/text fields to 512, metric
keys to 128, and metric strings to 512. Numbers must be finite and counters
nonnegative. Invalid producer payloads raise `ValueError`. Publish selected
observations, rather than raw controller requests, credentials, or transcripts.

Reporters retain their operation across threads. Delivery is serialized per
operation. Subscriber callbacks run synchronously, must return promptly, and
should enqueue UI work. Callback exceptions are logged without aborting the
operation; callback-generated feedback is not recursively delivered. Join
producer threads and exit heartbeat contexts before returning from an operation.
Throttle repeated counters; publish stage changes and verified outcomes promptly.

## Library and daemon consumers

```python
from sparkrun.application import initialize
from sparkrun import api

sctx = initialize()
sctx.events = lambda event: event_queue.put(event)
result = api.run(api.RunOptions(recipe="my-recipe", detached=True), sctx=sctx)
```

An event subscriber also enables the existing launch phase tracker when no
tracker was supplied. Legacy core `type: "progress"` events and specialized
benchmark/setup events remain supported alongside the new feedback envelope.

Daemon workers persist feedback in the existing bounded job journal. Subscribe
to `GET /v1/jobs/{job_id}/events` (or `sparkrun daemon jobs events JOB_ID`). The
stream supports cursor replay and reports expired cursors; it is not an unlimited
transfer trace. Existing clients can display the top-level `message` immediately.
`daemon call --wait` polls results/prompts; use the events command for feedback.
Workload `/v1/streams/logs` is a separate stream.

## Legacy logging compatibility

An operation with an event subscriber forwards accepted Python log records at
PROGRESS (25), WARNING, and ERROR levels as `type: "log"`, with source, message,
level, timestamp, and operation ID. Records already represented by structured
feedback or launch phase/step events are not duplicated. INFO/DEBUG diagnostics
are not automatically persisted by this bridge.

Library use preserves the application's logging configuration. Its legacy log
records must pass their normal logger levels, and background work must propagate
operation context. Isolated daemon workers enable PROGRESS and bind context-free
legacy plugin threads to their single job. Structured reporters avoid both
limitations. Worker feedback follows the existing prompt-secret redaction path.

OCI Relay publishes receiver phases, upper-bound byte counters, extraction/cache
metrics, heartbeats, and verification outcomes. ColdSnap publishes preparation
stages and controller span hierarchies, including host/unit identity and duration,
with controller outcomes checked against the execution result and receipt.

Scheduled benchmarks share this interface. All frameworks, including llama-benchy,
report task start/finish with source `benchmark.<framework_name>`. Tool-eval-bench
also reports live scenario counts and outcomes. A new activity ID identifies each
attempt, and benchmark/task IDs are scalar metrics. Direct benchmark callbacks
receive the same envelope as `task_progress`; daemon delivery retains one copy.
See [benchmark progress](BENCHMARK_API.md#scenario-progress) for counter semantics
and the optional framework parser interface.
