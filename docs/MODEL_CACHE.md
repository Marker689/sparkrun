# Model cache validation and distribution

Manifest preparation binds a model's endpoint, immutable commit, selected files,
checksums and runtime paths together. Core validates every required host before
launch; a download, rsync exit code or plugin success receipt is insufficient.

## Default behavior and policy

No configuration change is needed to adopt manifests. The default `legacy`
policy uses metadata validation with compatibility fallback. Optional settings
in `config.yaml`:

```yaml
model_cache_validation: legacy        # legacy (default) | metadata | checksum
model_file_selection: auto            # auto | all | safetensors
model_distribution_provider: auto     # auto | builtin | registered provider name
model_transfer_streams_per_host: 2     # concurrent rsync processes per destination
model_transfer_max_streams: 16         # total rsync processes per copy operation
# model_distribution_fallback: true   # optional override of provider policy
```

Both `legacy` and `metadata` resolve authoritative inventories, select required
files, check existence/type/readability/size, and persist manifests automatically.
Manifests are stored in `.sparkrun-model-artifacts/` under the HF cache home on
the inventory source and launch targets. Lookups cover both the requested ref and
resolved commit, so a later pinned recipe can reuse an inventory learned from an
unpinned launch. Inventory is saved before acquisition/repair, so even an
interrupted or failed preparation retains the expected file list for a later
offline run. Stored manifests describe expected files; they never replace
checking the current files or certify that a download completed.

The difference is what happens when inventory is unavailable:

- `legacy` uses saved inventory if available, warning when an online mutable-ref
  refresh fails. Without inventory it warns that the cache is unverified and uses
  the previous cache/download checks. This lets existing offline caches keep
  working and migrate automatically on a later successful online preparation.
- `metadata` requires authoritative inventory; missing inventory or a failed
  online mutable-ref refresh fails preparation.
- `checksum` has the same inventory requirement and additionally reads selected
  payloads to check authoritative SHA-256 or Git blob SHA-1 hashes. Missing
  digests are unverified, never successful checksum checks.

Every policy rejects invalid manifests and requires repair or failure for known
missing/damaged files. Repair, transport and workload access failures never
downgrade to the old checks. Selecting a model provider requires the strict
inventory contract even if the configured validation policy is `legacy`.

Named clusters can override those settings without changing global configuration:

```yaml
distribution:
  model:
    validation: metadata
    file_selection: auto
    provider: builtin
    streams_per_host: 2
    max_streams: 16
    # fallback: false
```

Existing `enabled`, `preserve_perms` and `skip_fan_out` remain compatible with the
compatibility fallback. Manifest preparation reobserves each host even on shared storage;
already-complete destinations need no copy. It does not recursively chown caches.
Its snapshot copies preserve times when supported, without preserving ownership
or permissions. Permission failures fail preparation rather than triggering a
large redownload under a different identity.

## Select files without redundant weights

With inventory available, the default `auto` selection reads the root
`model.safetensors.index.json` at the resolved commit and
includes every shard named by its `weight_map`. It excludes unreferenced root
checkpoints and recognized `original/` and `metal/` alternative weight payloads.
Configuration, tokenizer, license and other ancillary files remain included.
Unfamiliar nested component trees remain included to avoid discarding separate
vision/encoder weights. No shard count is inferred from safetensors filenames.
Without a root index, the complete repository inventory is selected.

This addresses [#142](https://github.com/spark-arena/sparkrun/issues/142) without
checking for a GPT-OSS model name. At GPT-OSS-120B commit
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, metadata and the index select 15 root
shards: about **65.28 GB instead of 195.76 GB**, avoiding about 130.49 GB of
alternative weights. This measurement uses inventory sizes, not a weight download.

Use `all` for a loader requiring additional weight formats; `safetensors` requires
a root index and fails if none exists. GGUF uses its quantization selection,
checks multipart completeness, and binds its selected weight and projector from
that same revision. It never picks an arbitrary cached snapshot at launch.

## Inspect, migrate and repair

These commands print structured JSON reports; inspection/import return nonzero
when a target is incomplete or unverified. `--host` is repeatable and uses the
configured SSH credentials; `--cache-dir` is the target's HF home, not `hub/`.

```sh
# Metadata only, offline and read-only; no recursive cache traversal.
sparkrun model-cache inspect org/model --revision COMMIT --cache-dir /models/hf

# Resolve authoritative inventory online (plus a small index, at most 16 MiB).
sparkrun model-cache resolve org/model --revision COMMIT --output manifest.json

# Import trusted inventory for an existing/offline cache. Import is not repair.
sparkrun model-cache import org/model --revision COMMIT --manifest manifest.json

# Repair exactly the missing/wrong-size files and revalidate each target.
sparkrun model-cache repair org/model --revision COMMIT --host node1 --host node2

# Explicit checksum audit/repair, including same-size corruption.
sparkrun model-cache inspect org/model --revision COMMIT --checksum
sparkrun model-cache repair org/model --revision COMMIT --checksum
```

Pass the same `--file-selection` and GGUF `--projector` used for preparation when
inspecting or importing. `repair` accepts `--transfer-mode`, `--local-cache-dir`,
`--offline` and `--dry-run`. Offline preparation uses saved/imported inventory and
existing controller/cluster files; it never calls the Hub or installs tools. A
manifest-less offline cache is **unverified**, not implicitly valid or corrupt;
only `legacy` launch/distribution may proceed using the previous presence checks.
Dry-run makes no metadata requests, probes or cache writes. Origin resolution
for mutable refs is online even when the payload is already cached. A pinned
commit with saved inventory needs no origin request.

The Python surface is `sparkrun.api.model_cache`: `resolve_manifest`,
`inspect_model`, `import_manifest`, and `repair_model`, with typed requests,
manifests, reports and prepared bindings. `api.materialize()` remains read-only.

For a pre-placed directory, put the trusted manifest at
`<directory>/.sparkrun-model-manifest.json`, with inventory paths relative to that
directory. For a pre-placed GGUF file, use
`<file>.sparkrun-model-manifest.json`, paths relative to its parent, and set
`weight_path` to that filename. These inventories are checked before launch,
including under the default policy; missing inventories produce an explicit unverified
migration warning. Core does not acquire or repair pre-placed paths automatically.

## Usage, retention and LRU cleanup

Model preparation records usage beside each cache in
`HF_HOME/.sparkrun-model-artifacts/usage/`. These mutable records are separate
from the immutable file manifests. They track preparation, acquisition,
distribution-source use, launch attempts and confirmed workload release, keyed
by repository and immutable revision. Inspection and manifest import do not
refresh usage or enroll unknown files in cleanup. Preparation needs a writable
Sparkrun metadata directory, including for a warm cache.

The records belong to the cache host, not the controller's daemon database.
Delegated downloads and fan-out therefore record usage on the head and workers
without storing model weights on the controller. Shared cache mounts share the
same usage records and claims; each host's inventory report is a view of that
cache, not an additional copy to add to a cluster-wide byte total.

```sh
# Read-only JSON inventory, including untracked snapshots and protection reasons.
sparkrun model-cache list --host node1 --host node2 --cache-dir /models/hf

# Preview LRU cleanup (the default); a byte budget applies per cache.
sparkrun model-cache prune --host node1 --cache-dir /models/hf \
  --min-age-days 7 --max-bytes 536870912000

# Explicitly apply the same policy. Nothing deletes automatically by default.
sparkrun model-cache prune --host node1 --cache-dir /models/hf \
  --min-age-days 7 --max-bytes 536870912000 --apply

# Keep an immutable revision; use --remove to remove its retention pin.
sparkrun model-cache pin org/model --revision FULL_COMMIT \
  --host node1 --cache-dir /models/hf
```

`prune` defaults to a dry run and a seven-day minimum age. Without a byte
target it selects all eligible revisions older than that age. `--max-bytes`
budgets private allocations in tracked snapshots, excluding canonical/shared
HF blobs and untracked data. `--min-free-bytes` instead targets actual free
space on the cache filesystem. With both targets, cleanup continues until
both are met or no eligible revision remains. The CLI exits nonzero if a
target cannot be met; it never overrides protections to reach a target.

Reports separate logical file sizes, reclaimable snapshot allocations, planned
removals and filesystem free space. Snapshot allocation counts exclude symbolic
links and multiply linked files. They are estimates of disk reclamation, not a
promise about filesystem compression, reflinks or concurrent disk activity;
`free_bytes_after` is measured again when applying cleanup. Dry-run free-space
projections use the allocation estimates. Shared HF blobs are **never deleted**.
Removing a symlink-based snapshot may therefore reclaim no weight bytes.

Only tracked revisions with a known file inventory are eligible. Unknown
snapshots, unexpected files, pinned revisions, busy publication locks, invalid
metadata and active workload claims are protected. Arbitrary pre-placed model
directories are outside the cleanup scope. To enroll an existing HF snapshot,
prepare/repair it successfully; importing a manifest alone does not grant
cleanup ownership. Select `--file-selection all` if all repository files need
to be included in that preparation.

Claims are written before preparation releases its repository locks and remain
after the launching CLI exits. All prepared models, including draft models and
projectors in their selected inventory, are covered. A manifest-less legacy
launch conservatively protects every previously tracked revision of its model.
Confirmed `stop`/`stop-all` releases the corresponding claims and restarts the
retention age. A failed or interrupted activation retains its claims; neither
age nor a dead controller proves that remote workloads stopped.

Retry a normal stop to reconcile a known workload. If job metadata is missing,
inspect **every host listed in the outstanding claim**, stop the workload and
any in-flight launch, then explicitly release the exact claim shown by `list`:

```sh
sparkrun model-cache reconcile org/model --revision FULL_COMMIT \
  --claim CLAIM_ID --host node1 --cache-dir /models/hf --confirm-inactive
```

Reconciliation removes a protection record, not model files, and refreshes its
retention age. The confirmation is an operator assertion; this command does
not infer process inactivity. External applications and older Sparkrun
processes do not participate in these claims/locks: pin revisions they use,
or leave automatic cleanup disabled on their shared caches.

Optional automatic cleanup runs before manifest-based model preparation:

```yaml
model_cache_prune:
  enabled: false                  # opt in explicitly
  min_age_days: 7
  min_free_bytes: 107374182400     # 100 GiB free-space target per cache
  # max_bytes: 536870912000       # optional 500 GiB private snapshot budget
```

Enabling it requires at least one byte target. It reuses the explicit prune
implementation, excludes the repository being prepared, and retains pins and
workload claims. It warns if protected content prevents meeting the target;
this is not a guarantee that a subsequent download will fit. It does not run
on dry-run launches, and no background daemon is required. Cluster-seed
retention can currently be implemented by pinning the revision on the desired
head host; there is no automatic cluster-wide replica placement policy.

Inventory and cleanup use Python 3 and standard filesystem tools on Linux cache
hosts, through the existing SSH/host session. They never contact the model
origin or install tools. Existing metadata inspection remains independent of
this Python helper. CLI and daemon callers use the same Python API:
`inventory_cache`, `prune_cache`, `pin_revision`, and `reconcile_claim`.
The daemon exposes `model_cache.list`, `model_cache.prune`, `model_cache.pin`
and `model_cache.reconcile` (and matching `/v1/model_cache/...` routes).
Prune and reconciliation require admin scope; prune defaults to dry-run over
the API as well.

## Parallel model copies

The builtin manifest transport copies to multiple destinations concurrently.
Each destination can use two rsync processes by default, with sixteen processes
in total per copy operation (one source). Set `model_transfer_streams_per_host`
and `model_transfer_max_streams` in global configuration, or `streams_per_host`
and `max_streams` under a cluster's `distribution.model`. Values must be positive
integers. Set both to `1` for serial transfers. These limits apply to the builtin
manifest copy path; plugins and legacy inventory-unavailable fallback retain
their own transfer behavior. Separate operations have separate limits.

Only files that fail the selected validation policy enter the queue. Files at
least 64 MiB run as individual jobs; smaller files are grouped into batches up
to 64 MiB. Admission rotates across destinations, with larger files first.
Multiple streams use independent SSH connections over the selected transfer
addresses, retaining SSH authentication and host-key verification. Rsync
compression is off; bulk SSH connections disable multiplexing and default
to no compression.

Shared-cache aliases identified by the held publication leases receive one
copy. Jobs use disjoint file lists, retain rsync partial files for retries, and
replace snapshot files atomically. Cancellation and deadlines stop owned copy
processes without closing a borrowed session or releasing publication leases
before workers finish. Core revalidates every target before publishing runtime
bindings, including aliases. A single large file stays a single transfer;
this scheduler does not split GGUF files into byte ranges.

CLI and API feedback report aggregate/per-host logical bytes processed,
active stream counts, and logical bytes per second. Counters include data
reused by rsync and remain monotonic across retries; they do not measure bytes
sent on the network. Completion of a copy is followed by core validation.
Tune stream counts against actual disk and network throughput; additional
streams cannot exceed the source NIC or storage bandwidth.

## Hugging Face transports

Weight downloads prefer Xet when the repository supports it and `hf_xet` is
available; the pinned Hub dependency installs it on supported architectures.
Xet handles chunk-based transfers, and cache-backed downloads can share blobs
across repositories. Snapshot repair uses a private staging directory and
still verifies the selected files before launch.

Hub uses HTTP when Xet is unavailable, when the file is not Xet-backed, or when
`HF_HUB_DISABLE_XET=1` is set in the downloading process. It does not silently
retry every failed Xet transfer over HTTP: authentication, access, integrity,
and transfer failures remain failures. Set the override on the controller or
remote downloading host as appropriate. An offline run does not try either
network transport.

Advisory metadata reads deliberately use bounded HTTP instead of Xet's separate
Rust transport. The overall metadata deadline remains in force even when the
HTTP library retries a request. Hub 2.x uses `httpx2`; Sparkrun takes its timeout
and exception types from `huggingface_hub.utils.httpx` and preserves Hub's client
factory, redirect policy, and event hooks. HTTPS uses the OS trust store by
default; `SSL_CERT_FILE` / `SSL_CERT_DIR` can supply custom trust on the process
performing the download. Certificate verification remains enabled.

Legacy remote cache acquisition supports both `hf` and `huggingface-cli`. It
retains the older command when both are installed, preserving its argument
format, and pins its isolated `uvx` fallback to the same Hub version as the
manifest adapter.

## Transfer and launch guarantees

Local, push, delegated and per-node pull share preparation. Delegated acquisition
runs on the designated head and need not contact the Hub from the controller.
Each distribution entry has its own revision, including speculative draft models.
Required metadata failures are separate from advisory metadata budgets. Remote
acquisition uses the pinned `huggingface_hub==2.2.0` adapter; offline observation
requires only Bash and ordinary Linux utilities, not Python or uv on targets.

Copies enumerate selected snapshot entries and dereference them into regular
snapshot files. They exclude other revisions, live blob-directory churn, locks,
partial downloads and unrelated cache state. This representation works alongside
classic and shared-blob HF layouts, without copying both snapshots and blobs.
Repair stages one affected file and atomically replaces its snapshot entry; it
never deletes a shared canonical blob. Copied files are not skipped based only
on equal sizes when a checksum repair has identified corruption.

Core holds per-cache/repository `flock` leases across acquisition and provider
calls. An operation nonce recognizes leases reached through shared-cache aliases;
complete aliases are omitted from transfer after observation. External downloaders
and garbage collectors do not participate in these leases. A final observation is
still required, and no receipt promises protection against later external changes.

The reproduced self-copy defect underlying the PR #308 review is also guarded in
the existing local rsync wrapper using directory identity, including symlink
aliases. Vanished-source retries are bounded to one and require a still-complete
manifest source. Diagnostics from the first failure remain logged, the original
time budget remains in force, and target validation is mandatory after success.
`SPARKRUN_NO_RSYNC_RETRY=1` disables retries. The resilience idea follows
[Aisoipheo's PR #308](https://github.com/spark-arena/sparkrun/pull/308); this does
not import that PR's observed-file verifier or repo-wide success marker.

Standard vLLM and SGLang commands consume the prepared snapshot while retaining
the recipe's served model name and immutable revision. Draft revision pins remain
independent. Custom commands must consume `{model}` or the validated revision;
a literal mutable ref cannot bypass the binding. Env `launch.model_path` and
`launch.model_revision` consume the same binding. Hooks additionally receive
`SPARKRUN_MODEL_PATH` (workload path, blank for controller post-commands) and the
resolved `SPARKRUN_MODEL_REVISION`. Compilation/runtime caches remain separate
and are not distributed with model payloads.

Docker access checks run the selected image with its final UID and structured
mounts, without GPU allocation or the image entrypoint. Local checks use the
host execution identity. Unavailable or shadowed mounts fail before eviction.
Custom executors need an equivalent `model_access_command` capability; raw mount
options cannot bypass mapping checks. Delegating runtimes that own preparation
retain that responsibility. ColdSnap native restores still bypass HF preparation
through their asset policy; capture/safetensors preparation uses the normal path.

## Rollout evidence and remaining acceptance

Tiny real-rsync fixtures cover missing files, dangling shared links, aliases,
same-size repair, offline reuse and all four modes. Provider tests reject false
success, changed identities and incomplete outcomes. Workload tests cover pinned
commands, env reuse, UID options and mount shadowing.

Local warm preparation (three samples, two observations per host) measured median
41 ms / 60 ms / 186 ms for 10 / 100 / 1,000 selected files using saved pinned
inventory under the default compatibility policy. It hashes no weights
and batches metadata calls; these are local filesystem measurements, not NFS or
multi-node startup promises. Actual NFS locking/latency and large-model startup
remain deployment acceptance checks before removing the default compatibility fallback.

## Retention report contract and lifecycle

Python retention methods return `CacheReport` objects with typed `CacheEntry`
entries; mapping access is also supported. CLI JSON and daemon results use the
same fields. `reason_code` is the stable machine-readable eligibility outcome;
`reason` is explanatory text. `budget_scope: private_managed_snapshots` makes the
scope of `max_bytes` explicit. `reclaimed_bytes_kind: allocated_bytes_estimate`
distinguishes removed allocation estimates from actual filesystem free space;
`free_bytes_after_kind` identifies a dry-run projection or measured result.
A successful prune operation can return `target_met: false` when protection or
shared blobs prevent reaching the requested target. CLI reports that condition
with a nonzero exit; API clients inspect `target_met` independently of job success.
Model-origin operations accept `endpoint` consistently through Python and daemon
requests; pin/reconcile reject an origin mismatch against the saved receipt.

Confirmed stop, stop-all and stop-after-hooks share claim release and metadata
finalization. Failed or partial teardown preserves recovery metadata. Age-based
metadata pruning never discards outstanding claim references. Cache claims have
no expiry: an unavailable controller is not proof of a stopped workload.

A model revision identifies an immutable origin commit. A plugin generation
identifies one managed plugin installation. Neither is a daemon job ID. Cache
claims protect workload usage on cache hosts; runtime leases pin controller code,
and operation reservations coordinate concurrent mutations. Their storage and
lifetimes remain separate.

Cache `pin` and `reconcile` also accept `--endpoint` (or `HF_ENDPOINT`), matching
the model-origin argument used by inspection and repair. A different origin cannot
mutate another origin's usage receipt merely because repo/revision names match.
