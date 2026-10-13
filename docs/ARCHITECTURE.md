# Architecture (Multiplatform)

Contributor-facing map of the layers introduced for multiplatform support. Pair
with `MULTIPLATFORM.md`, `EXECUTORS.md`, and `SECURITY.md` for the deeper dives.
The shared workload log reader, resolved targets, paging, and daemon filters are
documented in [LOGGING_API.md](LOGGING_API.md).

## Layer diagram

```
                       +---------------------------+
                       | cli.* / launcher entry    |
                       +-------------+-------------+
                                     |
                                     v
+--------------------+   +-----------+-----------+   +--------------------+
| core/recipe.py     |   | core/launcher.py      |   | core/config.py     |
| (executor field)   |-->| launch_inference()    |<--| SparkrunConfig     |
+--------------------+   |  - resolve_recipe_..  |   +--------------------+
                         |  - resolve_per_host_  |
                         |    backends           |
                         |  - resolve_executor   |
                         |  - platform validate  |
                         +-----------+-----------+
                                     |
                +--------------------+--------------------+
                v                                         v
   +-----------------------+                  +-----------------------+
   | core/hardware_probe.py|                  | core/backend_select.py|
   | probe_host (combined  |                  | select_backends() ->  |
   |  accel + IB SSH probe)|                  | BackendBundle         |
   +-----------+-----------+                  +-----------+-----------+
               |                                          |
               v                                          v
   +-----------------------+         +---------------------------------+
   | core/hardware.py      |         | orchestration/collectives/      |
   | HostHardware / Accel  |         |  base.py  (CollectiveBackend)   |
   |  Spec / capabilities  |         |  nccl.py  (NVIDIA, default)     |
   +-----------+-----------+         |  rccl.py  (AMD scaffold)        |
               |                     |  hccl.py  (Intel scaffold)      |
               v                     +---------------------------------+
   +-----------------------+
   | core/fingerprint.py   |         +---------------------------------+
   | core/scheduler.py     |         | orchestration/executors/        |
   | core/layout.py        |         |  _base.py (Executor ABC)        |
   +-----------------------+         |  docker.py (default)            |
                                     |  local.py  (experimental)       |
                                     |  plugins/k8s (experimental)    |
                                     +---------------------------------+
                                                  ^
                                                  |
                                     +---------------------------------+
                                     | platforms/                      |
                                     |  base.py  (HardwarePlatformPlug)|
                                     |  dgx_spark.py                   |
                                     |  nvidia_generic.py              |
                                     +---------------------------------+
```

## Module map (new + reshuffled layers)

### `core/`

| Module               | Purpose                                                                                                                                      |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `hardware.py`        | `AcceleratorSpec`, `HostHardware`, `default_dgx_spark_hardware()`. Vendor/model/capability records consumed by every multiplatform path.     |
| `fingerprint.py`     | Generates accelerator probes, parses their KV output into `HostHardware`, and computes stable hardware fingerprints.  |
| `hardware_probe.py`  | `probe_host()` / `probe_hosts()` — single SSH script that emits both the fingerprint section and the IB section, split by sentinel markers.  |
| `backend_select.py`  | `select_backends(host_hardware) -> BackendBundle`. Raises `NoMatchingBackendError` when no collective matches.                               |
| `scheduler.py`       | Placement records and scheduler contracts; `schedulers/greedy.py:pack()` implements capacity packing and explicit layouts.      |
| `layout.py`          | `RecipeLayout` / `Placement` dataclasses (per-rank `(host, local_gpu)`). Parsed from recipe `layout:` block.                                 |
| `image_preparation.py` | Shared builder, per-node image-plan, distribution-entry, image-only staging, and immutable identity receipts used by launch and integrations. |
| `launcher.py`        | `launch_inference()`. Resolves trust, per-host backends, runs the central compatibility check, threads `BackendBundle` to `runtime.run()`.   |

### `orchestration/`

| Subpath                       | Purpose                                                                                                                                  |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `executor.py`                 | Public facade. Exports `Executor`, `ExecutorConfig`, `EXT_EXECUTOR`, `get_executor()`, `list_executors()`, `resolve_executor()`.         |
| `executors/_base.py`          | `Executor` ABC, `ExecutorConfig` dataclass, `_registered_executor_names()` SAF lookup, `accelerator_vendor_for()` helper.                |
| `executors/docker.py`         | `DockerExecutor` (default). Owns `DOCKER_DEFAULTS` and the `rootless` / `auto_user` adjustment layer.                                    |
| `executors/local.py`          | `LocalExecutor` — experimental, native subprocess via `setsid`, pid/log-file lifecycle. No images.                                       |
| `collectives/base.py`         | `CollectiveBackend` ABC, `UnsupportedCollectiveError`.                                                                                   |
| `collectives/nccl.py`         | `NcclBackend` — wraps `infiniband.generate_nccl_env` / `generate_ring_nccl_overrides`. Byte-identical to legacy DGX output.               |
| `collectives/rccl.py`         | `RcclBackend` scaffold (AMD). Raises `NotImplementedError`.                                                                              |
| `collectives/hccl.py`         | `HcclBackend` scaffold (Intel Gaudi). Raises `NotImplementedError`.                                                                      |
| `comm_env.py`                 | `ClusterCommEnv` dataclass. Carries per-host env produced by collective backends through `runtime.run()`.                                |
| `infiniband.py`               | Legacy Mellanox IB probe parser + NCCL env emitter. Still authoritative; `NcclBackend` delegates here.                                   |

### `plugins/k8s/`

The optional Kubernetes integration owns its executor and config model, setup
commands, API, kubectl helpers, and Kueue/JobSet launch path. Enable it with
`sparkrun plugins enable k8s`; child features register only when that plugin loads.
See the [plugin guide](../src/sparkrun/plugins/k8s/README.md).

### `platforms/`

| Module               | Purpose                                                                                                                                            |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `base.py`            | `HardwarePlatformPlugin` ABC, `EXT_PLATFORM` constant. `matches()`, `accelerator_vendor()`, `collective_backend()`, `default_image()`, `validate_host()`. |
| `__init__.py`        | `_REGISTRY` ordered list (most-specific first). `register_platform()`, `iter_platforms()`, `resolve_platform()`.                                   |
| `dgx_spark.py`       | `DgxSparkPlatform` — matches `vendor=nvidia, model=gb10`. Publishes Spark Arena image defaults. `validate_host` warns on missing `rdma:roce-v2`.   |
| `nvidia_generic.py`  | `GenericNvidiaPlatform` — catch-all for any NVIDIA accelerator. Publishes upstream images. `validate_host` warns when no NVIDIA accelerator found. |

## Resolution chains

### Executor

`orchestration/executor.py:resolve_executor()` resolves the selector and builds
one executor per launch. See the [executor resolution chain](EXECUTORS.md#resolution-chain)
for precedence, including cluster settings, builder activation, runtime defaults,
and hardware-platform defaults. Application defaults enter through `SparkrunConfig`.

Explicit unknown or disabled selectors raise `ExecutorUnavailableError`; they do
not silently select Docker. SAF discovers executor classes. Only Docker and Local
have a static fallback for bootstrap-free test harnesses; Kubernetes is supplied
by its plugin.

### Collective backend

`orchestration/collectives/__init__.py:get_backend(vendor)` maps:

- `vendor in {"nvidia", None}` → `NcclBackend()`
- `vendor == "amd"` → `RcclBackend()` (scaffold; raises on `env_for_host`)
- `vendor == "intel"` → `HcclBackend()` (scaffold; raises on `env_for_host`)
- everything else → `UnsupportedCollectiveError`

`core/backend_select.py:select_backends(HostHardware)` derives the vendor via
`accelerator_vendor_for(...)`, calls `get_backend()`, and packages the result
into a `BackendBundle(accelerator_vendor, collective)`. `launcher.py`
materializes this once per host before calling `runtime.run(..., backends=...)`.

### Platform

`platforms/resolve_platform(host_hardware)` iterates `_REGISTRY` in order and
returns the first plugin whose `matches()` returns true. Ordering is
most-specific first: `DgxSparkPlatform` (`model="gb10"`) pre-empts
`GenericNvidiaPlatform`. `launcher.py` calls `validate_host()` per host and
logs warnings (does not raise).

## Launch flow (multiplatform additions)

`core/launcher.py:launch_inference()` order:

1. Resolve trust (`resolve_recipe_trust`).
2. SSH kwargs + transfer mode + serve port.
3. Generate `cluster_id`.
4. Validate declared per-host image/runtime/builder constraints.
5. Expand `recipe.mods` into `pre_exec`.
6. Shared image preparation (`prepare_images`): resolve per-host runtime/platform
   defaults beneath explicit recipe images, apply an optional builder, then derive
   the final image plan and distribution entries.
7. **`resolve_per_host_backends(host_list, cluster=...)`** — returns
   `dict[host, BackendBundle]`. Hosts that fail to resolve are dropped from the
   dict; per-host env then flows through
   `_cluster_ops.resolve_comm_env(ctx, comm_env, backends)`, which falls back
   to the legacy NCCL generator for those hosts (byte-identical for NVIDIA).
   The old `resolve_ib_env` wrapper has been removed.
8. **Compatibility check** — for runtimes with `requires_capability`, walk every
   host, accumulate errors, raise `IncompatibleHardwareError` if any.
9. **Platform validation** — `resolve_platform(hw).validate_host(hw)` per host;
   warnings logged, never raised.
10. Persist job metadata (includes serialized `backends`).
11. Distribution phase (the prepared container image plan + model + IB probe).
12. Tuning sync.
13. GGUF resolution.
14. `runtime.generate_command()`.
15. Page-cache clear.
16. `resolve_executor(...)` — single sanctioned entry point.
17. `runtime.run(..., backends=backends, executor=executor, trust=recipe_trusted)`.

## Key reference files

- `src/sparkrun/core/launcher.py:130-170` — `resolve_per_host_backends`.
- `src/sparkrun/core/launcher.py:381-422` — backend resolution + compatibility
  + platform validation block.
- `src/sparkrun/orchestration/executor.py:260-309` — `resolve_executor` chain.
- `src/sparkrun/orchestration/executors/_base.py:59-207` — `ExecutorConfig`
  dataclass and `from_chain` parsing.
- `src/sparkrun/core/backend_select.py:42-83` — `BackendBundle` + `select_backends`.
- `src/sparkrun/core/hardware_probe.py:299-416` — `probe_host` / `probe_hosts`
  and the combined script + sentinel splitter.
- `src/sparkrun/platforms/__init__.py:31-80` — registry + `resolve_platform`.

## Where new abstractions plug in

| Adding…                          | Touch                                                                                                                                   |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| A new executor                   | `orchestration/executors/<name>.py` (subclass `Executor`, set `executor_name`). SAF discovers it via `find_types_in_modules`.           |
| A new collective backend         | `orchestration/collectives/<name>.py` (subclass `CollectiveBackend`); register in `collectives/__init__._BY_VENDOR`.                    |
| A new platform                   | `platforms/<name>.py` (subclass `HardwarePlatformPlugin`); register via `register_platform(MyPlatform(), prepend=True)` if it must win. |
| A new accelerator vendor probe   | Extend `core/hardware_probe.py` (combined script), `core/fingerprint.py` (parsing), and `core/hardware.py` (vendor in `vendors`).       |
| A new runtime                    | `runtimes/<name>.py` (subclass `RuntimePlugin`); use `_make_node_command_args` template for native multi-node, set `requires_capability`. |


## Application profiles

`application.run_cli()` and `application.initialize()` select an immutable
`core.application_profile.ApplicationProfile` before configuration, SAF, feature
resolution or plugin registration. One profile is active per process. Implicit
entry points choose built-in Sparkrun, and children can restore an installed
profile reference before importing application consumers.

`core.config` keeps raw user settings separate from effective application profile
defaults. Call-time path/environment accessors replace operational copied
constants. `core.installed_plugins` discovers metadata without importing disabled
integrations, then uses the existing plugin module ABI and tracks failures,
provenance and implementation conflicts. Recipe and extension identities remain
shared; `core.ownership` keeps deployment ownership separate from recipe intent.
`core.application_identity` provides serializable application/controller descriptors
for plugins and shared services. One opaque controller identity is created lazily
per application and canonical config directory, shared across config filenames
and processes, and recorded in new job metadata. It is independent of telemetry
identity and recipe fingerprints and cannot be overridden in config.

Application profile registry/update policy is explicit and separate from plugin trust
and core feature maturity. Version diagnostics distinguish the running product,
core dependency and loaded integrations. The public contract, path policy,
legacy compatibility and wheel evidence are documented in
[APPLICATION_PROFILES.md](APPLICATION_PROFILES.md).


## Source type checks

Run the pinned checker over all source with the locked development environment:

```bash
uv sync --locked --group dev
uv run python scripts/check-types.py
```

The gate compares full Pyright diagnostics with `pyright-baseline.json`. It fails
on additional diagnostics and on resolved entries still recorded in the baseline,
so type cleanup ratchets the accepted debt downward. Line shifts alone do not
invalidate entries; duplicate diagnostics in one file retain their counts.

After fixing diagnostics, regenerate and review the baseline diff:

```bash
uv run python scripts/check-types.py --update-baseline
```

Regeneration records the current checker version and diagnostics; it is not
permission to accept new errors. Direct `uv run pyright src/sparkrun` still reports every remaining issue.
The separate source-type CI workflow checks the baseline with Python 3.12; runtime
and installed-wheel tests continue covering both supported Python versions.


## Dormant native monitor backend

Bash remains the working default for host monitoring. The `nv-monitor` backend
is dormant and this distribution does not ship its binary bundle. Selecting it
fails synchronously with instructions to use bash, before monitor threads start.
The old implementation needs separate rework before it can be supported again:
binary packaging and provenance, application/controller ownership, deployment and
cleanup, and telemetry-provider lifecycle tests. Its remaining optional-import
warning is recorded in the type baseline; the warning does not establish support.
