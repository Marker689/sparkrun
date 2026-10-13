# Executors

How sparkrun selects and configures an executor for a launch. Three are shipped:

| Selector | Class            | Status       | Notes                                                                   |
|----------|------------------|--------------|-------------------------------------------------------------------------|
| `docker` | `DockerExecutor` | Stable       | Default. Used by every previously-released launch path.                 |
| `local`  | `LocalExecutor`  | Alpha        | Native subprocess (no container). Hand-coded process-group lifecycle.    |
| `k8s`    | `K8sExecutor`    | Experimental | Native JobSet lifecycle; drops Docker-specific options.                    |

## Resolution chain

`orchestration/executor.py:resolve_executor()` is the single sanctioned entry
point. It layers (highest priority first):

1. **CLI overrides** — `cli_overrides` dict (`-o executor=local`, `-o
   k8s_namespace=...`, etc.).
2. **Recipe** — `recipe.executor` (selector) + `recipe.executor_config` (dict).
3. **Builder activation** — `builder.default_env_file()` supplies only `env_file`,
   below recipe/CLI overrides and above the cluster environment.
4. **Cluster** — `cluster.executor` (selector) + `cluster.executor_config` (dict).
5. **Runtime executor selector** — `runtime.default_executor()` (`None` by default; runtimes can force a non-Docker executor).
6. **Per-executor adjustments** — `cls.apply_runtime_adjustments(rootless=,
   auto_user=, defaults=)`. Docker reads these here; Local/K8s ignore.
   `defaults` contains the lower-priority configuration layers, allowing Docker
   to retain their security options while adding `no-new-privileges`.
7. **Runtime executor-config defaults** — `runtime.default_executor_config()` (`{}` by default; runtimes can set overridable executor defaults).
8. **`SparkrunConfig`** — `config.default_executor` + `config.executor_config`.
9. **Platform** — `platform.default_executor_config(<name>)` for the platform
   resolved from the launching host's hardware (`host_hardware=`, the head
   node's). Hardware-conditional container plumbing that should still lose to
   anything the user wrote — e.g. DGX Spark pins `gpu_access_mode: gpus`.
   Dropped entirely when no hardware is threaded (naming / teardown / log paths).
10. **Per-executor defaults** — `cls.default_config()` (e.g. `DOCKER_DEFAULTS`).
11. **Dataclass field defaults** — `ExecutorConfig` declares the floor.

A selector that is unknown — or names a real executor whose feature flag is
off — raises `ExecutorUnavailableError` naming the flag to enable. Resolution
never silently degrades to `"docker"`: running an explicitly-requested workload
on the wrong backend is worse than failing loudly.

When *no* layer names an executor, `_default_executor_name` returns `docker`
when enabled, else the sole remaining enabled executor, else raises. So
disabling `executor.docker` is honored rather than being overridden by a
hardcoded baseline.

The set of known selectors is queried from SAF via
`get_extensions(EXT_EXECUTOR, v=v)`; the hardcoded `_KNOWN_EXECUTORS` set was
retired in 0.3.0.

## SAF discovery

```python
EXT_EXECUTOR = "sparkrun.executor"
```

`core/bootstrap.py` calls
`find_types_in_modules("sparkrun.orchestration.executors", Executor)` and
registers each discovered subclass. Subclasses must set:

- `executor_name: ClassVar[str]` — the selector string (must be unique).
- `is_multi_extension(v)` → `True` and `is_enabled(v)` → `False` (inherited;
  prevents SAF's single-extension cache from short-circuiting).

Look-up helpers:

- `get_executor(name, v=None) -> type[Executor]` — returns the class, not an
  instance. Falls back to a static map (`docker`/`local`) when SAF isn't
  initialized (test paths). Kubernetes requires plugin registration.
- `list_executors(v=None) -> list[str]` — sorted selectors.

## `ExecutorConfig.from_chain` field reference

Every field is parsed from a chain layer with the same name (`chain.get(key)`).
Bool fields use `ext_parse_bool` and preserve explicit `False`. Plain string
fields preserve empty strings; nullable strings fall back when falsy. List fields
promote bare strings to single-item lists and normalize empty lists to `None`.
`entrypoint: ""` explicitly clears the image entrypoint. The table describes
resolved executor defaults; direct `ExecutorConfig()` construction uses its
dataclass floor (for example, `shm_size="25gb"` versus Docker's `"32gb"`).

### Common (Docker + K8s read; Local ignores most)

| Field                 | Type        | Consumed by         | Default       | Notes                                                                                          |
|-----------------------|-------------|---------------------|---------------|------------------------------------------------------------------------------------------------|
| `executor`            | str         | resolver            | `"docker"`    | Selector. Also accepts `executor_type` as an alias for forward compat.                         |
| `auto_remove`         | bool        | Docker              | `True`        | Adds `--rm`. Force-flipped to `False` when `restart_policy` is set.                            |
| `restart_policy`      | str?        | Docker              | `None`        | Docker `--restart` value.                                                                      |
| `privileged`          | bool        | Docker              | `True`        | Off in rootless mode.                                                                          |
| `gpus`                | str         | Docker, Local, K8s  | `"all"`       | The GPU spec. Docker spells it per `gpu_access_mode`. Local translates `device=0,2` → `CUDA_VISIBLE_DEVICES`. K8s extracts a count for `nvidia.com/gpu`. |
| `gpu_access_mode`     | str         | Docker              | `"cdi"`       | `cdi` → `--device nvidia.com/gpu=<id>`; `gpus` → `--gpus <gpus>`. Platform-defaulted (DGX Spark pins `gpus`). |
| `ipc`                 | str         | Docker              | `"shareable"` | `--ipc`. Own IPC namespace + own `/dev/shm`, joinable via `--ipc=container:<name>`. **Not `host`**: see below. K8s drops. |
| `shm_size`            | str         | Docker              | `"32gb"`      | `--shm-size`. Only applies when `ipc` is not `host` (Docker ignores it otherwise). K8s drops.  |
| `network`             | str         | Docker              | `"host"`      | `--network`. K8s drops.                                                                        |
| `user`                | str?        | Docker              | `None`        | `--user`. Sentinel `"$SHELL_USER"` expands to `$(id -u):$(id -g)` + bind-mounts passwd/group.  |
| `security_opt`        | list[str]?  | Docker              | `None`        | Repeated `--security-opt`. Docker adds the io_uring profile; rootless mode also defaults to `no-new-privileges`.               |
| `cap_add`             | list[str]?  | Docker              | `None`        | Repeated `--cap-add`.                                                                          |
| `ulimit`              | list[str]?  | Docker              | `None`        | Repeated `--ulimit`. Rootless mode sets `memlock=-1:-1`, `stack=67108864`.                     |
| `devices`             | list[str]?  | Docker              | `None`        | Repeated `--device`. Rootless mode adds `/dev/infiniband`.                                     |
| `memory_limit`        | str?        | Docker, K8s         | `None`        | Docker `--memory`; K8s `--limits=memory=...`.                                                  |
| `labels`              | list[str]?  | Docker, K8s         | `None`        | Repeated `--label` / `--labels`.                                                               |
| `entrypoint`          | str?        | Docker, K8s         | `None`        | Docker emits `--entrypoint`; `""` clears the image ENTRYPOINT. K8s emits `--command`; `""` uses `bash -c`. |
| `accelerator_vendor`  | str?        | Docker              | `None`        | `nvidia` / `amd` / `intel` / `apple` / `cpu`. Drives accelerator-flag emission.                |

### Local-only (Docker / K8s ignore)

| Field             | Default                                        | Notes                                                                                                |
|-------------------|------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `working_dir`     | `None`                                         | `cd <working_dir>` before launch.                                                                    |
| `log_dir`         | `$HOME/.cache/sparkrun/local/logs`             | Per-container `<log_dir>/<container_name>.log`.                                                      |
| `log_file`        | `None`                                         | Optional shared append log; otherwise logs are per-container.                                                                  |
| `pid_dir`         | `$HOME/.cache/sparkrun/local/pids`             | Per-container `<pid_dir>/<container_name>.pid`.                                                      |
| `pid_file`        | `None`                                         | Legacy command recovery only. Managed operations reject it; use `pid_dir`.                                                                  |
| `env_file`        | `None`                                         | Sourced via `set -a; . <env_file>; set +a` before launch.                                            |
| `command_prefix`  | `None`                                         | Prepended verbatim (e.g. `nice -n 10 ionice -c2`).                                                   |

See the [native path, state, and lifecycle contract](#native-paths-state-and-lifecycle)
for required absolute/home-relative control paths, setup failures, ownership,
process-group status, and legacy recovery. Workload settings do not relocate PID/log control paths.

### K8s-only (Docker / Local ignore)

| Field                    | Default | Notes                                                                                       |
|--------------------------|---------|---------------------------------------------------------------------------------------------|
| `k8s_namespace`          | `None`  | `kubectl -n <ns>`.                                                                          |
| `k8s_context`            | `None`  | `kubectl --context <ctx>`.                                                                  |
| `k8s_node_selector`      | `None`  | `key=value[,key=value]`. Emitted as `--overrides` JSON because `--node-selector` was removed. |
| `k8s_image_pull_policy`  | `None`  | `--image-pull-policy`.                                                                      |
| `kubeconfig`             | `None`  | `--kubeconfig`.                                                                             |

## `DockerExecutor` (default)

`orchestration/executors/docker.py`. Owns `DOCKER_DEFAULTS` and the
`apply_runtime_adjustments(rootless=, auto_user=)` lever:

- `rootless=True` (default) → flips `privileged=False`, adds
  `no-new-privileges` / `memlock` ulimit / `/dev/infiniband` device, sets
  `auto_user="$SHELL_USER"` when paired with `auto_user=True`.
- `auto_user=True` (default) → `--user $(id -u):$(id -g)` + bind-mounts
  `/etc/passwd:/etc/passwd:ro` and `/etc/group:/etc/group:ro` (kernel reads UID
  names) and sets `HOME=/tmp`.

`_accelerator_opts()` emits device flags based on `accelerator_vendor`:

| Vendor      | Flags                                                              |
|-------------|--------------------------------------------------------------------|
| `nvidia`/None | per `gpu_access_mode` — see below                                |
| `amd`         | `--device /dev/kfd --device /dev/dri --group-add video`           |
| `intel`       | `--device /dev/accel`                                             |
| `apple`/`cpu` | (none — route to a non-Docker executor)                           |

### ENTRYPOINT preflight

`Executor.verify_command_passthrough(image, hosts)` is the second write-path
preflight alongside `verify_mount_sources` — that one asks "do the paths I will
mount exist on this substrate?", this one asks "will the command I append
actually run on this image?".

sparkrun composes every workload as `docker run <opts> <image> bash -c <b64 cmd>`,
so the command is always CMD *arguments* and the image's ENTRYPOINT decides what
happens to them. Two opposite idioms are in wide use and `docker image inspect`
cannot tell them apart — both are simply "a non-empty ENTRYPOINT":

- **passthrough** — `/opt/nvidia/nvidia_entrypoint.sh` and friends: do setup,
  then `exec "$@"`. Inherited by nearly every NGC-derived image, so this is the
  *common* case. Clearing it would skip the setup.
- **consuming** — `ENTRYPOINT ["vllm","serve"]`: the appended `bash -c …` is
  parsed as that program's own flags, so the workload never starts.

The verdict is therefore established empirically (`containers/entrypoint.py` +
`scripts/image_entrypoint_probe.sh`) rather than by inspection or an allowlist:

1. No ENTRYPOINT → `absent`, no container started (the cheap exit for most images).
2. Run the real argv shape and look for a **computed** sentinel on stdout. Found
   → `pass`. The sentinel is computed, not literal, because a consuming
   entrypoint typically echoes the argv it rejected — an echo can reproduce a
   literal token but never the evaluated one.
3. Not found → re-run byte-identically with `--entrypoint ''`. Only if *that*
   succeeds is the entrypoint provably the cause (`fail`); otherwise the fault
   is elsewhere — stale CDI, no GPU, no bash — and the verdict is `unknown`.

Only `fail` blocks, and `launcher._verify_image_command_passthrough` raises a
`RecipeError` naming both fixes (`executor_config: {entrypoint: ""}` or
`-o entrypoint=''`) rather than auto-clearing: the probe shows that clearing
*works*, not that it is *harmless* — a consuming entrypoint may also be doing
setup the workload needs.

It runs from `distribute_from_config`'s `after_container_sync` hook — the image
is resident on every target but the long, routinely-interrupted model sync has
not started yet. One host is probed (the verdict is a property of the image, and
distribution already established the image matches everywhere), under the
launch's own accelerator flags. Everything is fail-open: unreachable host,
timeout, unresolvable executor, or `SPARKRUN_NO_IMAGE_PROBE=1` all read as
"proceed". The base implementation returns `None`, so container-less (`local`)
and provider executors never block a launch.

### IPC namespace (`ipc`) and `shm_size`

The default is `shareable`, **not** `host`. The two settings are coupled: Docker
ignores `--shm-size` whenever `--ipc=host`, because the container then simply
gets the host's `/dev/shm` (typically 50% of RAM). Change one and you change
what the other means.

`ipc: host` is unsafe in sparkrun's default configuration. The container runs as
the SSH user (`--user $(id -u):$(id -g)`), so under a host IPC namespace every
POSIX semaphore and shared-memory segment the workload creates is a host file
owned by a regular UID. systemd-logind with `RemoveIPC=yes` — the Ubuntu 24.04 /
DGX OS default — deletes exactly those objects `UserStopDelaySec` (10s) after
that user's last login session ends. sparkrun launches detached (`docker exec -d`)
and then closes its SSH session, so a workload still in Python's multiprocessing
bootstrap when the reaper runs dies with:

```text
File "/usr/lib/python3.12/multiprocessing/synchronize.py", line 115, in __setstate__
    self._semlock = _multiprocessing.SemLock._rebuild(*state)
FileNotFoundError: [Errno 2] No such file or directory
```

A workload that gets past that point first survives indefinitely — unlinking a
name does not invalidate an already-open handle — which is why the failure is
intermittent, why longer startups (large models, graph capture) fail more often
than short ones, and why holding an unrelated SSH session open makes it vanish.
Running as root also masks it, since logind never reaps system UIDs.

A container-owned namespace is immune: logind walks `/dev/shm` itself, not
arbitrary mounts. `shareable` is preferred over `private` because it costs
nothing and still lets a second container on the same host join with
`--ipc=container:<name>`.

Keep `ipc: host` only for cross-*container* shared memory on one host (e.g.
intra-node NCCL between separate containers). When you do, either enable
lingering for the SSH user (`sudo loginctl enable-linger <user>`) or set
`RemoveIPC=no` in `/etc/systemd/logind.conf` on every host; `sparkrun setup
check` reports this combination, and the executor warns at launch. Note the
`local` executor has the same exposure by construction — it runs natively, with
no namespace to hide in.

**`ipc` is trust-gated on its value.** A recipe may narrow the namespace freely
(`private`, `shareable`, `none`), but an untrusted recipe — one from a
non-default registry or a URL — cannot select `host` or `container:<name>`
without `--trust`. Both reach *outside* the container: the host's `/dev/shm`
holds every other tenant's POSIX shared memory and semaphores, and a container
name is derivable from a cluster_id. See [SECURITY.md](SECURITY.md).

### NVIDIA GPU access mode

There are two ways to ask Docker for NVIDIA GPUs and neither works everywhere,
so `gpu_access_mode` selects between them:

| Mode           | Flags for `gpus: all`          | Flags for `gpus: device=0,1`                                       |
|----------------|--------------------------------|---------------------------------------------------------------------|
| `cdi` (default) | `--device nvidia.com/gpu=all` | `--device nvidia.com/gpu=0 --device nvidia.com/gpu=1`               |
| `gpus`          | `--gpus all`                  | `--gpus device=0,1` (value passed through verbatim)                 |

A falsy `gpus` still means "no GPU request" in both modes. An unrecognised mode
warns and falls back to `cdi`.

CDI (Container Device Interface, Docker >= 25) is the portable path and is
*required* on daemons that reject `--gpus` (e.g. Thunder Compute), but it
depends on a present, non-stale `/etc/cdi/nvidia.yaml` — the spec pins versioned
absolute paths, so a driver upgrade can leave it dangling and containers then
fail to start. `--gpus` resolves through the container runtime at launch instead.

`sparkrun setup check` selects the CDI probe and check only when the resolved
GPU access mode requires CDI and the hardware plan supports it. A missing or
empty spec is a **FAIL**; a present spec with missing referenced paths is a
**WARN**. Under `gpus`, the CDI probe and finding are omitted entirely. Switching
the executor to `cdi` makes them applicable again. An unresolved or unrecognized
mode retains the CDI requirement, matching Docker's fallback.

The default comes from the resolved hardware platform
(`HardwarePlatformPlugin.default_executor_config("docker")`), which sits just
above `DOCKER_DEFAULTS` in the chain: **DGX Spark / GB10 pins `gpus`**, every
other platform inherits `cdi`. Override anywhere higher — recipe, cluster, or
`config.yaml`:

```yaml
executor_config:
  gpu_access_mode: cdi
```

## `LocalExecutor` (experimental)

`orchestration/executors/local.py`. Native subprocess; no container.

### Native paths, state, and lifecycle

Managed native workloads require absolute or home-relative `pid_dir`, `log_dir`,
and `log_file` paths. Ordinary relative control paths are rejected during target
resolution and before launch/preflight script generation: their spelling cannot
identify the same saved destination across invocation directories. Defaults remain
home-relative. Legacy relative-path records and cached observations cannot establish
absence or authorize automatic metadata pruning.

PID and log locations are bound before workload setup. Absolute paths retain their
location; a leading `~/`, `$HOME/`, or `${HOME}/` expands using the execution account's
entry home. `working_dir`, activation scripts, and explicit workload `HOME` do not
move the PID/owner/lock or log files. Workload `working_dir` and `env_file` may still
be relative; they are execution inputs, not saved control-state locations.

Only an originally leading home prefix expands. `./$HOME/pids` and `./~/pids`
name literal relative directories in low-level recovery helpers; managed paths
reject those spellings as relative. Spaces and
other shell metacharacters remain literal. Normalization unifies equivalent home
prefixes, redundant separators, and harmless current-directory components while
retaining every parent (`..`) step. For example, `/base/link/../pids` can differ
from `/base/pids` through a symlink; destination identity preserves that distinction.
Normalization neither uses the controller machine's home nor resolves remote
symlinks. Explicit file paths ending in `/` or `/.` keep their directory requirement.

Launch and exec hooks apply setup in this order: change to `working_dir`, source
`env_file` with automatic environment export, select GPU visibility, then export
explicit workload variables. A relative `env_file` uses the resulting working
directory, including a bare filename (never a PATH-selected activation script);
its supported home prefix expands before sourcing. A failed directory change,
nonzero source result, or rejected environment export aborts without running the payload or overwriting
an existing PID/owner claim. Setup runs in a subshell, leaving the caller's
working directory and environment intact. PID/log bindings remain fixed inside it.

Launch and stop serialize changes with a per-PID `flock` from util-linux. The lock
covers validated ownership, process checks, and PID/owner writes. The workload
closes the lock descriptor; the OS releases the operation's lock at exit. Lock
files remain so concurrent operations use the same inode. Launch refuses to
replace a live workload, including workers whose original leader has exited.

Owner and PID records are written to temporary files in their destination directory
and atomically replaced only after a complete write. A PID-write or rename failure
after spawn returns failure and rolls back the new process group with the same
verified shutdown used by stop. Failed commits preserve previous records and never
publish a partial numeric PID. If rollback cannot be confirmed, the error identifies
the new PID/group for manual recovery. A successful launch means its complete PID
claim was recorded; serving-endpoint readiness is a separate operation.

Shared state readers distinguish present, confirmed absent, and failed acquisition.
Unreadable directories, unreadable or invalid PID/owner records, and failed process
inspection produce errors. An absent entry in a readable parent can establish
absence; a failed stat alone cannot. Discovery reports incomplete observation
on failure, which cannot authorize free capacity or automatic metadata pruning.
A missing owner marker permits legacy fallback only for an eligible legacy
application-owned name. Present empty, malformed, unreadable, or foreign markers
cannot authorize replacement or deletion. PID 1 is invalid workload state.

`setsid` launches the workload with PID equal to its process-group ID. Liveness
includes both the recorded PID and live members of that group, so a dead leader
does not hide surviving workers. Legacy standalone PID records remain recoverable
without signalling the unrelated group containing that PID. Groups containing
only exited/zombie processes count as stopped.

| Command | Contract |
| --- | --- |
| `run_cmd` | Requires `detach=True`; foreground requests raise before script generation. Bind paths, validate/lock state, complete setup, then launch and commit its PID claim or roll back. |
| `generate_launch_script` / `generate_exec_serve_script` | Preflight and serving launch respectively. Only detached workload mode is supported; false `detach` / `detached` arguments raise before script generation. |
| `exec_cmd` | Run a foreground hook with the same setup/failure policy; it does not create a workload claim. Its `detach` argument is ignored. |
| `status_cmd` | Exit **0** for a live PID/group, **1** for confirmed absent/dead, **2** for acquisition failure. |
| `stop_cmd` / `teardown_script` | Send TERM, wait up to about 10 seconds, then KILL if needed and verify the group. Delete recovery records only after confirmed stop; teardown reports actual stopped workloads. |
| `logs_cmd` | Read the same fixed log location using `tail`, optionally following it or limiting lines. |

Unknown liveness or surviving workers fail teardown and retain recovery records
and job metadata. Public observation errors and stop result types carry these
outcomes. A successful stale-record cleanup counts zero stopped workloads.

Native PID/owner record writes execute in a writer subshell. A failed write,
including a writer killed by `SIGXFSZ`, leaves the owning launch shell able to
remove its pending record and invoke process-group rollback. Only a complete
atomic replacement becomes a committed PID claim. This is a record-write failure
contract, not a guarantee of recovery if the owning launcher itself is killed.

#### Legacy relative-path recovery

Retain the old job metadata. Use low-level `LocalExecutor.status_cmd()`,
`logs_cmd()`, and `stop_cmd()` with the saved configuration, workload name,
application profile, and SSH user, **from the original execution directory/home**.
These helpers preserve ordinary relative-path semantics for explicit recovery.
Managed lifecycle operations reject the unanchored configuration, including stop
by job ID, rather than reporting a successful stop in another directory. After
confirming the old workload has stopped, configure absolute or home-relative
control paths and launch again. Do not rewrite a live job's saved destination to
make it appear migrated.

#### Legacy fixed-file recovery

Managed local workloads require `pid_dir`, with one PID file per workload/rank.
A singular `pid_file` cannot identify multiple workloads during metadata-free
discovery, so 0.4.0 rejects it during target resolution and before any launch
script is generated. Status reports incomplete coverage for an old fixed-file
configuration and cannot authorize automatic metadata pruning. Old fixed-file records
also remain ineligible for absence matching after cluster configuration changes
to a PID directory. `log_file` remains an optional shared append log; use `log_dir` for separate workload logs.

For a legacy fixed-file workload, retain its metadata and use the low-level
`LocalExecutor.status_cmd()`, `logs_cmd()`, and `stop_cmd()` with its saved executor
configuration, workload name, application profile, and SSH user. These recovery
helpers still honor `pid_file`; stop retains its application ownership guard. Managed stop/log/liveness
operations require a resolvable target and therefore reject that old configuration.
After confirming the workload has stopped, replace `pid_file` with `pid_dir` in its
recipe/cluster configuration and launch again. Do not rename a live PID record or
rewrite its destination metadata to make it appear migrated. Ordinary directory
records remain readable; older noncanonical destination keys remain conservative
for absence matching until replaced by a new launch.

### Known limitations

- **No images or mounts**: `image` and Docker `extra_opts` are ignored;
  `pull_cmd` / `inspect_exists_cmd` return `true`. Volume mappings translate
  container-style environment paths to host paths; they do not mount anything.
- **Process-group lifecycle**: requires Bash, util-linux `setsid`/`flock`, and
  process inspection with `ps`. Descendants that leave the launched group are
  outside lifecycle tracking. No supervisor, systemd unit, or restart on crash.
- **No Ray strategy**: `generate_ray_head_script` and
  `generate_ray_worker_script` raise `NotImplementedError`. Use a native
  runtime (`vllm-distributed`, `sglang`) or fall back to Docker.
- **GPU visibility is best-effort**: only `gpus="all"` / `gpus="device=0,2"`
  are translated. `count=2` and capability filters log a warning and leave
  visibility to the workload.

Multi-host native cluster runtimes work because each host's
`container_name` becomes `<cluster_id>_node_<rank>`, which is the basename for
per-rank pid/log files. The Ray restriction means `vllm-ray` is the only
runtime that can't pair with Local.

## `K8sExecutor` (experimental draft)

`plugins/k8s/executor.py`, owned by the [Kubernetes plugin](../src/sparkrun/plugins/k8s/README.md). `kubectl run`-based — every lifecycle
operation is a `kubectl` invocation.

Run `sparkrun plugins enable k8s` first (default on only for Sparkrun alpha). The plugin
registers `executor.k8s`, `cli.setup.k8s`, and `api.run.k8s`, each defaulting on.
Its typed executor settings live in `K8sExecutorConfig`. The native API run path
uses JobSet for supported single-pod launches; the commands below describe the
Pod-level executor used for lifecycle operations and the legacy launch path.

### What it does

- `run_cmd`: `kubectl [--kubeconfig …] [--context …] [-n …] run <name>
  --image=<image> --restart=Never [--image-pull-policy=…] [--overrides=<JSON
  nodeSelector>] [--limits=nvidia.com/gpu=N] [--limits=memory=…] [--env=K=V …]
  [--labels=…] [--command when `entrypoint` is set] -- bash -c <base64 cmd>`.
- `exec_cmd`: `kubectl exec` (`-d`-like behavior synthesized via nohup).
- `stop_cmd`: `kubectl delete pod --ignore-not-found [--grace-period=0 --force]`.
- `logs_cmd`: `kubectl logs [-f] [--tail=N]`.
- `status_cmd`: `kubectl get pod -o jsonpath='{.status.phase}'` equality test
  vs `"Running"`.
- `inspect_exists_cmd` / `pull_cmd`: no-ops (the cluster pulls on Pod creation).

### Known limitations

- **`kubectl run`, not manifests**: init containers, sidecars, custom scheduler
  hints, volume claims, services — all unreachable. Use `k8s_node_selector`
  and any `--overrides`-compatible `extra_opts` entries to wedge in extras.
- **Docker-specific options dropped silently**: `--privileged`, `--shm-size`,
  `--ipc`, `--network` aren't translated.
- **GPU mapping is conservative**: `gpus="all"` → `nvidia.com/gpu=1`,
  `device=0,1` → `nvidia.com/gpu=2`. Anything fancier logs a warning and ships
  no GPU resource request.
- **No Ray strategy**: same restriction as Local; runtime authors can either
  target `vllm-distributed` / `sglang` or stay on Docker.
- **Image plumbing**: `generate_exec_serve_script` requires the runtime to put
  the image into the env block under `SPARKRUN_K8S_IMAGE`. Without it the
  generated command points at `sparkrun-k8s-image-not-configured` to fail
  loudly.
- **No StatefulSet / Job**: one Pod per host_list entry; multi-host
  orchestration is still loop-driven from `runtime.run()`.

## Quick recipe field cheat sheet

```yaml
# Local executor (no container)
executor: local
executor_config:
  working_dir: /opt/myproject
  log_dir: /var/log/sparkrun
  env_file: /etc/sparkrun.env
  command_prefix: nice -n 10
  gpus: "device=0,2"     # → CUDA_VISIBLE_DEVICES=0,2

# K8s executor
executor: k8s
executor_config:
  k8s_namespace: ml-prod
  k8s_context: prod-east
  k8s_node_selector: nodepool=dgx-spark
  k8s_image_pull_policy: IfNotPresent
  kubeconfig: /etc/k8s/admin.conf
  memory_limit: 128Gi
  entrypoint: ""        # emits --command -- bash -c ...
```

Docker fields (the existing ones — `privileged`, `cap_add`, `devices`, etc.)
keep their previous behavior and ship under the same `executor_config:` block
when `executor: docker` (or unset).


## Docker seccomp profiles (0.4)

The Docker executor defaults to `seccomp=io-uring`: a bundled, pinned copy of
[Moby's default profile](https://github.com/moby/profiles/blob/61eaf32614c7c71b60bd8927d3e6a4ffc8ff1f31/seccomp/default.json)
with an allow rule for `io_uring_setup`, `io_uring_enter`, and
`io_uring_register`. The default deny action, architecture mappings, and other
rules remain intact. Atlas uses this shared policy and no longer adds
`seccomp=unconfined` itself. Docker's treatment of explicitly privileged
containers still applies; normal sparkrun launches default to non-privileged.

Use the existing `executor_config.security_opt` in application configuration,
a cluster, a recipe, or API overrides to select a locally defined profile:

```yaml
executor_config:
  security_opt:
    - no-new-privileges
    - seccomp=/home/operator/policies/inference.json
```

The path is on the controller running sparkrun. `~` expands to that user's
home; relative paths resolve against its working directory. sparkrun validates
basic JSON/profile structure and snapshots the policy before replacing the
running workload. Docker validates target-specific syscall/architecture support
when creating the container. A custom policy is used as supplied: sparkrun does
not add io_uring permissions to it.

The policy contents travel with every generated Docker launch script, including
head and worker scripts. Bash process substitution gives Docker a readable
file descriptor on the node invoking its CLI. No shared directory, persistent
remote policy cache, or manual copy to the nodes is required. This follows
[Docker's client-side profile loading](https://docs.docker.com/engine/security/seccomp/).
The source file can change or disappear after preparation without changing the
policy used by later workers in the same launch. Preview generation reads and
validates the local policy but performs no remote writes.

An explicit `seccomp=builtin` selects Docker's own default profile, and
`seccomp=unconfined` explicitly disables filtering. If no seccomp option is
specified, including an empty security-options list, sparkrun supplies
`io-uring`. Other security options remain in force. Raw `extra_docker_opts`
seccomp flags use the same local-file delivery and override the configured
policy; duplicate policies within either list are rejected.

Executors may implement `prepare_launch(extra_opts=...)` to validate and snapshot
local inputs. The shared runtime launcher calls it before replacement, including
previews; it must not start workloads or mutate remote state. Direct command
and script generation also works without prior preparation. Generated Docker
commands require Bash, as do the existing host launch scripts.

## Operation targets and endpoint support

`resolve_executor_target()` returns an `ExecutorTarget` before placement and
reuse queries. Its `executor` selects the provider, `config` is a read-only
snapshot of destination and connection settings, and `destination_key` is the
provider-derived destination identity. `overrides` returns a detached mapping
for the highest-priority layer of `resolve_executor()`. Hardware and resource
settings continue to resolve after placement.

Provider executors override `resolve_target(dry_run=...)`. The hook may read
local configuration but must not start workloads or mutate remote state. Include
connection settings needed by discovery as well as launch. Exclude credentials
and launch policy from the destination key: changing a kubectl executable, for
example, must not create a different workload identity.

`RunPlan.executor_target` carries this snapshot. The plan's cluster is scoped to
it for occupancy, ensure, and replacement. Native handlers receive options with
the same target at caller precedence; they must not choose a new target. Native
Kubernetes keys include the canonical kubeconfig path, context, and namespace.
Separate destinations have separate deterministic workload IDs and saved lifecycle
records. Default host destinations retain their existing deterministic IDs.

A destination key does not change application/controller identity: one canonical
configuration directory still has one controller. It also does not describe a
reachable inference endpoint. Executors whose hosts refer only to a control plane
set `supports_host_endpoint = False`; benchmarking then rejects that executor
before submission or endpoint probing. Native Kubernetes currently does so.

The shared launcher writes the complete resolved executor and IP mappings only
after preparation and its replacement callback succeed, immediately before
submission. Writes atomically replace an owner-only file. An initial persistence
failure aborts submission; an interrupted submission retains the saved record for
recovery. Later version-info updates preserve the same target and IP mappings.

### Status observation coverage

`query_status_for_cluster()` pins each executor's target before querying it.
`ClusterStatus.coverage` contains immutable `ExecutorCoverage` values: the
`ExecutorTarget`, status scope, requested hosts, and successfully observed hosts.
Failures remain unobserved even if another executor answers for the same host.
`ClusterStatus.observation` combines that coverage with running IDs in a
`RunningSnapshot`. Both models live in `sparkrun.core.status_observation`.

Automatic metadata pruning and shell completion use the same absence test.
A job is confirmed absent only when its executor, destination, and complete host
set were successfully inspected. An old timestamp is insufficient. Metadata
without an executor selector or a destination key,
is unknown and is preserved by automatic cleanup. Explicit age-based cleanup
remains available through the setup command.

The local executor pins its PID/log directories. A custom PID directory is a
separate destination; changing only the log directory preserves identity.
Local deterministic IDs also include the resolved SSH user, including with the
default PID path. Docker's default IDs remain unchanged. Remote paths are not
expanded against the controller's home directory. A local target with no known
SSH user receives a fresh placement token rather than claiming deterministic
identity. Existing jobs remain addressable by their saved IDs.

`status_report()` retains coverage on its typed `ClusterStatusResult`.
`stop_all(discovered=result)` uses its recorded executor targets, including for
native teardown after defaults change or job metadata disappears. A legacy
manually constructed result without coverage still requires the matching
`cluster`/transport context or authoritative saved job metadata. `to_dict()` is
presentation output, not a serialization format for replaying lifecycle actions.

The completion cache stores the same observation model. Both normal cache reuse
and stale fallback require a matching destination. Old host-only cache files
are ignored and replaced by the next observation. Failed or missing coverage
never authorizes hiding a job.


An `ExecutorTarget` can set `user_scoped=True` when the SSH login determines its
observable namespace. The local executor does so because PID files and process
permissions depend on the remote user. This is derived destination evidence,
not another controller ID. `ExecutorCoverage.ssh_user` participates in matching
for those targets; absent users and legacy records cannot authorize automatic
pruning. Default and relative local PID paths obey the same rule. Credential-key
rotation does not change namespace identity.

Coverage also retains detached, immutable SSH arguments in memory for subsequent
teardown. Those arguments are not written to the completion cache: version 2
stores only the SSH user alongside target/host coverage. Older caches are ignored.
Cache matching checks user-sensitive peer backends even when Docker is the
primary executor for the host sweep.

Use `ClusterStatus.observation_errors` for incomplete discovery, including a
failed backend on a host that answered through another backend. `errors` retains
reachability/target failures; `for_host()` can still return partial observations.
`free_slots()` reports zero on incomplete hosts. Occupancy schedulers exclude those
hosts, and a failed status acquisition remains an explicit unknown observation
rather than selecting the no-status greedy fallback. Strict recipe-based
stop/log discovery and strict replacement reject incomplete observations.
`api.find_running_intent()` and ensure remain best effort: they return positive
matches from available observations, and a miss does not prove absence. A caller
requiring verified absence must query status, reject `observation_errors`, and
pass the complete snapshot to `find_running_intent(status=...)`. Callers that
intentionally do not request occupancy can still supply `status=None` to a scheduler.

Docker status preserves the exit status and diagnostics of `docker ps`. A missing
executable, permission denial, or unavailable daemon is incomplete coverage,
including when another host executor succeeds. Only a successful empty query
establishes absence; optional peer failures are not treated as idle capacity.

Placement, coverage matching, and saved-job recovery share the same private
destination definition. User-scoped job metadata records the namespace policy
alongside its SSH principal. Stop, logs, and liveness keep that recorded user
when cluster defaults change, while credentials may rotate. An explicitly different
user is rejected; a missing recorded principal requires fresh status discovery
before teardown. Legacy local records with a known user retain this protection.
The principal is checked again after transport preparation, which can refresh
connection settings. No additional controller identity is introduced.


New user-scoped launches materialize the effective SSH principal before submission.
Local dispatch uses the current OS account; remote aliases/defaults are resolved
through OpenSSH's local `-G` configuration evaluation, including `ssh.options`.
This opens no SSH session. Clusters with different implicit users must select
one explicit `cluster.user` or `ssh.user`; unresolved identity fails before launch.
Use those user fields rather than embedding `user@` in host addresses. An explicit
or recorded user takes precedence over `-o User=...`, `-l ...`, and SSH configuration.
This rule is shared by direct SSH, embedded transfer scripts, pipelines, and rsync.
Without an explicit user, OpenSSH options/configuration still supply the default.

`api.plan()` retains that resolved principal independently of mutable defaults and
its cluster object. Reusing the plan keeps discovery, replacement, and submission
in that namespace. Authentication keys may rotate without changing the ID. Direct
launcher calls apply the same principal resolution before submission and metadata
recording. An unknown user on an older record remains insufficient evidence for
automatic teardown; this fallback is not used to create new unmanaged workloads.

`api.open_telemetry()` and `api.open_live_monitor()` use the same operation-local
transport resolution as `api.status()`: prepare transport, apply cluster settings,
then overlay explicit SSH kwargs per key. Omitted contexts are initialized normally;
explicit None/empty values clear the corresponding configured key. Monitor frames
retain successfully observed workloads but expose incomplete peer coverage through
`status_error` and report zero confirmed free slots until observation succeeds.

A failed monitor poll also retains last-known workloads with an explicit error
and zero confirmed free capacity. A successful subsequent poll clears that error.
When reusing a plan, conflicting SSH `User` options are rejected before ensure
or handler dispatch, as well as before a direct launcher submission.


### Durable GPU assignments

Accepted GPU allocations are stored in the `sparkrun.gpu_allocations` Docker
label or a native `.pid.allocations` sidecar. The versioned record includes global
rank, physical local GPU index, utilization reservation, and shared memory
reservation. Job metadata also includes per-host allocation records for
inspection. Worker status recovery does not require the controller's metadata.

Launch commands expose the assigned devices to the workload. Status preserves
the assignments through Docker/local merging, startup, idle serving, and partial
job survival. Native teardown removes the sidecar with the PID record. Absent or
invalid records never grant permission to share; legacy workloads remain
conservatively occupied until their actual device assignments are known.
