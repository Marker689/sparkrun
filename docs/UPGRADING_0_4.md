# Upgrading to Sparkrun 0.4.0

This guide covers changes from 0.3.x and earlier 0.4.0 previews. Python 3.12 or
newer is required. The [Python API migration guide](DISTRIBUTION_API_MIGRATION.md)
documents breaking imports and request/result contracts for application authors.

## Upgrade and check plugins

Drain daemon workers and stop API instances using the installation before
replacing it. Stopping only the API does not stop independently running workers.
Core updates and forced setup installs refuse replacement while those runtime
leases are live, including leases for another config binding.

```sh
sparkrun update
sparkrun plugins list
```

The update stays on your configured channel. After the 0.4.0 stable release is
published, `sparkrun update --stable` switches to it. Git preview channels can
advance beyond 0.4.0; switching back to stable may downgrade.

Updates include compatible revisions of trusted managed plugins by default.
`--no-update-plugins` keeps plugin files unchanged and performs a local
compatibility audit. Pinned tags/commits are not advanced automatically.
Unknown or incompatible candidates produce warnings and leave the current
plugin intact. Check these warnings before restarting long-running consumers.

## SparkRoute and ColdSnap are separate installs

OCI Relay remains bundled and enabled by default on every channel. SparkRoute
and ColdSnap are installed independently of the core environment:

```sh
sparkrun plugins add https://github.com/sparksq/sparkrun-sparkroute-plugin --enable
sparkrun setup features enable gateway.sparkroute
sparkrun plugins add https://github.com/sparksq/sparkrun-coldsnap-plugin --enable
```

Only install the plugins you use. Review each source at the trust prompt;
consent also covers future compatible updates from that source. Adding a plugin
can enable external loading, while `--enable` enables plugin loading and preserves capability settings. Neither
capability defaults on. Existing manually configured copies are not converted
automatically; resolve duplicate providers before installing a managed copy.

Managed plugins live beside your config, outside the Python installation. The
default `plugins.retain_generations: 2` retains current and previous versions.
Use `sparkrun plugins rollback NAME` to roll back; it validates compatibility
with the current core. Live daemon/worker leases protect generations from
removal. Restart consumers after plugin changes.

LiteLLM is the default proxy on every channel. To use installed SparkRoute,
select `sparkrun proxy start --gateway sparkroute` (add `--restart` to replace a
running gateway). An existing SparkRoute selection is preserved even if its
plugin is missing. See [plugins](PLUGINS.md) and [SparkRoute](SPARKROUTE.md) for
installation constraints, trust, and gateway selection.

## Optional daemon and application APIs

The daemon provides authenticated HTTP APIs, durable jobs, events, and plugin
operation namespaces. Enable it explicitly with
`sparkrun setup features enable daemon` where the feature is off; alpha enables
it by default. CLI commands still execute locally unless a documented daemon
client or passthrough mode is selected. MCP remains disabled.

TCP controllers run on Linux, macOS, and Windows; Unix sockets and the systemd
installer are Linux-specific. Model hosts and inference runtimes retain their
Linux/runtime requirements. macOS/Windows service installers are not included.
Measurement subprocesses are portable, subject to each benchmark framework's
dependencies. See the [daemon guide](DAEMON.md) and [API reference](DAEMON_API.md).

Applications should use `sparkrun.application.initialize()` and the public
`sparkrun.api` contracts. [Application profiles](APPLICATION_PROFILES.md) provide
independent identity, config paths, defaults, and plugin composition.
Plugin daemon operations use loader-owned namespaces; workload orchestration
belongs to core. A daemon job ID identifies an API execution, not a workload.

## Model caches, readiness, and feedback

Manifest preparation verifies selected model files. Delegated acquisition can
download on a model host and distribute from there without putting weights on
the controller. Cache usage and workload claims live with each host's cache,
separately from daemon job storage. Legacy pre-placed model paths without a
manifest produce an unverified migration warning; see [model caches](MODEL_CACHE.md)
for importing their inventories.

Cleanup is disabled by default. `model-cache prune` previews changes unless
`--apply` is given; optional automatic cleanup requires an explicit policy.
Pins, active claims, unknown content, and shared Hugging Face blobs are protected.
Set `SPARKRUN_ADVANCED=1` to expose the model-cache CLI. Declared compatibility,
verified files, and a live inference endpoint are separate checks; a failed or
lost daemon job does not prove that its workload stopped.

Run and benchmark share launch feedback. Benchmark frameworks can publish
structured progress without making progress events authoritative results;
saved measurement artifacts/checkpoints remain authoritative. Plugins can use
the public [feedback interface](FEEDBACK.md), and custom frontends can use the
[benchmark API](BENCHMARK_API.md). Applications should tolerate additional event
fields and unknown advisory event kinds.

Docker launches now stage a seccomp profile that permits io_uring; see
[executor configuration](EXECUTORS.md#docker-seccomp-profiles-04). vLLM no longer
sets `OMP_NUM_THREADS` by default; recipes can set it explicitly. Hugging Face
Hub 2.2.0 uses Xet where supported; the HTTP selection policy and recovery limits
are documented in [model caches](MODEL_CACHE.md).
