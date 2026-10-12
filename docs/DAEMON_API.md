# Daemon operation and HTTP reference

Reviewed: 2026-10-12, for the unreleased Sparkrun 0.4.0 daemon/discovery migration.
This inventory was rendered from the actual operation declarations in the local
host and SparkRoute plugin worktrees, without running handlers or a live daemon.
Operational setup and examples are in [DAEMON.md](DAEMON.md).

## Contract and terminology

- HTTP protocol: **1** (`/v1`); every operation listed here currently has operation **version 1**.
  The old SparkRoute stdio bridge protocol **5** is a separate, explicitly selected transport.
- The canonical operation name selects a handler. Both `POST /v1/ops/{name}`
  and the literal alias listed below call the same dispatcher, authorization,
  schema validation, idempotency and worker admission path. They take the same
  JSON object. A query still uses POST; these aliases are RPC conveniences.
- `query` returns the declared result with HTTP **200**. `job` returns HTTP **202**
  with `{"job_id":"…","events_url":"/v1/jobs/…/events"}`. The final result is
  `GET /v1/jobs/{id}` → `result`; acceptance is never execution success.
- `read < operate < admin` are hierarchical permissions. Secret-prompt permission
  is separate. Same-user Unix peers have local admin access when no bearer is supplied; an
  explicit bearer selects its own scope/manager on UDS too. TCP always requires a token. `/v1/health` alone is unauthenticated.
- A non-admin principal sees only its own jobs and must retain the job's operation
  permission. `X-Sparkrun-Client` is an advisory label, never an identity or owner.
- `Idempotency-Key` applies to **job submissions**, scoped to the authenticated
  principal. The same key/operation/version/arguments/manager returns the same job;
  changed arguments conflict. Retention bounds this guarantee. A disconnect does
  not cancel the job. A later intentional action requires a new key.
- `auto` infers execution reservations; unresolved mutation destinations use `*`.
  `controller` requests that conservative reservation explicitly. `none` skips
  the worker's resource reservation, but a handler can still take domain locks
  or public-API reservations. It does not mean authorization is skipped.
- `proxy.watch` and `proxy.serve` are host-owned persistent service jobs, admitted
  against `daemon.max_services` (default 2), separately from `daemon.max_jobs`
  (default 4). They use the same durable job/event/cancellation infrastructure.
- Top-level request objects reject unknown fields. `!` marks a required field
  below; all others can be omitted. `null` means the published schema accepts
  it, not that every underlying API normalizes it identically.
  Maps, nested objects, bounds and final result schemas are published in `/v1/ops`
  and OpenAPI. This compact table is not a replacement for those schemas.

## Workload management identity

Lifecycle belongs to core `workloads.*`, for SparkRoute and other automation clients.
The daemon authenticates a stable **manager** from the locally issued credential;
`/v1/info` reports it, job admission persists it, and the isolated worker reuses it.
The default is `daemon`; managed SparkRoute credentials use `sparkroute`. A caller
cannot select ownership through operation arguments or `X-Sparkrun-Client`.
Local provisioning (`daemon token create NAME --manager sparkroute`) is the grant.
Non-default manager tokens share stable principal `manager:<name>` when provisioned
through the CLI/integration helper, so replacement credentials can resume retained
jobs and retry keys. Revocation prevents new submissions; accepted jobs keep their
admitted identity. Integration cutover drains those jobs before revoking credentials.

`workloads.run` and `workloads.ensure_ready` use this manager for new workload
ownership. The shared workload row (inventory, lifecycle control and the
`workload` field of ensure-ready) reports deployed `cluster_id`, revision, owner,
`owned` relative to this manager, provider, permitted actions, and recorded state.
`ensure_ready` separately returns a verified endpoint, original `observed_at` and
`adopted`. Adoption never rewrites an existing owner. A different manager may route
to a healthy endpoint and inspect its provider status, but cannot use the guarded
stop/sleep/wake/reconcile operations to manage it.

This is an automation guard within the existing single-account permission model,
not tenant isolation. `workloads.stop_matching`/`stop_all` deliberately retain broad
operator control. Managed gateway credentials retain admin scope for the admin
catalog (without secret-prompt permission); manager identity does not narrow scope.
Namespaced plugin `owner` in `/v1/ops` means declaration ownership, distinct from a
workload's management owner and a daemon job's authenticated principal.

SparkRoute registers only `endpoints.discover`, `endpoints.status`, and
`endpoints.metadata` projections. The first two add gateway model-card fields to
verified host observations; metadata is an advisory, revision-checked read of saved
recipe fields, never readiness evidence. Activation and lifecycle DTOs remain
core; the Go client adapts them to its internal gateway types. No duplicate
`plugins.sparkroute.workloads.*`, binding-resolution or capability routes remain.
ColdSnap continues implementing the common lifecycle provider without HTTP routes.

## Infrastructure routes (not operation aliases)

| Method | Path | Permission | Behavior / input |
| --- | --- | --- | --- |
| GET | `/v1/health` | public | Liveness and protocol; not worker readiness. |
| GET | `/v1/info` | read | Profile/config binding, runtime identity, restart-required state. |
| GET | `/v1/ops` | read | Authorized, feature-filtered operation metadata and request/result schemas. |
| POST | `/v1/ops/{name}` | selected operation | Generic dispatcher; JSON arguments, query 200 or job 202. |
| GET | `/v1/jobs` | read + job visibility | Optional `state`, `limit` (default 200; clamped 1–1000); bounded list, no cursor. |
| GET | `/v1/jobs/{id}` | job owner/admin + operation scope | Durable state, result/error, public prompt metadata; no raw arguments, worker nonce or runtime lease. |
| DELETE | `/v1/jobs/{id}` | job access + operate | Requests cancellation; does **not** delete history or promise remote rollback. |
| POST | `/v1/jobs/{id}/input` | job access + operation scope; secret if needed | Exactly `prompt_id`, `value`; answer a live worker prompt. |
| GET | `/v1/jobs/{id}/events` | job access | Durable SSE; `Last-Event-ID`; expired cursor → 410, later expiry → `gap` event. |
| GET | `/v1/events` | read + job visibility | SSE snapshots of visible jobs every 2 seconds; no inventory stream or replay cursor. |
| GET | `/v1/streams/logs` | read | SSE; optional `cluster_id`, `recipe`, comma-separated `hosts`, `cluster`, `scope`, `tail`, `follow=true/false`. Disconnect closes reader. |
| GET | `/v1/streams/monitor` | read | SSE; optional comma-separated `hosts`, `cluster`, `interval` (1–60 seconds). Shared latest frames, not durable history. |
| GET | `/v1/reservations` | admin | Execution claims and verified owner liveness; dead claims remain until reconciled. |
| POST | `/v1/reservations/{owner}/reconcile` | admin | `{"acknowledge_effects":true}`; releases only a verified dead owner's claims. |
| GET | `/v1/snapshots/{name}` | read | Advisory snapshot of `discovery.snapshot`; no other selector is accepted. |
| POST | `/v1/maintenance` | admin | `{"drain":true/false}`; admission control, not worker cancellation. |
| POST | `/v1/shutdown` | admin | `{}`; stops API process, workers continue. |
| GET | `/openapi.json` | authenticated | Full registered operation schemas, including feature-disabled or higher-scope declarations; not the filtered capability view. |
| GET | `/docs` | read | Swagger UI; same authorization boundary. |

No MCP route is mounted. No combined logs+monitor route exists. Tokens are issued,
listed and revoked through local CLI tooling, not an HTTP token-administration API.
Raw recipe-trust and credentials are not accepted as arbitrary request fields;
trust and secret prompts retain their separate server policies.

This snapshot contains **92 operations: 81 core, 8 K8s and 3 SparkRoute**.

## Domain vocabulary

Names use `domain.action`, with underscores inside multiword actions. These are
unreleased canonical operations; retired flat names have no compatibility aliases.
Python library names are independent of the daemon operation vocabulary.

| Surface | Meaning |
| --- | --- |
| `/v1/jobs` | Durable daemon invocation history, prompts, results and events. |
| `workloads.list` | Saved deployment metadata; does not probe current health. |
| `workloads.inventory` | Recorded provider lifecycle/ownership/action inventory. |
| `cluster.observe` / `cluster.report` | Raw host occupancy and coverage / grouped, metadata-enriched display report. |
| `workloads.plan` / `workloads.run` | Resolve placement / launch and optionally await readiness and post-launch hooks. |
| `workloads.ensure_ready` | Adopt, recover or launch with durable effect receipts and exact revision identity. |
| `workloads.stop` | Exact-ID/revision teardown of an integration-owned workload. |
| `workloads.stop_matching` / `workloads.stop_all` | Explicit selector-based teardown / host-wide teardown; distinct from ownership-guarded automation. |
| `workloads.materialize` | Resolve container launch units without launching. |
| `recipes.plan_build` / `recipes.build` | Plan / prepare native or image environments and stage assets without starting inference. |
| `recipes.validate` / `recipes.realize` | Validate a recipe / export a placed, pinned realization. |
| `catalog.search` / `catalog.recipes` | Simple recipe summaries / paginated catalog browsing and facets. |
| `catalog.registries.*` | Named admin actions: add, remove, enable, disable, trust, untrust; no action-dispatch field. |
| `benchmarks.run` / `benchmarks.resume` | Execute a benchmark / resume by benchmark ID, separate from the daemon job ID. |
| `discovery.snapshot` | Scoped endpoint observations with original timestamp, host coverage and errors. |

Registry `add` requires name and URL and never grants trust. `trust` requires
`acknowledge_trust: true`. Each registry action rejects irrelevant fields. Advisory
snapshots use `/v1/snapshots/discovery.snapshot`, with the same coverage-bearing
result as the corresponding operation. Plugin endpoint projections retain their
own namespaces and routing-policy fields.

<!-- BEGIN GENERATED DAEMON OPERATIONS -->

## Core operations

### benchmarks

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `benchmarks.resume` | `/v1/benchmarks/resume` | job / operate | auto | `benchmark_id!: string`, `dry_run: boolean/null`, `integrations: object/null`, `export_files: boolean/null`, `output_file: string/null`, `timeout: integer/null`, `exit_on_first_fail: boolean/null`, `api_key_env: string/null` |
| `benchmarks.run` | `/v1/benchmarks/run` | job / operate | auto | `recipe!: string`, `category: string/null`, `framework: string/null`, `profile: string/null`, `bench_args: object/null`, `hosts: string[]/null`, `cluster: string/null`, `overrides: object/null`, `resume: string/null`, `skip_run: boolean/null`, `no_stop: boolean/null`, `exit_on_first_fail: boolean/null`, `timeout: integer/null`, `api_key_env: string/null`, `integrations: object/null`, `output_file: string/null`, `export_files: boolean/null`, `solo: boolean/null`, `dry_run: boolean/null`, `scheduler: string/null`, `rootful: boolean/null`, `trust: boolean/null`, `offline: boolean/null`, `sync_tuning: boolean/null`, `extra_docker_opts: string[]/null` |

### catalog

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `catalog.capacity` | `/v1/catalog/capacity` | job / read | auto | `cluster!: string` |
| `catalog.cleanup` | `/v1/catalog/cleanup` | job / operate | auto | `{}` |
| `catalog.clusters` | `/v1/catalog/clusters` | query / read | auto | `{}` |
| `catalog.import` | `/v1/catalog/import` | job / operate | auto | `content!: string` |
| `catalog.recipe` | `/v1/catalog/recipe` | job / read | auto | `reference!: string`, `overrides: object/null`, `expected_revision: string/null` |
| `catalog.recipes` | `/v1/catalog/recipes` | query / read | auto | `query: string/null`, `registry: string/null`, `runtime: string/null`, `local_only: boolean/null`, `offset: integer/null`, `limit: integer/null`, `filters: object/null` |
| `catalog.refresh` | `/v1/catalog/refresh` | job / operate | auto | `{}` |
| `catalog.registries` | `/v1/catalog/registries` | query / read | auto | `{}` |
| `catalog.registries.add` | `/v1/catalog/registries/add` | job / admin | auto | `name!: string`, `url!: string`, `subpath: string` |
| `catalog.registries.disable` | `/v1/catalog/registries/disable` | job / admin | auto | `name!: string` |
| `catalog.registries.enable` | `/v1/catalog/registries/enable` | job / admin | auto | `name!: string` |
| `catalog.registries.remove` | `/v1/catalog/registries/remove` | job / admin | auto | `name!: string` |
| `catalog.registries.trust` | `/v1/catalog/registries/trust` | job / admin | auto | `name!: string`, `acknowledge_trust!: boolean` |
| `catalog.registries.untrust` | `/v1/catalog/registries/untrust` | job / admin | auto | `name!: string` |
| `catalog.retain` | `/v1/catalog/retain` | job / operate | auto | `reference!: string` |
| `catalog.search` | `/v1/catalog/search` | query / read | auto | `query: string/null`, `registry: string/null`, `runtime: string/null`, `include_hidden: boolean/null`, `include_local: boolean/null`, `unique_names: boolean/null` |

### cluster

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `cluster.create` | `/v1/cluster/create` | job / admin | auto | `name!: string`, `hosts!: string[]`, `description: string/null`, `user: string/null` |
| `cluster.default` | `/v1/cluster/default` | job / admin | auto | `name: string/null` |
| `cluster.delete` | `/v1/cluster/delete` | job / admin | auto | `name!: string` |
| `cluster.observe` | `/v1/cluster/observe` | job / read | auto | `hosts!: string[]`, `cluster: string/null`, `executor: string/null` |
| `cluster.report` | `/v1/cluster/report` | job / read | auto | `hosts!: string[]`, `cluster: string/null`, `executor: string/null` |
| `cluster.update` | `/v1/cluster/update` | job / admin | auto | `name!: string`, `hosts: string[]/null`, `description: string/null`, `user: string/null` |

### discovery

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `discovery.snapshot` | `/v1/discovery/snapshot` | job / read | auto | `hosts: string[]/null`, `cluster: string/null`, `cluster_ids: string[]/null` |

### model_cache

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `model_cache.import` | `/v1/model_cache/import` | job / operate | auto | `model!: string`, `revision: string/null`, `projector: string/null`, `file_selection: string`, `manifest!: object`, `cache_dir!: string`, `hosts: string[]/null`, `checksums: boolean/null` |
| `model_cache.inspect` | `/v1/model_cache/inspect` | job / read | auto | `model!: string`, `revision: string/null`, `projector: string/null`, `file_selection: string`, `cache_dir!: string`, `hosts: string[]/null`, `checksums: boolean/null` |
| `model_cache.repair` | `/v1/model_cache/repair` | job / operate | auto | `model!: string`, `revision: string/null`, `projector: string/null`, `file_selection: string`, `cache_dir!: string`, `hosts: string[]/null`, `local_cache_dir: string/null`, `transfer_mode: string/null`, `checksums: boolean/null`, `offline: boolean/null`, `dry_run: boolean/null` |
| `model_cache.resolve` | `/v1/model_cache/resolve` | job / read | auto | `model!: string`, `revision: string/null`, `projector: string/null`, `file_selection: string`, `cache_dir: string/null` |

### proxy

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `proxy.add_alias` | `/v1/proxy/add_alias` | job / admin | none | `alias!: string`, `target!: string` |
| `proxy.discovery.ensure` | `/v1/proxy/discovery/ensure` | query / admin | none | `generation!: string` |
| `proxy.discovery.status` | `/v1/proxy/discovery/status` | query / admin | none | `{}` |
| `proxy.discovery.stop` | `/v1/proxy/discovery/stop` | query / admin | none | `{}` |
| `proxy.install_service` | `/v1/proxy/install_service` | job / admin | none | `scope: string/null`, `cluster: string/null`, `now: boolean/null`, `dry_run: boolean/null` |
| `proxy.list_aliases` | `/v1/proxy/list_aliases` | query / read | none | `{}` |
| `proxy.list_gateways` | `/v1/proxy/list_gateways` | query / read | none | `{}` |
| `proxy.models` | `/v1/proxy/models` | job / read | none | `{}` |
| `proxy.register_loaded_model` | `/v1/proxy/register_loaded_model` | job / operate | none | `recipe!: string`, `overrides: object/null`, `cluster: string/null` |
| `proxy.remove_alias` | `/v1/proxy/remove_alias` | job / admin | none | `alias!: string` |
| `proxy.resolve_gateway` | `/v1/proxy/resolve_gateway` | query / read | none | `name: string/null` |
| `proxy.serve` | `/v1/proxy/serve` | job / admin | none | `generation!: string` |
| `proxy.service_status` | `/v1/proxy/service_status` | job / admin | none | `scope: string/null`, `journal_lines: integer/null` |
| `proxy.start` | `/v1/proxy/start` | job / operate | none | `host: string/null`, `port: integer/null`, `gateway: string/null`, `host_filter: string[]/null`, `cluster: string/null`, `clear_cluster: boolean/null`, `auto_discover: boolean/null`, `discovery_backend: string/null`, `discover_interval: integer/null`, `discover_removal_grace_sweeps: integer/null`, `restart: boolean/null`, `persist: boolean/null`, `dry_run: boolean/null` |
| `proxy.status` | `/v1/proxy/status` | job / read | none | `{}` |
| `proxy.stop` | `/v1/proxy/stop` | job / operate | none | `dry_run: boolean/null` |
| `proxy.sync` | `/v1/proxy/sync` | job / operate | none | `aliases: object/null`, `host_filter: string[]/null` |
| `proxy.uninstall_service` | `/v1/proxy/uninstall_service` | job / admin | none | `scope: string/null`, `dry_run: boolean/null` |
| `proxy.unregister_loaded_model` | `/v1/proxy/unregister_loaded_model` | job / operate | none | `recipe!: string` |
| `proxy.watch` | `/v1/proxy/watch` | job / admin | none | `generation!: string` |

### recipes

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `recipes.build` | `/v1/recipes/build` | job / operate | auto | `recipe!: string`, `hosts: string[]/null`, `cluster: string/null`, `overrides: object/null`, `dry_run: boolean/null`, `trust: boolean/null`, `rebuild: boolean/null`, `executor: string/null`, `executor_config: object/null`, `cache_dir: string/null`, `local_cache_dir: string/null`, `transfer_mode: string/null`, `transfer_interface: string/null`, `topology: string/null`, `sync_tuning: boolean/null`, `preserve_model_perms: boolean/null`, `skip_model_fan_out: boolean/null`, `builder_context: object/null` |
| `recipes.plan_build` | `/v1/recipes/plan_build` | job / read | auto | `recipe!: string`, `hosts: string[]/null`, `cluster: string/null`, `overrides: object/null`, `dry_run: boolean/null`, `trust: boolean/null`, `rebuild: boolean/null`, `executor: string/null`, `executor_config: object/null`, `cache_dir: string/null`, `local_cache_dir: string/null`, `transfer_mode: string/null`, `transfer_interface: string/null`, `topology: string/null`, `sync_tuning: boolean/null`, `preserve_model_perms: boolean/null`, `skip_model_fan_out: boolean/null`, `builder_context: object/null` |
| `recipes.realize` | `/v1/recipes/realize` | job / operate | auto | `recipe!: string`, `hosts: string[]/null`, `cluster: string/null`, `overrides: object/null`, `strategy_options: object/null`, `solo: boolean/null`, `dry_run: boolean/null`, `trust: boolean/null`, `ensure: boolean/null`, `scheduler: string/null`, `transfer_mode: string/null`, `cache_dir: string/null`, `offline: boolean/null`, `runtime_cache: boolean/null`, `auto_port: boolean/null`, `ray_port: integer/null`, `dashboard_port: integer/null`, `dashboard: boolean/null`, `init_port: integer/null`, `executor: string/null`, `executor_config: object/null`, `rootful: boolean/null`, `cluster_id_override: string/null`, `transfer_interface: string/null`, `local_cache_dir: string/null`, `sync_tuning: boolean/null`, `extra_docker_opts: string[]/null`, `topology: string/null`, `preserve_model_perms: boolean/null`, `skip_model_fan_out: boolean/null`, `recipe_ref: string/null` |
| `recipes.validate` | `/v1/recipes/validate` | job / read | auto | `recipe!: string`, `overrides: object/null` |

### setup

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `setup.ensure_local_key` | `/v1/setup/ensure_local_key` | job / admin | auto | `preferred: string/null`, `generate: boolean/null` |
| `setup.mesh_ssh_keys` | `/v1/setup/mesh_ssh_keys` | job / admin | auto | `hosts!: string[]`, `user: string/null`, `key: string/null`, `timeout: integer/null`, `dry_run: boolean/null` |
| `setup.plan` | `/v1/setup/plan` | job / read | auto | `hosts: string[]/null`, `cluster: string/null` |
| `setup.probe_hosts` | `/v1/setup/probe_hosts` | job / read | auto | `hosts: string[]/null`, `cluster: string/null` |
| `setup.probe_ssh_access` | `/v1/setup/probe_ssh_access` | job / read | auto | `hosts!: string[]`, `user: string/null`, `key: string/null`, `timeout: integer/null`, `max_workers: integer/null` |
| `setup.rdma_test` | `/v1/setup/rdma_test` | job / operate | auto | `hosts: string[]/null`, `cluster: string/null`, `suite: string/null`, `parallel_pairs: integer/null`, `full: boolean/null`, `image: string/null`, `duration: integer/null`, `queue_pairs: integer/null`, `msg_size: string/null`, `nccl_binary: string/null`, `gid_index: integer/null`, `link_type: string/null`, `force_container: boolean/null`, `keep_containers: boolean/null`, `dry_run: boolean/null` |
| `setup.run_steps` | `/v1/setup/run_steps` | job / admin | auto | `hosts: string[]/null`, `cluster: string/null`, `dry_run: boolean/null`, `only_steps: string[]/null` |
| `setup.undo` | `/v1/setup/undo` | job / admin | auto | `cluster!: string`, `dry_run: boolean/null`, `only_steps: string[]/null` |

### tailscale

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `tailscale.down` | `/v1/tailscale/down` | job / admin | auto | `hosts: string[]/null`, `cluster: string/null`, `remove: boolean/null`, `dry_run: boolean/null`, `ask_sudo: boolean` |
| `tailscale.expose` | `/v1/tailscale/expose` | job / admin | auto | `cluster: string/null`, `proxy: boolean/null`, `head_host: string/null`, `port: integer/null`, `set_proxy_host: boolean/null`, `ask_sudo: boolean` |
| `tailscale.join` | `/v1/tailscale/join` | job / admin | auto | `hosts: string[]/null`, `cluster: string/null`, `tag: string/null`, `ephemeral: boolean/null`, `enable_ssh: boolean/null`, `hostname: string/null`, `dry_run: boolean/null`, `ask_sudo: boolean` |
| `tailscale.status` | `/v1/tailscale/status` | job / read | auto | `hosts: string[]/null`, `cluster: string/null` |

### workloads

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `workloads.endpoints` | `/v1/workloads/endpoints` | job / read | none | `recipe_revision: string`, `cluster_candidates: string[]`, `cluster_id: string` |
| `workloads.ensure_ready` | `/v1/workloads/ensure_ready` | job / operate | controller | `recipe!: string`, `recipe_revision: string`, `cluster_candidates: string[]`, `overrides: object`, `timeout: number` |
| `workloads.find_intent` | `/v1/workloads/find_intent` | job / read | auto | `intent_id!: string`, `hosts!: string[]`, `cluster: string/null` |
| `workloads.inventory` | `/v1/workloads/inventory` | job / read | none | `{}` |
| `workloads.list` | `/v1/workloads/list` | query / read | auto | `limit: integer/null` |
| `workloads.materialize` | `/v1/workloads/materialize` | job / operate | auto | `recipe!: string`, `hosts: string[]/null`, `cluster: string/null`, `overrides: object/null`, `strategy_options: object/null`, `solo: boolean/null`, `dry_run: boolean/null`, `trust: boolean/null`, `ensure: boolean/null`, `scheduler: string/null`, `transfer_mode: string/null`, `cache_dir: string/null`, `offline: boolean/null`, `runtime_cache: boolean/null`, `auto_port: boolean/null`, `ray_port: integer/null`, `dashboard_port: integer/null`, `dashboard: boolean/null`, `init_port: integer/null`, `executor: string/null`, `executor_config: object/null`, `rootful: boolean/null`, `cluster_id_override: string/null`, `transfer_interface: string/null`, `local_cache_dir: string/null`, `sync_tuning: boolean/null`, `extra_docker_opts: string[]/null`, `topology: string/null`, `preserve_model_perms: boolean/null`, `skip_model_fan_out: boolean/null`, `recipe_ref: string/null` |
| `workloads.plan` | `/v1/workloads/plan` | job / read | auto | `recipe!: string`, `hosts: string[]/null`, `cluster: string/null`, `overrides: object/null`, `strategy_options: object/null`, `solo: boolean/null`, `dry_run: boolean/null`, `trust: boolean/null`, `ensure: boolean/null`, `scheduler: string/null`, `transfer_mode: string/null`, `cache_dir: string/null`, `offline: boolean/null`, `runtime_cache: boolean/null`, `auto_port: boolean/null`, `ray_port: integer/null`, `dashboard_port: integer/null`, `dashboard: boolean/null`, `init_port: integer/null`, `executor: string/null`, `executor_config: object/null`, `rootful: boolean/null`, `cluster_id_override: string/null`, `transfer_interface: string/null`, `local_cache_dir: string/null`, `sync_tuning: boolean/null`, `extra_docker_opts: string[]/null`, `topology: string/null`, `preserve_model_perms: boolean/null`, `skip_model_fan_out: boolean/null`, `recipe_ref: string/null` |
| `workloads.plugins` | `/v1/workloads/plugins` | job / read | none | `{}` |
| `workloads.reconcile` | `/v1/workloads/reconcile` | job / admin | controller | `recipe!: string`, `recipe_revision!: string`, `cluster_candidates: string[]`, `overrides: object`, `cluster_id!: string`, `acknowledge_effects!: boolean` |
| `workloads.run` | `/v1/workloads/run` | job / operate | auto | `recipe!: string`, `hosts: string[]/null`, `cluster: string/null`, `overrides: object/null`, `strategy_options: object/null`, `solo: boolean/null`, `dry_run: boolean/null`, `trust: boolean/null`, `ensure: boolean/null`, `scheduler: string/null`, `transfer_mode: string/null`, `cache_dir: string/null`, `offline: boolean/null`, `runtime_cache: boolean/null`, `auto_port: boolean/null`, `ray_port: integer/null`, `dashboard_port: integer/null`, `dashboard: boolean/null`, `init_port: integer/null`, `executor: string/null`, `executor_config: object/null`, `rootful: boolean/null`, `cluster_id_override: string/null`, `transfer_interface: string/null`, `local_cache_dir: string/null`, `sync_tuning: boolean/null`, `extra_docker_opts: string[]/null`, `topology: string/null`, `preserve_model_perms: boolean/null`, `skip_model_fan_out: boolean/null`, `recipe_ref: string/null`, `confirm: boolean`, `wait_ready: boolean`, `post_launch: boolean` |
| `workloads.sleep` | `/v1/workloads/sleep` | job / operate | controller | `recipe!: string`, `recipe_revision!: string`, `cluster_candidates: string[]`, `overrides: object`, `cluster_id!: string` |
| `workloads.status` | `/v1/workloads/status` | job / read | none | `recipe!: string`, `recipe_revision!: string`, `cluster_candidates: string[]`, `overrides: object`, `cluster_id!: string` |
| `workloads.stop` | `/v1/workloads/stop` | job / operate | controller | `recipe!: string`, `recipe_revision!: string`, `cluster_candidates: string[]`, `overrides: object`, `cluster_id!: string` |
| `workloads.stop_all` | `/v1/workloads/stop_all` | job / operate | auto | `hosts!: string[]`, `cluster: string/null`, `dry_run: boolean/null` |
| `workloads.stop_matching` | `/v1/workloads/stop_matching` | job / operate | auto | `cluster_id: string/null`, `recipe: string/null`, `hosts: string[]/null`, `cluster: string/null`, `overrides: object/null` |
| `workloads.wake` | `/v1/workloads/wake` | job / operate | controller | `recipe!: string`, `recipe_revision!: string`, `cluster_candidates: string[]`, `overrides: object`, `cluster_id!: string` |

## Bundled Kubernetes plugin

Requires the loaded `k8s` integration and `integration.k8s`.

### operations

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `plugins.k8s.cluster_info` | `/v1/plugins/k8s/ops/cluster_info` | job / operate | auto | `kubeconfig: string`, `context: string`, `namespace: string`, `pin: boolean` |
| `plugins.k8s.configure_service_account` | `/v1/plugins/k8s/ops/configure_service_account` | job / admin | auto | `name: string`, `namespace: string`, `kubeconfig: string`, `context: string`, `create_namespace: boolean`, `token_duration: string`, `write_kubeconfig: boolean`, `dry_run: boolean` |
| `plugins.k8s.ensure_kubectl` | `/v1/plugins/k8s/ops/ensure_kubectl` | job / admin | auto | `version: string`, `context: string`, `download: boolean` |
| `plugins.k8s.jobset_status` | `/v1/plugins/k8s/ops/jobset_status` | job / operate | auto | `name!: string`, `namespace: string`, `kubeconfig: string`, `context: string` |
| `plugins.k8s.kueue_status` | `/v1/plugins/k8s/ops/kueue_status` | job / operate | auto | `kubeconfig: string`, `context: string` |
| `plugins.k8s.list_nodes` | `/v1/plugins/k8s/ops/list_nodes` | job / operate | auto | `kubeconfig: string`, `context: string`, `selector: string`, `gpu_only: boolean` |
| `plugins.k8s.setup_kueue` | `/v1/plugins/k8s/ops/setup_kueue` | job / admin | auto | `install: boolean`, `kueue_version: string`, `jobset_version: string`, `namespace: string`, `kubeconfig: string`, `context: string`, `dry_run: boolean` |
| `plugins.k8s.stop_jobset` | `/v1/plugins/k8s/ops/stop_jobset` | job / operate | auto | `name!: string`, `namespace: string`, `kubeconfig: string`, `context: string` |

## External SparkRoute plugin

Requires the external SparkRoute plugin and `gateway.sparkroute`; it is not bundled with Sparkrun.

### endpoints

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `plugins.sparkroute.endpoints.discover` | `/v1/plugins/sparkroute/ops/endpoints/discover` | job / read | none | `binding: object` |
| `plugins.sparkroute.endpoints.metadata` | `/v1/plugins/sparkroute/ops/endpoints/metadata` | query / read | none | `cluster_id!: string`, `recipe_revision!: string`, `served_models!: string[]` |
| `plugins.sparkroute.endpoints.status` | `/v1/plugins/sparkroute/ops/endpoints/status` | job / read | none | `binding!: object`, `cluster_id!: string` |

<!-- END GENERATED DAEMON OPERATIONS -->

## Result families and interpretation

| Family | Final result / meaning |
| --- | --- |
| Catalog | Bounded recipe listings/facets, registry or cluster arrays, recipe details with stable `source_kind`, normalized overrides, revision, plugin items and issues; refresh/capacity/import mutations have their own schemas. |
| Planning/execution | Explicit Plan/Run/Build/Benchmark DTOs. Workload `cluster_id` is distinct from the daemon `job_id`. A serialized plan is a preview, not a resumable execution handle. |
| Generic workloads | `ensure_ready`: ready state, exact workload ID, credential-free endpoint, original observation timestamp, adoption flag and shared workload row. Guarded stop: offline state/IDs; selector/all teardown retain their separate Stop/StopAll DTOs. Inventory: recorded state/providers/actions. Control: the same workload row after provider observation/action. Reconcile: ID/state/receipt count. |
| Discovery | `discovery.snapshot` and `workloads.endpoints`: sweep ID/time, live/metadata mode, hosts/coverage/errors/completeness, safe endpoints. A partial or metadata-only sweep is not authoritative absence. |
| Proxy | State/models/start/stop/sync/alias/service DTOs; status includes discovery owner/backend/job/scope/health. `proxy.discovery.ensure` returns an existing/new watch job ID, not that watch's result. |
| SparkRoute | Gateway-only endpoint/model-card projections; optional metadata enrichment never changes readiness or observation time. Lifecycle and plugin/provider inventory use core `workloads.*`; binding resolution uses `catalog.recipe`. |
| Setup/network/cache/Kubernetes | Operation-specific result schemas; some are maps/arrays or booleans, not a universal envelope. Check `/v1/ops` rather than guessing from operation name. |

Query results and successful job results do not have the same HTTP envelope.
Transport errors use `detail` and usually a typed `error` record; job failures are
stored in the job's `error`. Terminal job states are `succeeded`, `failed`,
`cancelled`, `lost`; nonterminal states are `queued`, `starting`, `running`,
`waiting_input`, `cancelling`. Lost/uncertain effects require reconciliation;
`retryable` does not mean that replaying a mutation with a fresh key is safe.

## Extension model and current limits

Plugins register **local** names inside their loader-assigned identity. The host
binds `endpoints.discover` from SparkRoute to
`plugins.sparkroute.endpoints.discover`, with alias
`/v1/plugins/sparkroute/ops/endpoints/discover`. Registering another owner's
name, conflicting paths, or reserved infrastructure roots fails registration.
An explicit plugin `DaemonRoute` can adapt a closed envelope into declared
operations; authorization is still checked on the selected target. No production
custom adapter routes are currently registered by the reviewed core/K8s/SparkRoute set.

ColdSnap contributes the version-1 workload lifecycle **provider** interface;
it does not currently contribute `/v1/plugins/coldsnap/...` daemon operations.
Its receipt-backed status/sleep/wake are reached through core `workloads.*`
operations. The bundled LiL, OCI Relay and Sparkarena integrations do
not register independent daemon operations in this snapshot. Their effects and
structured feedback flow through the enclosing public API operation/job.

The live `/v1/ops` response is authoritative for an installation: disabled or
missing plugins change the available set. Compatibility pins for the coordinated
migration must be updated to committed, tested host/gateway revisions before
release; this worktree inventory is not a declaration that old pins support it.

## Updating this reference

The operation tables come from declarations, without executing handlers. From the
host checkout, with the matching reference plugin checkout available:

```sh
python scripts/render-daemon-api.py --sparkroute-checkout /path/to/sparkrun-sparkroute-plugin
```

Use `--check` to detect drift without writing. The prose/infrastructure table is
reviewed separately when routes or semantics change. `/v1/ops` and OpenAPI expose
full request/result schemas for the running installation.
