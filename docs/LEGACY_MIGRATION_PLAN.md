# Legacy CLI and stored-data migration plan

Sparkrun 0.4 removes obsolete Python aliases and private adapters in one breaking
API release; see [the API migration guide](DISTRIBUTION_API_MIGRATION.md).
Existing command-line automation and saved workloads/results have a separate
retirement policy. The migrations described below are planned work, not commands
that already exist.

## CLI retirement window

| Surface | Supported replacement | Retirement boundary |
| --- | --- | --- |
| `cluster import --from-spark-vllm-docker-env PATH` | `cluster import svd PATH` (place `--name`, `--default`, and `--dry-run` after `svd`) | Retain the warning and behavior throughout 0.4.x. Remove no earlier than 0.5.0, with a release-note reminder and migration tests. |
| `recipe update [--registry NAME]` | `registry update [NAME]` for registries; `update` for the application | Removed in 0.4.0 (hidden and deprecated since 0.1.0). Use the mappings below. |
| `run-recipe.sh` | Native `sparkrun` commands | Retain through 0.5.x. Remove no earlier than 0.6.0, after a full release announces removal and maintained automation has migrated. The script now calls `cluster import svd` internally. |

These are earliest removal versions, not automatic deadlines. Before removal,
check maintained scripts, CI examples, and docs for callers and publish the exact
release containing the removal. Keep the `benchmark RECIPE` default invocation
and `perf` convenience alias; they remain intentional CLI conveniences.

### Command mappings

- `sparkrun recipe update` becomes `sparkrun registry update` to refresh all
  configured registries. For updating Sparkrun itself, use `sparkrun update`.
- `sparkrun recipe update --registry NAME` becomes `sparkrun registry update NAME`.
- `run-recipe.sh --list` becomes `sparkrun recipe list`.
- `run-recipe.sh RECIPE -n h1,h2 -t IMAGE --tp 2 -d` becomes
  `sparkrun run RECIPE --hosts h1,h2 --image IMAGE --tp 2`. Native `run` detaches
  by default; use `--foreground` for the shim's default foreground behavior.
- `run-recipe.sh RECIPE --config FILE.env` becomes an explicit
  `sparkrun cluster import svd FILE.env`, followed by
  `sparkrun run RECIPE --cluster NAME` using the returned name. Import remains
  idempotent through the stored source path; keep secrets in the `.env` file.
- Translate engine arguments after `--` to supported `-o key=value` options.
  Use `--executor-args` for Docker volume/publish arguments. Consult
  `sparkrun run --help` for quoting and backend-specific options.
- `--discover`, `--show-env`, `--build-only`, and `--download-only` already lack
  equivalent behavior in the shim. Its retirement must not promise those modes.

## Saved-data migrations required before reader removal

| Data/read surface | Required migration | Verification before removing the old reader |
| --- | --- | --- |
| Unlabelled Sparkrun containers, Kubernetes ownership, historical local PID/log paths, old exported services | Inventory and explicitly adopt supported resources into canonical ownership/metadata, or provide an operator recovery command that can inspect and stop them. Resolve the application/config identity before adoption. | Existing jobs remain inspectable, stoppable, and log-readable; foreign applications and ambiguous resources are never adopted automatically. Test interrupted adoption and missing/stale metadata. |
| Benchmark identities, older host fields, saved specification, restored measurement timestamps | Add a versioned state converter, with preview and backup, that preserves benchmark IDs, completed measurements, failed/pending tasks, original timestamps, and execution provenance. Write the canonical version after conversion. | Load fixtures from supported releases, migrate twice, resume only pending work, and export equivalent prior results. Reject newer unsupported schemas without overwriting them. |
| `BenchmarkStateSnapshot.extras` | Move each integration's historical data into its own `context.data` namespace while preserving publication identities. Arena currently reads the historical `submission_id` when resuming. No SparkRoute consumer was found in the maintained checkout. | An interrupted or retried migration cannot generate a second submission ID or duplicate an upload. Confirm all maintained integration readers are migrated before removing `extras`. |
| v1 recipe fields/topology, `eugr-vllm`, persisted `cluster_config` | Audit maintained recipe registries and saved recipe snapshots. Supply a converter to canonical recipe/runtime/builder fields, reporting ambiguous translations for manual resolution. | Compare resolved commands, images, placement, environment, workload identity, and stop/log lookup before and after conversion. Update registry content before retiring parsing/runtime support. |

Do not delete these readers solely because the CLI window has elapsed. Each
removal needs versioned fixtures, an implemented conversion or recovery path,
and a release note specifying supported source versions. Preserve legacy
ownership recognition within its existing Sparkrun-only boundary.

## Migration implementation sequence

1. Inventory the source schema/version and referenced resources without writes.
   Keep credentials out of reports and do not probe unrelated application roots.
2. Produce a deterministic preview with conflicts and unsupported cases. Back up
   the original data before applying a supported conversion.
3. Apply under the existing per-state lock, using atomic replacement. Journal
   multi-file changes so interruption can be resumed or rolled back.
4. Verify IDs, ownership, measurements, and references; keep original data until
   the operator has verified recovery. Never silently regenerate an identity.
5. Publish the converter/recovery tool for at least one release before removing
   the corresponding reader. Retain fixtures and conversion tests thereafter.
