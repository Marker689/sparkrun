# Spark Arena integration

The `sparkarena` in-tree plugin (`sparkrun.plugins.sparkarena`) owns
authentication, the `arena` command group, benchmark
submission flags, the pinned Arena profile, and result upload. It defaults on;
`sparkrun plugins disable sparkarena` disables the whole integration. Application
profiles can choose its loading default. `integration.arena` is a legacy fallback. It does not register a measurement framework.

```bash
sparkrun arena login
sparkrun benchmark perf my-recipe --arena
sparkrun benchmark perf my-recipe --arena --local-test
sparkrun benchmark resume <benchmark-id>
```

`arena benchmark <recipe>` and `arena benchmark resume <benchmark-id>` are aliases
for the same benchmark execution and integration lifecycle. They are hidden from
help by default; `SPARKRUN_ADVANCED=1` shows them. Prefer `sparkrun benchmark <recipe> --arena`
and `sparkrun benchmark resume <benchmark-id>`. `--local-test`
skips authentication and upload, allows the recipe's own benchmark profile, and
writes the submission artifacts to the active application's cache. The ordinary
benchmark path requires `--arena` alongside `--local-test`.

Real submissions default to `@official/spark-arena-v2` and the performance
category; explicit benchmark profile/category choices remain authoritative.
Dry runs preview the flow without authenticating, uploading, or saving submission
state. Credentials live under the active application's config directory.

The plugin saves its submission ID and recipe/metadata snapshot before measuring,
then finalizes only successful benchmarks. Failed uploads retain the same ID for
retry through either resume command. Once an upload succeeds, later retries skip
it. Partial uploads retry the same remote object paths, rather than minting a
second submission. A process killed after the server accepts an upload but before
local success is saved can repeat that upload; remote object names still use the
same submission ID.

Published host metadata uses stable pseudonyms. Detailed launch timeline spans
stay local because their clock labels and arbitrary attributes can identify
hosts; numeric startup summaries remain available for fresh runs. Resumed data
is marked as such and excludes startup metrics from the retry's invocation.

See [benchmark integration contracts](../../../../docs/PLUGINS.md#benchmark-integrations)
for the host lifecycle and option registration API. Auth/upload network calls are
mocked in the regression suite; no live leaderboard submission is required.
