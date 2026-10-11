# Application profile CLI responsiveness

Measured 2026-09-11 (UTC) using Python 3.12.3 on
`Linux-6.14.0-1013-nvidia-aarch64-with-glibc2.39`. The baseline source is
`03b9114c78560d251889c5097cd6c1dfc09ea4d7`; the measured variant was the application-profile and plugin worktree snapshot
at the measurement date, not a benchmark of every subsequent branch revision. [Raw samples and environment metadata](benchmarks/application-profile-cli-startup.json)
are retained with this report.

## Method

The [benchmark script](../scripts/benchmark-cli-startup.py) extracts the baseline
with `git archive` and runs every variant with the same interpreter and dependencies.
Each process has isolated configuration and an empty registry catalog. Telemetry,
external source plugins and installed integration loading are disabled. Built-in
plugins still follow their normal feature gates.

The alternate variants construct a small `ApplicationProfile` in memory. They
have separate namespaces, an empty catalog, and explicit tuning/Arena/Kubernetes
defaults. `alternate-next` additionally selects a declared `next` release channel
and a channel-specific feature default. They require no downstream application
package and perform no self-update. Installed package and plugin composition is
covered separately by the offline wheel tests.

Every sample starts a fresh Python interpreter, including Click's actual Bash
completion protocol. Expected help and completion candidates are checked. There
are 3 warmups and 20 measured samples per variant/scenario, randomly interleaved
(320 measured processes). No test suite ran concurrently. OS and bytecode caches are warm.

## Median elapsed time (milliseconds)

| Request | Baseline | Current Sparkrun | Alternate profile | Alternate next |
| --- | ---: | ---: | ---: | ---: |
| help | 152.79 | 158.88 | 152.77 | 155.80 |
| root-completion | 153.06 | 158.30 | 157.40 | 154.21 |
| nested-completion | 156.13 | 159.13 | 149.45 | 156.20 |
| option-completion | 151.33 | 159.79 | 154.48 | 156.65 |

## 95th percentile (milliseconds)

| Request | Baseline | Current Sparkrun | Alternate profile | Alternate next |
| --- | ---: | ---: | ---: | ---: |
| help | 165.17 | 170.22 | 167.15 | 168.68 |
| root-completion | 164.45 | 169.77 | 167.99 | 165.56 |
| nested-completion | 169.06 | 171.00 | 166.22 | 164.76 |
| option-completion | 169.47 | 171.38 | 168.94 | 169.31 |

## Interpretation and limits

Current Sparkrun's median cost is 3.0–8.5 ms above the baseline for these
requests. This compares the complete core changes with the baseline; it does not
isolate the profile wrapper, setup registry or individual plugins. Shared-machine
scheduling affects the measurements, especially the tails. These results do not
establish a zero-cost abstraction or a timing guarantee.

Selected downstream integrations, large user catalogs, network/SSH completion,
cold caches, shell startup and inference are outside this measurement. Completion
still pays interpreter/import/bootstrap costs for each subprocess. Setup step
registration performs no target probes; probing starts only when setup runs.

## Reproduce

From a prepared Sparkrun development environment:

```bash
.venv/bin/python scripts/benchmark-cli-startup.py \
  --baseline 03b9114 --samples 20 --warmups 3 \
  --output /tmp/application-profile-cli-startup.json
```

No external application checkout or hardware plugin is required. There is no
automated timing threshold on a shared controller.


## Daemon integration check (2026-10-10)

Compared the daemon implementation worktree with pre-daemon `857c784` using
this same script, Python 3.12, two warmups and ten interleaved samples per
variant/scenario. No other test suite ran during the measured rounds. Timing is
local wall time with warm filesystem/bytecode caches, not a release threshold.

| Request | Pre-daemon median (ms) | Daemon worktree median (ms) | Delta (ms) |
| --- | ---: | ---: | ---: |
| help | 206.30 | 217.14 | +10.84 |
| root-completion | 213.55 | 218.97 | +5.42 |
| nested-completion | 214.73 | 224.05 | +9.31 |
| option-completion | 220.66 | 223.94 | +3.28 |

The measured increase is 1.5–5.3%. Fresh-process import guards separately prove
that CLI/completion, API bootstrap, plugin registration and worker startup do
not import FastAPI, Starlette, uvicorn, Pydantic or AnyIO. The implementation
adds framework-free command/coordination metadata; the web stack loads only on
server startup. This sample measures the complete change, not a causal breakdown
of those imports. Large catalogs, network calls, inference and RSS are outside
this comparison. It does not justify introducing reusable workers; v1 retains
a fresh isolated process per mutating/long operation.

```bash
.venv/bin/python scripts/benchmark-cli-startup.py \
  --baseline 857c784 --samples 10 --warmups 2 \
  --output /tmp/daemon-cli-startup.json
```

A separate empty-catalog check compared 30 warm local `Client.call` requests
(including authenticated identity verification) with ten fresh-process
`initialize()` + `api.list_clusters()` calls, after two warmups each. Medians
were **11.78 ms warm** and **202.35 ms cold**; API startup was 396 ms. This fixture
uses no remote hosts or registry network access. It demonstrates query bootstrap
reuse, not inference speed or a case for reusable mutation workers. Measurements
are saved locally in `/tmp/daemon-warm-benchmark.json`.
