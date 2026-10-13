# Optional SparkRoute integration

SparkRoute is not bundled with Sparkrun. Its source and releases are documented in the [SparkRoute plugin instructions](https://github.com/sparksq/sparkrun-sparkroute-plugin).
ColdSnap is also distributed separately through its
[plugin repository](https://github.com/sparksq/sparkrun-coldsnap-plugin).
OCI Relay remains bundled with Sparkrun.

Managed plugins live outside the Sparkrun installation and receive compatible
updates during `sparkrun update` by default. Install ColdSnap with
`sparkrun plugins add https://github.com/sparksq/sparkrun-coldsnap-plugin --enable`.
See [plugin management](PLUGINS.md#install-and-manage-external-plugins) for source
trust, pins, `--no-update-plugins`, and rollback.

## Choose SparkRoute

LiteLLM is the default gateway on stable, beta, and alpha. After installing a
compatible SparkRoute plugin, select it explicitly:

```sh
sparkrun plugins add https://github.com/sparksq/sparkrun-sparkroute-plugin --enable
sparkrun setup features enable gateway.sparkroute
sparkrun proxy start --host 127.0.0.1
```

`--enable` enables plugin loading, preserving its capability settings. Explicitly
enabling `gateway.sparkroute` enables that capability and saves the gateway
selection. It does not disable LiteLLM or restart a running gateway. Plugin updates
preserve selection. `plugins disable sparkroute` preserves all settings while
preventing plugin loading; `plugins enable sparkroute` restores that configuration.

Shared `gateway` support is on by default. If disabled, enable it before activating
a provider. To select the bundled LiteLLM plugin again:

```sh
sparkrun setup features enable gateway.litellm
sparkrun proxy start
```

Use `--restart` to replace an already running gateway. Process-level status and
stop remain available for a recorded gateway whose plugin is missing; querying
its models or changing its configuration requires reinstalling the plugin.
See [proxy management](PROXY.md) for details.
