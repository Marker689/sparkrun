"""Bundled LiteLLM gateway; shared proxy contracts and supervision stay in core."""

from sparkrun import __version__ as __version__

SPARKRUN_PLUGIN_API_VERSION = 1
SPARKRUN_PLUGIN_SCAN_SUBMODULES = False
SPARKRUN_PLUGIN_MANIFEST = {
    "api": 1,
    "features": {
        "gateway.litellm": {"description": "LiteLLM inference gateway", "default": True},
    },
    "hooks": {"feature.enable": "sparkrun.plugins.litellm.lifecycle:enable_feature"},
}


def _load_engine():
    from sparkrun.plugins.litellm.engine import ProxyEngine

    return ProxyEngine


def register(v):
    from sparkrun.core.features import feature_gate_enabled

    if not feature_gate_enabled("gateway", v):
        return
    from sparkrun.proxy.gateway import register_gateway

    register_gateway("litellm", feature_flag="gateway.litellm", loader=_load_engine)
