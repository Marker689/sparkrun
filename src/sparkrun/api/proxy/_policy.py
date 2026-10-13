"""Gateway capability admission shared by Python and daemon API callers."""

from functools import wraps


def gateway_operation(function):
    @wraps(function)
    def call(*args, **kwargs):
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.core.features import is_feature_enabled
        from sparkrun.api.proxy._ops import GatewayUnavailable

        sctx = kwargs.get("sctx")
        config = sctx.config if sctx is not None else SparkrunConfig()
        if not is_feature_enabled("gateway", config=config):
            raise GatewayUnavailable("Gateway support is disabled. Enable it with: sparkrun setup features enable gateway")
        return function(*args, **kwargs)

    return call
