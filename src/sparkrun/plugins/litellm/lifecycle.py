"""Explicit activation of the LiteLLM capability."""

from sparkrun.plugin_lifecycle import select_gateway


def enable_feature(context):
    if context.feature == "gateway.litellm":
        return select_gateway(context, "litellm")
