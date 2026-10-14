// OpenClaw supplies this SDK when it loads the plugin.
// @ts-ignore -- the host SDK is not a build dependency of this source-only package.
import {definePluginEntry, type OpenClawPluginApi} from "openclaw/plugin-sdk/plugin-entry";
import {createSparkrunTool} from "./tool.mjs";

export default definePluginEntry({
    id: "@sparkarena/sparkrun",
    name: "sparkrun",
    description: "Manage inference workloads using the Sparkrun 0.4.0 CLI.",
    register(api: OpenClawPluginApi) {
        api.registerTool(createSparkrunTool());
    },
});
