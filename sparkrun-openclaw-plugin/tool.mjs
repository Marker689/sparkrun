import {execFile} from "node:child_process";

const DEFAULT_TIMEOUT = 120_000;
const MAX_TIMEOUT = 3_600_000;

/** Build separately from the SDK entrypoint so execution can be tested offline. */
export function createSparkrunTool(run = execFile) {
    return {
        name: "sparkrun_exec",
        label: "Sparkrun",
        description:
            "Execute a finite Sparkrun CLI command. Supply args without 'sparkrun'; no shell expansion occurs. " +
            "Use ['logs', workloadId, '--tail', '200'] for bounded logs. Run --no-follow still waits for readiness; " +
            "--no-ready-wait skips that wait, not preparation. Set a longer timeout for downloads/benchmarks. " +
            "Interactive setup, sudo, trust, and login need a user terminal.",
        parameters: {
            type: "object",
            properties: {
                args: {
                    type: "array",
                    items: {type: "string"},
                    minItems: 1,
                    description: "CLI arguments, e.g. ['benchmark', 'tools', 'my-recipe', '--skip-run'].",
                },
                timeout: {
                    type: "integer",
                    minimum: 1,
                    maximum: MAX_TIMEOUT,
                    description: "Local CLI timeout in ms (default 120000, maximum 3600000).",
                },
            },
            required: ["args"],
            additionalProperties: false,
        },
        // registerTool passes the call ID first, then parameters and cancellation.
        async execute(_toolCallId, params, signal) {
            const {args, timeout = DEFAULT_TIMEOUT} = params ?? {};
            if (!Array.isArray(args) || !args.length || args.some(arg => typeof arg !== "string" || arg.includes("\0"))) {
                return failure("Provide args as a non-empty array of CLI argument strings, without 'sparkrun'.");
            }
            if (args[0] === "sparkrun") {
                return failure("Omit 'sparkrun' from args; the executable is selected by the tool.");
            }
            if (!Number.isInteger(timeout) || timeout < 1 || timeout > MAX_TIMEOUT) {
                return failure("timeout must be an integer between 1 and 3600000 milliseconds.");
            }
            return new Promise(resolve => {
                const finish = (error, stdout = "", stderr = "") => {
                    const aborted = error?.name === "AbortError";
                    const truncated = error?.code === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER";
                    const timedOut = Boolean(error?.killed && !aborted && !truncated);
                    const exitCode = error ? (typeof error.code === "number" ? error.code : null) : 0;
                    const reason = aborted ? "Local command cancelled."
                        : timedOut ? "Local command timed out."
                        : truncated ? "Command output exceeded the capture limit; request a smaller log tail."
                        : error?.code === "ENOENT" ? "Sparkrun executable not found on the OpenClaw host PATH."
                        : error ? `Sparkrun command failed${exitCode === null ? "" : ` (exit ${exitCode})`}.` : "";
                    const note = (aborted || timedOut)
                        ? "Remote workloads may still be running; inspect sparkrun status before retrying." : "";
                    const output = [reason, stdout, stderr, note].filter(Boolean).join("\n");
                    resolve({
                        content: [{type: "text", text: output || "Command completed with no output."}],
                        isError: Boolean(error),
                        details: {exitCode, signal: error?.signal ?? null, timedOut, cancelled: aborted, truncated},
                    });
                };
                try {
                    const child = run("sparkrun", args, {
                        encoding: "utf8",
                        timeout,
                        signal,
                        shell: false,
                        windowsHide: true,
                        maxBuffer: 4 * 1024 * 1024,
                    }, finish);
                    // Captured commands cannot answer interactive prompts; deliver EOF.
                    child.stdin?.end();
                } catch (error) {
                    finish(error);
                }
            });
        },
    };
}

function failure(message) {
    return {content: [{type: "text", text: message}], isError: true};
}
