# Captured tool-eval-bench progress

Captured from the actual upstream CLIs against the local fake endpoint in
`tests/test_tool_eval_cli_integration.py`; no remote model was used. Only recognized
progress events are retained and `json_file` paths are replaced with `suite.json`.
Timing values are illustrative. Both versions use zero-based indices; parallel
result indices follow completion order, independently of scenario start order.

- v2.6.0: `992a6978ecbee2d72fa2ead9ccc509436769d088`
- v2.8.0: `0a1ba68c7709f429298bc92417a9e003f61de4e3`

The multi-trial recording deliberately repeats scenario IDs without trial IDs.
Regenerate with `SPARKRUN_TEST_TOOL_EVAL_CLI=1` and an optional
`SPARKRUN_TOOL_EVAL_CAPTURE_DIR` pointing to a scratch directory.
