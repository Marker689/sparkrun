#!/usr/bin/env python3
"""Generate OpenClaw skills from Claude Code references, changing tool guidance only."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TOOL_USAGE = """<Tool_Usage>
Use `sparkrun_exec` for finite Sparkrun commands. Pass `args` as an array of CLI
arguments without the executable name, e.g. {"args": ["logs", "<workload_id>",
"--tail", "200"]}. Replace placeholders with actual values. Shell operators,
environment assignments, and shell expansion are not supported. The process
inherits the OpenClaw host's environment and PATH.

Default timeout is 120000 ms; set `timeout` up to 3600000 ms for preparation,
readiness, or benchmarks. Cancellation/timeouts stop the local CLI process, not
necessarily remote workloads; inspect status before retrying. Use an interactive
user terminal for setup/SSH/sudo/trust/login prompts. For indefinite monitoring,
use the host's cancellable background execution or an explicit short timeout.
Never print credentials or claim a timed-out launch was rolled back.
</Tool_Usage>"""


def rendered_skills():
    for source in sorted((ROOT / "sparkrun-cc-plugin/skills").glob("*/SKILL.md")):
        content, count = re.subn(r"<Tool_Usage>.*?</Tool_Usage>", lambda _: TOOL_USAGE, source.read_text(), flags=re.S)
        if count != 1:
            raise ValueError(f"Expected one Tool_Usage section in {source}")
        target = ROOT / "sparkrun-openclaw-plugin/skills" / source.parent.name / source.name
        yield target, content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated skills differ; do not write files.")
    args = parser.parse_args()
    stale = []
    for target, content in rendered_skills():
        if target.exists() and target.read_text() == content:
            continue
        stale.append(str(target.relative_to(ROOT)))
        if not args.check:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    if args.check and stale:
        parser.exit(1, "Regenerate with python scripts/sync-assistant-skills.py:\n" + "\n".join(stale) + "\n")


if __name__ == "__main__":
    main()
