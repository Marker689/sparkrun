"""Keep shipped assistant references aligned with the CLI without executing workflows."""

from contextlib import ExitStack
from pathlib import Path
import re
import runpy
import shlex
import json

import click
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = sorted(
    path
    for package in ("sparkrun-cc-plugin", "sparkrun-openclaw-plugin")
    for pattern in ("*.md", "commands/*.md", "skills/*/SKILL.md")
    for path in (ROOT / package).glob(pattern)
)


def test_generated_openclaw_skills_match_source():
    generator = runpy.run_path(str(ROOT / "scripts/sync-assistant-skills.py"))
    pairs = list(generator["rendered_skills"]())
    assert len(pairs) == 3
    for target, expected in pairs:
        assert target.read_text() == expected, f"Regenerate {target} with scripts/sync-assistant-skills.py"


def test_skill_names_do_not_shadow_claude_action_commands():
    package = ROOT / "sparkrun-cc-plugin"
    commands = {path.stem for path in (package / "commands").glob("*.md")}
    names = set()
    for path in (package / "skills").glob("*/SKILL.md"):
        metadata = yaml.safe_load(path.read_text().split("---", 2)[1])
        assert metadata["name"] == path.parent.name
        assert metadata["name"] not in commands
        assert metadata["name"] not in names
        names.add(metadata["name"])
    manifest = json.loads((ROOT / "sparkrun-openclaw-plugin/openclaw.plugin.json").read_text())
    assert set(manifest["skills"]) == {f"skills/{name}" for name in names}


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda path: str(path.relative_to(ROOT)))
def test_documented_cli_examples_parse_without_dispatch(path, tmp_path):
    from sparkrun.cli import main

    token_file = tmp_path / "token"
    token_file.write_text("test-token")
    for block in re.findall(r"```bash\n(.*?)```", path.read_text(), re.S):
        for line in block.splitlines():
            if not line.startswith("sparkrun "):
                continue
            example = re.sub(r"<([^>]+)>", lambda match: str(token_file) if match[1] == "token_file" else "example", line)
            args = shlex.split(example, comments=True)[1:]
            if args == ["--version"]:
                continue  # Eager version rendering has no command syntax to check.
            command = main
            name = "sparkrun"
            parent = None
            with ExitStack() as stack:
                while True:
                    try:
                        ctx = stack.enter_context(command.make_context(name, args, parent=parent))
                        if not isinstance(command, click.Group):
                            break
                        remaining = [*ctx._protected_args, *ctx.args]
                        if not remaining:
                            break
                        name, command, args = command.resolve_command(ctx, remaining)
                        assert command is not None, line
                        parent = ctx
                    except click.ClickException as error:
                        pytest.fail(f"{path.relative_to(ROOT)}: {line}\n{error.format_message()}")
            # Deliberately never call invoke(): install/stop/cleanup examples only parse.
