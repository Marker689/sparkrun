"""Tests for ``sparkrun setup completion`` snippet generation.

Covers issue #198: the completion snippet must embed the absolute,
shell-resolved path to ``sparkrun`` so it is robust against PATH ordering
in the user's shell rc file, rather than the bare ``sparkrun`` command.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.cli import main


@pytest.fixture
def rc_file(tmp_path: Path):
    """Redirect the shell rc file to a temp path for the duration of a test."""
    target = tmp_path / "rc"
    with mock.patch("sparkrun.cli._setup._commands._shell_rc_file", return_value=target):
        yield target


@mock.patch("shutil.which", return_value="/home/u/.local/bin/sparkrun")
def test_bash_embeds_absolute_path(mock_which, rc_file: Path):
    result = CliRunner().invoke(main, ["setup", "completion", "--shell", "bash"])
    assert result.exit_code == 0, result.output

    contents = rc_file.read_text()
    assert "_SPARKRUN_COMPLETE=bash_source /home/u/.local/bin/sparkrun)" in contents
    # Must not fall back to the bare command when the path resolves.
    assert "=bash_source sparkrun)" not in contents


@mock.patch("shutil.which", return_value="/home/u/.local/bin/sparkrun")
def test_zsh_embeds_absolute_path(mock_which, rc_file: Path):
    result = CliRunner().invoke(main, ["setup", "completion", "--shell", "zsh"])
    assert result.exit_code == 0, result.output
    assert "_SPARKRUN_COMPLETE=zsh_source /home/u/.local/bin/sparkrun)" in rc_file.read_text()


@mock.patch("shutil.which", return_value="/home/u/.local/bin/sparkrun")
def test_fish_embeds_absolute_path(mock_which, rc_file: Path):
    result = CliRunner().invoke(main, ["setup", "completion", "--shell", "fish"])
    assert result.exit_code == 0, result.output
    assert "_SPARKRUN_COMPLETE=fish_source /home/u/.local/bin/sparkrun | source" in rc_file.read_text()


@mock.patch("shutil.which", return_value=None)
def test_falls_back_to_bare_command_when_unresolved(mock_which, rc_file: Path):
    result = CliRunner().invoke(main, ["setup", "completion", "--shell", "bash"])
    assert result.exit_code == 0, result.output
    assert "_SPARKRUN_COMPLETE=bash_source sparkrun)" in rc_file.read_text()


@mock.patch("shutil.which", return_value="/opt/my tools/sparkrun")
def test_path_with_spaces_is_quoted(mock_which, rc_file: Path):
    result = CliRunner().invoke(main, ["setup", "completion", "--shell", "bash"])
    assert result.exit_code == 0, result.output
    # shlex.quote wraps the space-containing path in single quotes.
    assert "_SPARKRUN_COMPLETE=bash_source '/opt/my tools/sparkrun')" in rc_file.read_text()


def test_setup_install_embeds_the_installed_tool_not_the_uvx_environment(tmp_path, rc_file: Path):
    """``uvx sparkrun setup install`` runs with uvx's throwaway environment first on PATH.

    Completion must name the tool it just installed. The PATH lookup embedded the
    uv cache copy, which ``uv cache clean`` deletes, breaking every new shell.
    """
    uvx_copy = str(tmp_path / ".cache/uv/archive-v0/abc/bin/sparkrun")
    installed = tmp_path / "tools/sparkrun"

    def which(name):
        return "/usr/bin/uv" if name == "uv" else uvx_copy

    def run(argv, **kwargs):
        stdout = "sparkrun v0.4.1 (%s)\n" % installed if argv[1:3] == ["tool", "list"] else ""
        return mock.Mock(returncode=0, stdout=stdout, stderr="")

    with mock.patch("shutil.which", side_effect=which), mock.patch("subprocess.run", side_effect=run):
        result = CliRunner().invoke(main, ["setup", "install", "--no-update-registries", "--shell", "bash"])
    assert result.exit_code == 0, result.output

    contents = rc_file.read_text()
    assert "_SPARKRUN_COMPLETE=bash_source %s)" % (installed / "bin/sparkrun") in contents
    assert uvx_copy not in contents


@mock.patch("shutil.which", return_value="/home/u/.local/bin/sparkrun")
def test_idempotent_when_already_configured(mock_which, rc_file: Path):
    rc_file.write_text('# existing\neval "$(_SPARKRUN_COMPLETE=bash_source sparkrun)"\n')
    before = rc_file.read_text()
    result = CliRunner().invoke(main, ["setup", "completion", "--shell", "bash"])
    assert result.exit_code == 0, result.output
    assert "already configured" in result.output
    assert rc_file.read_text() == before  # not appended twice
