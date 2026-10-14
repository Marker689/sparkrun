"""Tests for the image ENTRYPOINT passthrough preflight.

sparkrun appends its launcher as CMD *arguments*, so an image whose ENTRYPOINT
consumes them runs a different program than intended.  Inspection cannot
distinguish that from the harmless passthrough wrappers most NGC images ship,
so the probe settles it empirically — these tests pin both the bash semantics
(against a fake ``docker``) and the Python plumbing around it.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess

import pytest

from sparkrun.containers.entrypoint import (
    NO_PROBE_ENV,
    PROBE_TOKEN,
    VERDICT_ABSENT,
    VERDICT_CONSUMES,
    VERDICT_PASSTHROUGH,
    VERDICT_UNKNOWN,
    EntrypointProbe,
    build_probe_script,
    parse_probe_output,
    probe_image_entrypoint,
)
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.ssh import RemoteResult


# --------------------------------------------------------------------------
# Script rendering
# --------------------------------------------------------------------------


def _code_lines(script: str) -> list[str]:
    """Script lines with comments stripped — the probe's prose mentions the same tokens."""
    return [line for line in script.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_build_probe_script_renders_placeholders():
    script = build_probe_script("ghcr.io/acme/img:1.0", ["--gpus", "all"])
    code = _code_lines(script)

    assert script.startswith("#!/bin/bash")
    assert any("ghcr.io/acme/img:1.0" in line for line in code)
    assert any("--gpus all" in line for line in code)
    # Docker's Go template must survive .format() un-doubled.
    assert "--format '{{json .Config.Entrypoint}}'" in script
    # Both runs present: the real argv shape, then the cleared-entrypoint control.
    assert sum(line.count("docker run") for line in code) == 2
    assert sum(line.count("--entrypoint ''") for line in code) == 1


def test_probe_sentinel_is_computed_not_literal():
    """An entrypoint echoing its argv back must not be able to fake a pass.

    The command embeds ``$((21*2))``; only a shell that actually *ran* it can
    print the evaluated token.  If the literal token appeared in the command,
    a consuming entrypoint that echoes the rejected argv would match.
    """
    run_lines = [line for line in _code_lines(build_probe_script("img", [])) if "docker run" in line]

    assert run_lines, "expected the probe to start a container"
    for line in run_lines:
        assert PROBE_TOKEN not in line, "sentinel must never appear literally in the probed argv"
        assert "$((21*2))" in line


def test_build_probe_script_quotes_image():
    script = build_probe_script("img; rm -rf /", [])
    assert "'img; rm -rf /'" in script


# --------------------------------------------------------------------------
# Script semantics, executed against a fake docker
# --------------------------------------------------------------------------

_FAKE_DOCKER = r"""#!/bin/bash
# Fake docker standing in for a %(mode)s image.
last_arg() { for x in "$@"; do :; done; echo "$x"; }

if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
    echo '%(entrypoint)s'
    exit 0
fi

if [ "$1" = "run" ]; then
    cleared=0
    for a in "$@"; do
        [ "$a" = "--entrypoint" ] && cleared=1
    done
    payload=$(last_arg "$@")
%(run_body)s
fi
exit 0
"""

# A passthrough wrapper (`exec "$@"`) runs sparkrun's command either way.
_RUN_PASSTHROUGH = '    exec bash -c "$payload"'

# A consuming entrypoint eats the args -- and echoes the value it rejected back
# on stderr, exactly as vLLM's argparse does.  Clearing the entrypoint fixes it.
_RUN_CONSUMING = """    if [ "$cleared" = "1" ]; then
        exec bash -c "$payload"
    fi
    echo "error: argument -cc: invalid JSON, input_value='$payload'" >&2
    exit 2"""

# Docker itself cannot start the container (stale CDI spec, GPU unavailable,
# ...) -- fails identically with and without the entrypoint, so the entrypoint
# is *not* provably the cause.
_RUN_BROKEN = '    echo "docker: error response from daemon" >&2; exit 125'


def _write_fake_docker(tmp_path, entrypoint: str, run_body: str, mode: str) -> dict:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    script = bindir / "docker"
    script.write_text(_FAKE_DOCKER % {"mode": mode, "entrypoint": entrypoint, "run_body": run_body})
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    env = dict(os.environ)
    env["PATH"] = "%s%s%s" % (bindir, os.pathsep, env.get("PATH", ""))
    return env


def _run_script(script: str, env: dict) -> str:
    proc = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, env=env)
    return proc.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
@pytest.mark.parametrize(
    "entrypoint,run_body,expected",
    [
        ("null", _RUN_PASSTHROUGH, VERDICT_ABSENT),
        ("[]", _RUN_PASSTHROUGH, VERDICT_ABSENT),
        ('["/opt/nvidia/nvidia_entrypoint.sh"]', _RUN_PASSTHROUGH, VERDICT_PASSTHROUGH),
        ('["vllm","serve"]', _RUN_CONSUMING, VERDICT_CONSUMES),
        ('["vllm","serve"]', _RUN_BROKEN, VERDICT_UNKNOWN),
    ],
    ids=["no-entrypoint", "empty-entrypoint", "passthrough", "consuming", "docker-broken"],
)
def test_probe_script_semantics(tmp_path, entrypoint, run_body, expected):
    """The bash itself classifies each idiom correctly."""
    env = _write_fake_docker(tmp_path, entrypoint, run_body, expected)
    out = _run_script(build_probe_script("img:tag", []), env)
    assert parse_probe_output(out)[0] == expected


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
def test_computed_sentinel_defeats_echoed_argv(tmp_path):
    """A consuming entrypoint that echoes the argv must not read as a pass.

    This is the concrete regression that motivated the computed sentinel:
    vLLM's argparse reports the value it rejected, so the probed argv comes
    *back* in the output.  Because the argv carries ``$((21*2))`` unevaluated
    and only a shell that ran it prints ``42``, the echo cannot match — proven
    here by merging stderr into the capture and still getting ``fail``.
    """
    env = _write_fake_docker(tmp_path, '["vllm","serve"]', _RUN_CONSUMING, "consuming")
    script = build_probe_script("img:tag", [])
    merged = script.replace("2>/dev/null || true)", "2>&1 || true)")
    assert merged != script, "probe no longer discards stderr — update this test"

    for variant in (script, merged):
        verdict, entrypoint = parse_probe_output(_run_script(variant, env))
        assert verdict == VERDICT_CONSUMES
        assert entrypoint == '["vllm","serve"]'


def test_probe_captures_stdout_only():
    """Defense in depth behind the computed sentinel; also keeps the probe quiet.

    The sentinel alone already defeats an echoed argv (see above), but a
    literal sentinel is an easy regression to introduce, and stderr is where
    consuming entrypoints put the argv they rejected.
    """
    run_lines = [line for line in _code_lines(build_probe_script("img", [])) if "docker run" in line]

    assert run_lines
    for line in run_lines:
        assert "2>/dev/null" in line


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SPARKRUN_PROBE=pass", VERDICT_PASSTHROUGH),
        ("SPARKRUN_PROBE=fail", VERDICT_CONSUMES),
        ("SPARKRUN_PROBE=absent", VERDICT_ABSENT),
        ("SPARKRUN_PROBE=unknown", VERDICT_UNKNOWN),
        ("SPARKRUN_PROBE=garbage", VERDICT_UNKNOWN),
        ("", VERDICT_UNKNOWN),
    ],
)
def test_parse_probe_output_verdicts(raw, expected):
    assert parse_probe_output(raw)[0] == expected


def test_parse_probe_output_normalizes_absent_entrypoint():
    assert parse_probe_output("SPARKRUN_ENTRYPOINT=null\nSPARKRUN_PROBE=absent")[1] == ""
    assert parse_probe_output('SPARKRUN_ENTRYPOINT=["vllm","serve"]\nSPARKRUN_PROBE=fail')[1] == '["vllm","serve"]'


# --------------------------------------------------------------------------
# probe_image_entrypoint plumbing (fail-open contract)
# --------------------------------------------------------------------------


def _patch_remote(monkeypatch, result):
    calls = {}

    def fake_run_remote_script(host, script, **kwargs):
        calls["host"] = host
        calls["script"] = script
        calls["kwargs"] = kwargs
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", fake_run_remote_script)
    return calls


def test_probe_reports_consuming_entrypoint(monkeypatch):
    _patch_remote(
        monkeypatch, RemoteResult(host="h1", returncode=0, stdout='SPARKRUN_ENTRYPOINT=["vllm","serve"]\nSPARKRUN_PROBE=fail', stderr="")
    )
    probe = probe_image_entrypoint("img", "h1")
    assert probe.consumes_command
    assert probe.host == "h1"


@pytest.mark.parametrize("stdout", ["SPARKRUN_PROBE=pass", "SPARKRUN_PROBE=absent", "SPARKRUN_PROBE=unknown"])
def test_probe_non_consuming_verdicts_do_not_flag(monkeypatch, stdout):
    _patch_remote(monkeypatch, RemoteResult(host="h1", returncode=0, stdout=stdout, stderr=""))
    assert not probe_image_entrypoint("img", "h1").consumes_command


def test_probe_failed_ssh_is_unknown_not_consuming(monkeypatch):
    """Could-not-verify must never block a launch."""
    _patch_remote(monkeypatch, RemoteResult(host="h1", returncode=255, stdout="", stderr="ssh: connect refused"))
    probe = probe_image_entrypoint("img", "h1")
    assert probe.verdict == VERDICT_UNKNOWN
    assert not probe.consumes_command


def test_probe_exception_is_unknown_not_consuming(monkeypatch):
    _patch_remote(monkeypatch, RuntimeError("boom"))
    assert probe_image_entrypoint("img", "h1").verdict == VERDICT_UNKNOWN


def test_probe_kill_switch(monkeypatch):
    calls = _patch_remote(monkeypatch, RemoteResult(host="h1", returncode=0, stdout="SPARKRUN_PROBE=fail", stderr=""))
    monkeypatch.setenv(NO_PROBE_ENV, "1")
    assert probe_image_entrypoint("img", "h1").verdict == VERDICT_UNKNOWN
    assert not calls, "kill switch must short-circuit before any SSH"


# --------------------------------------------------------------------------
# Executor dispatch
# --------------------------------------------------------------------------


def test_docker_executor_probes_first_host_with_accel_opts(monkeypatch):
    seen = {}

    def fake_probe(image, host, *, ssh_kwargs=None, accel_opts=None):
        seen.update(image=image, host=host, accel_opts=accel_opts, ssh_kwargs=ssh_kwargs)
        return EntrypointProbe(image=image, host=host, verdict=VERDICT_PASSTHROUGH)

    monkeypatch.setattr("sparkrun.containers.entrypoint.probe_image_entrypoint", fake_probe)

    executor = DockerExecutor()
    executor.config.gpu_access_mode = "gpus"
    probe = executor.verify_command_passthrough("img", ["h1", "h2", "h3"], ssh_kwargs={"ssh_user": "u"})

    assert probe is not None
    assert seen["host"] == "h1", "one probe only — the verdict is a property of the image"
    assert seen["accel_opts"] == ["--gpus", "all"]
    assert seen["ssh_kwargs"] == {"ssh_user": "u"}


@pytest.mark.parametrize("image,hosts", [("", ["h1"]), ("img", [])])
def test_docker_executor_no_image_or_hosts_returns_none(image, hosts):
    assert DockerExecutor().verify_command_passthrough(image, hosts) is None


@pytest.mark.parametrize("override", ["", "/opt/custom.sh"], ids=["cleared", "custom"])
def test_probe_skipped_when_launch_overrides_entrypoint(monkeypatch, override):
    """The recommended fix must not be rejected by the check that recommends it.

    ``entrypoint: ""`` / ``-o entrypoint=''`` makes the launch emit
    ``--entrypoint``, so the image's own ENTRYPOINT never runs — but the image
    still *declares* a consuming one, so a probe here would keep failing and the
    documented fix would be unusable.
    """

    def boom(*args, **kwargs):
        raise AssertionError("must not probe when the launch overrides the entrypoint")

    monkeypatch.setattr("sparkrun.containers.entrypoint.probe_image_entrypoint", boom)

    executor = DockerExecutor()
    executor.config.entrypoint = override
    assert executor.verify_command_passthrough("img", ["h1"]) is None


def test_base_executor_default_is_noop():
    """Container-less / provider executors have no opinion and never block."""
    from sparkrun.orchestration.executors._base import Executor

    assert Executor.verify_command_passthrough(DockerExecutor.__new__(DockerExecutor), "img", ["h1"]) is None


# --------------------------------------------------------------------------
# Launcher preflight
# --------------------------------------------------------------------------


def _call_preflight(monkeypatch, probe):
    from sparkrun.core import launcher

    class _FakeExecutor:
        def verify_command_passthrough(self, image, hosts, *, ssh_kwargs=None):
            return probe

    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", lambda **kw: _FakeExecutor())
    launcher._verify_image_command_passthrough(
        None,
        "img:tag",
        ["h1"],
        {},
        runtime=None,
        cluster=None,
        config=None,
        executor_config=None,
        rootless=True,
        auto_user=True,
        host_hardware=None,
        v=None,
    )


def test_preflight_raises_with_both_documented_fixes(monkeypatch):
    from sparkrun.core.recipe import RecipeError

    probe = EntrypointProbe(image="img:tag", host="h1", verdict=VERDICT_CONSUMES, entrypoint='["vllm","serve"]')
    with pytest.raises(RecipeError) as exc:
        _call_preflight(monkeypatch, probe)

    msg = str(exc.value)
    assert "img:tag" in msg
    assert '["vllm","serve"]' in msg
    assert 'entrypoint: ""' in msg, "recipe fix must be shown"
    assert "-o entrypoint=''" in msg, "CLI one-off fix must be shown"


@pytest.mark.parametrize(
    "probe",
    [
        None,
        EntrypointProbe(image="img:tag", host="h1", verdict=VERDICT_PASSTHROUGH),
        EntrypointProbe(image="img:tag", host="h1", verdict=VERDICT_ABSENT),
        EntrypointProbe(image="img:tag", host="h1", verdict=VERDICT_UNKNOWN),
    ],
    ids=["no-opinion", "passthrough", "absent", "unknown"],
)
def test_preflight_allows_everything_but_confirmed_consuming(monkeypatch, probe):
    _call_preflight(monkeypatch, probe)  # must not raise


def test_preflight_skips_when_executor_unresolvable(monkeypatch):
    from sparkrun.core import launcher

    def boom(**kwargs):
        raise RuntimeError("gated off")

    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", boom)
    launcher._verify_image_command_passthrough(
        None,
        "img:tag",
        ["h1"],
        {},
        runtime=None,
        cluster=None,
        config=None,
        executor_config=None,
        rootless=True,
        auto_user=True,
        host_hardware=None,
        v=None,
    )


def test_preflight_skips_without_image_or_hosts(monkeypatch):
    from sparkrun.core import launcher

    def boom(**kwargs):
        raise AssertionError("must not resolve an executor with nothing to probe")

    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", boom)
    for image, hosts in (("", ["h1"]), ("img", [])):
        launcher._verify_image_command_passthrough(
            None,
            image,
            hosts,
            {},
            runtime=None,
            cluster=None,
            config=None,
            executor_config=None,
            rootless=True,
            auto_user=True,
            host_hardware=None,
            v=None,
        )


# --------------------------------------------------------------------------
# CLI override key
# --------------------------------------------------------------------------


def test_entrypoint_is_an_executor_override_key():
    """``-o entrypoint=''`` must configure the executor, not the serve command."""
    from sparkrun.api import RunOptions
    from sparkrun.core.resolve import EXECUTOR_OVERRIDE_KEYS

    assert "entrypoint" in EXECUTOR_OVERRIDE_KEYS
    options = RunOptions(recipe="r", overrides={"entrypoint": "", "max_model_len": 8192})
    assert options.overrides == {"max_model_len": 8192}
    assert options.executor_config == {"entrypoint": ""}


def test_empty_entrypoint_survives_option_coercion():
    """Empty string is the meaningful value and must not coerce to None/False."""
    from sparkrun.cli._common import _parse_options

    assert _parse_options(("entrypoint=",)) == {"entrypoint": ""}


def test_empty_entrypoint_reaches_docker_run():
    """The full chain: executor_config entrypoint "" → ``docker run --entrypoint ''``."""
    from sparkrun.orchestration.executors._base import ExecutorConfig

    cfg = ExecutorConfig.from_chain({"entrypoint": ""})
    assert cfg.entrypoint == ""

    cmd = DockerExecutor(config=cfg).run_cmd(image="img:tag", container_name="c", command="echo hi")
    assert "--entrypoint ''" in cmd


def test_unset_entrypoint_emits_no_flag():
    from sparkrun.orchestration.executors._base import ExecutorConfig

    cfg = ExecutorConfig.from_chain({})
    assert cfg.entrypoint is None
    assert "--entrypoint" not in DockerExecutor(config=cfg).run_cmd(image="img:tag", container_name="c", command="echo hi")
