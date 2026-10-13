"""Kubernetes plugin isolation and application/channel policy in fresh processes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHILD_FEATURES = {"executor.k8s", "cli.setup.k8s", "api.run.k8s"}


def _run_policy(tmp_path, application, config, env_extra=None):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config))
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SPARKRUN_", "ALTERNATE_", "XDG_", "STATEFUL_")) and key not in {"PYTHONPATH", "VIRTUAL_ENV"}
    }
    env.update(
        HOME=str(tmp_path),
        SPARKRUN_NO_TELEMETRY="1",
        SPARKRUN_NO_EXTERNAL_PLUGINS="1",
        SPARKRUN_NO_INSTALLED_PLUGINS="1",
    )
    env.update(env_extra or {})
    script = """
import json, sys
from sparkrun.core.application_profile import SPARKRUN
from sparkrun.application import initialize
if sys.argv[1] == 'alternate':
    from sparkrun.core.application_profile import ApplicationProfile
    profile = ApplicationProfile(id="alternate", display_name="Alternate", command="alternate", package="alternate",
                                 feature_defaults={"integration.k8s": False})
else:
    profile = SPARKRUN
sctx = initialize(profile, config_path=sys.argv[2])
from sparkrun.core.features import all_features
from sparkrun.core.plugin_inventory import list_plugins
from sparkrun.core.run_handlers import registered_run_handlers
from sparkrun.orchestration.executor import list_executors, resolve_executor, ExecutorUnavailableError
flags = [flag.name for flag in all_features()]
plugins = {plugin.name: plugin.to_dict() for plugin in list_plugins(config=sctx.config, v=sctx.variables)}
modules = [name for name in sys.modules if name.startswith('sparkrun.plugins.k8s')]
assert 'click' not in sys.modules
executors = list_executors(sctx.variables)
try:
    executor = resolve_executor(cli_overrides={'executor': 'k8s', 'k8s_namespace': 'test-ns'}, config=sctx.config, v=sctx.variables)
    assert executor.config.k8s_namespace == 'test-ns'
    assert type(executor.config).__module__ == 'sparkrun.plugins.k8s.executor'
    error = None
except ExecutorUnavailableError as exc:
    error = str(exc)
assert [name for name in sys.modules if name.startswith('sparkrun.plugins.k8s')] == modules
from click.testing import CliRunner
from sparkrun.cli import main
runner = CliRunner()
help_result = runner.invoke(main, ['setup', '--help'])
assert help_result.exit_code == 0, help_result.output
completion = runner.invoke(main, [], prog_name=profile.command, env={
    '_' + profile.command.upper() + '_COMPLETE': 'bash_complete',
    'COMP_WORDS': profile.command + ' setup k', 'COMP_CWORD': '2',
})
assert completion.exit_code == 0, completion.output
feature_list = runner.invoke(main, ['setup', 'features', 'list', '--all', '--json'])
assert feature_list.exit_code == 0, feature_list.output
assert {row['name'] for row in json.loads(feature_list.output)} == set(flags)
setup = runner.invoke(main, ['setup', 'k8s', 'kubectl', '--list'])
print(json.dumps(dict(flags=flags, plugin=plugins['k8s'], modules=modules, executors=executors,
                     handlers=list(registered_run_handlers(sctx.config)), error=error,
                     help=help_result.output, completion=completion.output, setup_exit=setup.exit_code)))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, application, str(config_path)], env=env, cwd=ROOT, capture_output=True, text=True, timeout=40
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.parametrize("application", ["sparkrun", "alternate"])
@pytest.mark.parametrize("channel", ["stable", "beta", "alpha"])
def test_plugin_defaults_follow_channel_and_application(tmp_path, application, channel):
    data = _run_policy(tmp_path, application, {"features": {"channel": channel}})
    _assert_policy(data, enabled=application == "sparkrun" and channel == "alpha")


def _assert_policy(data, *, enabled):
    assert data["plugin"]["loaded"] is enabled
    assert data["plugin"]["enabled"] is enabled
    assert "integration.k8s" not in data["flags"]
    assert (CHILD_FEATURES & set(data["flags"])) == (CHILD_FEATURES if enabled else set())
    assert bool(data["modules"]) is enabled
    assert ("k8s" in data["executors"]) is enabled
    assert ("k8s" in data["handlers"]) is enabled
    assert ("  k8s " in data["help"]) is enabled
    assert ("plain,k8s" in data["completion"]) is enabled
    assert data["setup_exit"] == (0 if enabled else 2)
    if enabled:
        assert data["error"] is None
        from sparkrun import __version__

        assert data["plugin"]["version"] == __version__
    else:
        assert "integration.k8s" in data["error"]
        from sparkrun import __version__

        # Static release metadata is available even when the plugin is gated off.
        assert data["plugin"]["version"] == __version__


@pytest.mark.parametrize("application", ["sparkrun", "alternate"])
def test_config_opt_in_and_environment_override(tmp_path, application):
    config = {"features": {"channel": "stable", "integration.k8s": True}}
    _assert_policy(_run_policy(tmp_path, application, config), enabled=True)
    env = {application.upper() + "_FEATURE_INTEGRATION_K8S": "0"}
    _assert_policy(_run_policy(tmp_path, application, config, env), enabled=False)


@pytest.mark.parametrize("application", ["sparkrun", "alternate"])
def test_child_feature_overrides_do_not_load_disabled_parent(tmp_path, application):
    config = {"features": {"channel": "alpha", "integration.k8s": False, **dict.fromkeys(CHILD_FEATURES, True)}}
    env = {application.upper() + "_FEATURE_" + flag.upper().replace(".", "_"): "1" for flag in CHILD_FEATURES}
    _assert_policy(_run_policy(tmp_path, application, config, env), enabled=False)


def test_feature_channel_override_is_independent_of_release_channel(tmp_path):
    _assert_policy(_run_policy(tmp_path, "sparkrun", {"self_update": {"channel": "alpha"}}), enabled=True)
    config = {"self_update": {"channel": "alpha"}, "features": {"channel": "stable"}}
    _assert_policy(_run_policy(tmp_path, "sparkrun", config), enabled=False)


def test_enabled_plugin_keeps_individual_features_switchable(tmp_path):
    config = {"features": {"integration.k8s": True, **dict.fromkeys(CHILD_FEATURES, False)}}
    data = _run_policy(tmp_path, "sparkrun", config)
    assert data["plugin"]["loaded"]
    assert CHILD_FEATURES <= set(data["flags"])
    assert "k8s" not in data["executors"]
    assert "k8s" not in data["handlers"]
    assert "  k8s " not in data["help"]
    assert "plain,k8s" not in data["completion"]
    assert data["setup_exit"] == 1
    assert "executor.k8s" in data["error"]


def test_run_handler_registration_rolls_back_with_failed_plugin(monkeypatch):
    from scitrera_app_framework import Variables
    from sparkrun.core import run_handlers
    from sparkrun.core.installed_plugins import PluginConflictError
    from sparkrun.core.registration import registry_transaction

    monkeypatch.setattr(run_handlers, "_RUN_HANDLERS", {})
    handler = run_handlers.RunHandler("test", lambda *args: None)
    with pytest.raises(RuntimeError):
        with registry_transaction(Variables()):
            run_handlers.register_run_handler(handler)
            raise RuntimeError("plugin failure")
    assert not run_handlers._RUN_HANDLERS
    run_handlers.register_run_handler(handler)
    run_handlers.register_run_handler(handler)
    with pytest.raises(PluginConflictError):
        run_handlers.register_run_handler(run_handlers.RunHandler("test", lambda *args: None))
