"""Flags sparkrun appends to, forces into, or strips from a serve command."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.base import RuntimePlugin

reconcile = RuntimePlugin.reconcile_flag_in_command


def _argv(command: str) -> list[str]:
    """The argv bash builds from *command* (what the engine actually receives)."""
    result = subprocess.run(
        ["bash", "-c", 'args() { printf "%s\\n" "$@"; }; args ' + command],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


# ---------------------------------------------------------------------------
# reconcile_flag_in_command / command_has_flag / strip_flags_from_command
# ---------------------------------------------------------------------------


def test_forced_value_replaces_the_equals_spelling():
    assert reconcile("vllm serve m --distributed-executor-backend=ray", "--distributed-executor-backend", "mp", override=True) == (
        "vllm serve m --distributed-executor-backend=mp"
    )


def test_forced_value_replaces_the_space_spelling():
    assert reconcile("vllm serve m --port 8000 --x", "--port", 9000, override=True) == "vllm serve m --port 9000 --x"


def test_forced_value_is_inserted_literally():
    value = r"C:\path\1"
    assert reconcile("serve --root /old", "--root", value, override=True) == "serve --root " + value


def test_flag_presence_is_a_whole_token():
    # ``--port`` is not present in ``--port-range``: forcing it appends.
    assert reconcile("serve --port-range 5", "--port", 1, override=True) == "serve --port-range 5 --port 1"
    assert reconcile("serve --port-range 5", "--port", 1) == "serve --port-range 5 --port 1"
    assert reconcile("serve --port=2", "--port", 1) == "serve --port=2"
    assert RuntimePlugin.command_has_flag("serve --served-model-name=x", "--served-model-name")
    assert not RuntimePlugin.command_has_flag("serve --served-model-name-x y", "--served-model-name")


def test_forced_value_across_a_line_continuation():
    command = "serve m \\\n  --port \\\n  8000 \\\n  --x"
    assert _argv(reconcile(command, "--port", 9000, override=True)) == ["serve", "m", "--port", "9000", "--x"]


def test_strip_removes_both_spellings_and_only_whole_tokens():
    flag_map = {"port": "--port", "trust": "--trust"}
    command = "serve --port=8000 --port-range 5 --trust --trusted"
    stripped = RuntimePlugin.strip_flags_from_command(command, {"port", "trust"}, flag_map, bool_keys={"trust"})
    assert stripped.split() == ["serve", "--port-range", "5", "--trusted"]


def test_appended_flag_survives_a_dangling_continuation():
    assert _argv(reconcile("serve m \\", "--port", 1)) == ["serve", "m", "--port", "1"]
    assert _argv(RuntimePlugin.append_args("serve m \\\n", "--a 1", "--b")) == ["serve", "m", "--a", "1", "--b"]


# ---------------------------------------------------------------------------
# Native multi-node runtimes append their rendezvous flags through append_args
# ---------------------------------------------------------------------------


def _runtime(name):
    from sparkrun.runtimes.atlas import AtlasRuntime
    from sparkrun.runtimes.sglang import SglangRuntime
    from sparkrun.runtimes.tokenary import TokenaryRuntime
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    return {"vllm-distributed": VllmDistributedRuntime, "sglang": SglangRuntime, "atlas": AtlasRuntime, "tokenary": TokenaryRuntime}[name]()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(
    "runtime_name,flag",
    [("vllm-distributed", "--nnodes"), ("sglang", "--nnodes"), ("atlas", "--world-size"), ("tokenary", "--world-size")],
)
@pytest.mark.parametrize("suffix", ["\\", "\\\n", "\\  \n"])
def test_node_command_flags_survive_a_dangling_continuation(runtime_name, flag, suffix):
    recipe = Recipe.from_dict(
        {
            "model": "org/model",
            "runtime": runtime_name,
            "container": "img:v1",
            "defaults": {"tensor_parallel": 2},
            "command": "serve-it {model} --opt 1 " + suffix,
        }
    )
    command = _runtime(runtime_name).generate_node_command(recipe, {}, "10.0.0.1", 2, 1, hosts=["10.0.0.1", "10.0.0.2"])
    argv = _argv(command)
    assert argv[:4] == ["serve-it", "org/model", "--opt", "1"]
    # The first appended flag is the one an escaped space would swallow.
    assert argv[4].startswith("--")
    assert flag in argv


def test_atlas_node_command_uses_the_shared_rendezvous_template():
    """Atlas resolves its master address like every native runtime (hosts / placement)."""
    recipe = Recipe.from_dict({"model": "org/model", "runtime": "atlas", "container": "img:v1"})
    runtime = _runtime("atlas")
    command = runtime.generate_node_command(recipe, {}, "head-ip", 2, 1, init_port=29600, hosts=["fabric-0", "fabric-1"])
    expected = runtime._make_node_command_args(
        head_ip="head-ip", num_nodes=2, node_rank=1, init_port=29600, hosts=["fabric-0", "fabric-1"], replica_size=2
    )
    argv = command.split()
    assert argv[argv.index("--master-addr") + 1] == expected["master_addr"]
    assert argv[argv.index("--rank") + 1] == "1"
    assert argv[argv.index("--world-size") + 1] == "2"
    assert argv[argv.index("--master-port") + 1] == "29600"
