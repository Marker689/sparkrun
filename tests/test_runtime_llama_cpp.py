"""Unit tests for sparkrun.runtimes.llama_cpp (LlamaCppRuntime)."""

import pytest
from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.llama_cpp import LlamaCppRuntime
from sparkrun.core.log_source import MODE_FILE, SCOPE_ALL


def test_llama_cpp_runtime_name():
    """LlamaCppRuntime.runtime_name == 'llama-cpp'."""
    runtime = LlamaCppRuntime()
    assert runtime.runtime_name == "llama-cpp"


def test_llama_cpp_cluster_strategy():
    """LlamaCppRuntime uses native (RPC) clustering, not Ray."""
    runtime = LlamaCppRuntime()
    assert runtime.cluster_strategy() == "native"


# --- llama.cpp resolve_api_key Tests ---


def test_llama_cpp_resolve_api_key_from_defaults():
    """defaults.api_key is the recommended source for llama.cpp too."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "defaults": {"api_key": "sk-default"},
        }
    )
    assert LlamaCppRuntime().resolve_api_key(recipe) == "sk-default"


def test_llama_cpp_resolve_api_key_from_env():
    """env.LLAMA_API_KEY is honored when defaults.api_key is absent."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "env": {"LLAMA_API_KEY": "sk-env"},
        }
    )
    assert LlamaCppRuntime().resolve_api_key(recipe) == "sk-env"


def test_llama_cpp_resolve_api_key_overrides_take_priority():
    """CLI override beats defaults and env."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "defaults": {"api_key": "sk-default"},
            "env": {"LLAMA_API_KEY": "sk-env"},
        }
    )
    assert LlamaCppRuntime().resolve_api_key(recipe, {"api_key": "sk-cli"}) == "sk-cli"


def test_llama_cpp_resolve_api_key_defaults_beat_env():
    """defaults.api_key takes precedence over env.LLAMA_API_KEY."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "defaults": {"api_key": "sk-default"},
            "env": {"LLAMA_API_KEY": "sk-env"},
        }
    )
    assert LlamaCppRuntime().resolve_api_key(recipe) == "sk-default"


def test_llama_cpp_resolve_api_key_none_when_unset():
    """Returns None when no api_key is configured anywhere."""
    recipe = Recipe.from_dict({"name": "r", "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M", "runtime": "llama-cpp"})
    assert LlamaCppRuntime().resolve_api_key(recipe) is None


def test_llama_cpp_resolve_api_key_parses_inline_command_flag():
    """Literal --api-key in a fixed command string is extracted."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "command": "llama-server -hf m --api-key sk-inline --port 8080",
        }
    )
    assert LlamaCppRuntime().resolve_api_key(recipe) == "sk-inline"


def test_llama_cpp_resolve_api_key_ignores_placeholder_in_command():
    """`--api-key {api_key}` placeholder is ignored — defaults path handles it."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "command": "llama-server -hf m --api-key {api_key} --port 8080",
            "defaults": {"api_key": "sk-default"},
        }
    )
    assert LlamaCppRuntime().resolve_api_key(recipe) == "sk-default"


def test_llama_cpp_api_key_emitted_as_flag_for_structured_command():
    """defaults.api_key auto-emits as --api-key on structured (no-template) commands."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "defaults": {"port": 8080, "api_key": "sk-flag"},
        }
    )
    cmd = LlamaCppRuntime().generate_command(recipe, {}, is_cluster=False)
    assert "--api-key sk-flag" in cmd


def test_llama_cpp_resolve_container_from_recipe():
    """Recipe with container field."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "container": "custom-llama:v1.0",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "custom-llama:v1.0"


def test_llama_cpp_resolve_container_default():
    """Recipe without container uses default prefix."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "scitrera/dgx-spark-llama-cpp:latest"


def test_llama_cpp_generate_command_from_template():
    """Recipe with command template renders correctly."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "command": "llama-server -hf {model} --port {port}",
        "defaults": {"port": 8080},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd == "llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q4_K_M --port 8080"


def test_llama_cpp_generate_command_structured_hf():
    """HuggingFace model (contains '/') uses -hf flag."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {
            "port": 8080,
            "n_gpu_layers": 99,
            "ctx_size": 8192,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q4_K_M")
    assert "--port 8080" in cmd
    assert "--n-gpu-layers 99" in cmd
    assert "--ctx-size 8192" in cmd


def test_llama_cpp_generate_command_gguf_path():
    """Local .gguf path uses -m flag."""
    recipe_data = {
        "name": "test-recipe",
        "model": "/models/qwen3-1.7b-q4_k_m.gguf",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("llama-server -m /models/qwen3-1.7b-q4_k_m.gguf")
    assert "--port 8080" in cmd


def test_llama_cpp_generate_command_bool_flags():
    """Boolean flags flash_attn, jinja, no_webui are handled."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {
            "flash_attn": True,
            "jinja": True,
            "no_webui": True,
            "cont_batching": False,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--flash-attn" in cmd
    assert "--jinja" in cmd
    assert "--no-webui" in cmd
    # cont_batching is False, should not appear
    assert "--cont-batching" not in cmd


def test_llama_cpp_generate_command_overrides():
    """CLI overrides properly override defaults."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {"port": 9090}, is_cluster=False)
    assert "--port 9090" in cmd
    assert "--port 8080" not in cmd


def test_llama_cpp_validate_recipe_valid():
    """Valid recipe returns no issues."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    issues = runtime.validate_recipe(recipe)
    assert issues == []


def test_llama_cpp_validate_recipe_no_model():
    """Missing model returns issue."""
    recipe_data = {
        "name": "test-recipe",
        "runtime": "llama-cpp",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in str(issues[0])


def test_llama_cpp_build_rpc_head_command():
    """_build_rpc_head_command appends --rpc with worker addresses."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080, "n_gpu_layers": 99},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()
    config = recipe.build_config_chain({})

    cmd = runtime._build_rpc_head_command(
        recipe,
        config,
        worker_hosts=["10.0.0.2", "10.0.0.3"],
        rpc_port=50052,
    )
    assert "--rpc 10.0.0.2:50052,10.0.0.3:50052" in cmd
    assert cmd.startswith("llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q4_K_M")


def test_llama_cpp_build_rpc_worker_command():
    """_build_rpc_worker_command returns rpc-server with host and port."""
    cmd = LlamaCppRuntime._build_rpc_worker_command(50052)
    assert cmd == "rpc-server --host 0.0.0.0 --port 50052"


def test_llama_cpp_container_name():
    """_container_name returns {cluster_id}_{role}."""
    assert LlamaCppRuntime._container_name("spark0", "head") == "spark0_head"
    assert LlamaCppRuntime._container_name("spark0", "worker") == "spark0_worker"


def test_rpc_workers_exec_through_their_own_host_executor_with_labels(monkeypatch):
    """Each RPC worker is started by its host's executor and carries the workload labels."""
    from types import SimpleNamespace
    from unittest.mock import Mock
    from sparkrun.orchestration.executors.docker import DockerExecutor

    import sparkrun.runtimes._cluster_ops as ops

    monkeypatch.setattr(ops, "cleanup_named_containers", lambda *a, **k: None)
    monkeypatch.setattr(ops, "detect_ib_with_ips", lambda ctx, comm_env, ib_ip_map, backends=None: (None, {}, {}))
    monkeypatch.setattr(ops, "launch_containers_parallel", lambda *a, **k: 0)
    monkeypatch.setattr(ops, "run_pre_serve_hooks", lambda *a, **k: None)
    head_exec = Mock(return_value=0)
    monkeypatch.setattr(ops, "exec_serve_on_container", head_exec)
    sent = []
    monkeypatch.setattr(
        "sparkrun.orchestration.ssh.run_remote_script",
        lambda host, script, **k: sent.append((host, script)) or SimpleNamespace(success=True),
    )

    executors = {h: DockerExecutor() for h in ("head", "w1", "w2")}
    root = executors["head"]
    root.bind_host_executors(executors)
    calls = {}
    for host in ("w1", "w2"):

        def generate(*, _host=host, sparkrun_labels=None, **kwargs):
            calls[_host] = sparkrun_labels
            return "exec-" + _host

        monkeypatch.setattr(executors[host], "generate_exec_serve_script", generate)

    recipe = Recipe.from_dict({"model": "org/m-GGUF:Q4_K_M", "runtime": "llama-cpp", "container": "img"})
    runtime = LlamaCppRuntime()
    runtime.executor = root
    rc = runtime._run_cluster(["head", "w1", "w2"], "img", recipe=recipe, overrides={}, cluster_id="sparkrun_abc", dry_run=True)

    assert rc == 0
    assert sorted(sent) == [("w1", "exec-w1"), ("w2", "exec-w2")]
    assert all(labels and labels["sparkrun.cluster_id"] == "sparkrun_abc" for labels in calls.values())
    assert set(calls) == {"w1", "w2"}
    assert head_exec.call_args.kwargs["sparkrun_labels"]["sparkrun.cluster_id"] == "sparkrun_abc"


def test_llama_cpp_generate_command_gguf_presync_template():
    """When _gguf_model_path is set, template renders with resolved path
    and -hf is switched to -m."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {
            "port": 8080,
            "host": "0.0.0.0",
            "n_gpu_layers": 99,
            "ctx_size": 8192,
        },
        "command": (
            "llama-server \\\n"
            "    -hf {model} \\\n"
            "    --host {host} \\\n"
            "    --port {port} \\\n"
            "    --n-gpu-layers {n_gpu_layers} \\\n"
            "    --ctx-size {ctx_size} \\\n"
            "    --flash-attn \\\n"
            "    --jinja \\\n"
            "    --no-webui"
        ),
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    gguf_path = "/cache/huggingface/hub/models--Qwen--Qwen3-1.7B-GGUF/snapshots/abc123/q4_k_m.gguf"
    cmd = runtime.generate_command(
        recipe,
        {"_gguf_model_path": gguf_path, "model": gguf_path},
        is_cluster=False,
    )
    # Template is respected: -hf switched to -m, path substituted
    assert "-m " + gguf_path in cmd
    assert "-hf " not in cmd
    # Other flags from template are preserved
    assert "--host 0.0.0.0" in cmd
    assert "--port 8080" in cmd
    assert "--n-gpu-layers 99" in cmd
    assert "--ctx-size 8192" in cmd
    assert "--flash-attn" in cmd
    assert "--jinja" in cmd
    assert "--no-webui" in cmd


def test_llama_cpp_generate_command_gguf_presync_structured():
    """When _gguf_model_path is set and no template, structured build uses -m."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080, "n_gpu_layers": 99},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    gguf_path = "/cache/huggingface/hub/models--Qwen--Qwen3-1.7B-GGUF/snapshots/abc123/q4_k_m.gguf"
    cmd = runtime.generate_command(
        recipe,
        {"_gguf_model_path": gguf_path, "model": gguf_path},
        is_cluster=False,
    )
    assert cmd.startswith("llama-server -m " + gguf_path)
    assert "--port 8080" in cmd
    assert "--n-gpu-layers 99" in cmd


def test_llama_cpp_generate_command_no_presync_uses_hf():
    """Without _gguf_model_path, template renders -hf with original model."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080},
        "command": "llama-server -hf {model} --port {port}",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "-hf Qwen/Qwen3-1.7B-GGUF:Q4_K_M" in cmd
    assert "--port 8080" in cmd


# ---------------------------------------------------------------------------
# Vision GGUF / mmproj support (issue #204)
# ---------------------------------------------------------------------------

_MMPROJ = "/cache/huggingface/hub/models--unsloth--Qwen3-VL-8B-Instruct-GGUF/snapshots/abc/mmproj-F16.gguf"
_WEIGHTS = "/cache/huggingface/hub/models--unsloth--Qwen3-VL-8B-Instruct-GGUF/snapshots/abc/Q4_K_M.gguf"


def test_llama_cpp_mmproj_auto_injected_template():
    """Template without --mmproj gets the projector auto-appended."""
    recipe = Recipe.from_dict(
        {
            "name": "vl",
            "model": "unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "command": "llama-server -hf {model} --port {port}",
            "defaults": {"port": 8001},
        }
    )
    cmd = LlamaCppRuntime().generate_command(
        recipe,
        {"_gguf_model_path": _WEIGHTS, "model": _WEIGHTS, "_mmproj_path": _MMPROJ, "mmproj": _MMPROJ},
        is_cluster=False,
    )
    assert "-m " + _WEIGHTS in cmd
    assert "--mmproj " + _MMPROJ in cmd
    assert cmd.count("--mmproj") == 1


def test_llama_cpp_mmproj_placeholder_not_double_injected():
    """A template using {mmproj} keeps explicit placement; no duplicate flag."""
    recipe = Recipe.from_dict(
        {
            "name": "vl",
            "model": "unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "command": "llama-server -hf {model} --mmproj {mmproj} --port {port}",
            "defaults": {"port": 8001},
        }
    )
    cmd = LlamaCppRuntime().generate_command(
        recipe,
        {"_gguf_model_path": _WEIGHTS, "model": _WEIGHTS, "_mmproj_path": _MMPROJ, "mmproj": _MMPROJ},
        is_cluster=False,
    )
    assert cmd.count("--mmproj") == 1
    assert "--mmproj " + _MMPROJ in cmd


def test_llama_cpp_mmproj_literal_not_double_injected():
    """A template with a literal --mmproj is left untouched."""
    recipe = Recipe.from_dict(
        {
            "name": "vl",
            "model": "unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "command": "llama-server -hf {model} --mmproj /custom/mmproj.gguf",
        }
    )
    cmd = LlamaCppRuntime().generate_command(
        recipe,
        {"_gguf_model_path": _WEIGHTS, "model": _WEIGHTS, "_mmproj_path": _MMPROJ, "mmproj": _MMPROJ},
        is_cluster=False,
    )
    assert cmd.count("--mmproj") == 1
    assert "/custom/mmproj.gguf" in cmd
    assert _MMPROJ not in cmd


def test_llama_cpp_mmproj_structured():
    """Command-less recipe emits --mmproj alongside -m."""
    recipe = Recipe.from_dict(
        {
            "name": "vl",
            "model": "unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "defaults": {"port": 8001},
        }
    )
    cmd = LlamaCppRuntime().generate_command(
        recipe,
        {"_gguf_model_path": _WEIGHTS, "model": _WEIGHTS, "_mmproj_path": _MMPROJ, "mmproj": _MMPROJ},
        is_cluster=False,
    )
    assert cmd.startswith("llama-server -m " + _WEIGHTS)
    assert "--mmproj " + _MMPROJ in cmd


def test_llama_cpp_flash_attn_valued():
    """flash_attn is emitted as a valued flag (bool->on/off, str passthrough)."""
    runtime = LlamaCppRuntime()

    cmd_true = runtime.generate_command(
        Recipe.from_dict({"name": "r", "model": "/m.gguf", "runtime": "llama-cpp", "defaults": {"flash_attn": True}}),
        {},
        is_cluster=False,
    )
    assert "--flash-attn on" in cmd_true

    cmd_str = runtime.generate_command(
        Recipe.from_dict({"name": "r", "model": "/m.gguf", "runtime": "llama-cpp", "defaults": {"flash_attn": "auto"}}),
        {},
        is_cluster=False,
    )
    assert "--flash-attn auto" in cmd_str

    cmd_false = runtime.generate_command(
        Recipe.from_dict({"name": "r", "model": "/m.gguf", "runtime": "llama-cpp", "defaults": {"flash_attn": False}}),
        {},
        is_cluster=False,
    )
    assert "--flash-attn off" in cmd_false


def test_llama_cpp_inverted_webui_mmap_flags():
    """webui/mmap are inverted toggles: emit --no-* only when disabled."""
    runtime = LlamaCppRuntime()

    cmd_off = runtime.generate_command(
        Recipe.from_dict({"name": "r", "model": "/m.gguf", "runtime": "llama-cpp", "defaults": {"webui": False, "mmap": False}}),
        {},
        is_cluster=False,
    )
    assert "--no-webui" in cmd_off
    assert "--no-mmap" in cmd_off

    cmd_on = runtime.generate_command(
        Recipe.from_dict({"name": "r", "model": "/m.gguf", "runtime": "llama-cpp", "defaults": {"webui": True, "mmap": True}}),
        {},
        is_cluster=False,
    )
    assert "--no-webui" not in cmd_on
    assert "--no-mmap" not in cmd_on


def test_llama_cpp_sampling_flags_structured():
    """Sampling controls render so command-less recipes are viable."""
    recipe = Recipe.from_dict(
        {
            "name": "r",
            "model": "/m.gguf",
            "runtime": "llama-cpp",
            "defaults": {
                "top_p": 0.8,
                "top_k": 20,
                "temperature": 0.7,
                "min_p": 0.0,
                "presence_penalty": 1.5,
            },
        }
    )
    cmd = LlamaCppRuntime().generate_command(recipe, {}, is_cluster=False)
    assert "--top-p 0.8" in cmd
    assert "--top-k 20" in cmd
    assert "--temp 0.7" in cmd
    assert "--min-p 0.0" in cmd
    assert "--presence-penalty 1.5" in cmd


def test_llama_cpp_issue_204_commandless_equivalent():
    """The issue's recipe, expressed command-less, renders a complete command."""
    recipe = Recipe.from_dict(
        {
            "name": "qwen3-vl-8b",
            "model": "unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M",
            "runtime": "llama-cpp",
            "defaults": {
                "port": 8001,
                "host": "0.0.0.0",
                "served_model_name": "qwen3-vl-8b",
                "n_gpu_layers": 99,
                "ctx_size": 8192,
                "flash_attn": "on",
                "jinja": True,
                "top_p": 0.8,
                "top_k": 20,
                "temperature": 0.7,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "webui": False,
                "mmap": False,
            },
        }
    )
    cmd = LlamaCppRuntime().generate_command(
        recipe,
        {"_gguf_model_path": _WEIGHTS, "model": _WEIGHTS, "_mmproj_path": _MMPROJ, "mmproj": _MMPROJ},
        is_cluster=False,
    )
    for fragment in (
        "-m " + _WEIGHTS,
        "--mmproj " + _MMPROJ,
        "--host 0.0.0.0",
        "--port 8001",
        "--alias qwen3-vl-8b",
        "--n-gpu-layers 99",
        "--ctx-size 8192",
        "--flash-attn on",
        "--jinja",
        "--top-p 0.8",
        "--top-k 20",
        "--temp 0.7",
        "--min-p 0.0",
        "--presence-penalty 1.5",
        "--no-webui",
        "--no-mmap",
    ):
        assert fragment in cmd, fragment


def _vision_recipe(**extra):
    return Recipe.from_dict({"model": "unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M", "runtime": "llama-cpp", "container": "img", **extra})


def test_presynced_gguf_overrides_resolve_the_projector(monkeypatch):
    """The launcher asks the runtime for GGUF companions; llama.cpp supplies mmproj."""
    seen = {}

    def _resolve(model, cache_dir, selector=None):
        seen.update(model=model, cache_dir=cache_dir, selector=selector)
        return _MMPROJ

    monkeypatch.setattr("sparkrun.models.download.resolve_mmproj_container_path", _resolve)
    got = LlamaCppRuntime().presynced_gguf_overrides(_vision_recipe(mmproj="F16"), "/hf")
    assert got == {"_mmproj_path": _MMPROJ, "mmproj": _MMPROJ}
    assert seen == {"model": "unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M", "cache_dir": "/hf", "selector": "F16"}


@pytest.mark.parametrize("selector", [False, "none", "off", "disabled"])
def test_presynced_gguf_overrides_honor_a_disabled_projector(monkeypatch, selector):
    def _unexpected(*a, **kw):
        raise AssertionError("a disabled projector must not be resolved")

    monkeypatch.setattr("sparkrun.models.download.resolve_mmproj_container_path", _unexpected)
    assert LlamaCppRuntime().presynced_gguf_overrides(_vision_recipe(mmproj=selector), "/hf") == {}


def test_presynced_gguf_overrides_without_a_projector(monkeypatch):
    monkeypatch.setattr("sparkrun.models.download.resolve_mmproj_container_path", lambda *a, **kw: None)
    assert LlamaCppRuntime().presynced_gguf_overrides(_vision_recipe(), "/hf") == {}


def test_other_runtimes_add_no_gguf_companions():
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    assert VllmDistributedRuntime().presynced_gguf_overrides(_vision_recipe(mmproj="F16"), "/hf") == {}


def test_only_mpi_runtimes_refuse_per_node_env():
    from sparkrun.runtimes.trtllm import TrtllmRuntime
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    assert TrtllmRuntime.supports_per_node_env is False
    assert LlamaCppRuntime.supports_per_node_env is True
    assert VllmDistributedRuntime.supports_per_node_env is True


class TestLlamaCppFollowLogs:
    """Test LlamaCppRuntime.follow_logs()."""

    def test_follow_logs_solo_uses_file_logs(self, log_sources_spy):
        """Single-host llama-cpp tails serve log file inside solo container."""
        LlamaCppRuntime().follow_logs(hosts=["10.0.0.1"], cluster_id="test0")

        (source,) = log_sources_spy[0].sources
        assert (source.container, source.mode) == ("test0_solo", MODE_FILE)

    def test_follow_logs_cluster_uses_file_logs_on_head(self, log_sources_spy):
        """Multi-host llama-cpp follows file logs on _head container (sleep-infinity + exec)."""
        LlamaCppRuntime().follow_logs(hosts=["10.0.0.1", "10.0.0.2"], cluster_id="mycluster")

        (source,) = log_sources_spy[0].sources
        assert (source.host, source.container, source.mode) == ("10.0.0.1", "mycluster_head", MODE_FILE)

    def test_scope_all_uses_worker_names_not_ranked_names(self, log_sources_spy):
        """llama.cpp declares the ``native`` strategy but uses head/worker naming.

        Worker sources are derived from ``_head_container_name``, so its single
        override is enough — no second place to keep in sync.
        """
        LlamaCppRuntime().follow_logs(hosts=["10.0.0.1", "10.0.0.2"], cluster_id="mycluster", scope=SCOPE_ALL)

        sources = log_sources_spy[0].sources
        assert [s.container for s in sources] == ["mycluster_head", "mycluster_worker"]
        assert all(s.mode == MODE_FILE for s in sources)


class TestLlamaCppValidateRecipe:
    """Test LlamaCppRuntime.validate_recipe() — TP/PP exclusion + DP rejection.

    Node-count math is exercised via the scheduler (see ``test_scheduler.py``
    and ``test_placement.py``). These tests only verify that llama.cpp
    rejects recipe shapes the runtime cannot honor: TP and PP set
    simultaneously (``--split-mode`` is single-valued) and any DP > 1
    (llama.cpp has no multi-replica DP coordination).
    """

    def _make_recipe(self, defaults=None):
        data = {
            "name": "test",
            "runtime": "llama-cpp",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        }
        if defaults:
            data["defaults"] = defaults
        return Recipe.from_dict(data)

    def test_tp_only_passes(self):
        """TP=4 (no PP) is allowed → no issues."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 4})
        runtime = LlamaCppRuntime()
        assert runtime.validate_recipe(recipe) == []

    def test_pp_only_passes(self):
        """PP=3 (no TP) is allowed → no issues."""
        recipe = self._make_recipe(defaults={"pipeline_parallel": 3})
        runtime = LlamaCppRuntime()
        assert runtime.validate_recipe(recipe) == []

    def test_neither_passes(self):
        """No TP/PP/DP is allowed → no issues."""
        recipe = self._make_recipe()
        runtime = LlamaCppRuntime()
        assert runtime.validate_recipe(recipe) == []

    def test_tp_eq_1_and_pp_eq_1_passes(self):
        """TP=1 and PP=1 are no-ops → not flagged as mutual exclusion."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 1, "pipeline_parallel": 1})
        runtime = LlamaCppRuntime()
        assert runtime.validate_recipe(recipe) == []

    def test_both_tp_and_pp_gt_1_flagged(self):
        """TP>1 and PP>1 together → flagged as mutually exclusive."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 2, "pipeline_parallel": 2})
        runtime = LlamaCppRuntime()
        issues = runtime.validate_recipe(recipe)
        assert any("mutually" in str(issue) and "llama-cpp" in str(issue) for issue in issues)

    def test_dp_gt_1_flagged(self):
        """data_parallel > 1 → flagged (llama.cpp has no native DP coordination)."""
        recipe = self._make_recipe(defaults={"data_parallel": 2})
        runtime = LlamaCppRuntime()
        issues = runtime.validate_recipe(recipe)
        assert any("data_parallel" in str(issue) and "llama-cpp" in str(issue) for issue in issues)

    def test_dp_eq_1_passes(self):
        """data_parallel=1 is a no-op → not flagged."""
        recipe = self._make_recipe(defaults={"data_parallel": 1})
        runtime = LlamaCppRuntime()
        assert runtime.validate_recipe(recipe) == []


class TestLlamaCppSplitModeCommand:
    """Test that TP/PP correctly inject --split-mode in llama-server commands."""

    def _make_recipe(self, defaults=None, command=None):
        data = {
            "name": "test",
            "runtime": "llama-cpp",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        }
        if defaults:
            data["defaults"] = defaults
        if command:
            data["command"] = command
        return Recipe.from_dict(data)

    def test_tp_generates_split_mode_row(self):
        """--tp → --split-mode row in generated command."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 2, "port": 8080})
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode row" in cmd
        assert "--split-mode layer" not in cmd

    def test_pp_generates_split_mode_layer(self):
        """--pp → --split-mode layer in generated command."""
        recipe = self._make_recipe(defaults={"pipeline_parallel": 4, "port": 8080})
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode layer" in cmd
        assert "--split-mode row" not in cmd

    def test_neither_uses_default_layer(self):
        """No TP/PP → default split_mode=layer from _LLAMA_CPP_DEFAULTS."""
        recipe = self._make_recipe(defaults={"port": 8080})
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode layer" in cmd

    def test_tp_override_generates_row(self):
        """CLI override tensor_parallel → --split-mode row."""
        recipe = self._make_recipe(defaults={"port": 8080})
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {"tensor_parallel": 2}, is_cluster=False)
        assert "--split-mode row" in cmd

    def test_pp_override_generates_layer(self):
        """CLI override pipeline_parallel → --split-mode layer."""
        recipe = self._make_recipe(defaults={"port": 8080})
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {"pipeline_parallel": 2}, is_cluster=False)
        assert "--split-mode layer" in cmd

    def test_tp_overrides_recipe_split_mode(self):
        """TP takes precedence over recipe split_mode=layer."""
        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "split_mode": "layer",
            }
        )
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode row" in cmd
        assert "--split-mode layer" not in cmd

    def test_both_tp_pp_raises_in_generate(self):
        """Both TP and PP raises ValueError in generate_command too."""
        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
            }
        )
        runtime = LlamaCppRuntime()
        with pytest.raises(ValueError, match="simultaneously"):
            runtime.generate_command(recipe, {}, is_cluster=False)

    def test_template_split_mode_overridden_by_tp(self):
        """Command template with --split-mode layer is overridden by TP → row."""
        recipe = self._make_recipe(
            defaults={"tensor_parallel": 2},
            command="llama-server -hf {model} --split-mode layer --port 8080",
        )
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode row" in cmd
        assert "--split-mode layer" not in cmd

    def test_template_split_mode_overridden_by_pp(self):
        """Command template with --split-mode row is overridden by PP → layer."""
        recipe = self._make_recipe(
            defaults={"pipeline_parallel": 4},
            command="llama-server -hf {model} --split-mode row --port 8080",
        )
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode layer" in cmd
        assert "--split-mode row" not in cmd

    def test_template_no_tp_pp_preserves_existing_split_mode(self):
        """Template with --split-mode is preserved when no TP/PP set."""
        recipe = self._make_recipe(
            command="llama-server -hf {model} --split-mode row --port 8080",
        )
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode row" in cmd
