"""Shared public benchmark data projections; execution credentials are excluded."""

from collections.abc import Mapping
from copy import copy, deepcopy
from typing import Any, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe


def _credential_key(key: object) -> bool:
    return isinstance(key, str) and key.lower().replace("-", "_").endswith(("api_key", "auth_token"))


_AUTH_FLAG = re.compile(r"(--(?:api-key|auth-token)(?:=|\s+))(\"(?:\\.|[^\"\\])*\"|'[^']*'|[^\s\\]+)")


def _public_command(text: str) -> str:
    def redact(match):
        value = match[2].strip("\"'")
        return match[0] if value.startswith("{") and value.endswith("}") else match[1] + "[REDACTED]"

    return _AUTH_FLAG.sub(redact, text)


def public_benchmark_data(value: Any) -> Any:
    """Detach publication/state data without credential fields.

    Literal --api-key/--auth-token arguments are redacted, preserving templates.
    Serialized documents are interpreted only by their owning caller, never by
    guessing at field names inside arbitrary JSON data.
    Do not replace arbitrary substrings: a short key must not corrupt names,
    identifiers, measurements, or unrelated plugin data during migration.
    """
    if isinstance(value, Mapping):
        return {key: public_benchmark_data(child) for key, child in value.items() if not _credential_key(key)}
    if isinstance(value, (list, tuple)):
        return [public_benchmark_data(child) for child in value]
    if isinstance(value, str):
        return _public_command(value)
    return deepcopy(value)


#: ``executor_config`` keys a published recipe carries.  A published recipe is
#: both the record of a run and the prototype someone else launches, so it
#: keeps what reproduces the run anywhere (``entrypoint: ""`` decides whether
#: the image starts at all) and drops what belongs to one site.  An allowlist:
#: a key nobody classified (including a plugin executor's own, such as a
#: kubeconfig path) is omitted and named in the trailing comment, never
#: published by default.  ``tests/test_benchmark_container_pin.py`` requires
#: every ``ExecutorConfig`` field to be in one of the two sets.
PORTABLE_EXECUTOR_KEYS = frozenset(
    {
        "executor",
        "executor_type",
        "auto_remove",
        "restart_policy",
        "privileged",
        "gpus",
        "gpu_access_mode",
        "accelerator_vendor",
        "ipc",
        "shm_size",
        "memory_limit",
        "network",
        "security_opt",
        "cap_add",
        "ulimit",
        "entrypoint",
        # Whether the workload runs as root is part of the run, not of the site.
        "user",
    }
)

#: Host paths and site metadata: omitted from published recipes.
HOST_SPECIFIC_EXECUTOR_KEYS = frozenset(
    {
        "devices",
        "volumes",
        "labels",
        "working_dir",
        "log_dir",
        "log_file",
        "pid_dir",
        "pid_file",
        "env_file",
        "command_prefix",
    }
)


def _container_side(volume: object) -> str:
    """``/host:/ctr:ro`` → ``<host path>:/ctr:ro``; a bare path is an identity mount."""
    parts = str(volume).split(":")
    target = parts[1:] if len(parts) > 1 else parts
    return ":".join(["<host path>", *target])


def portable_recipe_text(text: str) -> str:
    """Drop host-specific ``executor_config`` keys from a published recipe.

    What was dropped is listed in a trailing YAML comment: the key names, and
    for ``volumes`` the container side of each mount, so whoever replicates
    the run knows a mount was there without learning this site's paths.
    Text with nothing to drop is returned unchanged.
    """
    import yaml

    data = yaml.safe_load(text)
    executor_config = data.get("executor_config") if isinstance(data, Mapping) else None
    if not isinstance(executor_config, Mapping):
        return text
    omitted = [key for key in executor_config if key not in PORTABLE_EXECUTOR_KEYS]
    if not omitted:
        return text
    kept = {key: value for key, value in executor_config.items() if key in PORTABLE_EXECUTOR_KEYS}
    data = dict(data)
    if kept:
        data["executor_config"] = kept
    else:
        data.pop("executor_config")
    notes = []
    for key in omitted:
        if key == "volumes" and isinstance(executor_config[key], list):
            notes.extend("#   volumes: %s" % _container_side(volume) for volume in executor_config[key])
        else:
            notes.append("#   %s" % key)
    comment = "# Host-specific executor_config omitted (set your own to replicate):\n%s\n" % "\n".join(notes)
    return yaml.safe_dump(data, sort_keys=False) + comment


def model_metadata(recipe: "Recipe") -> dict[str, Any]:
    """Shared model fields for local export and publication provenance."""
    fields = {
        "dtype": "model_dtype",
        "params": "model_params",
        "num_layers": "num_layers",
        "num_kv_heads": "num_kv_heads",
        "head_dim": "head_dim",
        "quantization": "quantization",
        "quant_bits": "quant_bits",
        "kv_dtype": "kv_dtype",
    }
    result = {key: recipe.metadata[source] for key, source in fields.items() if recipe.metadata.get(source)}
    if recipe.model_revision:
        result["revision"] = recipe.model_revision
    return result


def public_recipe_text(text: str) -> str:
    """Remove credential fields from the serialized recipe carried by results."""
    import yaml

    if not isinstance(text, str):
        raise TypeError("Serialized recipe must be a YAML string")
    data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError("Serialized recipe must contain a mapping")
    safe = public_benchmark_data(data)
    return text if safe == data else yaml.safe_dump(safe, sort_keys=False)


def benchmark_recipe_fingerprint(recipe: "Recipe", overrides: dict[str, Any] | None = None) -> str:
    """Hash declared serving configuration without execution credential fields.

    Keep the general deployment fingerprint unchanged: deployment artifacts can
    depend on authentication configuration. A benchmark instead compares the
    measurement configuration across credential rotations. The shallow recipe
    copy preserves resolved state and plugin objects without mutating the caller.
    """
    from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint

    measurement_recipe = copy(recipe)
    for field in ("defaults", "env", "runtime_config", "command", "_raw"):
        setattr(measurement_recipe, field, public_benchmark_data(getattr(recipe, field)))
    # With `overrides:` applied, identity reads the declared snapshot: redact it too.
    declared = getattr(recipe, "_declared", None)
    if declared is not None:
        measurement_recipe._declared = {
            **declared,
            "defaults": public_benchmark_data(declared["defaults"]),
            "env": public_benchmark_data(declared["env"]),
        }
    return derive_recipe_fingerprint(measurement_recipe, public_benchmark_data(overrides))
