"""Reusable local daemon startup for the CLI and gateway integrations."""

from sparkrun.core.config import SparkrunConfig
from sparkrun.daemon.client import Client
from sparkrun.utils.fs import file_lock


def ensure_daemon(binding, launch_options=None):
    """Attach or start once for this binding; never change feature settings."""
    binding.prepare()
    with file_lock(binding.directory / "startup.lock"):
        return _start(binding, Client(binding=binding), launch_options)


def _start(binding, client, launch_options=None):
    import os
    import subprocess
    import sys
    import time
    from sparkrun.core.application_profile import child_environment
    from sparkrun.core.process_identity import alive, detached_process_options

    binding = binding.prepare()
    if not SparkrunConfig(binding.config_path).is_feature_enabled("daemon"):
        raise RuntimeError("Daemon feature is disabled; enable it with setup features enable daemon")
    state = binding.read_state()
    if state and alive(state.get("process")):
        return client.info()
    from sparkrun.daemon.supervisor import installed_service, unit_name

    if installed_service(binding):
        subprocess.run(["systemctl", "--user", "start", unit_name(binding)], check=True, timeout=30)
        for _ in range(100):
            try:
                return client.info()
            except (OSError, ValueError):
                time.sleep(0.1)
        raise RuntimeError("Service started but API is not ready; inspect daemon systemd status")
    options = launch_options or {}
    extra = ["--unix" if options["unix"] else "--no-unix"] if "unix" in options else []
    if options.get("tcp"):
        host, port = options["tcp"]
        extra += ["--tcp", f"[{host}]:{port}" if ":" in host else f"{host}:{port}"]
    for key, flag in (
        ("cert", "--tls-cert"),
        ("key", "--tls-key"),
        ("conn_file", "--conn-file"),
        ("parent_pid", "--parent-pid"),
        ("worker_supervisor", "--worker-supervisor"),
    ):
        if options.get(key):
            extra += [flag, str(options[key])]
    if options.get("allow_request_trust"):
        extra += ["--allow-request-trust"]
    if options.get("allow_insecure"):
        extra += ["--allow-insecure"]
    child = subprocess.Popen(
        [sys.executable, "-m", "sparkrun.daemon", "--config", str(binding.config_path), "serve", *extra],
        env={**os.environ, **child_environment(config_path=binding.config_path)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **detached_process_options(),
    )
    for _ in range(100):
        if child.poll() is not None:
            raise RuntimeError("Daemon failed to start; run 'daemon serve' for diagnostics")
        try:
            return client.info()
        except (OSError, ValueError):
            time.sleep(0.1)
    raise RuntimeError("Daemon startup timed out; inspect 'daemon status'")


def write_connection_descriptor(binding, path, *, manager=None):
    """Provision private connection references and an optional manager credential.

    A dedicated credential binds automation ownership on both UDS and TCP. Its
    principal and manager survive rotation; the local daemon's admin credential
    is never handed to integrations. This helper requires local account access.
    """
    import json
    from pathlib import Path
    from sparkrun.daemon.auth import TokenStore
    from sparkrun.daemon.state import private_directory
    from sparkrun.utils.fs import atomic_private_write

    path = Path(path)
    private_directory(path.parent)
    actions = private_directory(path.parent / "daemon-actions")
    value = {
        "protocol": 1,
        "binding": binding.key,
        "profile": binding.profile,
        "state_file": str(binding.directory / "server.json"),
        "connection_file": str(binding.connection),
        "action_directory": str(actions),
    }
    if manager is None:
        atomic_private_write(path, json.dumps(value))
        return path
    with file_lock(path.with_suffix(".lock")):
        old = json.loads(path.read_text()) if path.exists() else {}
        tokens = TokenStore(binding)
        # Registry administration is part of gateway admin catalog access.
        # Scope and workload ownership are independent; this is not tenancy.
        identifier, token = tokens.create(
            "integration:" + manager,
            scope="admin",
            principal_id="manager:" + manager,
            manager=manager,
        )
        try:
            token_path = path.with_name(path.stem + "." + identifier + ".token")
            atomic_private_write(token_path, token)
            value.update(manager=manager, token_file=str(token_path), credential_id=identifier)
            atomic_private_write(path, json.dumps(value))
        except BaseException:
            tokens.revoke(identifier)
            raise
        if old.get("credential_id"):
            try:
                tokens.revoke(old["credential_id"])
            except KeyError:
                pass
            old_token = path.with_name(path.stem + "." + old["credential_id"] + ".token")
            old_token.unlink(missing_ok=True)
    return path
