"""Worker launch/reconciliation independent of the API service's lifetime."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from sparkrun.core.application_profile import child_environment
from sparkrun.core.process_identity import alive, terminate_tree, detached_process_options
from sparkrun.daemon.control import request


def worker_argv(binding, job):
    return [sys.executable, "-m", "sparkrun.daemon.worker", "--config", str(binding.config_path), job]


def launch(binding, job, *, systemd=False):
    env = {**os.environ, **child_environment(config_path=binding.config_path)}
    argv = worker_argv(binding, job)
    if systemd:
        # A separate transient service has its own cgroup; no PartOf/API unit tie.
        unit = f"sparkrun-job-{binding.key}-{job}"
        command = [
            "systemd-run",
            "--user",
            "--quiet",
            "--collect",
            f"--unit={unit}",
            "--property=Type=exec",
            "--property=Restart=no",
            "--property=KillMode=control-group",
        ]
        for key, value in child_environment(config_path=binding.config_path).items():
            command.append(f"--setenv={key}={value}")
        from sparkrun.telemetry.config import worker_environment

        # systemd launches the service from its manager environment, not the
        # systemd-run client's environment. Preserve explicit opt-outs/routing.
        for key, value in worker_environment().items():
            command.append(f"--setenv={key}={value}")
        from sparkrun.core.runtime_lease import TEST_SESSION_MARKER_ENV, TEST_STATE_ROOT_ENV

        # Editable development installs need their import path as well. The
        # test-only lease root must reach workers too, or a suite's worker
        # would take its lease in the developer's real runtime directory.
        for key in ("PYTHONPATH", "XDG_RUNTIME_DIR", "SPARKRUN_NO_EXTERNAL_PLUGINS", TEST_STATE_ROOT_ENV, TEST_SESSION_MARKER_ENV):
            if key in env:
                command.append(f"--setenv={key}={env[key]}")
        subprocess.run([*command, "--", *argv], env=env, check=True, capture_output=True, timeout=20)
        return None
    if os.environ.get("INVOCATION_ID"):
        raise RuntimeError("Service mode requires --worker-supervisor systemd to keep workers outside the API cgroup")
    return subprocess.Popen(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **detached_process_options(),
        close_fds=True,
    )


class Coordinator:
    def __init__(self, binding, store, *, systemd=False, max_active=4, max_services=2, cancel_grace: float = 15, start_timeout: float = 60):
        self.binding, self.store = binding, store
        self.systemd, self.max_active, self.cancel_grace = systemd, max_active, cancel_grace
        self.max_services = max_services
        self.start_timeout = start_timeout
        self.children = {}
        self.maintenance = False

    def tick(self, runtime):
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.core.runtime_lease import runtime_identity

        if runtime_identity(SparkrunConfig(self.binding.config_path)) != runtime:
            self.maintenance = True
        for job, child in list(self.children.items()):
            if child.poll() is not None:
                del self.children[job]
        for job in self.store.active():
            worker = job["worker"]
            if not worker:
                # A launch may have crossed the spawn boundary. Never blindly retry.
                if time.time() - job["updated"] > self.start_timeout:
                    self.store.finish(job["id"], "lost", error={"detail": "Worker did not register; execution was not authorized"})
                continue
            if not alive(worker["process"]):
                self.store.finish(job["id"], "lost", error={"detail": "Worker disappeared; remote effects require reconciliation"})
                continue
            if job["state"] == "cancelling" and time.time() - (job["cancelled"] or time.time()) > self.cancel_grace:
                if self.systemd:
                    subprocess.run(
                        ["systemctl", "--user", "kill", "--signal=SIGKILL", f"sparkrun-job-{self.binding.key}-{job['id']}.service"],
                        check=False,
                        capture_output=True,
                        timeout=10,
                    )
                elif os.name == "nt":
                    from sparkrun.utils.windows import terminate_worker_job

                    terminate_worker_job(worker["process"])
                else:
                    terminate_tree(worker["process"])
                self.store.finish(job["id"], "lost", error={"detail": "Forced termination; remote effects require reconciliation"})
                continue
            try:
                hello = request(worker, job["nonce"], "hello")
                if hello.get("job") != job["id"] or hello.get("process") != worker["process"] or hello.get("runtime") != job["runtime"]:
                    continue
                if job["state"] == "starting":
                    if self.maintenance or job["runtime"] != runtime:
                        self.store.cancel(job["id"])
                    else:
                        request(worker, job["nonce"], "start")
                elif job["state"] == "cancelling":
                    request(worker, job["nonce"], "cancel")
            except (OSError, ValueError):
                # A live but unresponsive worker is not permission to duplicate execution.
                continue
        if self.maintenance:
            return
        while job := self.store.claim(max_active=self.max_active, max_services=self.max_services):
            if job["runtime"] != runtime:
                self.store.finish(
                    job["id"], "failed", error={"detail": "Runtime changed before this queued job started; resubmit after restart"}
                )
                continue
            try:
                child = launch(self.binding, job["id"], systemd=self.systemd)
                if child is not None:
                    self.children[job["id"]] = child
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                # Subprocess exceptions include the complete systemd command,
                # including --setenv telemetry credentials. Log the failure
                # without exposing command arguments or captured output.
                if isinstance(exc, subprocess.CalledProcessError):
                    reason = f"supervisor exited with status {exc.returncode}"
                elif isinstance(exc, subprocess.TimeoutExpired):
                    reason = f"supervisor timed out after {exc.timeout} seconds"
                else:
                    reason = str(exc)
                logging.getLogger(__name__).error("Could not launch daemon worker for job %s: %s", job["id"], reason)
                self.store.finish(
                    job["id"], "lost", error={"detail": "Worker launch failed; inspect supervisor diagnostics before resubmitting"}
                )


def unit_name(binding):
    return f"sparkrun-daemon-{binding.key}.service"


def unit_text(binding):
    argv = [sys.executable, "-m", "sparkrun.daemon", "--config", str(binding.config_path), "serve", "--worker-supervisor", "systemd"]

    # systemd does not invoke a shell. Its quoted words use backslash escaping.
    def quote(value):
        if any(c in str(value) for c in ("\n", "\r", "\0")):
            raise ValueError("Unit arguments must not contain control characters")
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'

    env = {**child_environment(config_path=binding.config_path), "SPARKRUN_DAEMON_UNIT": unit_name(binding)}
    return "\n".join(
        [
            "[Unit]",
            "Description=Sparkrun API (" + binding.key + ")",
            "",
            "[Service]",
            "Type=simple",
            "ExecStart=" + " ".join(quote(v) for v in argv),
            *("Environment=" + quote(k + "=" + v) for k, v in env.items()),
            "Restart=on-failure",
            "KillMode=mixed",
            "TimeoutStopSec=30",
            "UMask=0077",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def install_service(binding, *, now=False):
    from sparkrun.utils.fs import atomic_private_write

    state = binding.read_state() or {}
    if now and alive(state.get("process")) and not state.get("service_unit"):
        raise ValueError("Stop this binding's foreground/GUI daemon before installing a running service")
    directory = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / unit_name(binding)
    text = unit_text(binding)
    if path.exists() and "Description=Sparkrun API (" not in path.read_text():
        raise ValueError("Refusing to replace an unrelated service")
    # Verify before replacing an installed unit.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sparkrun-unit-") as temporary:
        candidate = Path(temporary) / path.name
        candidate.write_text(text)
        subprocess.run(["systemd-analyze", "--user", "verify", str(candidate)], check=True, capture_output=True, timeout=20)
    atomic_private_write(path, text)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=20)
    subprocess.run(["systemctl", "--user", "enable", *(["--now"] if now else []), path.name], check=True, timeout=30)
    return str(path)


def uninstall_service(binding):
    directory = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    path = directory / unit_name(binding)
    if path.exists() and "Description=Sparkrun API (" not in path.read_text():
        raise ValueError("Refusing to remove an unrelated service")
    subprocess.run(["systemctl", "--user", "disable", "--now", path.name], check=True, timeout=30)
    path.unlink(missing_ok=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=20)


def installed_service(binding):
    if not sys.platform.startswith("linux"):
        return False
    directory = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    path = directory / unit_name(binding)
    return path.is_file() and "Description=Sparkrun API (" in path.read_text()
