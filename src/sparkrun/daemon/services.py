"""Opt-in periodic core services and advisory snapshots backed by durable jobs."""

from __future__ import annotations

from copy import deepcopy
import threading
import time


class QueryCache:
    """Small bounded cache; never used as authoritative input to a mutation."""

    def __init__(self, *, size=128, ttl=2):
        self.size, self.ttl = size, ttl
        self.lock = threading.Lock()
        self.values = {}

    def get(self, key):
        with self.lock:
            value = self.values.get(key)
            if value and time.monotonic() - value[0] < self.ttl:
                return deepcopy(value[1])
            self.values.pop(key, None)
            return None

    def put(self, key, value):
        with self.lock:
            if len(self.values) >= self.size:
                self.values.pop(next(iter(self.values)))
            self.values[key] = (time.monotonic(), deepcopy(value))

    def clear(self):
        with self.lock:
            self.values.clear()


class Services:
    def __init__(self, store):
        self.store = store
        self.snapshots = {}
        self.lock = threading.Lock()
        self.last_update = 0

    def tick(self, config, runtime, *, maintenance=False):
        from sparkrun.core.daemon_registry import get_daemon_op
        from sparkrun.daemon.jobs import TERMINAL

        interval = float(config.get("daemon.discovery_interval", 0))
        if interval <= 0 or maintenance:
            return
        interval = max(interval, 5)
        names = ["discovery.snapshot"]
        jobs = self.store.list(principal="service:daemon", limit=1000)
        for name in names:
            previous = next((j for j in jobs if j["operation"] == name), None)
            if previous:
                with self.lock:
                    self.snapshots[name] = {
                        "job_id": previous["id"],
                        "state": previous["state"],
                        "observed_at": previous["updated"],
                        "advisory": True,
                        "result": previous["result"],
                        "error": previous["error"],
                    }
                if previous["state"] not in TERMINAL:
                    continue
                last_attempt = previous["created"]
                if previous["state"] == "lost":
                    from sparkrun.core.process_identity import alive

                    # Only this read-only periodic observation is retried. A
                    # registered worker must be gone; an unregistered worker
                    # never received authorization to execute the lost job.
                    if previous["worker"] and alive(previous["worker"]["process"]):
                        continue
                    last_attempt = previous["updated"]
                if time.time() - last_attempt < interval:
                    continue
            op = get_daemon_op(name)
            from sparkrun.core.runtime_lease import consistent_runtime

            with consistent_runtime(config, runtime):
                self.store.submit(op, {}, "service:daemon", "daemon services", runtime, key=f"{name}:{int(time.time() // interval)}")

    def snapshot(self, name):
        with self.lock:
            return deepcopy(self.snapshots.get(name, {"advisory": True, "state": "unavailable"}))
