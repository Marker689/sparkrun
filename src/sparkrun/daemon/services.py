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
        names = ["discover_endpoints"]
        if config.get("daemon.proxy_autodiscover", False):
            names += ["proxy.status", "proxy.watch"]
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
                if previous["state"] not in TERMINAL or previous["state"] == "lost" or time.time() - previous["created"] < interval:
                    continue
            if name == "proxy.watch":
                status = next((j for j in jobs if j["operation"] == "proxy.status" and j["state"] == "succeeded"), None)
                # Never compete with the existing standalone autodiscover process.
                if not status or not status["result"].get("running") or status["result"].get("autodiscover_running"):
                    continue
                if time.time() - status["updated"] > 2 * interval:
                    continue
            op = get_daemon_op(name)
            from sparkrun.core.runtime_lease import consistent_runtime

            with consistent_runtime(config, runtime):
                self.store.submit(op, {}, "service:daemon", "daemon services", runtime, key=f"{name}:{int(time.time() // interval)}")

    def snapshot(self, name):
        with self.lock:
            return deepcopy(self.snapshots.get(name, {"advisory": True, "state": "unavailable"}))


def watch_proxy(arguments, sctx, prompt=None):
    """Durable, cancellable replacement for the standalone discovery loop.

    The removal-grace window and any endpoint credentials remain in this
    worker's memory. Daemon restart adopts this job rather than restarting it.
    """
    from sparkrun import api
    from sparkrun.api.proxy._ops import _discovery_args, _running_engine
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.operation import CURRENT
    from sparkrun.proxy.autodiscover import _EndpointRemovalGrace, sync_discovered
    from sparkrun.proxy.discovery import discover_endpoints

    operation = CURRENT.get()
    assert operation is not None
    grace = _EndpointRemovalGrace(sctx.config.get("proxy.removal_grace_sweeps", 2))
    scope_key = None
    while True:
        operation.check()
        config = SparkrunConfig(sctx.config.config_path)
        sctx.config = config
        if not config.get("daemon.proxy_autodiscover", False):
            return {"stopped": "disabled"}
        state = api.proxy.status(sctx=sctx)
        if not state.running or state.autodiscover_running:
            return {"stopped": "proxy stopped or standalone autodiscover active"}
        if not getattr(_running_engine(sctx), "supports_autodiscover", True):
            return {"stopped": "gateway manages its own desired state"}
        scope, host_filter, hosts, ssh, cluster, _ = _discovery_args(api.proxy.ProxyStartOptions(), sctx)
        current_key = (state.gateway, scope.cluster, tuple(scope.hosts))
        if current_key != scope_key:
            grace = _EndpointRemovalGrace(config.get("proxy.removal_grace_sweeps", 2))
            scope_key = current_key
        endpoints = discover_endpoints(host_filter=host_filter, host_list=hosts, ssh_kwargs=ssh, cluster_def=cluster, sctx=sctx)
        result, deferred = sync_discovered(endpoints, grace, sctx)
        operation.emit("discovery", added=result.added, removed=result.removed, deferred_removals=deferred)
        operation.cancel.wait(max(5, float(config.get("daemon.discovery_interval", 30))))
