"""Bounded stream producers and shared, reference-counted monitor sessions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import queue
import threading
import time

from sparkrun.api._wire import to_wire
from sparkrun.core.operation import OperationState, operation_scope


class LogStream:
    def __init__(self, arguments, sctx):
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.queue = queue.Queue(maxsize=128)
        self.error = None
        self.dropped = 0

        def produce():
            from sparkrun.api import logs

            try:
                with operation_scope(sctx, state=OperationState(via_daemon=True)):
                    iterator = logs(**arguments, sctx=sctx, cancel=self.cancel)
                    try:
                        for line in iterator:
                            if self.cancel.is_set():
                                break
                            try:
                                self.queue.put_nowait({"type": "log", **to_wire(line)})
                            except queue.Full:
                                self.dropped += 1
                    finally:
                        close = getattr(iterator, "close", None)
                        if close:
                            close()
            except Exception as exc:
                from sparkrun.api._wire import error_body

                self.error = error_body(exc)
            finally:
                self.done.set()

        self.thread = threading.Thread(target=produce, name="sparkrun-log-stream", daemon=True)
        self.thread.start()

    def close(self):
        self.cancel.set()
        self.thread.join(timeout=15)


class MonitorSubscription:
    """One subscriber's view over its shared per-host monitor sessions.

    ``frame()`` is the live event.  ``history()`` is the backfill sent once at
    subscribe time: the shared session has been recording since it was opened
    (possibly for an earlier subscriber), so a late client can draw sparklines
    at once.  Its series sit under ``series``, not ``hosts``, so a client that
    reads every event's ``hosts`` as frame rows ignores it.
    """

    def __init__(self, sessions):
        self._sessions = sessions

    def frame(self):
        return {
            "type": "frame",
            "queried_at": time.time(),
            "hosts": [host for session in self._sessions for host in to_wire(session.frame())["hosts"]],
        }

    def history(self):
        return {
            "type": "history",
            "queried_at": time.time(),
            "series": [series for session in self._sessions for series in to_wire(session.history_snapshot())],
        }


class MonitorHub:
    def __init__(self, *, max_sessions=128, linger=5):
        self.lock = threading.RLock()
        self.entries = {}
        self.max_sessions, self.linger = max_sessions, linger

    @contextmanager
    def subscribe(self, arguments, sctx):
        from sparkrun.api import open_live_monitor
        from sparkrun.api._resolve import resolve_cluster

        cluster = resolve_cluster(arguments.get("cluster"), arguments.get("hosts"), sctx=sctx)
        sctx = sctx.for_cluster(cluster)
        cluster_settings = asdict(cluster)
        # Pool per host even when subscribers request different host subsets,
        # but never reuse a session after its executor or transport changes.
        cluster_settings.pop("hosts")
        settings = {"config": sctx.config.effective_data, "cluster": cluster_settings}
        revision = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()
        keys = []
        try:
            for host in cluster.hosts:
                key = (host, cluster.name, cluster.user, revision, arguments.get("interval", 2))
                with self.lock:
                    entry = self.entries.get(key)
                    if entry is None:
                        if len(self.entries) >= self.max_sessions:
                            raise ValueError("Monitor session limit reached")
                        # Opening occurs in a bounded server worker, never on the event loop.
                        from copy import copy

                        scoped = copy(sctx)
                        with operation_scope(scoped, state=OperationState(via_daemon=True)):
                            session = open_live_monitor([host], cluster=cluster, interval=arguments.get("interval", 2), sctx=scoped)
                        entry = {"session": session, "refs": 0, "released": None}
                        self.entries[key] = entry
                    entry["refs"] += 1
                    entry["released"] = None
                    keys.append(key)
            yield MonitorSubscription([self.entries[key]["session"] for key in keys])
        finally:
            with self.lock:
                for key in keys:
                    entry = self.entries.get(key)
                    if entry:
                        entry["refs"] -= 1
                        if entry["refs"] == 0:
                            entry["released"] = time.monotonic()

    def prune(self, *, all_sessions=False):
        with self.lock:
            for key, entry in list(self.entries.items()):
                if all_sessions or (entry["refs"] == 0 and time.monotonic() - entry["released"] >= self.linger):
                    entry["session"].close()
                    del self.entries[key]
