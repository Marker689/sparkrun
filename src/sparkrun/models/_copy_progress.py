"""Logical copy progress shared by rsync workers and CLI/API feedback."""

from __future__ import annotations

import re
import threading
import time


class CopyProgress:
    def __init__(self, request, jobs):
        self.reporter = request.feedback.child(source="model-rsync") if request.feedback else None
        self.children = {t.host: self.reporter.child(host=t.host) for t in request.targets} if self.reporter else {}
        self.jobs = jobs
        self.counts = {key: 0 for key in jobs}
        self.active = set()
        self.completed = set()
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.last_event = self.last_log = 0.0

    def start(self, key):
        with self.lock:
            self.active.add(key)

    def advance(self, key, count):
        with self.lock:
            self.counts[key] = max(self.counts[key], min(count, self.jobs[key][1]))

    def finish(self, key, success):
        with self.lock:
            self.active.discard(key)
            if success:
                self.counts[key] = self.jobs[key][1]
                self.completed.add(key)

    def parser(self, key, watch=None):
        """Feed *key*'s displayed count, and *watch* the attempt's raw count.

        The display keeps the highest count across attempts; the watch needs
        this attempt's own, which restarts from zero on a retry.
        """
        pending = b""

        def feed(data):
            nonlocal pending
            lines = re.split(rb"[\r\n]", pending + data)
            pending = lines.pop()[-4096:]
            for line in lines:
                match = re.match(rb"\s*([0-9,]+)\s+[0-9]+%\s", line)
                if match:
                    count = int(match[1].replace(b",", b""))
                    self.advance(key, count)
                    if watch is not None:
                        watch.advance(count)

        return feed

    def emit(self, *, outcomes=None, status="running", force=False):
        now = time.monotonic()
        if self.reporter is None or (not force and now - self.last_event < 0.5):
            return
        self.last_event = now
        with self.lock:
            counts, active, completed = dict(self.counts), set(self.active), set(self.completed)
        elapsed = max(now - self.started, 0.001)
        total = sum(size for _, size in self.jobs.values())
        current = sum(counts.values())
        metrics = {
            "active_streams": len(active),
            "completed_files": sum(len(self.jobs[k][0].required_files) for k in completed),
            "logical_bytes_per_second": current / elapsed,
        }
        log = force or now - self.last_log >= 10
        if log:
            self.last_log = now
        self.reporter.report(
            "Model copy: %.2f / %.2f GiB (%d streams)" % (current / 2**30, total / 2**30, len(active)),
            stage="copy",
            status=status,
            current=current,
            total=total,
            total_kind="exact",
            unit="bytes",
            metrics=metrics,
            log_message=log,
        )
        for host, reporter in self.children.items():
            keys = [key for key, (target, _) in self.jobs.items() if target.host == host]
            done = sum(counts[key] for key in keys)
            state = (outcomes or {}).get(host)
            host_status = {"complete": "succeeded", "already_present": "skipped", "failed": "failed"}.get(state or "", status)
            reporter.report(
                "Copying model files to " + host,
                stage="copy",
                status=host_status,
                current=done,
                total=sum(self.jobs[key][1] for key in keys),
                total_kind="exact",
                unit="bytes",
                metrics={"active_streams": len(active.intersection(keys)), "logical_bytes_per_second": done / elapsed},
                log_message=False,
            )
