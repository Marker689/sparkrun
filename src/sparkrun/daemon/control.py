"""Bounded, nonce-authenticated worker control; independent of API lifetime."""

from __future__ import annotations

import json
import socket
import sys


def request(worker, nonce, action, **values):
    from sparkrun.core.process_identity import alive

    if not worker or not alive(worker["process"]):
        raise ConnectionError("Worker identity no longer matches")
    endpoint = worker.get("endpoint")
    if endpoint:
        if endpoint.get("host") != "127.0.0.1" or endpoint.get("transport") != "tcp":
            raise ValueError("Worker control must use local TCP")
        sock = socket.create_connection((endpoint["host"], endpoint["port"]), timeout=5)
    else:
        # Adopt workers launched by earlier Linux daemons until they finish.
        if sys.platform == "win32":
            raise ValueError("Legacy Unix workers cannot be adopted on Windows")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            sock.connect(worker["socket"])
        except BaseException:
            sock.close()
            raise
    with sock:
        payload = json.dumps({"nonce": nonce, "action": action, **values}).encode() + b"\n"
        if len(payload) > 65536:
            raise ValueError("Control message exceeds 64 KiB")
        sock.sendall(payload)
        with sock.makefile("rb") as stream:
            raw = stream.readline(65537)
        if len(raw) > 65536 or not raw.endswith(b"\n"):
            raise ValueError("Invalid worker response")
        result = json.loads(raw)
        if not result.get("ok"):
            raise ValueError(result.get("detail", "Worker rejected control request"))
        return result
