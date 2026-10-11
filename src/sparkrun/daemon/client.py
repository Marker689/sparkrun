"""Small synchronous reference client; intentionally independent of ASGI libraries."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import sys
import time
from urllib.parse import urlsplit


class DaemonError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(body.get("detail", f"Daemon returned HTTP {status}"))
        self.status, self.body = status, body


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout=30):
        super().__init__("localhost", timeout=timeout)
        self.path = str(path)

    def connect(self):
        if sys.platform == "win32":
            raise ValueError("Use TCP daemon discovery on Windows")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


class Client:
    def __init__(self, *, binding=None, socket_path=None, url=None, token=None, timeout=30, ca_cert=None):
        if socket_path is not None and url is not None:
            raise ValueError("Choose either a daemon URL or Unix socket")
        if token is not None and (not token or len(token) > 4096 or not token.isascii() or any(c.isspace() for c in token)):
            raise ValueError("Bearer token must be a nonempty single-line ASCII value of at most 4096 characters")
        if ca_cert is not None and (url is None or urlsplit(url).scheme != "https"):
            raise ValueError("A CA certificate requires an HTTPS daemon URL")
        self.binding, self.socket_path, self.url, self.token, self.timeout = binding, socket_path, url, token, timeout
        self.discover = binding is not None and socket_path is None and url is None
        self.tls_context = ssl.create_default_context(cafile=ca_cert) if ca_cert is not None else None

    def connection(self):
        if self.discover and self.binding is not None:
            from sparkrun.daemon.state import read_connection
            from sparkrun.core.process_identity import alive

            state = self.binding.read_state()
            if not state or not alive(state.get("process")):
                raise ConnectionError("Daemon is not running")
            if state.get("socket"):
                self.socket_path, self.token = state["socket"], None
            else:
                data = read_connection(self.binding.connection)
                if data.get("binding") != self.binding.key or data.get("process") != state["process"]:
                    raise ValueError("Connection file does not match this daemon")
                self.socket_path, self.url, self.token = None, data["url"], data["token"]
        if self.socket_path is not None:
            return UnixConnection(self.socket_path, self.timeout)
        parsed = urlsplit(self.url or "")
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Expected an HTTP(S) daemon origin without credentials, a path prefix, query, or fragment")
        if parsed.scheme == "https":
            return http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=self.timeout, context=self.tls_context)
        return http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=self.timeout)

    def headers(self):
        return {
            "Content-Type": "application/json",
            "X-Sparkrun-Client": "cli",
            **({"Authorization": "Bearer " + self.token} if self.token else {}),
        }

    def request(self, method, path, body=None, *, headers=None):
        connection = self.connection()
        try:
            connection.request(
                method, path, body=json.dumps(body).encode() if body is not None else None, headers={**self.headers(), **(headers or {})}
            )
            response = connection.getresponse()
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("Daemon response exceeds 4 MiB")
            result = json.loads(raw)
            if response.status >= 400:
                raise DaemonError(response.status, result)
            return result
        finally:
            connection.close()

    def info(self):
        info = self.request("GET", "/v1/info")
        if info.get("protocol") != 1 or (self.binding and info.get("binding") != self.binding.key):
            raise ValueError("Daemon protocol/config binding does not match")
        return info

    def call(self, name, arguments=None, *, key=None):
        self.info()
        from urllib.parse import quote

        return self.request("POST", "/v1/ops/" + quote(name, safe="."), arguments or {}, headers={"Idempotency-Key": key} if key else None)

    def wait(self, identifier, *, timeout=None, prompt=None):
        deadline = time.monotonic() + timeout if timeout else None
        while deadline is None or time.monotonic() < deadline:
            job = self.request("GET", "/v1/jobs/" + identifier)
            if job["state"] == "succeeded":
                return job["result"]
            if job["state"] in {"failed", "cancelled", "lost"}:
                raise DaemonError(409, job["error"] or {"detail": job["state"]})
            if job["prompt"]:
                if prompt is None:
                    return job
                value = prompt(job["prompt"])
                self.request("POST", f"/v1/jobs/{identifier}/input", {"prompt_id": job["prompt"]["prompt_id"], "value": value})
            time.sleep(0.25)
        raise TimeoutError("Job is still running; inspect its ID before retrying")

    def events(self, identifier, *, after=0):
        connection = self.connection()
        try:
            headers = self.headers()
            if after:
                headers["Last-Event-ID"] = str(after)
            connection.request("GET", f"/v1/jobs/{identifier}/events", headers=headers)
            response = connection.getresponse()
            if response.status >= 400:
                raise DaemonError(response.status, json.loads(response.read()))
            while line := response.readline(524289):
                if len(line) > 524288:
                    raise ValueError("Event line exceeds limit")
                if line.startswith(b"data: "):
                    yield json.loads(line[6:])
        finally:
            connection.close()
