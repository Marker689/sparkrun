"""One redaction boundary for all bundle records, including text and errors."""

import os
import re
from typing import Any

_SENSITIVE = re.compile(r"password|passwd|secret|token|(?:api|master|access|signing|private)[-_]?key|authorization|credential", re.I)
_ASSIGNMENT = re.compile(
    # Start at identifier boundaries and test sensitivity independently of the
    # delimiter/value. Avoid competing greedy prefixes/suffixes (including for
    # long words containing repeated "token" or runs of whitespace).
    r"""(?<![\w-])(?=[\w-]*(?:password|passwd|secret|token|(?:api|master|access|signing|private)[_-]?key|authorization|credential))([\w-]+["']?(?:\s*[:=]\s*|\s+))(?:"[^"]*"|'[^']*'|[^\s,;}]+)""",
    re.I,
)


class Redactor:
    def __init__(self, secrets=()):
        self.secrets = set()
        self.operation_secrets = set()
        for value in secrets:
            self.add(value)
        for key, value in os.environ.items():
            if _SENSITIVE.search(key):
                self.add(value)

    def add(self, value):
        if isinstance(value, str) and len(value) >= 4:
            self.secrets.add(value)

    def __call__(self, value: Any) -> Any:
        # Copy the built-in sets before traversing: concurrent prompt registration
        # affects subsequent records, and sorting happens once per record.
        secrets = sorted(self.secrets.copy() | self.operation_secrets.copy(), key=len, reverse=True)
        return self._redact(value, secrets)

    def _redact(self, value, secrets):
        if isinstance(value, dict):
            return {
                self._redact(str(key), secrets): "[redacted]" if _SENSITIVE.search(str(key)) else self._redact(item, secrets)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [self._redact(item, secrets) for item in value]
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if not isinstance(value, str):
            return self._redact(str(value), secrets)
        for secret in secrets:
            if not secret:
                continue
            value = value.replace(secret, "[redacted]")
        value = re.sub(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", "[redacted private key]", value, flags=re.S)
        value = re.sub(r"(https?://)[^/\s@]+@", r"\1[redacted]@", value)
        value = re.sub(r"(?i)\bBearer\s+[^\s,\"']+", "Bearer [redacted]", value)
        return _ASSIGNMENT.sub(r"\1[redacted]", value)
