"""Token principals and authorization independent of HTTP/framework imports."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import secrets
import time

from sparkrun.utils.fs import atomic_private_write, file_lock


@dataclass(frozen=True)
class Principal:
    id: str
    scope: str = "read"
    secret: bool = False
    local: bool = False
    credential_id: str | None = None

    def allows(self, scope, *, secret=False):
        levels = {"read": 0, "operate": 1, "admin": 2}
        return levels[self.scope] >= levels[scope] and (not secret or self.secret)

    def owns(self, job):
        return self.local or self.scope == "admin" or job["principal"] == self.id


class TokenStore:
    def __init__(self, binding):
        binding.prepare()
        self.path = binding.directory / "tokens.json"
        self.lock = binding.directory / "tokens.lock"

    def _read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def create(self, name, *, scope="operate", secret=False, expires=None, owner_process=None, principal_id=None):
        if scope not in {"read", "operate", "admin"}:
            raise ValueError("Invalid token scope")
        token = secrets.token_urlsafe(32)
        identifier = secrets.token_hex(8)
        with file_lock(self.lock):
            records = self._read()
            from sparkrun.core.process_identity import alive

            records = {key: value for key, value in records.items() if not value.get("owner_process") or alive(value["owner_process"])}
            records[identifier] = {
                "name": name,
                "scope": scope,
                "secret": bool(secret),
                "expires": expires,
                "hash": hashlib.sha256(token.encode()).hexdigest(),
                "created": time.time(),
                "owner_process": owner_process,
                "principal_id": principal_id,
            }
            atomic_private_write(self.path, json.dumps(records))
        return identifier, token

    def authenticate(self, token):
        digest = hashlib.sha256(token.encode()).hexdigest()
        for identifier, record in self._read().items():
            if hmac.compare_digest(digest, record["hash"]) and (record["expires"] is None or record["expires"] > time.time()):
                return self.validate_principal(Principal(identifier, record["scope"], record["secret"]))
        raise PermissionError("Invalid or expired bearer token")

    def validate_principal(self, principal):
        if principal.local:
            return principal
        credential = principal.credential_id or principal.id
        record = self._read().get(credential)
        from sparkrun.core.process_identity import alive

        if (
            not record
            or (record["expires"] is not None and record["expires"] <= time.time())
            or (record.get("owner_process") and not alive(record["owner_process"]))
        ):
            raise PermissionError("Token was revoked or expired")
        return Principal(record.get("principal_id") or credential, record["scope"], record["secret"], credential_id=credential)

    def list(self):
        return [{"id": key, **{k: v for k, v in value.items() if k != "hash"}} for key, value in self._read().items()]

    def revoke(self, identifier):
        with file_lock(self.lock):
            records = self._read()
            if identifier not in records:
                raise KeyError(identifier)
            del records[identifier]
            atomic_private_write(self.path, json.dumps(records))
