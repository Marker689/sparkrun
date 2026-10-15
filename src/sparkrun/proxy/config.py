"""Proxy configuration management.

Reads/writes ``~/.config/sparkrun/proxy.yaml`` for proxy settings,
model aliases, and (future) default recipe mappings.

**Two processes write this file.** The auto-discover daemon re-reads it every
sweep, and a ``sparkrun proxy`` command may save an alias or listener setting
at any moment.  A whole-document last-writer-wins save therefore silently
discards the other's change, so :meth:`ProxyConfig.save` locks, re-reads, and
merges only the sections *this* instance modified.
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from sparkrun.proxy import (
    DEFAULT_DISCOVER_INTERVAL,
    DEFAULT_DISCOVER_REMOVAL_GRACE_SWEEPS,
    DEFAULT_MASTER_KEY,
    DEFAULT_PROXY_HOST,
    DEFAULT_PROXY_PORT,
)

logger = logging.getLogger(__name__)

_DELETE = object()


class ProxyConfig:
    """Manages proxy configuration stored in ``proxy.yaml``.

    The canonical access path is :attr:`sparkrun.core.context.SparkrunContext.proxy_config`
    (or :meth:`sparkrun.core.config.SparkrunConfig.get_proxy_config` for direct
    callers without a ``sctx``).  Direct construction is reserved for tests
    that need to point at a temporary config path.
    """

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            from sparkrun.core.config import get_config_root

            config_path = get_config_root() / "proxy.yaml"
        self.config_path = config_path
        self._data: dict[str, Any] = {}
        self._reset_pending()
        self._load()

    def _reset_pending(self) -> None:
        """Forget which sections this instance has modified."""
        self._pending_proxy: dict[str, Any] = {}
        self._pending_aliases: dict[str, Any] = {}

    def _read(self, *, strict: bool = False) -> dict[str, Any]:
        """Parse the file as it is on disk right now.

        Absent is ``{}``.  An unreadable file is ``{}`` too when only reading,
        but ``strict`` (the save path) raises instead: merging pending keys
        into ``{}`` and writing that back would erase the bind host and master
        key, and the next start would listen on every interface unauthenticated.
        """
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path) as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            if strict:
                raise ValueError(
                    "Cannot update %s: the file could not be parsed (%s); fix or remove it first" % (self.config_path, exc)
                ) from exc
            logger.warning("Failed to load proxy config %s; using defaults: %s", self.config_path, exc)
            return {}
        if data is None:
            return {}
        if not isinstance(data, dict):
            if strict:
                raise ValueError("Cannot update %s: the document is not a mapping; fix or remove it first" % self.config_path)
            logger.warning("Proxy config %s is not a mapping; using defaults", self.config_path)
            return {}
        return data

    def _load(self) -> None:
        self._data = self._read()
        self._reset_pending()

    @property
    def _lock_path(self) -> Path:
        return Path(str(self.config_path) + ".lock")

    @contextmanager
    def _write_lock(self):
        """Serialize all proxy writers, including plugin activation, on every OS."""
        from sparkrun.utils.fs import file_lock

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self._lock_path):
            yield

    def _write_atomic(self, data: dict[str, Any]) -> None:
        """Replace the config in one step, owner-only, never leaving a partial file."""
        from sparkrun.utils.fs import atomic_private_write

        atomic_private_write(self.config_path, yaml.safe_dump(data, default_flow_style=False, sort_keys=False))

    def save(self) -> None:
        """Atomically merge this instance's pending changes into the file.

        Writers lock a stable sidecar, re-read the newest document, and apply
        only the sections changed through this instance.  An auto-discovery
        sweep therefore cannot overwrite an alias or listener setting saved
        concurrently by a CLI process, and vice versa.

        With no pending changes this still rewrites the document held in
        memory, preserving the historical whole-file semantics for a caller
        that mutated ``_data`` directly.
        """
        has_pending = bool(self._pending_proxy or self._pending_aliases)

        from sparkrun.core.config import _lifecycle_snapshot

        if _lifecycle_snapshot is not None:
            raise ValueError("Lifecycle hooks must return configuration plans")
        with self._write_lock():
            journal = self.config_path.parent / "plugins-transaction.yaml"
            if journal.exists():
                pending = yaml.safe_load(journal.read_text())
                if not isinstance(pending, dict) or pending.get("state") != "committed":
                    raise ValueError("Interrupted plugin activation; run a plugin management command to recover")
            latest = self._read(strict=True) if has_pending else copy.deepcopy(self._data)

            if self._pending_proxy:
                proxy = latest.setdefault("proxy", {})
                for key, value in self._pending_proxy.items():
                    if value is _DELETE:
                        proxy.pop(key, None)
                    else:
                        proxy[key] = copy.deepcopy(value)

            if self._pending_aliases:
                aliases = latest.setdefault("aliases", {})
                for alias, target in self._pending_aliases.items():
                    if target is _DELETE:
                        aliases.pop(alias, None)
                    else:
                        aliases[alias] = target
                if not aliases:
                    latest.pop("aliases", None)

            self._write_atomic(latest)

        self._data = latest
        self._reset_pending()
        logger.debug("Saved proxy config to %s", self.config_path)

    # -- Proxy settings --

    @property
    def port(self) -> int:
        return int(self._data.get("proxy", {}).get("port", DEFAULT_PROXY_PORT))

    @property
    def host(self) -> str:
        return str(self._data.get("proxy", {}).get("host", DEFAULT_PROXY_HOST))

    @property
    def host_configured(self) -> bool:
        """True when a bind host has been explicitly persisted.

        Distinguishes "user chose a bind host" from "fell back to the legacy
        ``0.0.0.0`` default".  When False, the proxy keeps the legacy 0.0.0.0
        bind for backward compatibility but emits a loud security warning.
        """
        return "host" in self._data.get("proxy", {})

    @property
    def gateway(self) -> str | None:
        """Explicitly pinned gateway implementation, or ``None``.

        The selector half of :mod:`sparkrun.proxy.gateway` — availability is a
        feature flag in ``config.yaml``, *which* gateway to use is this key.
        ``None`` (the normal case) means "resolve the default"; pinning a
        gateway that is disabled is an error rather than a silent fallback.
        """
        val = self._data.get("proxy", {}).get("gateway")
        return str(val) if val else None

    @property
    def cluster(self) -> str | None:
        """Discovery cluster saved by an explicit ``--cluster``, or ``None``.

        ``None`` means "follow the default cluster", resolved at every start so
        a later ``sparkrun cluster set-default`` is picked up rather than
        frozen.  Written only when a cluster is named explicitly.
        """
        val = self._data.get("proxy", {}).get("cluster")
        return str(val) if val else None

    @property
    def master_key(self) -> str | None:
        val = self._data.get("proxy", {}).get("master_key", DEFAULT_MASTER_KEY)
        return str(val) if val is not None else None

    @property
    def auto_discover(self) -> bool:
        return bool(self._data.get("proxy", {}).get("auto_discover", True))

    @property
    def discovery_backend(self) -> str:
        return self.discovery_backend_for(self.gateway)

    def discovery_backend_for(self, gateway: str | None) -> str:
        """Resolve shared discovery settings for the gateway actually selected.

        SparkRoute daemon transport defaults to daemon discovery. Keep that
        default without applying SparkRoute settings to another
        gateway, including when --gateway overrides the saved selector.
        """
        proxy = getattr(self, "_data", {}).get("proxy", {})
        if "discovery_backend" in proxy:
            value = proxy["discovery_backend"]
        elif gateway == "sparkroute":
            transport = self.gateway_settings("sparkroute").get("transport", "bridge")
            if transport not in {"bridge", "daemon"}:
                raise ValueError("gateways.sparkroute.transport must be bridge or daemon")
            value = "daemon" if transport == "daemon" else "process"
        else:
            value = "process"
        if value not in {"process", "daemon"}:
            raise ValueError("proxy.discovery_backend must be process or daemon")
        return value

    @property
    def discover_interval(self) -> int:
        return int(self._data.get("proxy", {}).get("discover_interval", DEFAULT_DISCOVER_INTERVAL))

    @property
    def discover_removal_grace_sweeps(self) -> int:
        """Consecutive absent discovery sweeps required before removal.

        Clamped to at least 1 — zero would mean "remove before you have looked".
        """
        return max(
            1,
            int(
                self._data.get("proxy", {}).get(
                    "discover_removal_grace_sweeps",
                    DEFAULT_DISCOVER_REMOVAL_GRACE_SWEEPS,
                )
            ),
        )

    @property
    def enable_ui(self) -> bool:
        """True when an obsolete ``enable_ui`` is still set in proxy.yaml.

        The LiteLLM ``/ui`` is not supported — it is DB-backed and its
        ``schema.prisma`` requires PostgreSQL.  This survives only so
        ``proxy start`` can warn about (and ignore) a stale key rather than
        silently dropping a setting the user believes is active.
        """
        return bool(self._data.get("proxy", {}).get("enable_ui", False))

    def set_proxy(self, **kwargs: Any) -> None:
        """Update proxy settings (port, host, master_key, etc.).

        Recorded as pending so :meth:`save` merges exactly these keys into the
        newest document rather than replacing the whole ``proxy`` section.
        """
        proxy = self._data.setdefault("proxy", {})
        proxy.update(kwargs)
        self._pending_proxy.update(copy.deepcopy(kwargs))

    def unset_proxy(self, *keys: str) -> None:
        """Remove proxy settings, recorded as explicit deletions for :meth:`save`.

        "Absent from my copy" is not an instruction to delete a key another
        writer saved, so a removal has to be pending like any other change.
        """
        proxy = self._data.get("proxy", {})
        for key in keys:
            proxy.pop(key, None)
            self._pending_proxy[key] = _DELETE

    # -- Gateway-owned settings --

    def gateway_settings(self, name: str) -> dict[str, Any]:
        """Settings owned by gateway *name*, under ``gateways.<name>`` ({} when unset).

        Gateways are interchangeable, so a setting only one implementation
        understands is namespaced by its selector rather than placed at the top
        level, where it would read as applying to whichever gateway runs. The
        content is the gateway's to interpret; sparkrun neither validates it nor
        writes it. A value that is not a mapping is ignored.
        """
        gateways = self._data.get("gateways")
        value = gateways.get(name) if isinstance(gateways, dict) else None
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    # -- Alias management --

    @property
    def aliases(self) -> dict[str, str]:
        """Return alias -> model group mapping."""
        return dict(self._data.get("aliases", {}))

    def add_alias(self, alias: str, target: str) -> None:
        """Add or update an alias mapping."""
        aliases = self._data.setdefault("aliases", {})
        aliases[alias] = target
        self._pending_aliases[alias] = target

    def remove_alias(self, alias: str) -> bool:
        """Remove an alias. Returns True if it existed."""
        aliases = self._data.get("aliases", {})
        if alias in aliases:
            del aliases[alias]
            # Recorded as an explicit deletion rather than by absence: save()
            # merges into the newest document, where the alias may well still
            # be present, and "not in my copy" is not an instruction to delete.
            self._pending_aliases[alias] = _DELETE
            return True
        return False

    def list_aliases(self) -> list[tuple[str, str]]:
        """Return list of (alias, target) pairs."""
        return list(self._data.get("aliases", {}).items())

    # -- Default recipes (schema only, not wired up) --

    @property
    def default_recipes(self) -> dict[str, dict]:
        """Return default recipe mappings (future use)."""
        return dict(self._data.get("default_recipes", {}))
