"""Durable plugin generations, atomic activation, bounded history, and recovery."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import uuid

import yaml

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.plugin_manifest import PluginError, PluginMetadata, module_parts, safe_name, safe_relative
from sparkrun.utils.fs import atomic_private_write, file_lock

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1


def _read(path: Path) -> str | None:
    if path.is_symlink():
        raise PluginError("Refusing symlink in managed plugin state: %s" % path)
    return path.read_text(encoding="utf-8") if path.exists() else None


def _yaml(value: dict) -> str:
    return yaml.safe_dump(value, sort_keys=False)


def generation_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise PluginError("Invalid managed generation identifier")
    return value


def retention(config: SparkrunConfig) -> int:
    value = config.get("plugins.retain_generations", 2)
    if type(value) is not int or value < 1:
        raise PluginError("plugins.retain_generations must be a positive integer (default 2)")
    return value


def file_hashes(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PluginError("Symlink in managed plugin generation: %s" % path)
        if path.is_file() and path != root / "receipt.json" and "__pycache__" not in path.parts:
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif not path.is_dir() and not path.is_file():
            raise PluginError("Special file in managed plugin generation")
    return result


class PluginStore:
    def __init__(self, config: SparkrunConfig):
        self.config_path = config.config_path
        self.root = config.config_path.parent
        self.directory = self.root / "plugins"
        self.ledger = self.root / "plugins.yaml"
        self.journal = self.root / "plugins-transaction.yaml"
        self.lock = self.root / "plugins.lock"
        self.operation_lock = self.root / "plugins-operation.lock"

    def _ensure_root(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise PluginError("Managed plugins directory must not be a symlink")

    def _records(self) -> dict[str, dict]:
        raw = _read(self.ledger)
        if raw is None:
            return {}
        try:
            data = yaml.safe_load(raw)
            if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != SCHEMA_VERSION:
                raise PluginError("Unsupported plugins.yaml schema; upgrade Sparkrun before editing it")
            records = data["plugins"]
            if not isinstance(records, dict):
                raise PluginError("plugins.yaml plugins must be a mapping")
            for name, record in records.items():
                safe_name(name)
                if not isinstance(record, dict) or not isinstance(record.get("previous_generations"), list):
                    raise PluginError("Plugin records must contain a generation history list")
                metadata = PluginMetadata.from_dict(record["metadata"])
                if metadata.name != name:
                    raise PluginError("Plugin ledger name does not match its metadata")
                ids = [record["active_generation"], *record["previous_generations"]]
                if len(set(ids)) != len(ids):
                    raise PluginError("Plugin ledger repeats a generation")
                for identifier in ids:
                    generation_id(identifier)
                if not isinstance(record["source"], dict) or record["source"].get("kind") not in {"git", "wheel", "pypi"}:
                    raise PluginError("Invalid managed plugin source")
                source = record["source"]
                if source["kind"] != "git":
                    raise PluginError("This release cannot manage non-Git plugin records")
                if not isinstance(source.get("url"), str) or not isinstance(source.get("ref"), str):
                    raise PluginError("Plugin source requires a URL and ref")
                if source.get("ref_kind") not in {"branch", "tag", "commit"}:
                    raise PluginError("Invalid Git tracking policy")
                if not isinstance(record.get("revision"), str) or not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", record["revision"]):
                    raise PluginError("Invalid installed Git revision")
                if not isinstance(record.get("trust"), dict) or record["trust"].get("granted") is not True:
                    raise PluginError("Managed plugin has no source trust grant")
            return records
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise PluginError("Invalid plugin ledger: %s" % exc) from exc

    def _journal(self) -> dict | None:
        raw = _read(self.journal)
        if raw is None:
            return None
        try:
            data = yaml.safe_load(raw)
            if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("state") not in {"prepared", "committed"}:
                raise PluginError("Unknown plugin transaction; preserve the journal for recovery")
            if data.get("config_path") != str(self.config_path):
                raise PluginError("Plugin transaction belongs to another configuration file")
            for field in ("before_config", "after_config", "before_ledger", "after_ledger"):
                if field not in data or (data[field] is not None and not isinstance(data[field], str)):
                    raise PluginError("Invalid transaction snapshots; preserve the journal for recovery")
            owned = data.get("owned_generations", [])
            if not isinstance(owned, list):
                raise PluginError("Invalid transaction ownership records")
            for pair in owned:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise PluginError("Invalid transaction ownership record")
                safe_name(pair[0])
                generation_id(pair[1])
            return data
        except yaml.YAMLError as exc:
            raise PluginError("Invalid plugin transaction journal") from exc

    def snapshot(self) -> tuple[SparkrunConfig, dict[str, dict]]:
        # A stock install has no store and pays no filesystem creation cost.
        if not self.ledger.exists() and not self.journal.exists():
            return SparkrunConfig(self.config_path), {}
        self._ensure_root()
        with file_lock(self.lock):
            pending = self._journal()
            if pending and pending["state"] == "prepared":
                raise PluginError("Interrupted plugin activation; run sparkrun plugins update/remove to recover")
            return SparkrunConfig(self.config_path), self._records()

    def generation_path(self, name: str, identifier: str) -> Path:
        path = self.directory / safe_name(name) / generation_id(identifier)
        for parent in (self.directory, path.parent, path):
            if parent.is_symlink():
                raise PluginError("Symlink in managed plugin path: %s" % parent)
        return path

    def receipt(self, name: str, identifier: str, *, verify: bool = True) -> dict:
        path = self.generation_path(name, identifier)
        try:
            data = json.loads(_read(path / "receipt.json") or "{}")
            metadata = PluginMetadata.from_dict(data["metadata"])
            if metadata.name != name or data.get("generation") != identifier or data.get("schema_version") != 1:
                raise PluginError("Generation receipt ownership mismatch")
            if not isinstance(data["hashes"], dict):
                raise PluginError("Generation hashes must be a mapping")
            for relative, digest in data["hashes"].items():
                safe_relative(relative)
                if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                    raise PluginError("Invalid generation digest")
            if verify and file_hashes(path) != data["hashes"]:
                raise PluginError("Plugin generation was modified or is incomplete: %s" % path)
            return data
        except (OSError, TypeError, KeyError, ValueError) as exc:
            raise PluginError("Cannot validate generation %s/%s: %s" % (name, identifier, exc)) from exc

    @contextmanager
    def mutation(self):
        self._ensure_root()
        with file_lock(self.operation_lock):
            with file_lock(self.lock):
                self._recover()
                config = SparkrunConfig(self.config_path)
                retention(config)
                records = self._records()
                self._clean_staging()
                # Unpublished generations from an interrupted rename are safe to
                # clean only while no operation can still publish them.
                self._prune(records)
                baseline = (_read(self.config_path), _read(self.ledger))
            try:
                yield config, records, baseline
            finally:
                # A rejected commit must not leave an unpublished payload behind.
                with file_lock(self.lock):
                    if not self.journal.exists():
                        self._clean_staging()
                        self._prune(self._records())

    def _clean_staging(self):
        stage = self.directory / ".staging"
        if stage.is_symlink():
            raise PluginError("Refusing symlink in staging")
        if stage.exists():
            for path in stage.iterdir():
                generation_id(path.name)
                if path.is_symlink() or not path.is_dir():
                    raise PluginError("Unexpected staging entry")
                shutil.rmtree(path)

    def publish(self, candidate) -> tuple[str, dict]:
        identifier = uuid.uuid4().hex
        self.directory.mkdir(exist_ok=True)
        staging = self.directory / ".staging" / identifier
        staging.mkdir(parents=True)
        metadata = candidate.metadata
        try:
            payload = staging / "payload" / Path(*module_parts(metadata.module))
            payload.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(candidate.package_path, payload, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            # Preserve metadata and all exported licenses/notices outside the package.
            for source in candidate.root.rglob("*"):
                if source.is_symlink():
                    raise PluginError("Cannot publish symlinked plugin content")
                if not source.is_file() or source.is_relative_to(candidate.package_path):
                    continue
                relative = source.relative_to(candidate.root)
                target = staging / "metadata" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            receipt = {
                "schema_version": 1,
                "generation": identifier,
                "metadata": metadata.to_dict(),
                "source": candidate.source,
                "revision": candidate.revision,
                "hashes": file_hashes(staging),
            }
            atomic_private_write(staging / "receipt.json", json.dumps(receipt, indent=2))
            target = self.generation_path(metadata.name, identifier)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(target)
            return identifier, receipt
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def commit(self, records: dict[str, dict], config: SparkrunConfig, baseline: tuple[str | None, str | None]) -> str | None:
        """Commit activation, then prune. Return an honest cleanup-pending warning."""
        limit = retention(config)
        self.history_pruned = any(len(record["previous_generations"]) >= limit for record in records.values())
        for record in records.values():
            record["previous_generations"] = record["previous_generations"][: limit - 1]
        after_config = _yaml(config._data)
        # Avoid creating/reformatting config for pure plugin updates.
        before_config, before_ledger = baseline
        if yaml.safe_load(before_config or "{}") == config._data:
            after_config = before_config
        after_ledger = _yaml({"schema_version": SCHEMA_VERSION, "plugins": records})
        with file_lock(self.lock):
            if (_read(self.config_path), _read(self.ledger)) != baseline:
                raise PluginError("Plugin/config state changed during validation; retry the operation")
            before_records = (yaml.safe_load(before_ledger or "{}") or {}).get("plugins", {})
            owned = [
                [name, identifier]
                for name, record in before_records.items()
                for identifier in [record["active_generation"], *record["previous_generations"]]
            ]
            from sparkrun.core.runtime_lease import assert_generations_available

            assert_generations_available(self.root, records)
            journal = {
                "owned_generations": owned,
                "schema_version": 1,
                "config_path": str(self.config_path),
                "state": "prepared",
                "before_config": before_config,
                "before_ledger": before_ledger,
                "after_config": after_config,
                "after_ledger": after_ledger,
            }
            atomic_private_write(self.journal, _yaml(journal))
            try:
                self._restore(self.config_path, after_config)
                self._restore(self.ledger, after_ledger)
                journal["state"] = "committed"
                atomic_private_write(self.journal, _yaml(journal))
            except BaseException:
                self._recover()
                raise
            try:
                self._prune(records, owned=owned)
                self.journal.unlink()
            except (OSError, PluginError) as exc:
                return "Updated; cleanup pending (no further generations will be installed until cleanup succeeds): %s" % exc
        return None

    @staticmethod
    def _restore(path: Path, content: str | None):
        if content is None:
            path.unlink(missing_ok=True)
        else:
            atomic_private_write(path, content)

    def _recover(self):
        pending = self._journal()
        if pending is None:
            return
        if pending["state"] == "prepared":
            # Manual edits outside our lock must never be overwritten by recovery.
            for path, field in ((self.config_path, "config"), (self.ledger, "ledger")):
                if _read(path) not in (pending["before_" + field], pending["after_" + field]):
                    raise PluginError("Plugin recovery found outside edits in %s; reconcile the journal before retrying" % path)
            self._restore(self.config_path, pending["before_config"])
            self._restore(self.ledger, pending["before_ledger"])
        self._prune(self._records(), owned=pending.get("owned_generations", []))
        self._clean_staging()
        self.journal.unlink()

    def _prune(self, records: dict[str, dict], *, owned=()):
        from sparkrun.core.runtime_lease import assert_generations_available

        assert_generations_available(self.root, records)
        owned = {(safe_name(name), generation_id(identifier)) for name, identifier in owned}
        if not self.directory.exists():
            return
        for plugin in self.directory.iterdir():
            if plugin.name == ".staging":
                continue
            safe_name(plugin.name)
            if plugin.is_symlink() or not plugin.is_dir():
                raise PluginError("Unexpected entry in managed store: %s" % plugin)
            record = records.get(plugin.name)
            keep = {record["active_generation"], *record["previous_generations"]} if record else set()
            for path in plugin.iterdir():
                generation_id(path.name)
                if path.name in keep:
                    continue
                # Ownership is in the receipt; payload hashes can be incomplete
                # after a partial prune, so verify ownership rather than contents.
                if path.is_symlink():
                    raise PluginError("Refusing to delete a generation symlink")
                if not path.is_dir():
                    raise PluginError("Generation must be a directory")
                if not any(path.iterdir()):
                    path.rmdir()  # Recover deletion interrupted after receipt removal.
                    continue
                if (plugin.name, path.name) not in owned:
                    self.receipt(plugin.name, path.name, verify=False)
                # Keep the receipt until the last step for idempotent deletion.
                for child in path.iterdir():
                    if child.name == "receipt.json":
                        continue
                    if child.is_symlink():
                        raise PluginError("Refusing symlink in generation cleanup")
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
                (path / "receipt.json").unlink(missing_ok=True)
                path.rmdir()
            if not any(plugin.iterdir()):
                plugin.rmdir()

    @staticmethod
    def select(records: dict[str, dict], selector: str) -> tuple[str, dict]:
        if selector in records:
            return selector, deepcopy(records[selector])
        from sparkrun.core.plugin_manifest import normalize_source

        try:
            normalized = normalize_source(selector, local=True)
        except PluginError:
            normalized = selector
        matches = [(name, deepcopy(record)) for name, record in records.items() if record["source"].get("url") == normalized]
        if len(matches) != 1:
            raise PluginError(
                "No unique managed plugin matches %r; use sparkrun plugins list (unmanaged packages are not modified)" % selector
            )
        return matches[0]
