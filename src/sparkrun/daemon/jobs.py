"""Transactional jobs/events/idempotency, usable without an API server."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import sqlite3
import time
import uuid

TERMINAL = frozenset({"succeeded", "failed", "cancelled", "lost"})
SERVICE_OPERATIONS = frozenset({"proxy.watch", "proxy.serve"})
ACTIVE = frozenset({"starting", "running", "waiting_input", "cancelling"})


class Conflict(ValueError):
    pass


class ReplayGap(ValueError):
    pass


class JobStore:
    def __init__(self, directory, *, max_queued=128, max_events=4096, max_event_bytes=8 * 1024 * 1024):
        from sparkrun.daemon.state import private_directory

        if max_queued < 1 or max_events < 1 or max_event_bytes < 1024:
            raise ValueError("Job queue/events limits must be positive (event bytes >= 1024)")
        self.directory = private_directory(directory)
        self.path = self.directory / "jobs.sqlite3"
        self.max_queued = max_queued
        self.max_events = max(1, max_events)
        self.max_event_bytes = max(1024, max_event_bytes)
        with self.transaction() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > 2:
                raise RuntimeError("Job database was written by a newer daemon; upgrade before opening it")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, operation TEXT NOT NULL, version INTEGER NOT NULL,
                    arguments TEXT NOT NULL, principal TEXT NOT NULL, client TEXT NOT NULL,
                    scope TEXT NOT NULL, runtime TEXT NOT NULL, state TEXT NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL, result TEXT, error TEXT,
                    worker TEXT, nonce TEXT, prompt TEXT, cancelled REAL, seq INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS events (
                    job TEXT NOT NULL, seq INTEGER NOT NULL, created REAL NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(job, seq)
                );
                CREATE TABLE IF NOT EXISTS submissions (
                    principal TEXT NOT NULL, key TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    job TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY(principal, key)
                );
                CREATE TABLE IF NOT EXISTS prompts (job TEXT, id TEXT, metadata TEXT, PRIMARY KEY(job,id));
                CREATE INDEX IF NOT EXISTS events_created ON events(created);
            """)
            if not db.in_transaction:
                db.execute("BEGIN IMMEDIATE")
            if "manager" not in {row[1] for row in db.execute("PRAGMA table_info(jobs)")}:
                db.execute("ALTER TABLE jobs ADD COLUMN manager TEXT NOT NULL DEFAULT 'daemon'")
            db.execute("PRAGMA user_version=2")
        self.path.chmod(0o600)

    @classmethod
    def from_config(cls, directory, config):
        return cls(
            directory,
            max_queued=int(config.get("daemon.max_queued_jobs", 128)),
            max_events=int(config.get("daemon.max_events", 4096)),
            max_event_bytes=int(config.get("daemon.max_event_bytes", 8 * 1024 * 1024)),
        )

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("BEGIN IMMEDIATE")
        try:
            yield db
            if db.in_transaction:
                db.commit()
        except BaseException:
            if db.in_transaction:
                db.rollback()
            raise
        finally:
            db.close()

    def _event(self, db, job, payload):
        text = json.dumps(payload, allow_nan=False)
        if len(text.encode()) > 256 * 1024:
            raise ValueError("Event exceeds 256 KiB")
        db.execute("UPDATE jobs SET seq=seq+1, updated=? WHERE id=?", (time.time(), job))
        row = db.execute("SELECT seq FROM jobs WHERE id=?", (job,)).fetchone()
        if row is None:
            raise KeyError(job)
        db.execute("INSERT INTO events VALUES(?,?,?,?)", (job, row[0], time.time(), text))
        db.execute("DELETE FROM events WHERE job=? AND seq<=?", (job, row[0] - self.max_events))
        size = db.execute("SELECT COALESCE(SUM(length(payload)),0) FROM events WHERE job=?", (job,)).fetchone()[0]
        while size > self.max_event_bytes:
            old = db.execute("SELECT seq,length(payload) FROM events WHERE job=? ORDER BY seq LIMIT 1", (job,)).fetchone()
            db.execute("DELETE FROM events WHERE job=? AND seq=?", (job, old[0]))
            size -= old[1]
        return row[0]

    def emit(self, job, payload):
        with self.transaction() as db:
            return self._event(db, job, payload)

    def submit(self, op, arguments, principal, client, runtime, *, key=None, manager="daemon"):
        request = json.dumps(
            {"op": op.name, "version": op.version, "arguments": arguments, "manager": manager}, sort_keys=True, allow_nan=False
        )
        digest = hashlib.sha256(request.encode()).hexdigest()
        with self.transaction() as db:
            if key:
                if len(key) > 200:
                    raise ValueError("Idempotency key exceeds 200 characters")
                previous = db.execute("SELECT * FROM submissions WHERE principal=? AND key=?", (principal, key)).fetchone()
                if previous:
                    if previous["fingerprint"] != digest:
                        raise Conflict("Idempotency key was already used for different arguments")
                    if not db.execute("SELECT 1 FROM jobs WHERE id=?", (previous["job"],)).fetchone():
                        raise Conflict("The previous result expired; this idempotency key cannot execute again")
                    return previous["job"]
            if db.execute("SELECT count(*) FROM jobs WHERE state='queued'").fetchone()[0] >= self.max_queued:
                raise Conflict("Job queue is full")
            job = uuid.uuid4().hex
            now = time.time()
            db.execute(
                "INSERT INTO jobs(id,operation,version,arguments,principal,client,scope,runtime,state,created,updated,manager) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job,
                    op.name,
                    op.version,
                    json.dumps(arguments, allow_nan=False),
                    principal,
                    client[:100],
                    op.scope,
                    json.dumps(runtime),
                    "queued",
                    now,
                    now,
                    manager,
                ),
            )
            if key:
                db.execute("INSERT INTO submissions VALUES(?,?,?,?,?)", (principal, key, digest, job, now))
            self._event(db, job, {"type": "state", "state": "queued"})
            return job

    @staticmethod
    def _decode(row):
        if row is None:
            raise KeyError("Unknown job")
        data = dict(row)
        for key in ("arguments", "runtime", "result", "error", "worker", "prompt"):
            data[key] = json.loads(data[key]) if data[key] else None
        return data

    def get(self, job):
        with self.transaction() as db:
            return self._decode(db.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone())

    def list(self, *, principal=None, state=None, limit=200):
        with self.transaction() as db:
            sql = "SELECT * FROM jobs WHERE 1=1"
            args = []
            if principal:
                sql += " AND principal=?"
                args.append(principal)
            if state:
                sql += " AND state=?"
                args.append(state)
            args.append(min(max(int(limit), 1), 1000))
            order = " ORDER BY CASE WHEN state IN ('starting','running','waiting_input','cancelling') THEN 0 WHEN state='queued' THEN 1 ELSE 2 END, created DESC LIMIT ?"
            return [self._decode(row) for row in db.execute(sql + order, args)]

    def active(self):
        """All admitted workers, independently of the bounded history listing."""
        with self.transaction() as db:
            return [
                self._decode(row)
                for row in db.execute(
                    "SELECT * FROM jobs WHERE state IN ('starting','running','waiting_input','cancelling') ORDER BY created"
                )
            ]

    def claim(self, *, max_active=4, max_services=2):
        with self.transaction() as db:
            counts = {False: 0, True: 0}
            for active in db.execute("SELECT operation FROM jobs WHERE state IN ('starting','running','waiting_input','cancelling')"):
                counts[active[0] in SERVICE_OPERATIONS] += 1
            row = None
            for queued in db.execute("SELECT id,operation FROM jobs WHERE state='queued' ORDER BY created"):
                service = queued["operation"] in SERVICE_OPERATIONS
                if counts[service] < (max_services if service else max_active):
                    row = queued
                    break
            if row is None:
                return None
            job, nonce = row[0], uuid.uuid4().hex + uuid.uuid4().hex
            db.execute("UPDATE jobs SET state='starting', nonce=? WHERE id=?", (nonce, job))
            self._event(db, job, {"type": "state", "state": "starting"})
            return self._decode(db.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone())

    def register_worker(self, job, nonce, worker):
        with self.transaction() as db:
            row = db.execute("SELECT state,nonce,worker FROM jobs WHERE id=?", (job,)).fetchone()
            if row is None or row["state"] != "starting" or row["nonce"] != nonce or row["worker"]:
                raise Conflict("Job claim is no longer available")
            db.execute("UPDATE jobs SET worker=?,updated=? WHERE id=?", (json.dumps(worker), time.time(), job))

    def authorize(self, job, nonce):
        with self.transaction() as db:
            row = db.execute("SELECT state,nonce,worker FROM jobs WHERE id=?", (job,)).fetchone()
            if row is None or row["nonce"] != nonce or not row["worker"]:
                raise Conflict("Invalid worker claim")
            if row["state"] == "running":
                return
            if row["state"] != "starting":
                raise Conflict("Job cannot start")
            db.execute("UPDATE jobs SET state='running' WHERE id=?", (job,))
            self._event(db, job, {"type": "state", "state": "running"})

    def finish(self, job, state, *, result=None, error=None):
        if state not in TERMINAL:
            raise ValueError("Expected terminal state")
        encoded = json.dumps(result, allow_nan=False) if result is not None else None
        if encoded and len(encoded.encode()) > 2 * 1024 * 1024:
            state, encoded, error = "failed", None, {"detail": "Result exceeds 2 MiB"}
        with self.transaction() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=?", (job,)).fetchone()
            if row is None:
                raise KeyError(job)
            if row[0] in TERMINAL:
                return False
            db.execute(
                "UPDATE jobs SET state=?,result=?,error=?,prompt=NULL WHERE id=?",
                (state, encoded, json.dumps(error) if error else None, job),
            )
            if result is not None and encoded:
                # Large results are fetched by GET rather than duplicated into the event ring.
                self._event(db, job, {"type": "result", "available": True})
            if error:
                self._event(db, job, {"type": "error", "error": error})
            self._event(db, job, {"type": "state", "state": state})
            return True

    def cancel(self, job):
        with self.transaction() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=?", (job,)).fetchone()
            if row is None:
                raise KeyError(job)
            if row[0] in TERMINAL:
                raise Conflict("Job is already finished")
            state = "cancelled" if row[0] == "queued" else "cancelling"
            db.execute("UPDATE jobs SET state=?,cancelled=COALESCE(cancelled,?) WHERE id=?", (state, time.time(), job))
            self._event(db, job, {"type": "state", "state": state})

    def prompt(self, job, metadata):
        with self.transaction() as db:
            row = db.execute("SELECT state FROM jobs WHERE id=?", (job,)).fetchone()
            if not row or row[0] != "running":
                raise Conflict("Job cannot request input")
            db.execute("UPDATE jobs SET state='waiting_input',prompt=? WHERE id=?", (json.dumps(metadata), job))
            db.execute("INSERT INTO prompts VALUES(?,?,?)", (job, metadata["prompt_id"], json.dumps(metadata)))
            self._event(db, job, {"type": "prompt", **metadata})
            self._event(db, job, {"type": "state", "state": "waiting_input"})

    def answer(self, job, prompt_id):
        with self.transaction() as db:
            row = db.execute("SELECT state,prompt FROM jobs WHERE id=?", (job,)).fetchone()
            if not row or row[0] != "waiting_input" or json.loads(row[1])["prompt_id"] != prompt_id:
                raise Conflict("Prompt has expired or was already answered")
            db.execute("UPDATE jobs SET state='running',prompt=NULL WHERE id=?", (job,))
            self._event(db, job, {"type": "answered", "prompt_id": prompt_id})
            self._event(db, job, {"type": "state", "state": "running"})

    def prompt_info(self, job, prompt_id):
        with self.transaction() as db:
            row = db.execute("SELECT metadata FROM prompts WHERE job=? AND id=?", (job, prompt_id)).fetchone()
            if not row:
                raise Conflict("Prompt expired or unavailable")
            return json.loads(row[0])

    def events(self, job, after=0):
        with self.transaction() as db:
            row = db.execute("SELECT min(seq),max(seq) FROM events WHERE job=?", (job,)).fetchone()
            if after and row[0] is not None and after < row[0] - 1:
                raise ReplayGap(f"Event cursor expired; retained events start at {row[0]}")
            return [
                {"id": v[0], **json.loads(v[1])}
                for v in db.execute("SELECT seq,payload FROM events WHERE job=? AND seq>? ORDER BY seq LIMIT 256", (job, after))
            ]

    def revision(self):
        with self.transaction() as db:
            return tuple(db.execute("SELECT count(*),max(updated),sum(seq) FROM jobs").fetchone())

    def prune(self, *, age: float = 7 * 86400, count=1000):
        with self.transaction() as db:
            rows = db.execute(
                "SELECT id,updated,worker FROM jobs WHERE state IN ('succeeded','failed','cancelled','lost') ORDER BY updated DESC"
            ).fetchall()
            from sparkrun.core.process_identity import alive

            for i, row in enumerate(rows):
                if row[2] and alive(json.loads(row[2])["process"]):
                    continue  # Completion cleanup still owns its runtime and resources.
                if i >= count or row[1] < time.time() - age:
                    db.execute("DELETE FROM events WHERE job=?", (row[0],))
                    db.execute("DELETE FROM jobs WHERE id=?", (row[0],))
                    db.execute("DELETE FROM prompts WHERE job=?", (row[0],))
                    (self.directory / (row[0] + ".error.log")).unlink(missing_ok=True)
            db.execute(
                "DELETE FROM submissions WHERE created<? AND job NOT IN "
                "(SELECT id FROM jobs WHERE state NOT IN ('succeeded','failed','cancelled','lost'))",
                (time.time() - 30 * 86400,),
            )
