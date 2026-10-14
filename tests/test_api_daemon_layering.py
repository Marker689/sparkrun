"""The api wire layer classifies daemon errors without importing the daemon.

``api._wire.error_body`` recognises the job store's ``Conflict``; that type
used to live in ``sparkrun.daemon.jobs``, so encoding any error pulled the
daemon package into api-only processes (an api ↔ daemon import cycle).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_wire_error_encoding_does_not_import_the_daemon():
    code = textwrap.dedent(
        """
        import sys
        import sparkrun.api._wire as wire
        from sparkrun.core.coordination import Conflict

        body = wire.error_body(Conflict("Job queue is full"))
        assert body["error"]["type"] == "Conflict", body
        assert body["detail"] == "Job queue is full", body
        wire.error_body(RuntimeError("boom"))
        leaked = sorted(name for name in sys.modules if name == "sparkrun.daemon" or name.startswith("sparkrun.daemon."))
        assert not leaked, leaked
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_daemon_jobs_reexports_the_same_conflict():
    from sparkrun.core.coordination import Conflict
    from sparkrun.daemon.jobs import Conflict as DaemonConflict

    assert DaemonConflict is Conflict
