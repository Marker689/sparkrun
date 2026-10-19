"""Failure output keeps its root cause (issue #325).

A hook's error used to carry ``stderr[:500]``: for a traceback that is the
header and the first frames, and never the exception line at the end. These
cover the shared head+tail excerpt, the redacted full-output file a hook
failure points at, and that file's bookkeeping.
"""

from __future__ import annotations

import os
import re
import stat
import time
from unittest import mock

import pytest

from sparkrun.orchestration import hooks
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.utils.text import excerpt

TRACEBACK = (
    "Traceback (most recent call last):\n"
    + "".join('  File "/usr/lib/python3/site-packages/pkg/mod%d.py", line %d, in fn\n    call()\n' % (i, i) for i in range(200))
    + "huggingface_hub.errors.GatedRepoError: 403 Client Error: access to model org/name is restricted"
)


# ---------------------------------------------------------------------------
# excerpt
# ---------------------------------------------------------------------------


def test_excerpt_leaves_short_output_alone():
    assert excerpt("  short output\n", 100) == "short output"


def test_excerpt_of_nothing_is_empty():
    assert excerpt(None, 100) == ""
    assert excerpt("   \n", 100) == ""


def test_excerpt_keeps_the_final_exception_line():
    text = excerpt(TRACEBACK, 500)
    assert text.startswith("Traceback (most recent call last):")
    assert text.endswith("GatedRepoError: 403 Client Error: access to model org/name is restricted")


def test_excerpt_marks_and_counts_the_omission():
    text = excerpt(TRACEBACK, 500)
    match = re.search(r"^… \[(\d+) chars omitted\] …$", text, re.M)
    assert match
    head, tail = text[: match.start()].rstrip("\n"), text[match.end() :].lstrip("\n")
    # The count is exact: head + omitted + tail is the whole text.
    assert len(head) + int(match.group(1)) + len(tail) == len(TRACEBACK)


def test_excerpt_cuts_at_line_boundaries():
    for line in excerpt(TRACEBACK, 500).splitlines():
        assert line in TRACEBACK.splitlines() or line.startswith("… [")


def test_excerpt_stays_near_its_budget():
    assert len(excerpt(TRACEBACK, 500)) <= 500 + 40


def test_excerpt_bounds_a_single_unbroken_line():
    text = excerpt("a" * 1000 + "END", 100)
    assert text.startswith("a")
    assert text.endswith("END")
    assert len(text) <= 100 + 40


# ---------------------------------------------------------------------------
# hook failures
# ---------------------------------------------------------------------------


def _fail(stderr="", stdout="", returncode=1):
    return RemoteResult(host="spark1", returncode=returncode, stdout=stdout, stderr=stderr)


def _run_failing_hook(result, *, cluster_id="sparkrun_abc123", host="spark1"):
    env = {"SPARKRUN_CLUSTER_ID": cluster_id}
    with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=result):
        with pytest.raises(RuntimeError) as excinfo:
            hooks._run_exec_command(host, "ctr", "python fetch.py", label="pre_exec[1]", env=env)
    return str(excinfo.value)


def _saved_path(message):
    marker = "(full output: "
    assert marker in message
    return message.split(marker, 1)[1].rstrip(")")


def test_hook_failure_message_keeps_the_root_cause():
    message = _run_failing_hook(_fail(stderr=TRACEBACK))
    assert message.startswith("pre_exec[1] failed (exit 1) on spark1/ctr:")
    assert "GatedRepoError: 403 Client Error" in message
    assert "chars omitted" in message


def test_hook_failure_saves_full_output_privately():
    message = _run_failing_hook(_fail(stderr=TRACEBACK, stdout="progress line"))
    path = _saved_path(message)
    content = open(path, encoding="utf-8").read()
    assert TRACEBACK in content
    assert "progress line" in content
    assert "python fetch.py" in content
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_hook_failure_file_is_under_the_sparkrun_cache():
    from sparkrun.core.config import resolve_sparkrun_cache_dir

    path = _saved_path(_run_failing_hook(_fail(stderr="boom")))
    assert os.path.dirname(path) == str(resolve_sparkrun_cache_dir() / "failures")
    assert os.path.basename(path) == "sparkrun_abc123-spark1.log"


def test_next_failure_overwrites_the_same_workload_and_host():
    first = _saved_path(_run_failing_hook(_fail(stderr="first failure")))
    second = _saved_path(_run_failing_hook(_fail(stderr="second failure")))
    assert first == second
    content = open(second, encoding="utf-8").read()
    assert "second failure" in content
    assert "first failure" not in content


def test_failures_on_other_hosts_are_kept_apart():
    one = _saved_path(_run_failing_hook(_fail(stderr="on one"), host="spark1"))
    two = _saved_path(_run_failing_hook(_fail(stderr="on two"), host="spark2"))
    assert one != two
    assert "on one" in open(one, encoding="utf-8").read()


def test_log_key_cannot_escape_the_failures_directory():
    path = _saved_path(_run_failing_hook(_fail(stderr="x"), cluster_id="../../etc", host="h/../x"))
    from sparkrun.core.config import resolve_sparkrun_cache_dir

    assert os.path.dirname(path) == str(resolve_sparkrun_cache_dir() / "failures")


def test_hook_failure_redacts_tokens(monkeypatch):
    token = "hf_" + "S3cretT0kenValue" * 2
    monkeypatch.setattr("sparkrun.core.config.resolve_hf_token", lambda: token)
    message = _run_failing_hook(_fail(stderr="auth with %s failed\nHF_TOKEN=%s" % (token, token)))
    assert token not in message
    assert token not in open(_saved_path(message), encoding="utf-8").read()


def test_unwritable_failure_dir_still_reports_the_error(monkeypatch, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setattr(hooks, "_failure_log_dir", lambda: blocker / "failures")
    message = _run_failing_hook(_fail(stderr=TRACEBACK))
    assert "GatedRepoError" in message
    assert "full output" not in message


def test_no_output_is_explicit():
    message = _run_failing_hook(_fail())
    assert "(no output)" in message


def test_old_failure_logs_are_pruned():
    path = _saved_path(_run_failing_hook(_fail(stderr="x"), host="spark1"))
    directory = os.path.dirname(path)
    stale = os.path.join(directory, "sparkrun_old-spark9.log")
    with open(stale, "w") as f:
        f.write("old")
    old = time.time() - hooks.FAILURE_LOG_MAX_AGE_S - 60
    os.utime(stale, (old, old))
    recent = os.path.join(directory, "sparkrun_recent-spark9.log")
    with open(recent, "w") as f:
        f.write("recent")

    _run_failing_hook(_fail(stderr="y"), host="spark2")

    assert not os.path.exists(stale)
    assert os.path.exists(recent)
    assert os.path.exists(path)


def test_post_commands_failure_carries_its_output():
    with mock.patch("sparkrun.orchestration.hooks._run_control_command") as run:
        run.return_value = (2, TRACEBACK)
        with pytest.raises(RuntimeError) as excinfo:
            hooks.run_post_commands(["./notify.sh"], {"cluster_id": "sparkrun_abc123"}, trust=True)
    message = str(excinfo.value)
    assert message.startswith("post_commands[1] failed (exit 2): ./notify.sh")
    assert "GatedRepoError" in message
    assert TRACEBACK in open(_saved_path(message), encoding="utf-8").read()


def test_copy_failure_keeps_the_end_of_its_output():
    result = _fail(stderr="docker cp: start\n" + "noise\n" * 300 + "no space left on device")
    message = hooks._copy_failure_message("pre_exec[2]", "spark1", "ctr", result)
    assert message.startswith("pre_exec[2] copy failed on spark1/ctr:")
    assert message.endswith("no space left on device")


def test_excerpt_does_not_snap_away_most_of_a_part():
    # A distant newline must not reduce a part to a stub.
    assert len(excerpt("a" * 1000 + "\nZ", 100)) > 60
    assert len(excerpt("x\n" + "y" * 1000, 100)) > 60


def test_token_straddling_the_cut_is_not_partially_leaked(monkeypatch):
    secret = "A" * 37
    token = "hf_" + secret
    monkeypatch.setattr("sparkrun.core.config.resolve_hf_token", lambda: token)
    # The tail keeps the last 3/4 of the budget; leave 20 of the token's
    # characters inside it, so cutting first would strand them.
    tail_budget = hooks.HOOK_FAILURE_EXCERPT_LIMIT - hooks.HOOK_FAILURE_EXCERPT_LIMIT // 4
    stderr = "x" * 5000 + token + "y" * (tail_budget - 20)
    message = _run_failing_hook(_fail(stderr=stderr))
    assert "A" * 10 not in message


def test_unreadable_token_file_does_not_hide_the_hook_error(monkeypatch):
    def boom():
        raise PermissionError("token file unreadable")

    monkeypatch.setattr("sparkrun.core.config.resolve_hf_token", boom)
    message = _run_failing_hook(_fail(stderr=TRACEBACK))
    assert message.startswith("pre_exec[1] failed (exit 1) on spark1/ctr:")
    assert "GatedRepoError" in message


def test_unresolvable_failure_dir_does_not_hide_the_hook_error(monkeypatch):
    def boom():
        raise RuntimeError("no cache dir")

    monkeypatch.setattr(hooks, "_failure_log_dir", boom)
    message = _run_failing_hook(_fail(stderr=TRACEBACK))
    assert "GatedRepoError" in message
    assert "full output" not in message
