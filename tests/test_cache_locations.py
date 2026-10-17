"""Cache locations: hosts that share one model cache are one cache to operate on.

The probe and the nonce round-trip are bash run on each host, so these run
them under real bash, with each fixture "host" given its own ``HOME`` (cache
paths are written ``$HOME/...`` and expanded on the host, as in production).
"""

from __future__ import annotations

from dataclasses import replace
import fcntl
import subprocess

import pytest

from sparkrun.models import cache as cache_mod
from sparkrun.models.cache_locations import identity, probe_script, resolve_locations, verify_locations
from sparkrun.models.cache_reports import CacheLocation
from sparkrun.models.host_io import ModelHostIO
from sparkrun.transports.session import HostCommandResult

NFS_A = 'FSTYPE="nfs4" SOURCE="nas:/export/hf" TARGET="/mnt/hf" FSROOT="/export/hf"'
BOOT_1, BOOT_2 = "11111111-1111", "22222222-2222"


def probe(*, real="/mnt/hf/cache", at=None, inode=42, device=50, boot=BOOT_1, mount=NFS_A):
    return "REAL=%s\nPROBE=%s\nINODE=%d\nDEVICE=%d\nBOOT=%s\n%s\n" % (real, at or real, inode, device, boot, mount)


# ---------------------------------------------------------------------------
# identity(): what may and may not be merged
# ---------------------------------------------------------------------------


def test_one_export_seen_from_two_machines_is_one_cache():
    a = identity("spark-01", probe(boot=BOOT_1, device=50))
    b = identity("spark-02", probe(boot=BOOT_2, device=77))  # client-local device numbers differ
    assert a[0] == b[0] and a[1] == "network" and a[2] == "nas:/export/hf"


def test_one_export_mounted_at_different_paths_is_one_cache():
    other = 'FSTYPE="nfs" SOURCE="nas:/export/hf" TARGET="/data" FSROOT="/export/hf"'
    a = identity("spark-01", probe(real="/mnt/hf/cache"))
    b = identity("spark-02", probe(real="/data/cache", mount=other))
    assert a[0] == b[0]


@pytest.mark.parametrize(
    "change",
    [
        {"inode": 43},  # a different directory on the same export
        {"real": "/mnt/hf/other"},  # a different subdirectory
        {"mount": NFS_A.replace("nas:", "nas2:")},  # a different server
    ],
)
def test_different_network_caches_are_kept_apart(change):
    assert identity("a", probe())[0] != identity("b", probe(**change))[0]


def test_local_disks_are_never_merged_across_machines():
    local = 'FSTYPE="ext4" SOURCE="/dev/nvme0n1p2" TARGET="/" FSROOT="/"'
    # Identically imaged hosts: same device numbers, same inode, different boots.
    a = identity("spark-01", probe(real="/home/u/.cache/huggingface", mount=local, boot=BOOT_1))
    b = identity("spark-02", probe(real="/home/u/.cache/huggingface", mount=local, boot=BOOT_2))
    assert a[1] == "local" and a[0] != b[0]


def test_one_machine_under_two_names_is_one_cache():
    local = 'FSTYPE="ext4" SOURCE="/dev/nvme0n1p2" TARGET="/" FSROOT="/"'
    a = identity("10.0.0.5", probe(real="/home/u/.cache/huggingface", mount=local))
    b = identity("spark-01", probe(real="/home/u/.cache/huggingface", mount=local))
    assert a[0] == b[0]


def test_a_cache_not_yet_created_keys_on_its_nearest_ancestor():
    a = identity("a", probe(real="/mnt/hf/cache", at="/mnt/hf"))
    b = identity("b", probe(real="/mnt/hf/cache", at="/mnt/hf"))
    c = identity("c", probe(real="/mnt/hf/elsewhere", at="/mnt/hf"))
    assert a[0] == b[0] != c[0]


@pytest.mark.parametrize("stdout", ["", "garbage", "REAL=relative\nPROBE=relative\nINODE=1\n"])
def test_an_unusable_probe_identifies_nothing(stdout):
    assert identity("a", stdout) is None


def test_the_probe_runs_under_bash(tmp_path):
    proc = subprocess.run(["bash"], input=probe_script(str(tmp_path / "hf")), text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    key, kind, source, path = identity("localhost", proc.stdout)
    assert kind in {"local", "network"} and path == str(tmp_path / "hf")


# ---------------------------------------------------------------------------
# resolve / verify / cache_operation on local fixture hosts
# ---------------------------------------------------------------------------


class Homes:
    """An io whose hosts are local bash shells, each with its own HOME."""

    def __init__(self, homes):
        self.homes = homes
        self.calls = []

    def root(self, host, cache_dir):
        return cache_dir.replace("$HOME", str(self.homes[host]))

    def execute(self, host, script, timeout=30, **_):
        self.calls.append(host)
        proc = subprocess.run(
            ["bash"], input=script.encode(), capture_output=True, env={"HOME": str(self.homes[host]), "PATH": "/usr/bin:/bin"}
        )
        return HostCommandResult(host, proc.returncode, proc.stdout, proc.stderr)


def test_hosts_seeing_one_directory_form_one_location(tmp_path):
    io = Homes({"a": tmp_path / "shared", "b": tmp_path / "shared", "c": tmp_path / "own"})
    for home in io.homes.values():
        home.mkdir(exist_ok=True)

    locations = resolve_locations(io, ["a", "b", "c"], "$HOME/hf")

    assert [loc.hosts for loc in locations] == [("a", "b"), ("c",)]
    assert locations[0].paths == {"a": str(tmp_path / "shared/hf"), "b": str(tmp_path / "shared/hf")}
    assert not locations[0].verified and locations[1].verified


def test_a_failed_probe_is_a_location_of_its_own(tmp_path):
    io = Homes({"a": tmp_path, "b": tmp_path})

    def execute(host, script, timeout=30, **_):
        if host == "b":
            return HostCommandResult(host, 255, b"", b"ssh: connect refused")
        return Homes.execute(io, host, script, timeout)

    io.execute = execute
    locations = resolve_locations(io, ["a", "b"], "$HOME/hf")
    assert [loc.hosts for loc in locations] == [("a",), ("b",)]
    assert locations[1].kind == "unknown"


def test_a_shared_location_is_confirmed_by_a_nonce(tmp_path):
    (tmp_path / "shared").mkdir()
    io = Homes({"a": tmp_path / "shared", "b": tmp_path / "shared"})

    (location,) = verify_locations(io, resolve_locations(io, ["a", "b"], "$HOME/hf"))

    assert location.hosts == ("a", "b") and location.verified
    assert (tmp_path / "shared/hf/.sparkrun-model-artifacts/location-probe").exists()


def test_a_host_that_does_not_see_the_nonce_is_split_off(tmp_path):
    """What the probed identity got wrong, the round-trip catches, in the safe direction."""
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
    io = Homes({"a": tmp_path / "one", "b": tmp_path / "two"})
    wrong = CacheLocation("k", "network", "nas:/x", ("a", "b"), {"a": str(tmp_path / "one/hf"), "b": str(tmp_path / "two/hf")})

    locations = verify_locations(io, [wrong], timeout=0.3)

    assert [loc.hosts for loc in locations] == [("a",), ("b",)]
    assert all(loc.verified for loc in locations)


class LocalHosts(ModelHostIO):
    homes: dict = {}
    agent_runs: list = []

    def execute(self, host, script, **kwargs):
        if "print(json.dumps(run(" in script:
            LocalHosts.agent_runs.append(host)
        return super().execute(None, "export HOME=%s\n" % self.homes[host] + script, **kwargs)


@pytest.fixture
def local_hosts(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    LocalHosts.homes = {"a": shared, "b": shared}
    LocalHosts.agent_runs = []
    monkeypatch.setattr(cache_mod, "ModelHostIO", LocalHosts)
    return shared


def test_inventory_runs_once_per_cache_and_names_every_host(local_hosts):
    from sparkrun.api import model_cache

    (report,) = model_cache.inventory_cache(cache_dir="$HOME/hf", hosts=("a", "b"))

    assert LocalHosts.agent_runs == ["a"]
    assert report.host == "a" and report.location.hosts == ("a", "b")
    assert not (local_hosts / "hf/.sparkrun-model-artifacts/location-probe").exists()  # list stays read-only


def test_a_mutating_operation_verifies_before_running_once(local_hosts):
    from sparkrun.api import model_cache

    (report,) = model_cache.prune_cache(cache_dir="$HOME/hf", hosts=("a", "b"), dry_run=False, min_age_days=0)

    assert LocalHosts.agent_runs == ["a"]
    assert report.location.verified and report.location.hosts == ("a", "b")


def test_reports_round_trip_through_the_wire(local_hosts):
    from sparkrun.api import model_cache
    from sparkrun.api._wire import to_wire, validate
    from sparkrun.api._wire_schemas import RESULT_SCHEMAS
    from sparkrun.models.cache_reports import CacheReport

    reports = model_cache.inventory_cache(cache_dir="$HOME/hf", hosts=("a", "b"))
    wire = to_wire(reports)
    validate(wire, RESULT_SCHEMAS["model_cache.list"])
    assert CacheReport.from_dict(wire[0]) == reports[0]


def test_a_shared_cache_is_rendered_once_with_its_hosts():
    from sparkrun.utils.cli_formatters import format_model_cache_reports
    from test_model_cache_cli import entry, report

    location = CacheLocation(
        "k", "network", "nas:/export/hf", ("spark-01", "spark-02"), {"spark-01": "/mnt/hf", "spark-02": "/mnt/hf"}, True
    )
    text = format_model_cache_reports([replace(report([entry()]), host="spark-01", cache_root="/mnt/hf", location=location)])

    assert text.splitlines()[0].startswith("nas:/export/hf")
    assert "shared by 2 hosts: spark-01, spark-02" in text
    assert "mounted at: /mnt/hf" in text


def test_delete_on_a_shared_cache_previews_and_applies_once(local_hosts):
    """Per host, the preview summed one cache's bytes once per host and asked to confirm that."""
    from sparkrun.api import model_cache
    from sparkrun.models.artifacts import ModelArtifactRequest
    from test_model_cache_delete import ONLY_A, REV_A, hf_repo

    hf_repo(local_hosts / "hf", "org/model", REV_A, {"w.safetensors": ONLY_A})
    request = ModelArtifactRequest("org/model")

    (preview,) = model_cache.delete_model(request, cache_dir="$HOME/hf", hosts=("a", "b"))
    assert preview.location.hosts == ("a", "b")
    assert len(ONLY_A) <= preview.planned_bytes < 2 * len(ONLY_A)

    (applied,) = model_cache.delete_model(request, cache_dir="$HOME/hf", hosts=("a", "b"), dry_run=False)
    assert applied.reclaimed_bytes == preview.planned_bytes and applied.location.verified
    assert LocalHosts.agent_runs == ["a", "a", "a"]  # preview; apply's own preview; apply
    assert not (local_hosts / "hf/hub/models--org--model").exists()


# ---------------------------------------------------------------------------
# Review regressions
# ---------------------------------------------------------------------------


def test_findmnt_escapes_are_decoded():
    """A mount point with a space is printed as \\x20; undecoded, the host dropped out of its share."""
    spaced = r'FSTYPE="nfs" SOURCE="nas:/export/hf" TARGET="/mnt/a\x20b" FSROOT="/export/hf"'
    a = identity("a", probe(real="/mnt/a b/cache", mount=spaced))
    b = identity("b", probe(real="/mnt/hf/cache"))
    assert a is not None and a[0] == b[0]


def test_an_unknown_location_still_gets_the_resolved_cache_root(tmp_path):
    """The agent refuses a relative or ~ path; the old per-host resolution must survive a failed probe."""
    io = Homes({"a": tmp_path})
    original = io.execute

    def execute(host, script, timeout=30, **_):
        return HostCommandResult(host, 0, b"garbage", b"") if "findmnt" in script else original(host, script, timeout)

    io.execute = execute
    io.root = lambda host, cache_dir: str(tmp_path / "hf")
    (location,) = resolve_locations(io, ["a"], "$HOME/hf")
    assert location.kind == "unknown" and location.paths == {"a": str(tmp_path / "hf")}


def test_a_down_first_host_does_not_split_the_rest(tmp_path):
    (tmp_path / "shared").mkdir()
    io = Homes({"down": tmp_path / "shared", "b": tmp_path / "shared", "c": tmp_path / "shared"})
    original = io.execute
    io.execute = lambda host, script, timeout=30, **_: (
        HostCommandResult(host, 255, b"", b"unreachable") if host == "down" else original(host, script, timeout)
    )
    location = CacheLocation("k", "local", "x", ("down", "b", "c"), {h: str(tmp_path / "shared/hf") for h in ("down", "b", "c")})

    locations = verify_locations(io, [location], timeout=0.3)

    assert [loc.hosts for loc in locations] == [("b", "c"), ("down",)]


def test_a_failed_mutation_is_reported_not_retried_on_another_host(local_hosts, monkeypatch):
    """The failed host may already have changed the cache; a second run would misreport it."""
    from sparkrun.api import model_cache
    from sparkrun.models.artifacts import ModelArtifactError

    calls = []

    def failing(io, host, root, **options):
        calls.append(host)
        raise ModelArtifactError("cache operation on %s failed: timed out" % host)

    monkeypatch.setattr(cache_mod, "host_cache", failing)
    with pytest.raises(ModelArtifactError):
        model_cache.prune_cache(cache_dir="$HOME/hf", hosts=("a", "b"), dry_run=False, min_age_days=0)
    assert calls == ["a"]
    calls.clear()
    with pytest.raises(ModelArtifactError):
        model_cache.inventory_cache(cache_dir="$HOME/hf", hosts=("a", "b"))
    assert calls == ["a", "b"]  # a read-only listing may ask another host


def test_delete_gates_its_apply_on_exactly_the_caches_it_changes(tmp_path, monkeypatch):
    """If verification splits a wrongly merged host, its refusal must still stop the whole delete."""
    from sparkrun.api import model_cache
    from sparkrun.models import cache_locations
    from sparkrun.models.artifacts import ModelArtifactRequest
    from test_model_cache_delete import ONLY_A, REV_A, hf_repo

    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        hf_repo(tmp_path / name / "hf", "org/model", REV_A, {"w.safetensors": ONLY_A})
    # A download in progress on b's (separate) cache: delete must refuse everywhere.
    locks = tmp_path / "two/hf/hub/.locks/models--org--model"
    locks.mkdir(parents=True)
    held = open(locks / "x.lock", "w")
    fcntl.flock(held, fcntl.LOCK_EX)
    LocalHosts.homes = {"a": tmp_path / "one", "b": tmp_path / "two"}
    LocalHosts.agent_runs = []
    monkeypatch.setattr(cache_mod, "ModelHostIO", LocalHosts)
    real_resolve = cache_locations.resolve_locations

    def merged(io, hosts, cache_dir):
        # The probe wrongly merges two different caches.
        found = real_resolve(io, hosts, cache_dir)
        paths = {h: p for loc in found for h, p in loc.paths.items()}
        return (CacheLocation("same", "network", "nas:/x", tuple(hosts), paths),)

    monkeypatch.setattr(cache_locations, "resolve_locations", merged)

    try:
        reports = model_cache.delete_model(ModelArtifactRequest("org/model"), cache_dir="$HOME/hf", hosts=("a", "b"), dry_run=False)
    finally:
        held.close()

    assert [r.location.hosts for r in reports] == [("a",), ("b",)]
    assert not all(r.target_met for r in reports)
    assert (tmp_path / "one/hf/hub/models--org--model").exists()  # nothing deleted anywhere
