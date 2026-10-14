"""Tests for the Kubernetes setup surface.

Covers kubectl binary acquisition/version-cache, the KubectlClient
wrapper, kube-target resolution, manifest + service-account generation,
config accessors, the api.k8s surface, and the ``setup k8s`` CLI. No real
cluster or network — urllib / subprocess are stubbed.
"""

from __future__ import annotations

from sparkrun import api
from sparkrun.plugins.k8s import api as k8s_api
from sparkrun.plugins.k8s.config import K8sSettings

import hashlib
import os

import pytest

from sparkrun.plugins.k8s.orchestration import kubectl, manifests, serviceaccount
from sparkrun.plugins.k8s.orchestration.client import KubectlClient
from sparkrun.plugins.k8s.orchestration.connect import ClusterInfo, probe_cluster
from sparkrun.plugins.k8s.orchestration.context import resolve_kube_target
from sparkrun.plugins.k8s.orchestration.errors import KubectlDownloadError, KubectlNotFoundError
from sparkrun.orchestration.ssh import RemoteResult


# ---------------------------------------------------------------------------
# Platform detection & cache layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "system,expected",
    [("Linux", "linux"), ("Darwin", "darwin"), ("Windows", "windows")],
)
def test_detect_os(monkeypatch, system, expected):
    monkeypatch.setattr(kubectl.platform, "system", lambda: system)
    assert kubectl.detect_os() == expected


@pytest.mark.parametrize(
    "machine,expected",
    [("x86_64", "amd64"), ("amd64", "amd64"), ("aarch64", "arm64"), ("arm64", "arm64")],
)
def test_detect_arch(monkeypatch, machine, expected):
    monkeypatch.setattr(kubectl.platform, "machine", lambda: machine)
    assert kubectl.detect_arch() == expected


def test_binary_name():
    assert kubectl.binary_name("linux") == "kubectl"
    assert kubectl.binary_name("windows") == "kubectl.exe"


def test_cached_binary_path_layout(tmp_path):
    p = kubectl.cached_binary_path(tmp_path, "v1.31.0", "linux", "arm64")
    assert p == tmp_path / "kubectl" / "v1.31.0" / "linux-arm64" / "kubectl"


def test_normalize_release_version():
    assert kubectl.normalize_release_version("v1.31.2") == "v1.31.2"
    assert kubectl.normalize_release_version("v1.31.2+ck1") == "v1.31.2"
    assert kubectl.normalize_release_version("v1.30.0-eks-abc") == "v1.30.0"
    assert kubectl.normalize_release_version("garbage") is None


def test_list_cached_sorted_newest_first(tmp_path):
    for version in ("v1.29.0", "v1.31.0", "v1.30.5"):
        p = kubectl.cached_binary_path(tmp_path, version, "linux", "arm64")
        p.parent.mkdir(parents=True)
        p.write_text("#!/bin/sh\n")
    cached = kubectl.list_cached(tmp_path, os_name="linux", arch="arm64")
    assert [b.version for b in cached] == ["v1.31.0", "v1.30.5", "v1.29.0"]


# ---------------------------------------------------------------------------
# Download + integrity
# ---------------------------------------------------------------------------


def _stub_urlopen(mapping):
    """Return a fake urlopen resolving *mapping* (url -> bytes)."""

    class _Resp:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _open(url, timeout=None):  # noqa: ARG001
        if url not in mapping:
            raise AssertionError("unexpected url: %s" % url)
        return _Resp(mapping[url])

    return _open


def test_download_kubectl_verifies_checksum(tmp_path, monkeypatch):
    payload = b"fake-kubectl-binary"
    digest = hashlib.sha256(payload).hexdigest()
    url = kubectl._release_url("v1.31.0", "linux", "arm64")
    monkeypatch.setattr(
        kubectl.urllib.request,
        "urlopen",
        _stub_urlopen({url: payload, url + ".sha256": ("%s\n" % digest).encode()}),
    )
    dest = kubectl.download_kubectl(tmp_path, "v1.31.0", "linux", "arm64")
    assert dest.read_bytes() == payload
    assert os.access(dest, os.X_OK)


def test_download_kubectl_rejects_bad_checksum(tmp_path, monkeypatch):
    payload = b"fake-kubectl-binary"
    url = kubectl._release_url("v1.31.0", "linux", "arm64")
    monkeypatch.setattr(
        kubectl.urllib.request,
        "urlopen",
        _stub_urlopen({url: payload, url + ".sha256": b"deadbeef\n"}),
    )
    with pytest.raises(KubectlDownloadError, match="checksum mismatch"):
        kubectl.download_kubectl(tmp_path, "v1.31.0", "linux", "arm64")


# ---------------------------------------------------------------------------
# ensure_kubectl resolution order
# ---------------------------------------------------------------------------


def test_ensure_kubectl_explicit_path(tmp_path):
    binary = tmp_path / "mykubectl"
    binary.write_text("#!/bin/sh\n")
    resolved = kubectl.ensure_kubectl(tmp_path, explicit_path=binary)
    assert resolved.source == "config"
    assert resolved.path == binary


def test_ensure_kubectl_explicit_path_missing(tmp_path):
    with pytest.raises(KubectlNotFoundError):
        kubectl.ensure_kubectl(tmp_path, explicit_path=tmp_path / "nope")


def test_ensure_kubectl_prefers_matching_cached_version(tmp_path):
    p = kubectl.cached_binary_path(tmp_path, "v1.31.0", "linux", "arm64")
    p.parent.mkdir(parents=True)
    p.write_text("#!/bin/sh\n")
    resolved = kubectl.ensure_kubectl(tmp_path, version="v1.31.0", os_name="linux", arch="arm64")
    assert resolved.source == "cache"
    assert resolved.version == "v1.31.0"


def test_ensure_kubectl_downloads_when_nothing_cached(tmp_path, monkeypatch):
    called = {}

    def _fake_download(cache_dir, version, os_name, arch, **kw):  # noqa: ARG001
        called["version"] = version
        p = kubectl.cached_binary_path(cache_dir, version, os_name, arch)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#!/bin/sh\n")
        return p

    monkeypatch.setattr(kubectl, "fetch_stable_version", lambda **kw: "v1.31.0")
    monkeypatch.setattr(kubectl, "download_kubectl", _fake_download)
    resolved = kubectl.ensure_kubectl(tmp_path, allow_path=False, os_name="linux", arch="arm64")
    assert resolved.source == "download"
    assert called["version"] == "v1.31.0"


def test_ensure_kubectl_no_download_raises(tmp_path):
    with pytest.raises(KubectlNotFoundError):
        kubectl.ensure_kubectl(tmp_path, version="v9.9.9", allow_path=False, allow_download=False)


# ---------------------------------------------------------------------------
# KubectlClient
# ---------------------------------------------------------------------------


def test_client_base_args_ordering():
    client = KubectlClient("/usr/bin/kubectl", kubeconfig="/k/cfg", context="ctx", namespace="ns")
    assert client.base_args() == ["/usr/bin/kubectl", "--kubeconfig", "/k/cfg", "--context", "ctx", "-n", "ns"]


def test_client_run_returns_remote_result(monkeypatch):
    client = KubectlClient("/usr/bin/kubectl")

    class _Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(
        "sparkrun.plugins.k8s.orchestration.client.subprocess.run",
        lambda *a, **k: _Proc(),
    )
    result = client.run(["get", "pods"])
    assert isinstance(result, RemoteResult)
    assert result.success and result.stdout == "ok"


def test_client_run_json_parses(monkeypatch):
    client = KubectlClient("/usr/bin/kubectl")

    class _Proc:
        returncode = 0
        stdout = '{"a": 1}'
        stderr = ""

    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.client.subprocess.run", lambda *a, **k: _Proc())
    assert client.run_json(["version", "-o", "json"]) == {"a": 1}


def test_client_dry_run_short_circuits():
    client = KubectlClient("/usr/bin/kubectl", dry_run=True)
    result = client.run(["apply", "-f", "-"])
    assert result.success and result.stdout == "[dry-run]"


def test_client_exec_builds_kubectl_exec(monkeypatch):
    client = KubectlClient("/usr/bin/kubectl", namespace="ns")
    captured = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(cmd, **k):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.client.subprocess.run", _run)
    client.exec("mypod", "echo hi", container="c1")
    cmd = captured["cmd"]
    assert "exec" in cmd and "mypod" in cmd and "-c" in cmd and "c1" in cmd and "--" in cmd


# ---------------------------------------------------------------------------
# Kube target resolution
# ---------------------------------------------------------------------------


def test_resolve_kube_target_arg_precedence(monkeypatch):
    monkeypatch.delenv("KUBECONFIG", raising=False)
    target = resolve_kube_target(None, kubeconfig="/explicit", context="c", namespace="n")
    assert target.kubeconfig == "/explicit"
    assert target.context == "c"
    assert target.namespace == "n"


def test_resolve_kube_target_env_kubeconfig(monkeypatch):
    monkeypatch.setenv("KUBECONFIG", "/env/cfg")
    target = resolve_kube_target(None)
    assert target.kubeconfig == "/env/cfg"


def test_resolve_kube_target_config_block(monkeypatch, tmp_path):
    from sparkrun.core.config import SparkrunConfig

    monkeypatch.delenv("KUBECONFIG", raising=False)
    cfg = SparkrunConfig(tmp_path / "config.yaml")
    cfg.set("k8s", {"context": "prod", "namespace": "sparkrun"})
    target = resolve_kube_target(cfg)
    assert target.context == "prod"
    assert target.namespace == "sparkrun"


# ---------------------------------------------------------------------------
# Manifests & RBAC
# ---------------------------------------------------------------------------


def test_default_runner_rules_include_jobs_and_exec():
    rules = manifests.default_runner_rules()
    flat = [(g, r, v) for rule in rules for g in rule["apiGroups"] for r in rule["resources"] for v in rule["verbs"]]
    assert ("batch", "jobs", "create") in flat
    assert ("", "pods/exec", "create") in flat
    assert ("", "pods", "watch") in flat


def test_build_manifests_no_cluster_admin(tmp_path):
    spec = serviceaccount.ServiceAccountSpec(name="sparkrun", namespace="sparkrun")
    rendered = serviceaccount.build_manifests(spec)
    assert "kind: ClusterRole" in rendered
    assert "kind: ClusterRoleBinding" in rendered
    assert "cluster-admin" not in rendered
    assert "app.kubernetes.io/managed-by: sparkrun" in rendered


def test_build_manifests_omits_namespace_when_disabled():
    spec = serviceaccount.ServiceAccountSpec(create_namespace=False)
    rendered = serviceaccount.build_manifests(spec)
    assert "kind: Namespace" not in rendered


# ---------------------------------------------------------------------------
# Service account: kubeconfig writing
# ---------------------------------------------------------------------------


def test_build_kubeconfig_binds_token():
    kc = serviceaccount.build_kubeconfig(
        server="https://api:6443",
        cluster={"certificate-authority-data": "CADATA"},
        token="tok123",
        context_name="sparkrun",
        namespace="sparkrun",
    )
    assert kc["users"][0]["user"]["token"] == "tok123"
    assert kc["clusters"][0]["cluster"]["certificate-authority-data"] == "CADATA"
    assert kc["current-context"] == "sparkrun"


def test_write_kubeconfig_is_0600(tmp_path):
    dest = tmp_path / "k8s" / "sparkrun.kubeconfig"
    written = serviceaccount.write_kubeconfig(dest, {"apiVersion": "v1", "kind": "Config"})
    mode = written.stat().st_mode & 0o777
    assert mode == 0o600


def test_configure_service_account_dry_run_does_not_apply():
    spec = serviceaccount.ServiceAccountSpec()

    class _Client:
        def apply(self, *a, **k):
            raise AssertionError("dry-run must not apply")

    result = serviceaccount.configure_service_account(_Client(), spec, dry_run=True)
    assert result.dry_run and not result.applied
    assert "ClusterRole" in result.manifests_yaml


def test_service_account_result_redacts_token():
    r = serviceaccount.ServiceAccountResult(
        name="sparkrun",
        namespace="sparkrun",
        cluster_role="sparkrun-runner",
        binding="sparkrun-runner",
        manifests_yaml="",
        dry_run=False,
        token="secret-token",
    )
    assert r.redacted().token == "***"
    assert r.token == "secret-token"


# ---------------------------------------------------------------------------
# probe_cluster
# ---------------------------------------------------------------------------


def test_probe_cluster_reachable(monkeypatch):
    client = KubectlClient("/usr/bin/kubectl", context="ctx")

    def _run(args, **k):
        if args[:1] == ["version"]:
            return RemoteResult(
                host="ctx",
                returncode=0,
                stdout='{"clientVersion":{"gitVersion":"v1.31.0"},"serverVersion":{"gitVersion":"v1.30.2"}}',
                stderr="",
            )
        return RemoteResult(host="ctx", returncode=0, stdout="ctx", stderr="")

    monkeypatch.setattr(client, "run", _run)
    info = probe_cluster(client)
    assert info.reachable
    assert info.server_version == "v1.30.2"
    assert info.client_version == "v1.31.0"


def test_probe_cluster_unreachable(monkeypatch):
    client = KubectlClient("/usr/bin/kubectl", context="ctx")

    def _run(args, **k):
        if args[:1] == ["version"]:
            return RemoteResult(
                host="ctx",
                returncode=1,
                stdout='{"clientVersion":{"gitVersion":"v1.31.0"}}',
                stderr="dial tcp: connection refused",
            )
        return RemoteResult(host="ctx", returncode=0, stdout="ctx", stderr="")

    monkeypatch.setattr(client, "run", _run)
    info = probe_cluster(client)
    assert not info.reachable
    assert "connection refused" in (info.message or "")


# ---------------------------------------------------------------------------
# Config accessors
# ---------------------------------------------------------------------------


def test_config_kubectl_accessors_and_pin(tmp_path):
    from sparkrun.core.config import SparkrunConfig

    cfg = SparkrunConfig(tmp_path / "config.yaml")
    assert K8sSettings(cfg).k8s_defaults == {}
    assert K8sSettings(cfg).kubectl_path is None
    cfg.set("k8s", {"kubectl": {"path": "/x/kubectl", "version": "v1.31.0"}})
    assert K8sSettings(cfg).kubectl_path == "/x/kubectl"
    assert K8sSettings(cfg).kubectl_version == "v1.31.0"

    K8sSettings(cfg).pin_kubectl_version("prod-ctx", "v1.30.2")
    assert K8sSettings(cfg).kubectl_pinned_version("prod-ctx") == "v1.30.2"
    assert K8sSettings(cfg).kubectl_pinned_version("other") is None
    # existing kubectl settings preserved after pin
    assert K8sSettings(cfg).kubectl_path == "/x/kubectl"


# ---------------------------------------------------------------------------
# api.k8s surface
# ---------------------------------------------------------------------------


def _sctx(tmp_path):
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.context import SparkrunContext

    return SparkrunContext(
        variables=init_sparkrun(),
        config=SparkrunConfig(tmp_path / "config.yaml"),
        verbose=False,
    )


def test_api_ensure_kubectl_translates_error(tmp_path):

    sctx = _sctx(tmp_path)
    with pytest.raises(k8s_api.KubectlUnavailable):
        k8s_api.ensure_kubectl(sctx, version="v9.9.9", download=False)


def test_api_cluster_info_pins_server_version(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s import api as apik8s

    sctx = _sctx(tmp_path)

    fake = ClusterInfo(reachable=True, current_context="prod", server_version="v1.30.2+ck1", client_version="v1.31.0")
    monkeypatch.setattr(apik8s._ops, "make_client", lambda *a, **k: object())
    monkeypatch.setattr(apik8s._ops, "probe_cluster", lambda client: fake)

    info = k8s_api.cluster_info(sctx)
    assert info.server_version == "v1.30.2+ck1"
    # normalized (+ck1 stripped) pin persisted
    assert K8sSettings(sctx.config).kubectl_pinned_version("prod") == "v1.30.2"


# ---------------------------------------------------------------------------
# K8sExecutor transition onto KubectlClient
# ---------------------------------------------------------------------------


def test_k8s_executor_prefix_uses_resolved_path():
    from sparkrun.plugins.k8s.executor import K8sExecutor, K8sExecutorConfig

    ex = K8sExecutor(K8sExecutorConfig(kubectl_path="/opt/kubectl", k8s_context="ctx", k8s_namespace="ns"))
    cmd = ex.run_cmd(image="img:tag", command="echo hi", container_name="pod1")
    assert cmd.startswith("/opt/kubectl ")
    assert "--context ctx" in cmd
    assert "-n ns" in cmd


def test_k8s_executor_prefix_falls_back_to_bare_kubectl():
    from sparkrun.plugins.k8s.executor import K8sExecutor, K8sExecutorConfig

    ex = K8sExecutor(K8sExecutorConfig(executor_type="k8s"))
    cmd = ex.run_cmd(image="img:tag", command="echo hi", container_name="pod1")
    assert cmd.startswith("kubectl ")


def test_k8s_executor_finalize_config_resolves_cached_binary(tmp_path):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.plugins.k8s.executor import K8sExecutor, K8sExecutorConfig

    os_name, arch = kubectl.detect_os(), kubectl.detect_arch()
    binary = kubectl.cached_binary_path(tmp_path, "v1.31.0", os_name, arch)
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")

    cfg = SparkrunConfig(tmp_path / "config.yaml")
    cfg.set("cache_dir", str(tmp_path))

    ex = K8sExecutor(K8sExecutorConfig(executor_type="k8s"))
    ex.finalize_config(config=cfg)
    assert ex.config.kubectl_path == str(binary)


def test_k8s_executor_finalize_config_skips_partial_config(tmp_path):
    from sparkrun.plugins.k8s.executor import K8sExecutor, K8sExecutorConfig

    class _PartialConfig:
        default_executor = "k8s"
        executor_config: dict = {}

    ex = K8sExecutor(K8sExecutorConfig(executor_type="k8s"))
    ex.finalize_config(config=_PartialConfig())  # must not raise
    assert ex.config.kubectl_path is None


def test_resolve_executor_wires_kubectl_path(tmp_path):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.executor import resolve_executor
    from sparkrun.plugins.k8s.executor import K8sExecutor

    os_name, arch = kubectl.detect_os(), kubectl.detect_arch()
    binary = kubectl.cached_binary_path(tmp_path, "v1.31.0", os_name, arch)
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")

    cfg = SparkrunConfig(tmp_path / "config.yaml")
    cfg.set("cache_dir", str(tmp_path))
    # executor.k8s is force-enabled by the isolate_stateful conftest fixture.

    ex = resolve_executor(cli_overrides={"executor": "k8s"}, config=cfg, rootless=False, auto_user=False)
    assert isinstance(ex, K8sExecutor)
    assert ex.config.kubectl_path == str(binary)


# ---------------------------------------------------------------------------
# Launcher Job (job-driven launch, Phase 1)
# ---------------------------------------------------------------------------


def test_launcher_job_command_form():
    from sparkrun.plugins.k8s.orchestration.job import LauncherJobSpec, build_launcher_manifests

    spec = LauncherJobSpec(
        name="cl-abc",
        image="ghcr.io/x/sparkrun:latest",
        namespace="sparkrun",
        command=["sparkrun", "run", "qwen"],
        env={"HF_TOKEN": "x"},
    )
    docs = build_launcher_manifests(spec)
    assert len(docs) == 1  # no ConfigMap for command form
    job = docs[0]
    assert job["kind"] == "Job"
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["ttlSecondsAfterFinished"] == 3600
    pod = job["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == "sparkrun"
    assert pod["restartPolicy"] == "Never"
    assert pod["containers"][0]["command"] == ["sparkrun", "run", "qwen"]
    assert {"name": "HF_TOKEN", "value": "x"} in pod["containers"][0]["env"]


def test_launcher_job_script_form_mounts_configmap():
    from sparkrun.plugins.k8s.orchestration.job import LauncherJobSpec, build_launcher_manifests

    spec = LauncherJobSpec(name="cl-def", image="kubectl:1", namespace="ns", script="echo hi")
    docs = build_launcher_manifests(spec)
    assert docs[0]["kind"] == "ConfigMap"
    assert docs[0]["data"]["launch.sh"] == "echo hi"
    assert docs[0]["metadata"]["name"] == "cl-def-script"
    pod = docs[1]["spec"]["template"]["spec"]
    assert pod["containers"][0]["command"] == ["bash", "/sparkrun/launch.sh"]
    assert pod["volumes"][0]["configMap"]["name"] == "cl-def-script"


def test_launcher_job_requires_exactly_one_payload():
    from sparkrun.plugins.k8s.orchestration.job import LauncherJobSpec

    with pytest.raises(ValueError, match="exactly one"):
        LauncherJobSpec(name="x", image="i")
    with pytest.raises(ValueError, match="exactly one"):
        LauncherJobSpec(name="x", image="i", command=["a"], script="b")


def test_launcher_job_active_deadline_optional():
    from sparkrun.plugins.k8s.orchestration.job import LauncherJobSpec, job_manifest

    without = job_manifest(LauncherJobSpec(name="j", image="i", command=["a"]))
    assert "activeDeadlineSeconds" not in without["spec"]
    with_deadline = job_manifest(LauncherJobSpec(name="j", image="i", command=["a"], active_deadline_seconds=300))
    assert with_deadline["spec"]["activeDeadlineSeconds"] == 300


def test_client_run_launcher_job_applies(monkeypatch):
    from sparkrun.plugins.k8s.orchestration.job import LauncherJobSpec

    client = KubectlClient("/usr/bin/kubectl", namespace="ns")
    captured = {}

    def _apply(manifest_yaml, **k):
        captured["yaml"] = manifest_yaml
        return RemoteResult(host="k8s", returncode=0, stdout="created", stderr="")

    monkeypatch.setattr(client, "apply", _apply)
    spec = LauncherJobSpec(name="cl-1", image="img", command=["sparkrun", "run"])
    res = client.run_launcher_job(spec)
    assert res.success
    assert "kind: Job" in captured["yaml"]


def test_api_run_launcher_job_requires_image(tmp_path):

    sctx = _sctx(tmp_path)
    with pytest.raises(k8s_api.LauncherJobError, match="launcher image"):
        k8s_api.run_launcher_job(sctx, name="cl-1", command=["sparkrun"], dry_run=True)


def test_api_run_launcher_job_uses_config_image(tmp_path):

    sctx = _sctx(tmp_path)
    sctx.config.set("k8s", {"launcher_image": "ghcr.io/x/sparkrun:pinned"})
    result = k8s_api.run_launcher_job(sctx, name="cl-1", command=["sparkrun", "run"], dry_run=True)
    assert result.dry_run and not result.applied
    assert result.image == "ghcr.io/x/sparkrun:pinned"
    assert "kind: Job" in result.manifests_yaml


def test_api_run_launcher_job_dry_run_does_not_build_client(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s import api as apik8s

    sctx = _sctx(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("dry-run must not build a client / resolve kubectl")

    monkeypatch.setattr(apik8s._ops, "make_client", _boom)
    result = k8s_api.run_launcher_job(sctx, name="cl-1", image="img", command=["x"], dry_run=True)
    assert result.dry_run


# ---------------------------------------------------------------------------
# Node inventory (k8s-native introspection spine)
# ---------------------------------------------------------------------------


def _node(name, labels, *, capacity_gpu=None, allocatable_gpu=None, unschedulable=False):
    capacity = {"nvidia.com/gpu": str(capacity_gpu)} if capacity_gpu is not None else {}
    allocatable = {"nvidia.com/gpu": str(allocatable_gpu)} if allocatable_gpu is not None else {}
    return {
        "metadata": {"name": name, "labels": labels},
        "status": {"capacity": capacity, "allocatable": allocatable},
        "spec": {"unschedulable": unschedulable} if unschedulable else {},
    }


_SPARK_LABELS = {
    "nvidia.com/gpu.present": "true",
    "nvidia.com/gpu.product": "NVIDIA-GB10",
    "nvidia.com/gpu.machine": "NVIDIA-DGX-Spark",
    "nvidia.com/gpu.count": "1",
    "nvidia.com/gpu.memory": "131072",
    "feature.node.kubernetes.io/pci-15b3.present": "true",
}
_RTX_LABELS = {
    "nvidia.com/gpu.present": "true",
    "nvidia.com/gpu.product": "NVIDIA-RTX-PRO-6000-Blackwell",
    "nvidia.com/gpu.count": "1",
    "nvidia.com/gpu.memory": "98304",
}


def test_inventory_dgx_spark_node_maps_to_gb10():
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info
    from sparkrun.platforms import resolve_platform

    info = build_node_info(_node("spark-0", _SPARK_LABELS, capacity_gpu=1, allocatable_gpu=1))
    accel = info.hardware.accelerators[0]
    assert accel.vendor == "nvidia"
    assert accel.model == "gb10"  # same token as the SSH fingerprint path
    assert accel.count == 1
    assert accel.memory_gb == 128.0  # 131072 MiB / 1024
    assert "rdma:roce-v2" in accel.capabilities  # Mellanox PCI present
    assert resolve_platform(info.hardware).platform_name == "dgx-spark"


def test_inventory_rtx_pro_6000_maps_to_generic_nvidia():
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info
    from sparkrun.platforms import resolve_platform

    info = build_node_info(_node("rtx-0", _RTX_LABELS, capacity_gpu=1, allocatable_gpu=1))
    accel = info.hardware.accelerators[0]
    assert accel.model == "rtx-pro-6000-blackwell"
    assert accel.memory_gb == 96.0
    # Not gb10 → generic NVIDIA platform, not DGX Spark
    assert resolve_platform(info.hardware).platform_name == "nvidia-generic"


def test_inventory_hybrid_cluster_distinguishes_node_classes():
    from sparkrun.plugins.k8s.orchestration.inventory import parse_nodes
    from sparkrun.platforms import resolve_platform

    nodes = parse_nodes({"items": [_node("spark-0", _SPARK_LABELS, capacity_gpu=1), _node("rtx-0", _RTX_LABELS, capacity_gpu=1)]})
    by_name = {n.name: n for n in nodes}
    assert resolve_platform(by_name["spark-0"].hardware).platform_name == "dgx-spark"
    assert resolve_platform(by_name["rtx-0"].hardware).platform_name == "nvidia-generic"
    # distinct models → the scheduler can allocate the right node class
    assert by_name["spark-0"].hardware.accelerators[0].model != by_name["rtx-0"].hardware.accelerators[0].model


def test_inventory_cordoned_and_allocatable():
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info

    info = build_node_info(_node("rtx-0", _RTX_LABELS, capacity_gpu=1, allocatable_gpu=0, unschedulable=True))
    assert info.schedulable is False
    assert info.capacity_gpus == 1
    assert info.allocatable_gpus == 0


def test_inventory_count_falls_back_to_capacity_when_label_absent():
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info

    labels = {"nvidia.com/gpu.present": "true", "nvidia.com/gpu.product": "NVIDIA-H200"}
    info = build_node_info(_node("h200-0", labels, capacity_gpu=8, allocatable_gpu=8))
    assert info.hardware.accelerators[0].count == 8
    assert info.hardware.accelerators[0].model == "h200"


def test_inventory_cpu_node_has_no_accelerators():
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info

    info = build_node_info(_node("cpu-0", {"kubernetes.io/arch": "amd64"}))
    assert info.hardware.accelerators == []
    assert info.has_accelerators is False


def test_probe_nodes_passes_selector_and_filters_gpu_only(monkeypatch):
    from sparkrun.plugins.k8s.orchestration.client import KubectlClient
    from sparkrun.plugins.k8s.orchestration.inventory import probe_nodes

    client = KubectlClient("/usr/bin/kubectl")
    captured = {}

    def _run_json(args, **k):
        captured["args"] = args
        return {"items": [_node("spark-0", _SPARK_LABELS, capacity_gpu=1), _node("cpu-0", {})]}

    monkeypatch.setattr(client, "run_json", _run_json)
    nodes = probe_nodes(client, selector="nvidia.com/gpu.present=true", gpu_only=True)
    assert "-l" in captured["args"] and "nvidia.com/gpu.present=true" in captured["args"]
    assert [n.name for n in nodes] == ["spark-0"]  # cpu-0 filtered out


def test_probe_node_hardware_returns_hosthardware_map(monkeypatch):
    from sparkrun.core.hardware import HostHardware
    from sparkrun.plugins.k8s.orchestration.client import KubectlClient
    from sparkrun.plugins.k8s.orchestration.inventory import probe_node_hardware

    client = KubectlClient("/usr/bin/kubectl")
    monkeypatch.setattr(client, "run_json", lambda args, **k: {"items": [_node("spark-0", _SPARK_LABELS, capacity_gpu=1)]})
    hw_map = probe_node_hardware(client)
    assert set(hw_map) == {"spark-0"}
    assert isinstance(hw_map["spark-0"], HostHardware)


def test_api_list_nodes_translates_error(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s import api as apik8s
    from sparkrun.plugins.k8s.orchestration.errors import K8sError

    sctx = _sctx(tmp_path)
    monkeypatch.setattr(apik8s._ops, "make_client", lambda *a, **k: object())

    def _boom(*a, **k):
        raise K8sError("connection refused")

    # list_nodes imports probe_nodes from the inventory module at call time.
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", _boom)
    with pytest.raises(k8s_api.ClusterUnreachable, match="connection refused"):
        k8s_api.list_nodes(sctx)


def test_cli_setup_k8s_nodes_renders(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli import main
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info

    monkeypatch.setenv("STATEFUL_ROOT", str(tmp_path / "stateful"))
    fake = [
        build_node_info(_node("spark-0", _SPARK_LABELS, capacity_gpu=1, allocatable_gpu=1)),
        build_node_info(_node("rtx-0", _RTX_LABELS, capacity_gpu=1, allocatable_gpu=1)),
    ]
    monkeypatch.setattr("sparkrun.plugins.k8s.api.list_nodes", lambda *a, **k: fake)
    result = CliRunner().invoke(main, ["setup", "k8s", "nodes"])
    assert result.exit_code == 0, result.output
    assert "spark-0" in result.output and "gb10" in result.output
    assert "rtx-0" in result.output and "rtx-pro-6000" in result.output
    assert "DGX Spark" in result.output


# ---------------------------------------------------------------------------
# Kueue setup (gang scheduling)
# ---------------------------------------------------------------------------


def test_kueue_derive_flavors_groups_by_product():
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info
    from sparkrun.plugins.k8s.orchestration.kueue import derive_flavors

    nodes = [
        build_node_info(_node("spark-0", _SPARK_LABELS, capacity_gpu=1)),
        build_node_info(_node("spark-1", _SPARK_LABELS, capacity_gpu=1)),
        build_node_info(_node("rtx-0", _RTX_LABELS, capacity_gpu=1)),
    ]
    flavors = {f.name: f for f in derive_flavors(nodes)}
    assert set(flavors) == {"sparkrun-gb10", "sparkrun-rtx-pro-6000-blackwell"}
    assert flavors["sparkrun-gb10"].gpu_quota == 2  # two spark nodes summed
    assert flavors["sparkrun-gb10"].node_labels == {"nvidia.com/gpu.product": "NVIDIA-GB10"}
    assert flavors["sparkrun-rtx-pro-6000-blackwell"].gpu_quota == 1


def test_kueue_provision_manifests_shape():
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info
    from sparkrun.plugins.k8s.orchestration.kueue import build_provision_manifests

    nodes = [build_node_info(_node("spark-0", _SPARK_LABELS, capacity_gpu=1)), build_node_info(_node("rtx-0", _RTX_LABELS, capacity_gpu=1))]
    docs, flavors = build_provision_manifests(nodes, namespace="sparkrun")
    kinds = [d["kind"] for d in docs]
    assert kinds.count("ResourceFlavor") == 2
    assert kinds.count("ClusterQueue") == 1 and kinds.count("LocalQueue") == 1
    cq = next(d for d in docs if d["kind"] == "ClusterQueue")
    quotas = {f["name"]: f["resources"][0]["nominalQuota"] for f in cq["spec"]["resourceGroups"][0]["flavors"]}
    assert quotas == {"sparkrun-gb10": "1", "sparkrun-rtx-pro-6000-blackwell": "1"}
    lq = next(d for d in docs if d["kind"] == "LocalQueue")
    assert lq["metadata"]["namespace"] == "sparkrun" and lq["spec"]["clusterQueue"] == "sparkrun"


def test_kueue_provision_raises_without_gpu_product_labels():
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info
    from sparkrun.plugins.k8s.orchestration.kueue import KueueError, build_provision_manifests

    nodes = [build_node_info(_node("cpu-0", {"kubernetes.io/arch": "amd64"}))]
    with pytest.raises(KueueError, match="GPU Feature Discovery"):
        build_provision_manifests(nodes)


def test_kueue_detect_reads_crds(monkeypatch):
    from sparkrun.plugins.k8s.orchestration.client import KubectlClient
    from sparkrun.plugins.k8s.orchestration.kueue import CRD_CLUSTERQUEUE, CRD_JOBSET, detect

    client = KubectlClient("/usr/bin/kubectl")
    present = {CRD_CLUSTERQUEUE: True, CRD_JOBSET: False}
    monkeypatch.setattr(client, "resource_exists", lambda kind, name: present.get(name, False))
    status = detect(client)
    assert status.kueue_installed is True and status.jobset_installed is False
    assert status.ready is False


def test_kueue_fetch_manifest_rejects_non_https():
    from sparkrun.plugins.k8s.orchestration.kueue import KueueError, fetch_manifest

    with pytest.raises(KueueError, match="non-https"):
        fetch_manifest("http://example.com/manifests.yaml")


def test_api_setup_kueue_dry_run_renders_without_install(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s import api as apik8s
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info

    from sparkrun.plugins.k8s.orchestration import kueue as kmod

    sctx = _sctx(tmp_path)
    nodes = [build_node_info(_node("spark-0", _SPARK_LABELS, capacity_gpu=1))]
    monkeypatch.setattr(apik8s._ops, "make_client", lambda *a, **k: object())
    monkeypatch.setattr(kmod, "detect", lambda client: kmod.KueueStatus(False, False))
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", lambda client, **k: nodes)

    result = k8s_api.setup_kueue(sctx, dry_run=True)
    assert result.dry_run and not result.provisioned
    assert not result.installed_kueue and not result.installed_jobset
    assert {f.name for f in result.flavors} == {"sparkrun-gb10"}
    assert "kind: ClusterQueue" in result.manifests_yaml


def test_api_setup_kueue_missing_without_install_raises(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s import api as apik8s
    from sparkrun.plugins.k8s.orchestration import kueue as kmod

    sctx = _sctx(tmp_path)
    monkeypatch.setattr(apik8s._ops, "make_client", lambda *a, **k: object())
    monkeypatch.setattr(kmod, "detect", lambda client: kmod.KueueStatus(False, False))
    with pytest.raises(k8s_api.KueueSetupError, match="not installed"):
        k8s_api.setup_kueue(sctx, install=False, dry_run=False)


def test_cli_setup_k8s_kueue_dry_run(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli import main
    from sparkrun.plugins.k8s.orchestration import kueue as kmod
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info

    monkeypatch.setenv("STATEFUL_ROOT", str(tmp_path / "stateful"))
    nodes = [build_node_info(_node("spark-0", _SPARK_LABELS, capacity_gpu=1)), build_node_info(_node("rtx-0", _RTX_LABELS, capacity_gpu=1))]
    monkeypatch.setattr("sparkrun.plugins.k8s.api._ops.make_client", lambda *a, **k: object())
    monkeypatch.setattr(kmod, "detect", lambda client: kmod.KueueStatus(True, True))
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", lambda client, **k: nodes)

    result = CliRunner().invoke(main, ["setup", "k8s", "kueue", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "kind: ResourceFlavor" in result.output and "kind: ClusterQueue" in result.output
    assert "dry-run" in result.output


# ---------------------------------------------------------------------------
# Scheduling / feasibility precheck
# ---------------------------------------------------------------------------


def _nodes_for(spec):
    """spec: list of (name, labels, capacity_gpu, allocatable_gpu, unschedulable)."""
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info

    return [
        build_node_info(_node(n, labels, capacity_gpu=cap, allocatable_gpu=alloc, unschedulable=unsched))
        for (n, labels, cap, alloc, unsched) in spec
    ]


def test_aggregate_gpu_classes_sums_allocatable_from_schedulable_only():
    from sparkrun.plugins.k8s.orchestration.scheduling import aggregate_gpu_classes

    nodes = _nodes_for(
        [
            ("spark-0", _SPARK_LABELS, 1, 1, False),
            ("spark-1", _SPARK_LABELS, 1, 1, True),  # cordoned → excluded from allocatable
            ("rtx-0", _RTX_LABELS, 1, 1, False),
        ]
    )
    classes = aggregate_gpu_classes(nodes)
    assert classes["gb10"].allocatable_gpus == 1  # only the schedulable spark node
    assert classes["gb10"].capacity_gpus == 2  # both count toward capacity
    assert classes["gb10"].schedulable_node_count == 1
    assert classes["gb10"].product == "NVIDIA-GB10"
    assert classes["rtx-pro-6000-blackwell"].allocatable_gpus == 1


def test_check_feasibility_hybrid_ok():
    from sparkrun.plugins.k8s.orchestration.scheduling import GpuRequest, check_feasibility

    nodes = _nodes_for(
        [("spark-0", _SPARK_LABELS, 1, 1, False), ("spark-1", _SPARK_LABELS, 1, 1, False), ("rtx-0", _RTX_LABELS, 1, 1, False)]
    )
    report = check_feasibility(nodes, [GpuRequest("gb10", 2), GpuRequest("rtx-pro-6000-blackwell", 1)])
    assert report.feasible is True
    assert all(c.ok for c in report.classes)


def test_check_feasibility_shortfall():
    from sparkrun.plugins.k8s.orchestration.scheduling import GpuRequest, check_feasibility

    nodes = _nodes_for([("spark-0", _SPARK_LABELS, 1, 1, False)])
    report = check_feasibility(nodes, [GpuRequest("gb10", 2)])
    assert report.feasible is False
    cf = report.classes[0]
    assert cf.shortfall == 1 and cf.allocatable == 1 and cf.required == 2
    assert "short by 1" in report.summary()


def test_check_feasibility_unknown_model():
    from sparkrun.plugins.k8s.orchestration.scheduling import GpuRequest, check_feasibility

    nodes = _nodes_for([("spark-0", _SPARK_LABELS, 1, 1, False)])
    report = check_feasibility(nodes, [GpuRequest("h200", 8)])
    assert report.feasible is False
    assert report.unknown_models == ["h200"]
    assert "no such GPU class" in report.summary()


def test_check_feasibility_aggregates_multiple_requests_same_model():
    from sparkrun.plugins.k8s.orchestration.scheduling import GpuRequest, check_feasibility

    nodes = _nodes_for([("spark-0", _SPARK_LABELS, 1, 1, False), ("spark-1", _SPARK_LABELS, 1, 1, False)])
    # two rank-groups on gb10 summing to 2 → fits exactly
    report = check_feasibility(nodes, [GpuRequest("gb10", 1, "rank0"), GpuRequest("gb10", 1, "rank1")])
    assert report.feasible is True
    assert report.classes[0].required == 2


def test_api_check_feasibility_reads_inventory(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s import api as apik8s

    sctx = _sctx(tmp_path)
    nodes = _nodes_for([("spark-0", _SPARK_LABELS, 1, 1, False)])
    monkeypatch.setattr(apik8s._ops, "make_client", lambda *a, **k: object())
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", lambda client, **k: nodes)
    report = k8s_api.check_feasibility(sctx, requests=[k8s_api.GpuRequest("gb10", 1)])
    assert report.feasible is True


# ---------------------------------------------------------------------------
# Placement -> JobSet translation
# ---------------------------------------------------------------------------


def test_node_selectors_from_inventory():
    from sparkrun.plugins.k8s.orchestration.inventory import build_node_info
    from sparkrun.plugins.k8s.orchestration.jobset import node_selectors_from_nodes

    nodes = [build_node_info(_node("s0", _SPARK_LABELS, capacity_gpu=1)), build_node_info(_node("r0", _RTX_LABELS, capacity_gpu=1))]
    selectors = node_selectors_from_nodes(nodes)
    assert selectors["gb10"] == {"nvidia.com/gpu.product": "NVIDIA-GB10"}
    assert selectors["rtx-pro-6000-blackwell"] == {"nvidia.com/gpu.product": "NVIDIA-RTX-PRO-6000-Blackwell"}


def test_plan_from_rank_models_groups_by_class():
    from sparkrun.plugins.k8s.orchestration.jobset import plan_from_rank_models

    plan = plan_from_rank_models(
        "job-1",
        ["gb10", "gb10", "rtx-pro-6000-blackwell"],
        image="img",
        node_selectors={
            "gb10": {"nvidia.com/gpu.product": "NVIDIA-GB10"},
            "rtx-pro-6000-blackwell": {"nvidia.com/gpu.product": "NVIDIA-RTX-PRO-6000-Blackwell"},
        },
    )
    by_name = {ps.name: ps for ps in plan.pod_sets}
    assert by_name["gb10"].replicas == 2
    assert by_name["rtx-pro-6000-blackwell"].replicas == 1
    assert plan.total_pods == 3
    assert {(r.model, r.gpus) for r in plan.gpu_requests} == {("gb10", 2), ("rtx-pro-6000-blackwell", 1)}


def test_build_jobset_hybrid_shape():
    from sparkrun.plugins.k8s.orchestration.jobset import QUEUE_LABEL, build_jobset, plan_from_rank_models

    plan = plan_from_rank_models(
        "job-1",
        ["gb10", "gb10", "rtx-pro-6000-blackwell"],
        image="ghcr.io/x/sparkrun:latest",
        node_selectors={
            "gb10": {"nvidia.com/gpu.product": "NVIDIA-GB10"},
            "rtx-pro-6000-blackwell": {"nvidia.com/gpu.product": "NVIDIA-RTX-PRO-6000-Blackwell"},
        },
    )
    js = build_jobset(plan)
    assert js["kind"] == "JobSet" and js["apiVersion"].startswith("jobset.x-k8s.io/")
    assert js["metadata"]["labels"][QUEUE_LABEL] == "sparkrun"
    rjs = {rj["name"]: rj for rj in js["spec"]["replicatedJobs"]}
    assert rjs["gb10"]["replicas"] == 2
    gb10_pod = rjs["gb10"]["template"]["spec"]["template"]["spec"]
    assert gb10_pod["nodeSelector"] == {"nvidia.com/gpu.product": "NVIDIA-GB10"}
    assert gb10_pod["serviceAccountName"] == "sparkrun"
    assert gb10_pod["restartPolicy"] == "Never"
    assert gb10_pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    # gang: JobSet job template does not retry a failed launch
    assert rjs["gb10"]["template"]["spec"]["backoffLimit"] == 0


def test_plan_multi_gpu_per_pod():
    from sparkrun.plugins.k8s.orchestration.jobset import build_jobset, plan_from_rank_models

    # single-node tp=4 on one RTX box → 1 pod requesting 4 GPUs
    plan = plan_from_rank_models("job-2", ["rtx-pro-6000-blackwell"], image="img", gpus_per_pod={"rtx-pro-6000-blackwell": 4})
    js = build_jobset(plan)
    rj = js["spec"]["replicatedJobs"][0]
    assert rj["replicas"] == 1
    pod = rj["template"]["spec"]["template"]["spec"]
    assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "4"


def test_plan_command_and_env_threaded_into_pod():
    from sparkrun.plugins.k8s.orchestration.jobset import PodSetPlan, JobSetPlan, build_jobset

    plan = JobSetPlan(
        name="job-3",
        pod_sets=[
            PodSetPlan(
                name="gb10", replicas=1, gpus_per_pod=1, image="img", model="gb10", command=["sparkrun", "run"], env={"HF_TOKEN": "x"}
            )
        ],
    )
    pod = build_jobset(plan)["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["command"] == ["sparkrun", "run"]
    assert {"name": "HF_TOKEN", "value": "x"} in container["env"]


def test_jobset_gpu_requests_feed_feasibility():
    from sparkrun.plugins.k8s.orchestration.jobset import plan_from_rank_models
    from sparkrun.plugins.k8s.orchestration.scheduling import check_feasibility

    # the plan's own gpu_requests drive the precheck against inventory
    plan = plan_from_rank_models("job-4", ["gb10", "gb10"], image="img")
    nodes = _nodes_for([("s0", _SPARK_LABELS, 1, 1, False)])  # only 1 gb10 GPU available
    report = check_feasibility(nodes, plan.gpu_requests)
    assert report.feasible is False
    assert report.classes[0].shortfall == 1


# ---------------------------------------------------------------------------
# NCCL + full launch (step 5)
# ---------------------------------------------------------------------------


def test_nccl_headless_dns_and_master():
    from sparkrun.plugins.k8s.orchestration import nccl

    assert nccl.headless_pod_dns("job-x", "gb10", job_index=0, pod_index=0) == "job-x-gb10-0-0.job-x"
    assert nccl.master_addr("job-x", "gb10") == "job-x-gb10-0-0.job-x"


def test_nccl_base_env_forces_tcp():
    from sparkrun.plugins.k8s.orchestration import nccl

    env = nccl.base_tcp_nccl_env(4, "job-x-gb10-0-0.job-x", master_port=29500)
    assert env["WORLD_SIZE"] == "4"
    assert env["MASTER_ADDR"] == "job-x-gb10-0-0.job-x"
    assert env["NCCL_IB_DISABLE"] == "1"  # TCP tier
    assert env["NCCL_SOCKET_IFNAME"] == "eth0"


def test_nccl_rank_prelude_offsets_base():
    from sparkrun.plugins.k8s.orchestration import nccl

    prelude = nccl.rank_prelude(2)
    assert "RANK=$(( 2 + SPARKRUN_JOB_INDEX ))" in prelude
    assert 'NODE_RANK="$RANK"' in prelude


def test_group_contiguous_ranks_rejects_interleaving():
    from sparkrun.plugins.k8s.orchestration.launch import group_contiguous_ranks

    groups = group_contiguous_ranks(["gb10", "gb10", "rtx-pro-6000-blackwell"])
    assert [(g.model, g.base_rank, g.count) for g in groups] == [("gb10", 0, 2), ("rtx-pro-6000-blackwell", 2, 1)]
    with pytest.raises(ValueError, match="not contiguous"):
        group_contiguous_ranks(["gb10", "rtx-pro-6000-blackwell", "gb10"])


def test_build_launch_jobset_hybrid_wires_ranks_and_master():
    from sparkrun.plugins.k8s.orchestration.jobset import build_jobset
    from sparkrun.plugins.k8s.orchestration.launch import build_launch_jobset

    selectors = {
        "gb10": {"nvidia.com/gpu.product": "NVIDIA-GB10"},
        "rtx-pro-6000-blackwell": {"nvidia.com/gpu.product": "NVIDIA-RTX-PRO-6000-Blackwell"},
    }
    plan = build_launch_jobset(
        "job-x", ["gb10", "gb10", "rtx-pro-6000-blackwell"], image="img", serve_command="vllm serve M", node_selectors=selectors
    )
    js = build_jobset(plan)
    rjs = {rj["name"]: rj for rj in js["spec"]["replicatedJobs"]}

    rtx_pod = rjs["rtx-pro-6000-blackwell"]["template"]["spec"]["template"]["spec"]
    env = {e["name"]: e for e in rtx_pod["containers"][0]["env"]}
    # rank 0 lives in the gb10 podset → everyone rendezvouses there
    assert env["MASTER_ADDR"]["value"] == "job-x-gb10-0-0.job-x"
    assert env["WORLD_SIZE"]["value"] == "3"
    # rtx block starts at global rank 2
    cmd = rtx_pod["containers"][0]["command"]
    assert cmd[0] == "bash" and "RANK=$(( 2 + SPARKRUN_JOB_INDEX ))" in cmd[2]
    assert "vllm serve M" in cmd[2]
    # job-index injected via downward API
    assert env["SPARKRUN_JOB_INDEX"]["valueFrom"]["fieldRef"]["fieldPath"].endswith("job-index']")


def test_build_launch_jobset_empty_rejected():
    from sparkrun.plugins.k8s.orchestration.launch import build_launch_jobset

    with pytest.raises(ValueError, match="non-empty"):
        build_launch_jobset("j", [], image="img", serve_command="x")


def test_api_launch_jobset_dry_run_reports_feasibility(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s import api as apik8s

    sctx = _sctx(tmp_path)
    nodes = _nodes_for([("s0", _SPARK_LABELS, 1, 1, False), ("s1", _SPARK_LABELS, 1, 1, False)])
    monkeypatch.setattr(apik8s._ops, "make_client", lambda *a, **k: object())
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", lambda client, **k: nodes)

    result = k8s_api.launch_jobset(sctx, name="job-x", rank_models=["gb10", "gb10"], image="img", serve_command="serve", dry_run=True)
    assert result.dry_run and not result.submitted
    assert result.feasible is True
    assert "kind: JobSet" in result.manifests_yaml


def test_api_launch_jobset_infeasible_raises(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s import api as apik8s

    sctx = _sctx(tmp_path)
    nodes = _nodes_for([("s0", _SPARK_LABELS, 1, 1, False)])  # only 1 gb10 GPU
    monkeypatch.setattr(apik8s._ops, "make_client", lambda *a, **k: object())
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", lambda client, **k: nodes)

    with pytest.raises(k8s_api.JobSetLaunchError, match="infeasible"):
        k8s_api.launch_jobset(sctx, name="job-x", rank_models=["gb10", "gb10"], image="img", serve_command="serve")


def test_api_launch_jobset_submits(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s import api as apik8s
    from sparkrun.plugins.k8s.orchestration import launch as launch_mod

    sctx = _sctx(tmp_path)
    nodes = _nodes_for([("s0", _SPARK_LABELS, 1, 1, False)])
    monkeypatch.setattr(apik8s._ops, "make_client", lambda *a, **k: object())
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", lambda client, **k: nodes)
    submitted = {}

    def _submit(client, plan, **k):
        submitted["name"] = plan.name
        return RemoteResult(host="k8s", returncode=0, stdout="created", stderr="")

    monkeypatch.setattr(launch_mod, "submit_jobset", _submit)
    result = k8s_api.launch_jobset(sctx, name="job-x", rank_models=["gb10"], image="img", serve_command="serve")
    assert result.submitted and submitted["name"] == "job-x"


def test_cli_setup_k8s_launch_dry_run(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli import main

    monkeypatch.setenv("STATEFUL_ROOT", str(tmp_path / "stateful"))
    nodes = _nodes_for([("s0", _SPARK_LABELS, 1, 1, False), ("s1", _SPARK_LABELS, 1, 1, False)])
    monkeypatch.setattr("sparkrun.plugins.k8s.api._ops.make_client", lambda *a, **k: object())
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", lambda client, **k: nodes)

    result = CliRunner().invoke(
        main,
        ["setup", "k8s", "launch", "--name", "job-x", "--image", "img", "--ranks", "gb10,gb10", "--serve", "vllm serve M", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "kind: JobSet" in result.output and "Feasibility" in result.output


# ---------------------------------------------------------------------------
# Step 6a: RDMA NCCL tier
# ---------------------------------------------------------------------------


def test_rdma_available_detection():
    from sparkrun.plugins.k8s.orchestration import nccl

    assert nccl.rdma_available({"feature.node.kubernetes.io/pci-15b3.present": "true"}, {"rdma/rdma_shared_device_a": "8"}) is True
    assert nccl.rdma_available({"feature.node.kubernetes.io/pci-15b3.present": "true"}, {}) is False  # no rdma resource
    assert nccl.rdma_available({}, {"rdma/rdma_shared_device_a": "8"}) is False  # no Mellanox NIC


def test_rdma_nccl_env_enables_ib():
    from sparkrun.plugins.k8s.orchestration import nccl

    env = nccl.base_rdma_nccl_env(2, "m", hca="mlx5_2")
    assert env["NCCL_IB_HCA"] == "mlx5_2"
    assert "NCCL_IB_DISABLE" not in env  # IB left enabled (unlike TCP tier)


def test_build_launch_jobset_rdma_requests_resource_and_caps():
    from sparkrun.plugins.k8s.orchestration.jobset import build_jobset
    from sparkrun.plugins.k8s.orchestration.launch import build_launch_jobset

    plan = build_launch_jobset("j", ["gb10", "gb10"], image="img", serve_command="serve", transport="rdma")
    pod = build_jobset(plan)["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["resources"]["limits"]["rdma/rdma_shared_device_a"] == "1"
    assert container["securityContext"]["capabilities"]["add"] == ["IPC_LOCK"]
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert env["NCCL_IB_HCA"] == "mlx5_0" and "NCCL_IB_DISABLE" not in env


def test_build_launch_jobset_tcp_has_no_rdma_bits():
    from sparkrun.plugins.k8s.orchestration.jobset import build_jobset
    from sparkrun.plugins.k8s.orchestration.launch import build_launch_jobset

    plan = build_launch_jobset("j", ["gb10"], image="img", serve_command="serve", transport="tcp")
    pod = build_jobset(plan)["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert "rdma/rdma_shared_device_a" not in container["resources"]["limits"]
    assert "securityContext" not in container


def test_build_launch_jobset_rejects_bad_transport():
    from sparkrun.plugins.k8s.orchestration.launch import build_launch_jobset

    with pytest.raises(ValueError, match="transport"):
        build_launch_jobset("j", ["gb10"], image="img", serve_command="s", transport="magic")


# ---------------------------------------------------------------------------
# Step 6b: privileged fallback probe
# ---------------------------------------------------------------------------


def test_probe_job_manifest_is_privileged_and_pinned():
    from sparkrun.plugins.k8s.orchestration.probe import probe_job_manifest

    manifest = probe_job_manifest("gpu-node-1", namespace="sparkrun", image="cuda:12")
    assert manifest["metadata"]["name"] == "sparkrun-probe-gpu-node-1"
    pod = manifest["spec"]["template"]["spec"]
    assert pod["nodeName"] == "gpu-node-1"
    assert pod["hostPID"] is True
    assert pod["containers"][0]["securityContext"]["privileged"] is True
    assert any(v["hostPath"]["path"] == "/dev" for v in pod["volumes"])


def test_probe_output_parsed_with_ssh_fingerprint_parser():
    from sparkrun.plugins.k8s.orchestration.probe import parse_probe_output

    stdout = (
        "SPARKRUN_PROBE_ACCEL_START\n"
        "NVIDIA_GPU_COUNT=1\n"
        "NVIDIA_GPU_0_NAME=NVIDIA GB10\n"
        "NVIDIA_GPU_0_MEMORY_MIB=131072\n"
        "SPARKRUN_PROBE_ACCEL_END\n"
        "SPARKRUN_PROBE_IB_START\n"
        "IB_PRESENT=0\n"
        "SPARKRUN_PROBE_IB_END\n"
    )
    hw = parse_probe_output(stdout)
    # same model token the label path + SSH path produce
    assert hw.accelerators[0].model == "gb10"
    assert hw.accelerators[0].memory_gb == 128.0


def test_probe_nodes_fallback_collects_per_node(monkeypatch):
    from sparkrun.plugins.k8s.orchestration.client import KubectlClient
    from sparkrun.plugins.k8s.orchestration import probe as probe_mod

    client = KubectlClient("/usr/bin/kubectl")
    calls = []

    def _apply(manifest, **k):
        calls.append("apply")
        return RemoteResult(host="k8s", returncode=0, stdout="created", stderr="")

    def _run(args, **k):
        if args[:1] == ["logs"]:
            return RemoteResult(
                host="k8s",
                returncode=0,
                stdout="SPARKRUN_PROBE_ACCEL_START\nNVIDIA_GPU_COUNT=1\nNVIDIA_GPU_0_NAME=NVIDIA GB10\nNVIDIA_GPU_0_MEMORY_MIB=131072\nSPARKRUN_PROBE_ACCEL_END\n",
                stderr="",
            )
        return RemoteResult(host="k8s", returncode=0, stdout="", stderr="")

    monkeypatch.setattr(client, "apply", _apply)
    monkeypatch.setattr(client, "run", _run)
    monkeypatch.setattr(client, "wait_for_job", lambda *a, **k: RemoteResult(host="k8s", returncode=0, stdout="", stderr=""))
    result = probe_mod.probe_nodes_fallback(client, ["node-a"], image="cuda:12")
    assert set(result) == {"node-a"}
    assert result["node-a"].accelerators[0].model == "gb10"


def test_api_probe_nodes_fallback(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s import api as apik8s
    from sparkrun.core.hardware import HostHardware

    sctx = _sctx(tmp_path)
    monkeypatch.setattr(apik8s._ops, "make_client", lambda *a, **k: object())
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.probe.probe_nodes_fallback", lambda client, nodes, **k: {"n": HostHardware()})
    out = k8s_api.probe_nodes_fallback(sctx, node_names=["n"], image="cuda:12")
    assert set(out) == {"n"}


# ---------------------------------------------------------------------------
# api.run k8s integration (gated by api.run.k8s)
# ---------------------------------------------------------------------------


class _FakeRuntime:
    runtime_name = "vllm"

    def resolve_container(self, recipe, *, host_hardware=None):
        return "ghcr.io/x/vllm:latest"

    def generate_command(self, recipe, overrides, is_cluster, num_nodes=1, head_ip=None, skip_keys=frozenset()):
        return "vllm serve M --port 8000"


class _FakeRecipe:
    container = ""
    containers = []
    executor = "k8s"
    env = {"HF_TOKEN": "x"}
    model = "M"
    name = "r"
    qualified_name = "r"

    def build_config_chain(self, overrides):
        return {"port": 8000, **overrides}


def _run_k8s_common(tmp_path, monkeypatch, nodes):
    from sparkrun.plugins.k8s import api as apik8s
    from sparkrun.plugins.k8s.orchestration.launch import LaunchJobsetResult

    sctx = _sctx(tmp_path)
    monkeypatch.setattr(apik8s._ops, "make_client", lambda *a, **k: object())
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", lambda client, **k: nodes)

    captured = {}

    def _fake_launch(_sctx, **kw):
        captured.update(kw)
        return LaunchJobsetResult(
            name=kw["name"], namespace="sparkrun", manifests_yaml="kind: JobSet", feasible=True, dry_run=kw.get("dry_run", False)
        )

    monkeypatch.setattr(apik8s._ops, "_launch_jobset", _fake_launch)
    return sctx, captured


def test_run_k8s_solo_homogeneous(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s.run import run_k8s

    nodes = _nodes_for([("s0", _SPARK_LABELS, 1, 1, False)])
    sctx, captured = _run_k8s_common(tmp_path, monkeypatch, nodes)

    result = run_k8s(
        api.RunOptions(recipe=_FakeRecipe(), overrides={"port": 8000}, dry_run=True),
        sctx,
        plan=api.RunPlan(
            recipe=_FakeRecipe(),
            runtime=_FakeRuntime(),
            cluster=None,
            candidate_hosts=tuple(["s0"]),
            host_list=tuple(["s0"]),
            placement=None,
            is_solo=True,
            workload_id="intent-token",
            intent_id="intent",
            placement_token="token",
            scheduler="greedy",
        ),
        started_at=0.0,
    )
    assert result.executor == "k8s" and result.is_solo
    assert result.serve_command == "vllm serve M --port 8000"
    assert result.container_image == "ghcr.io/x/vllm:latest"
    assert result.metadata["k8s_gpu_class"] == "gb10"
    assert result.serve_port == 8000  # from the port override
    # launch received a single-rank gb10 plan
    assert captured["rank_models"] == ["gb10"]
    assert captured["name"] == result.metadata["k8s_jobset"] != "intent-token"
    assert captured["annotations"]["sparkrun.cluster_id"] == "intent-token"
    assert "follow" not in captured


def test_run_k8s_rejects_multinode(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s.run import run_k8s

    nodes = _nodes_for([("s0", _SPARK_LABELS, 1, 1, False)])
    sctx, _ = _run_k8s_common(tmp_path, monkeypatch, nodes)

    class _P:
        total_ranks = 2

    with pytest.raises(api.SparkrunError, match="solo"):
        run_k8s(
            api.RunOptions(recipe=_FakeRecipe()),
            sctx,
            plan=api.RunPlan(
                recipe=_FakeRecipe(),
                runtime=_FakeRuntime(),
                cluster=None,
                candidate_hosts=tuple(["s0", "s1"]),
                host_list=tuple(["s0", "s1"]),
                placement=_P(),
                is_solo=False,
                workload_id="c",
                intent_id="i",
                placement_token="t",
                scheduler="greedy",
            ),
            started_at=0.0,
        )


def test_run_k8s_rejects_hybrid_cluster(tmp_path, monkeypatch):
    from sparkrun.plugins.k8s.run import run_k8s

    nodes = _nodes_for([("s0", _SPARK_LABELS, 1, 1, False), ("r0", _RTX_LABELS, 1, 1, False)])
    sctx, _ = _run_k8s_common(tmp_path, monkeypatch, nodes)

    with pytest.raises(api.SparkrunError, match="one GPU class"):
        run_k8s(
            api.RunOptions(recipe=_FakeRecipe(), dry_run=True),
            sctx,
            plan=api.RunPlan(
                recipe=_FakeRecipe(),
                runtime=_FakeRuntime(),
                cluster=None,
                candidate_hosts=tuple(["s0"]),
                host_list=tuple(["s0"]),
                placement=None,
                is_solo=True,
                workload_id="c",
                intent_id="i",
                placement_token="t",
                scheduler="greedy",
            ),
            started_at=0.0,
        )


def test_api_run_k8s_flag_registered_by_enabled_plugin():
    from sparkrun.core.features import get_feature

    from sparkrun.core.bootstrap import init_sparkrun

    init_sparkrun()
    flag = get_feature("api.run.k8s")
    assert flag is not None and flag.default is True


def test_api_run_branches_to_k8s_when_flag_on(tmp_path, monkeypatch):
    """api.run routes to run_k8s only when executor=k8s AND api.run.k8s is on."""
    from sparkrun import api
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.cluster_manager import ClusterDefinition

    monkeypatch.setenv("SPARKRUN_FEATURE_API_RUN_K8S", "1")
    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "M"})

    # Stub the heavy resolution pipeline so we reach the branch deterministically.
    monkeypatch.setattr("sparkrun.api._resolve.resolve_cluster", lambda *a, **k: ClusterDefinition(name="", hosts=["s0"]))
    monkeypatch.setattr("sparkrun.api._resolve.resolve_recipe", lambda *a, **k: recipe)
    monkeypatch.setattr("sparkrun.api._resolve.resolve_runtime", lambda *a, **k: _FakeRuntime())
    monkeypatch.setattr("sparkrun.api._hosts.resolve_effective_hosts", lambda *a, **k: (["s0"], True, [], None))
    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor_name", lambda **k: "k8s")

    sentinel = api.RunResult(
        workload_id="native-result",
        host_list=("s0",),
        placement=None,
        scheduler="greedy",
        runtime="fake",
        executor="k8s",
        started_at=0,
        dry_run=True,
        is_solo=True,
        rc=42,
    )
    called = {}

    def _fake_run_k8s(options, sctx, **kw):
        called["hit"] = True
        return sentinel

    monkeypatch.setattr("sparkrun.plugins.k8s.run.run_k8s", _fake_run_k8s)

    sctx = _sctx(tmp_path)
    out = api.run(api.RunOptions(recipe=_FakeRecipe(), hosts=("s0",), solo=True, dry_run=True, executor="k8s"), sctx=sctx)
    assert out.workload_id == sentinel.workload_id and out.rc == 42 and called.get("hit")
    assert out.timeline is not None and out.recipe_fingerprint


# ---------------------------------------------------------------------------
# Feature-flag gate on the whole `setup k8s` group
# ---------------------------------------------------------------------------


def test_cli_setup_k8s_feature_flag_registered():
    from sparkrun.core.features import get_feature

    from sparkrun.core.bootstrap import init_sparkrun

    init_sparkrun()
    flag = get_feature("cli.setup.k8s")
    assert flag is not None and flag.default is True


def test_cli_setup_k8s_gated_off_when_flag_disabled(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli import main

    monkeypatch.setenv("STATEFUL_ROOT", str(tmp_path / "stateful"))
    # env override is checked first → deterministically disabled regardless of config
    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_K8S", "0")
    result = CliRunner().invoke(main, ["setup", "k8s", "kubectl", "--list"])
    assert result.exit_code != 0
    assert "cli.setup.k8s" in result.output


def test_cli_setup_k8s_allowed_when_flag_enabled(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli import main

    monkeypatch.setenv("STATEFUL_ROOT", str(tmp_path / "stateful"))
    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_K8S", "1")
    result = CliRunner().invoke(main, ["setup", "k8s", "kubectl", "--list"])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_setup_k8s_sa_dry_run(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli import main

    # Route the CLI's context at a tmp config so nothing touches real home.
    monkeypatch.setenv("STATEFUL_ROOT", str(tmp_path / "stateful"))
    runner = CliRunner()
    result = runner.invoke(main, ["setup", "k8s", "sa", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "ClusterRole" in result.output
    assert "dry-run" in result.output


def test_cli_setup_k8s_kubectl_list_empty(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli import main

    monkeypatch.setenv("STATEFUL_ROOT", str(tmp_path / "stateful"))
    runner = CliRunner()
    result = runner.invoke(main, ["setup", "k8s", "kubectl", "--list"])
    assert result.exit_code == 0, result.output


def test_cli_setup_k8s_run_job_dry_run(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from sparkrun.cli import main

    monkeypatch.setenv("STATEFUL_ROOT", str(tmp_path / "stateful"))
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup", "k8s", "run-job", "--name", "cl-1", "--image", "img", "--command", "sparkrun run qwen", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "kind: Job" in result.output
    assert "dry-run" in result.output


@pytest.mark.parametrize("case", ["submit", "preview", "infeasible", "invalid", "replacement_failure"])
def test_jobset_replacement_follows_preflight(tmp_path, monkeypatch, case):
    from unittest.mock import Mock
    from sparkrun.plugins.k8s import api as apik8s
    from sparkrun.plugins.k8s.orchestration import launch as launch_mod

    sctx = _sctx(tmp_path)
    nodes = _nodes_for([("s0", _SPARK_LABELS, 1, 1, False)])
    monkeypatch.setattr(apik8s._ops, "make_client", lambda *a, **k: object())
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.inventory.probe_nodes", lambda client, **k: nodes)
    calls = []

    def replace():
        calls.append("replace")
        if case == "replacement_failure":
            raise RuntimeError("replacement failed")

    replacement = Mock(side_effect=replace)
    submit = Mock(side_effect=lambda *a: calls.append("submit") or RemoteResult(host="k8s", returncode=0, stdout="created", stderr=""))
    monkeypatch.setattr(launch_mod, "submit_jobset", submit)
    kwargs = dict(
        name="job-x",
        rank_models=["gb10"] * (2 if case == "infeasible" else 1),
        image="img",
        serve_command="serve",
        transport="invalid" if case == "invalid" else "tcp",
        dry_run=case == "preview",
        before_start=replacement,
    )
    if case in ("infeasible", "invalid", "replacement_failure"):
        with pytest.raises(RuntimeError if case == "replacement_failure" else apik8s.JobSetLaunchError):
            apik8s.launch_jobset(sctx, **kwargs)
    else:
        result = apik8s.launch_jobset(sctx, **kwargs)
        assert result.submitted is (case == "submit")
    assert calls == ({"submit": ["replace", "submit"], "replacement_failure": ["replace"]}.get(case, []))
    assert submit.call_count == (1 if case == "submit" else 0)
