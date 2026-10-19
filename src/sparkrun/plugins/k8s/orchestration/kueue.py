"""Kueue + JobSet: gang scheduling for k8s-native sparkrun launches.

Every k8s-mode launch goes through Kueue: the workload is a JobSet
labeled with a LocalQueue, and Kueue admits it all-or-nothing (gang) —
which both fixes the multi-pod TP partial-placement deadlock and does the
hybrid node-class targeting (each ResourceFlavor's ``nodeLabels`` select a
GPU class).

This module handles the *setup* side (an admin step, parallel to
``setup k8s sa``):

- **detect** — are the Kueue / JobSet CRDs present?
- **install** — ``kubectl apply --server-side`` the pinned release
  manifests (https-only fetch), then wait for the controllers.
- **provision** — derive one ResourceFlavor per detected GPU node-class
  (from the node inventory), a ClusterQueue with per-flavor GPU quota, and
  a LocalQueue in the sparkrun namespace.

Install + provision run under the operator's admin context (creating CRDs
/ cluster-scoped objects needs more than the scoped sparkrun SA); runtime
workloads then reference the LocalQueue via the SA.
"""

from __future__ import annotations

from sparkrun.core.application_profile import resource_name

import logging
import urllib.request
from dataclasses import dataclass

from .client import KubectlClient
from .errors import K8sError
from .inventory import GPU_PRODUCT_LABEL, NodeInfo
from .manifests import managed_by_labels, render_manifests
from sparkrun.utils.text import excerpt

logger = logging.getLogger(__name__)

KUEUE_API_VERSION = "kueue.x-k8s.io/v1beta1"

CRD_CLUSTERQUEUE = "clusterqueues.kueue.x-k8s.io"
CRD_JOBSET = "jobsets.jobset.x-k8s.io"

DEFAULT_KUEUE_VERSION = "v0.10.0"
DEFAULT_JOBSET_VERSION = "v0.7.0"

KUEUE_MANIFEST_URL = "https://github.com/kubernetes-sigs/kueue/releases/download/%s/manifests.yaml"
JOBSET_MANIFEST_URL = "https://github.com/kubernetes-sigs/jobset/releases/download/%s/manifests.yaml"

KUEUE_NAMESPACE, KUEUE_DEPLOYMENT = "kueue-system", "kueue-controller-manager"
JOBSET_NAMESPACE, JOBSET_DEPLOYMENT = "jobset-system", "jobset-controller-manager"

DEFAULT_QUEUE_NAME = "sparkrun"
DEFAULT_CLUSTER_QUEUE_NAME = "sparkrun"

_FETCH_TIMEOUT = 60
_WAIT_TIMEOUT = 300


class KueueError(K8sError):
    """Kueue detection / install / provisioning failed."""


@dataclass
class FlavorSpec:
    """One ResourceFlavor = one GPU node-class."""

    name: str
    node_labels: dict[str, str]
    gpu_quota: int
    model: str


@dataclass
class KueueStatus:
    kueue_installed: bool
    jobset_installed: bool

    @property
    def ready(self) -> bool:
        return self.kueue_installed and self.jobset_installed


@dataclass
class KueueSetupResult:
    namespace: str
    cluster_queue: str
    local_queue: str
    flavors: list[FlavorSpec]
    manifests_yaml: str
    dry_run: bool
    kueue_version: str
    jobset_version: str
    installed_kueue: bool = False
    installed_jobset: bool = False
    provisioned: bool = False


# ---------------------------------------------------------------------------
# Provisioning manifests (pure, from the node inventory)
# ---------------------------------------------------------------------------


def _labels() -> dict[str, str]:
    return managed_by_labels()


def derive_flavors(node_infos: list[NodeInfo], *, name_prefix: str | None = None) -> list[FlavorSpec]:
    """Group GPU nodes by ``nvidia.com/gpu.product`` into ResourceFlavors.

    Quota is the summed GPU capacity of the class.  Nodes without a product
    label or accelerators are skipped (they can't be selected by nodeLabels).
    """
    name_prefix = name_prefix if name_prefix is not None else resource_name()
    groups: dict[str, dict] = {}
    for node in node_infos:
        product = node.labels.get(GPU_PRODUCT_LABEL)
        if not product or not node.has_accelerators:
            continue
        model = node.hardware.accelerators[0].model
        group = groups.setdefault(product, {"model": model, "gpu": 0})
        group["gpu"] += node.capacity_gpus or node.hardware.total_gpus

    flavors: list[FlavorSpec] = []
    for product, group in sorted(groups.items()):
        flavors.append(
            FlavorSpec(
                name="%s-%s" % (name_prefix, group["model"]),
                node_labels={GPU_PRODUCT_LABEL: product},
                gpu_quota=int(group["gpu"]),
                model=group["model"],
            )
        )
    return flavors


def resource_flavor_manifest(flavor: FlavorSpec) -> dict:
    return {
        "apiVersion": KUEUE_API_VERSION,
        "kind": "ResourceFlavor",
        "metadata": {"name": flavor.name, "labels": _labels()},
        "spec": {"nodeLabels": flavor.node_labels},
    }


def cluster_queue_manifest(name: str, flavors: list[FlavorSpec]) -> dict:
    return {
        "apiVersion": KUEUE_API_VERSION,
        "kind": "ClusterQueue",
        "metadata": {"name": name, "labels": _labels()},
        "spec": {
            "namespaceSelector": {},
            "resourceGroups": [
                {
                    "coveredResources": ["nvidia.com/gpu"],
                    "flavors": [
                        {
                            "name": flavor.name,
                            "resources": [{"name": "nvidia.com/gpu", "nominalQuota": str(flavor.gpu_quota)}],
                        }
                        for flavor in flavors
                    ],
                }
            ],
        },
    }


def local_queue_manifest(name: str, namespace: str, cluster_queue: str) -> dict:
    return {
        "apiVersion": KUEUE_API_VERSION,
        "kind": "LocalQueue",
        "metadata": {"name": name, "namespace": namespace, "labels": _labels()},
        "spec": {"clusterQueue": cluster_queue},
    }


def build_provision_manifests(
    node_infos: list[NodeInfo],
    *,
    namespace: str | None = None,
    queue_name: str | None = None,
    cluster_queue_name: str | None = None,
) -> tuple[list[dict], list[FlavorSpec]]:
    """Build (manifest docs, flavors) for the sparkrun Kueue objects."""
    namespace = resource_name("") if namespace is None else namespace
    queue_name = resource_name("") if queue_name is None else queue_name
    cluster_queue_name = resource_name("") if cluster_queue_name is None else cluster_queue_name
    flavors = derive_flavors(node_infos)
    if not flavors:
        raise KueueError(
            "No GPU nodes with a nvidia.com/gpu.product label were found — is the NVIDIA GPU Operator (GPU Feature Discovery) installed?"
        )
    docs: list[dict] = [resource_flavor_manifest(f) for f in flavors]
    docs.append(cluster_queue_manifest(cluster_queue_name, flavors))
    docs.append(local_queue_manifest(queue_name, namespace, cluster_queue_name))
    return docs, flavors


# ---------------------------------------------------------------------------
# Detection + install (live)
# ---------------------------------------------------------------------------


def detect(client: KubectlClient) -> KueueStatus:
    """Report whether the Kueue and JobSet CRDs are present."""
    return KueueStatus(
        kueue_installed=client.resource_exists("crd", CRD_CLUSTERQUEUE),
        jobset_installed=client.resource_exists("crd", CRD_JOBSET),
    )


def fetch_manifest(url: str, *, timeout: int = _FETCH_TIMEOUT) -> str:
    """Fetch a release manifest over https."""
    if not url.startswith("https://"):
        raise KueueError("Refusing to fetch manifest from non-https URL: %s" % url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — https-guarded
            return resp.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise KueueError("Failed to fetch manifest from %s: %s" % (url, exc)) from exc


def install_component(
    client: KubectlClient,
    *,
    url: str,
    namespace: str,
    deployment: str,
    timeout: int = _WAIT_TIMEOUT,
) -> None:
    """Server-side apply a release manifest and wait for its controller."""
    manifest = fetch_manifest(url)
    apply_res = client.apply(manifest, server_side=True, timeout=timeout)
    if not apply_res.success:
        raise KueueError("Failed to apply manifest from %s: %s" % (url, excerpt(apply_res.stderr, 400)))
    wait_res = client.run(
        ["-n", namespace, "wait", "--for=condition=Available", "deploy/%s" % deployment, "--timeout=%ds" % timeout],
        timeout=timeout + 30,
    )
    if not wait_res.success:
        raise KueueError("Timed out waiting for %s/%s to become Available" % (namespace, deployment))


__all__ = [
    "KUEUE_API_VERSION",
    "DEFAULT_KUEUE_VERSION",
    "DEFAULT_JOBSET_VERSION",
    "KUEUE_MANIFEST_URL",
    "JOBSET_MANIFEST_URL",
    "KUEUE_NAMESPACE",
    "KUEUE_DEPLOYMENT",
    "JOBSET_NAMESPACE",
    "JOBSET_DEPLOYMENT",
    "DEFAULT_QUEUE_NAME",
    "DEFAULT_CLUSTER_QUEUE_NAME",
    "KueueError",
    "FlavorSpec",
    "KueueStatus",
    "KueueSetupResult",
    "derive_flavors",
    "resource_flavor_manifest",
    "cluster_queue_manifest",
    "local_queue_manifest",
    "build_provision_manifests",
    "render_manifests",
    "detect",
    "fetch_manifest",
    "install_component",
]
