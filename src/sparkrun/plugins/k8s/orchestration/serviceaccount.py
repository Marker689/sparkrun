"""Configure the sparkrun service account, RBAC, and a derived kubeconfig.

The high-level flow (:func:`configure_service_account`):

1. Render manifests (optional Namespace, ServiceAccount, ClusterRole,
   ClusterRoleBinding) and ``kubectl apply`` them.
2. Mint a short-or-long-lived token via ``kubectl create token``.
3. Read the current cluster endpoint + CA from the *admin* kubeconfig
   (``kubectl config view --raw --minify -o json``).
4. Assemble a self-contained kubeconfig bound to the service-account
   token and write it ``0600`` under ``~/.config/sparkrun/k8s/``.

The derived kubeconfig is what sparkrun subsequently points its executor
at (``executor_config.kubeconfig``) so day-to-day runs use the scoped SA
rather than the operator's admin credentials.
"""

from __future__ import annotations

from sparkrun.core.application_profile import resource_name

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import manifests
from .client import KubectlClient
from .errors import ServiceAccountSetupError
from sparkrun.utils.text import excerpt

logger = logging.getLogger(__name__)

DEFAULT_SA_NAME = "sparkrun"
DEFAULT_CLUSTER_ROLE = "sparkrun-runner"
DEFAULT_BINDING = "sparkrun-runner"


@dataclass
class ServiceAccountSpec:
    """Inputs describing the service account to configure."""

    name: str = field(default_factory=lambda: resource_name(""))
    namespace: str = field(default_factory=lambda: resource_name(""))
    cluster_role_name: str = field(default_factory=lambda: resource_name("-runner"))
    binding_name: str = field(default_factory=lambda: resource_name("-runner"))
    create_namespace: bool = True
    rules: list[dict] = field(default_factory=manifests.default_runner_rules)
    token_duration: str | None = None  # e.g. "8760h"; None → cluster default


@dataclass
class ServiceAccountResult:
    """Outcome of :func:`configure_service_account`."""

    name: str
    namespace: str
    cluster_role: str
    binding: str
    manifests_yaml: str
    dry_run: bool
    applied: bool = False
    server: str | None = None
    kubeconfig_path: str | None = None
    token: str | None = None  # present only when minted

    def redacted(self) -> "ServiceAccountResult":
        """A copy safe to log — the token is masked."""
        from dataclasses import replace

        return replace(self, token="***" if self.token else None)


def build_manifests(spec: ServiceAccountSpec) -> str:
    """Render the full manifest set for *spec* as multi-doc YAML."""
    docs: list[dict] = []
    if spec.create_namespace:
        docs.append(manifests.namespace_manifest(spec.namespace))
    docs.append(manifests.service_account_manifest(spec.name, spec.namespace))
    docs.append(manifests.cluster_role_manifest(spec.cluster_role_name, spec.rules))
    docs.append(manifests.cluster_role_binding_manifest(spec.binding_name, spec.cluster_role_name, spec.name, spec.namespace))
    return manifests.render_manifests(docs)


def mint_token(client: KubectlClient, spec: ServiceAccountSpec) -> str:
    """Mint a service-account token via ``kubectl create token``."""
    args = ["create", "token", spec.name, "-n", spec.namespace]
    if spec.token_duration:
        args += ["--duration", spec.token_duration]
    result = client.run(args)
    if not result.success:
        raise ServiceAccountSetupError("Failed to mint token for %s/%s: %s" % (spec.namespace, spec.name, excerpt(result.stderr, 400)))
    token = result.stdout.strip()
    if not token:
        raise ServiceAccountSetupError("kubectl create token returned an empty token")
    return token


def read_cluster_endpoint(client: KubectlClient) -> tuple[str, dict]:
    """Return ``(server_url, cluster_dict)`` for the client's current context.

    Reads the minified admin kubeconfig so the derived kubeconfig can
    reuse the exact server URL and CA data (or ``insecure-skip-tls-verify``).
    """
    payload = client.run_json(["config", "view", "--raw", "--minify", "-o", "json"])
    clusters = payload.get("clusters") or []
    if not clusters:
        raise ServiceAccountSetupError("No cluster found in kubeconfig for the current context")
    cluster = clusters[0].get("cluster") or {}
    server = cluster.get("server")
    if not server:
        raise ServiceAccountSetupError("Current-context cluster has no server URL")
    return server, cluster


def build_kubeconfig(
    *,
    server: str,
    cluster: dict,
    token: str,
    context_name: str,
    namespace: str,
    cluster_name: str | None = None,
    user_name: str | None = None,
) -> dict:
    """Assemble a standalone kubeconfig dict bound to *token*."""
    cluster_name = cluster_name if cluster_name is not None else resource_name("-cluster")
    user_name = user_name if user_name is not None else resource_name()
    cluster_entry: dict = {"server": server}
    if cluster.get("certificate-authority-data"):
        cluster_entry["certificate-authority-data"] = cluster["certificate-authority-data"]
    elif cluster.get("insecure-skip-tls-verify"):
        cluster_entry["insecure-skip-tls-verify"] = True
    return {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [{"name": cluster_name, "cluster": cluster_entry}],
        "users": [{"name": user_name, "user": {"token": token}}],
        "contexts": [
            {
                "name": context_name,
                "context": {"cluster": cluster_name, "user": user_name, "namespace": namespace},
            }
        ],
        "current-context": context_name,
    }


def write_kubeconfig(path: str | Path, kubeconfig: dict) -> Path:
    """Write *kubeconfig* to *path* with ``0600`` permissions."""
    dest = Path(path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Create with restrictive perms before writing token material.
    fd = os.open(str(dest), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as fh:
            yaml.safe_dump(kubeconfig, fh, default_flow_style=False, sort_keys=False)
    finally:
        # Re-assert perms in case an existing file had looser mode.
        os.chmod(dest, 0o600)
    return dest


def configure_service_account(
    client: KubectlClient | None,
    spec: ServiceAccountSpec,
    *,
    kubeconfig_out: str | Path | None = None,
    context_name: str | None = None,
    dry_run: bool = False,
) -> ServiceAccountResult:
    """Apply the SA + RBAC and (unless *dry_run*) mint a token + kubeconfig."""
    context_name = context_name if context_name is not None else resource_name()
    manifests_yaml = build_manifests(spec)
    result = ServiceAccountResult(
        name=spec.name,
        namespace=spec.namespace,
        cluster_role=spec.cluster_role_name,
        binding=spec.binding_name,
        manifests_yaml=manifests_yaml,
        dry_run=dry_run,
    )
    if dry_run:
        return result

    if client is None:
        raise ServiceAccountSetupError("A Kubernetes client is required to apply service-account manifests")
    apply_res = client.apply(manifests_yaml)
    if not apply_res.success:
        raise ServiceAccountSetupError("Failed to apply service-account manifests: %s" % excerpt(apply_res.stderr, 400))
    result.applied = True

    token = mint_token(client, spec)
    result.token = token

    server, cluster = read_cluster_endpoint(client)
    result.server = server

    if kubeconfig_out is not None:
        kubeconfig = build_kubeconfig(
            server=server,
            cluster=cluster,
            token=token,
            context_name=context_name,
            namespace=spec.namespace,
        )
        written = write_kubeconfig(kubeconfig_out, kubeconfig)
        result.kubeconfig_path = str(written)

    return result


__all__ = [
    "DEFAULT_SA_NAME",
    "DEFAULT_CLUSTER_ROLE",
    "DEFAULT_BINDING",
    "ServiceAccountSpec",
    "ServiceAccountResult",
    "build_manifests",
    "mint_token",
    "read_cluster_endpoint",
    "build_kubeconfig",
    "write_kubeconfig",
    "configure_service_account",
]
