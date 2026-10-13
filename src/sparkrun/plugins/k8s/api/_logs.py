"""Explicit, console-free log consumption for native Kubernetes resources."""

from __future__ import annotations

from collections.abc import Generator
import threading
from typing import TYPE_CHECKING, Literal

from sparkrun.api._context import resolve_sctx
from sparkrun.api._logs import _filter_sources, _nonnegative_int
from sparkrun.core.log_source import LogLine, SCOPE_ALL, SCOPE_HEAD
from sparkrun.orchestration.logs import read_log_sources
from ._errors import _operation_errors
from ..orchestration.names import validate_resource_name

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext


def logs(
    sctx: SparkrunContext | None = None,
    *,
    name: str,
    kind: Literal["jobset", "job"] = "jobset",
    namespace: str | None = None,
    kubeconfig: str | None = None,
    context: str | None = None,
    follow: bool = False,
    tail: int | None = None,
    scope: str = SCOPE_ALL,
    host: str | None = None,
    container: str | None = None,
    source_id: str | None = None,
    cancel: threading.Event | None = None,
    check: bool = True,
) -> Generator[LogLine, None, None]:
    """Read an owned JobSet or launcher Job through the shared logging backend.

    Resolve individual pod/container sources eagerly; read them lazily. Source
    discovery is a snapshot: reopen after pods appear, restart, or are replaced.
    Workload stdout/stderr are log records; kubectl diagnostics are separate.
    Cancellation defaults to the operation event and stops only local readers.
    For saved workloads, prefer ``sparkrun.api.open_log_target(workload_id)``.
    """
    from ._ops import make_client
    from ..executor import K8sExecutor, K8sExecutorConfig
    from ..orchestration.logs import resolve_sources
    from ..orchestration.context import _pin_client_target
    from sparkrun.core.application_profile import resource_name

    if kind not in ("jobset", "job"):
        raise ValueError("Kubernetes log kind must be 'jobset' or 'job'")
    validate_resource_name(name)
    if tail is not None:
        _nonnegative_int("tail", tail)
    if scope not in (SCOPE_HEAD, SCOPE_ALL):
        raise ValueError("Log scope must be 'head' or 'all'")
    for key, value in (("host", host), ("container", container), ("source_id", source_id)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("%s must be a non-empty string" % key)
    with _operation_errors("Kubernetes log resolution"):
        sctx = resolve_sctx(sctx)
        cancel = cancel if cancel is not None else getattr(getattr(sctx, "operation", None), "cancel", None)
        client = make_client(sctx, kubeconfig=kubeconfig, context=context, namespace=namespace)
        client.namespace = client.namespace or resource_name()
        _pin_client_target(client)
        filtered = any(value is not None for value in (host, container, source_id))
        sources = resolve_sources(client, name=name, kind=kind, scope=SCOPE_ALL if filtered else scope, cancel=cancel)
        if not (cancel is not None and cancel.is_set()):
            sources = _filter_sources(sources, name, host=host, container=container, source_id=source_id)
        executor = K8sExecutor(
            K8sExecutorConfig(
                kubectl_path=str(client.binary_path),
                kubeconfig=client.kubeconfig,
                k8s_context=client.context,
                k8s_namespace=client.namespace,
            )
        )

    def read():
        with _operation_errors("Kubernetes log reading"):
            yield from read_log_sources(executor, sources, follow=follow, tail=tail, cancel=cancel, check=check)

    return read()
