"""Resolve owned native resources to individual, instance-bound log sources."""

from __future__ import annotations

import json

from sparkrun.core.log_source import LogResource, LogSource, LogSourceChanged, MODE_STDOUT, SCOPE_HEAD, source_sort_key
from sparkrun.orchestration.logs import read_log_command

from .errors import K8sError, OwnershipError
from .launch import JOBSET_NAME_LABEL
from .manifests import require_application_owner
from .names import validate_resource_name
from .nccl import JOBSET_JOB_INDEX_ANNOTATION


def _query_json(client, args, *, cancel=None, timeout=None):
    """Cancellable, bounded control-plane lookup using the same reader cleanup."""
    source = LogSource(client.label, "discovery", mode=MODE_STDOUT, path=None)
    reader = read_log_command(
        client.base_args() + args,
        source,
        cancel=cancel,
        timeout=min(timeout, 10.0) if timeout is not None else 10.0,
        check=True,
        merge_stderr=False,
    )
    try:
        payload = "\n".join(line.text for line in reader)
    finally:
        reader.close()
    if cancel is not None and cancel.is_set():
        return {}
    result = json.loads(payload) if payload.strip() else {}
    if not isinstance(result, dict):
        raise K8sError("kubectl returned an invalid resource document")
    return result


def _owner_uid(resource, kind):
    return next(
        (
            ref.get("uid")
            for ref in resource.get("metadata", {}).get("ownerReferences", [])
            if ref.get("kind") == kind and ref.get("controller") is True
        ),
        None,
    )


def _instance(status):
    # Include both values: restart counts and runtime IDs have different
    # lifetimes, and neither a replacement Pod nor a new container is a cursor.
    return json.dumps([status.get("restartCount", 0), status.get("containerID", "")], separators=(",", ":"))


def _rank_layout(resource):
    offsets = {}
    total = 0
    for group in resource.get("spec", {}).get("replicatedJobs", []):
        replicas = group.get("replicas", 1)
        offsets[group["name"]] = (total, replicas)
        total += replicas
    return offsets, total


def _rank(pod, job, offsets):
    attributes = {}
    for resource in (job, pod):
        metadata = resource.get("metadata", {})
        attributes.update(metadata.get("annotations", {}))
        attributes.update(metadata.get("labels", {}))
    group = attributes.get("jobset.sigs.k8s.io/replicatedjob-name")
    try:
        offset, replicas = offsets[group]
        index = int(attributes[JOBSET_JOB_INDEX_ANNOTATION])
        return offset + index if 0 <= index < replicas else None
    except (KeyError, TypeError, ValueError):
        return None


def resolve_sources(client, *, name, kind="jobset", cluster_id=None, uid=None, scope="all", cancel=None):
    """Snapshot regular containers belonging to this exact controller instance.

    Label selectors bound discovery, but controller UID chains establish actual
    membership. Completed containers remain readable; waiting containers do not.
    No resource is changed and no replacement/new Pod is automatically followed.
    """
    if kind not in ("jobset", "job"):
        raise ValueError("Kubernetes log kind must be 'jobset' or 'job'")
    validate_resource_name(name)
    if cancel is not None and cancel.is_set():
        return []
    resource = _query_json(client, ["get", kind, name, "-o", "json", "--ignore-not-found"], cancel=cancel)
    if cancel is not None and cancel.is_set():
        return []
    if not resource:
        raise K8sError("No Kubernetes %s named %r" % (kind, name))
    require_application_owner(resource, kind=kind, name=name)
    meta = resource.get("metadata", {})
    if cluster_id is not None and meta.get("annotations", {}).get("sparkrun.cluster_id") != cluster_id:
        raise OwnershipError("Kubernetes %s %r belongs to another workload" % (kind, name))
    controller_uid = meta.get("uid")
    if not controller_uid:
        raise K8sError("Kubernetes controller has no UID")
    if uid is not None and controller_uid != uid:
        raise LogSourceChanged("Kubernetes controller was replaced; reopen the workload target")
    namespace = meta.get("namespace") or client.namespace
    if not namespace or namespace != client.namespace:
        raise K8sError("Kubernetes log namespace differs from the resolved destination")

    if kind == "jobset":
        selector = "%s=%s" % (JOBSET_NAME_LABEL, name)
        children = _query_json(client, ["get", "jobs", "-l", selector, "-o", "json"], cancel=cancel)
        jobs = {
            job["metadata"]["uid"]: job
            for job in children.get("items", [])
            if _owner_uid(job, "JobSet") == controller_uid and job.get("metadata", {}).get("uid")
        }
    else:
        selector = "batch.kubernetes.io/controller-uid=%s" % controller_uid
        jobs = {controller_uid: resource}
    if cancel is not None and cancel.is_set():
        return []
    pods = _query_json(client, ["get", "pods", "-l", selector, "-o", "json"], cancel=cancel)
    if cancel is not None and cancel.is_set():
        return []
    offsets, total = _rank_layout(resource)
    sources = []
    for pod in pods.get("items", []):
        job = jobs.get(_owner_uid(pod, "Job"))
        if job is None:
            continue
        pod_meta = pod.get("metadata", {})
        pod_name, pod_uid = pod_meta.get("name"), pod_meta.get("uid")
        if not pod_name or not pod_uid or pod_meta.get("namespace") != namespace:
            continue
        require_application_owner(pod, kind="pod", name=pod_name)
        rank = _rank(pod, job, offsets) if kind == "jobset" else 0
        statuses = {item["name"]: item for item in pod.get("status", {}).get("containerStatuses", [])}
        for index, container in enumerate(pod.get("spec", {}).get("containers", [])):
            status = statuses.get(container["name"], {})
            state = status.get("state", {})
            if "running" not in state and "terminated" not in state:
                continue
            if scope == SCOPE_HEAD and (rank != 0 or index != 0):
                continue
            role = "solo" if kind == "job" or total == 1 else ("head" if rank == 0 else "worker")
            if index != 0:
                role = "sidecar"
            sources.append(
                LogSource(
                    host=pod.get("spec", {}).get("nodeName", ""),
                    container=container["name"],
                    role=role,
                    rank=rank,
                    mode=MODE_STDOUT,
                    path=None,
                    resource=LogResource("Pod", namespace, pod_name, pod_uid, _instance(status)),
                )
            )
    if not sources:
        raise K8sError(
            "No readable %s logs for Kubernetes %s %r; containers may be pending, queued, or removed. "
            "Reopen the log target when pods are available, or use scope='all' to inspect other containers." % (scope, kind, name)
        )
    return sorted(sources, key=source_sort_key)


def read_command(client, source, *, follow=False, tail=None):
    resource = source.resource
    if resource is None or resource.kind != "Pod" or resource.namespace != client.namespace:
        raise ValueError("Kubernetes log source does not match this destination")
    command = client.base_args() + ["logs", resource.name, "-c", source.container, "--tail=%d" % (-1 if tail is None else tail)]
    if follow:
        command.append("--follow")
    return command


def validate_source(client, source, *, cancel=None, timeout=None):
    resource = source.resource
    if resource is None:
        return
    if resource.namespace != client.namespace:
        raise LogSourceChanged("Kubernetes log destination changed; reopen the log target")
    pod = _query_json(client, ["get", "pod", resource.name, "-o", "json", "--ignore-not-found"], cancel=cancel, timeout=timeout)
    if cancel is not None and cancel.is_set():
        return
    status = next((s for s in pod.get("status", {}).get("containerStatuses", []) if s.get("name") == source.container), {})
    meta = pod.get("metadata", {})
    if (
        meta.get("uid") != resource.uid
        or meta.get("namespace") != resource.namespace
        or not status
        or _instance(status) != resource.instance
    ):
        raise LogSourceChanged("Log source %s was removed, replaced, or restarted; reopen the log target" % source.label)
    require_application_owner(pod, kind="pod", name=resource.name)
