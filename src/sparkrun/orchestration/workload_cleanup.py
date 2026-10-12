"""Finalize a confirmed workload teardown before forgetting its recovery data."""


def finalize_workload(cluster_id, *, hosts, cache_dir=None, ssh_kwargs=None, session=None):
    """Release only confirmed hosts; retain metadata on partial cleanup or error.

    Callers must establish inactivity. A missing process, old timestamp, or
    missing metadata alone is not evidence that a remote workload stopped.
    """
    from sparkrun.models.usage import release_claims
    from sparkrun.orchestration.job_metadata import load_job_metadata, remove_job_metadata

    metadata = load_job_metadata(cluster_id, cache_dir=cache_dir) or {}
    connection = dict(ssh_kwargs or {})
    if metadata.get("ssh_user"):
        connection["ssh_user"] = metadata["ssh_user"]
    confirmed = set(hosts)
    claims = metadata.get("model_cache_claims", [])
    release_claims(claims, hosts=confirmed, ssh_kwargs=connection, session=session)
    expected = set(metadata.get("hosts", ())) | {claim["host"] for claim in claims}
    if not expected <= confirmed:
        return False
    remove_job_metadata(cluster_id, cache_dir=cache_dir)
    return True
