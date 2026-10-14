"""Valid operational requests share an API error boundary and ownership policy."""

from unittest.mock import Mock

import pytest

from sparkrun.api import SparkrunError
from sparkrun.plugins.k8s import api as k8s
from sparkrun.plugins.k8s.orchestration.client import KubectlClient
from sparkrun.plugins.k8s.orchestration.errors import K8sError, OwnershipError
from test_k8s_setup import _sctx


@pytest.mark.parametrize("operation", ["stop_jobset", "jobset_status", "logs"])
@pytest.mark.parametrize("failure", ["foreign", "network", "unexpected", "interrupt"])
def test_lifecycle_error_boundary_and_no_mutation(tmp_path, monkeypatch, operation, failure):
    client = KubectlClient("/unused/kubectl")
    if failure == "foreign":
        lookup = Mock(return_value={"metadata": {"labels": {"sparkrun.distribution": "jetsonrun"}}})
        cause = OwnershipError
    else:
        cause = {"network": K8sError, "unexpected": OSError, "interrupt": KeyboardInterrupt}[failure]
        lookup = Mock(side_effect=cause("lookup failed"))
    monkeypatch.setattr(client, "run_json", lookup)
    # Log discovery reads through the shared cancellable reader, not ``run_json``.
    monkeypatch.setattr("sparkrun.plugins.k8s.orchestration.logs._query_json", lambda client, args, cancel=None: lookup(args))
    monkeypatch.setattr(k8s._ops, "make_client", lambda *a, **kw: client)
    mutation = Mock(side_effect=AssertionError("must not mutate"))

    def run(args, **kwargs):
        # Pinning the current context is a read; anything else through ``run`` would mutate.
        if list(args) == ["config", "current-context"]:
            return Mock(stdout="fixture-context\n", returncode=0)
        return mutation(args, **kwargs)

    monkeypatch.setattr(client, "run", run)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else SparkrunError) as caught:
        getattr(k8s, operation)(_sctx(tmp_path), name="sample")
    if failure != "interrupt":
        assert isinstance(caught.value.__cause__, cause)
    mutation.assert_not_called()


@pytest.mark.parametrize("operation", ["stop_jobset", "jobset_status", "logs"])
@pytest.mark.parametrize("name", ["bad_name", "-bad", "a..b", "a.-b", "a" * 254])
def test_invalid_references_fail_before_client_resolution(tmp_path, monkeypatch, operation, name):
    client = Mock(side_effect=AssertionError("invalid names must be rejected eagerly"))
    monkeypatch.setattr(k8s._ops, "make_client", client)
    with pytest.raises(ValueError, match="resource name"):
        getattr(k8s, operation)(_sctx(tmp_path), name=name)
    client.assert_not_called()
