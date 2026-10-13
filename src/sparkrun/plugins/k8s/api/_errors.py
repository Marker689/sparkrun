"""Public Kubernetes errors for :mod:`sparkrun.plugins.k8s.api`.

Orchestration-level failures (:mod:`sparkrun.plugins.k8s.orchestration.errors`)
are translated into these :class:`~sparkrun.api._errors.SparkrunError`
subclasses at the api boundary, so callers can ``except SparkrunError``
uniformly.
"""

from __future__ import annotations

from contextlib import contextmanager

from sparkrun.api._errors import SparkrunError


class KubectlUnavailable(SparkrunError):
    """No usable ``kubectl`` binary could be resolved or downloaded."""


class ClusterUnreachable(SparkrunError):
    """The target cluster's API server did not respond."""


class ServiceAccountError(SparkrunError):
    """Configuring the sparkrun service account failed."""


class LauncherJobError(SparkrunError):
    """Building or applying an in-cluster launcher Job failed."""


class KueueSetupError(SparkrunError):
    """Detecting, installing, or provisioning Kueue / JobSet failed."""


class JobSetLaunchError(SparkrunError):
    """Building, prechecking, or submitting a k8s JobSet launch failed."""


@contextmanager
def _operation_errors(operation: str):
    """Translate operational failures; argument checks run outside this boundary."""
    from ..orchestration.errors import K8sError, OwnershipError
    from sparkrun.core.log_source import LogSourceChanged

    try:
        yield
    except SparkrunError:
        raise
    except LogSourceChanged:
        raise
    except OwnershipError as exc:
        raise SparkrunError(str(exc)) from exc
    except K8sError as exc:
        raise ClusterUnreachable(str(exc)) from exc
    except Exception as exc:
        raise SparkrunError("%s failed: %s" % (operation, exc)) from exc


__all__ = [
    "KubectlUnavailable",
    "ClusterUnreachable",
    "ServiceAccountError",
    "LauncherJobError",
    "KueueSetupError",
    "JobSetLaunchError",
]
