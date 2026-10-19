"""Thin ``kubectl`` subprocess wrapper.

:class:`KubectlClient` is deliberately shaped like the SSH transport
(:mod:`sparkrun.orchestration.ssh`): both return a
:class:`~sparkrun.orchestration.ssh.RemoteResult`, and
:meth:`KubectlClient.exec` mirrors ``run_remote_script(host, script)``.
This is the seam along which a future ``K8sTransport`` — treating
``kubectl exec`` into a pod as an alternative to ``ssh <host> bash -s``
— can slot in beside the SSH layer without reshaping callers.  For now
it is just a convenience wrapper the api / setup layers use.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.utils.shell import b64_wrap_bash, quote

from .errors import K8sError
from .kubectl import KubectlBinary
from sparkrun.utils.text import excerpt

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60


class KubectlClient:
    """Run ``kubectl`` against a resolved binary + kube target.

    The *label* (context name or ``"k8s"``) is threaded into the
    ``RemoteResult.host`` field so log lines and error handling match the
    SSH transport's shape.
    """

    def __init__(
        self,
        binary: KubectlBinary | str | Path,
        *,
        kubeconfig: str | None = None,
        context: str | None = None,
        namespace: str | None = None,
        dry_run: bool = False,
    ) -> None:
        self.binary_path = Path(getattr(binary, "path", binary))
        self.kubeconfig = kubeconfig
        self.context = context
        self.namespace = namespace
        self.dry_run = dry_run

    @property
    def label(self) -> str:
        return self.context or "k8s"

    def base_args(self) -> list[str]:
        """``[kubectl, --kubeconfig K, --context C, -n NS]`` argv prefix."""
        args = [str(self.binary_path)]
        if self.kubeconfig:
            args += ["--kubeconfig", str(self.kubeconfig)]
        if self.context:
            args += ["--context", self.context]
        if self.namespace:
            args += ["-n", self.namespace]
        return args

    def prefix_string(self) -> str:
        """Shell-quoted prefix for embedding in generated scripts."""
        return " ".join(quote(a) for a in self.base_args())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        args: list[str],
        *,
        input_data: str | None = None,
        timeout: int | None = _DEFAULT_TIMEOUT,
        check: bool = False,
    ) -> RemoteResult:
        """Run ``kubectl <args>`` and return a :class:`RemoteResult`."""
        cmd = self.base_args() + list(args)
        if self.dry_run:
            logger.debug("[dry-run] %s", " ".join(cmd))
            return RemoteResult(host=self.label, returncode=0, stdout="[dry-run]", stderr="")
        try:
            proc = subprocess.run(  # noqa: S603 — argv list, no shell
                cmd,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return RemoteResult(host=self.label, returncode=-1, stdout="", stderr="kubectl timed out")
        except FileNotFoundError as exc:
            raise K8sError("kubectl binary not found at %s" % self.binary_path) from exc
        result = RemoteResult(host=self.label, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        if not result.success:
            logger.debug("kubectl %s FAILED rc=%d: %s", args[:2], proc.returncode, excerpt(proc.stderr, 200))
            if check:
                raise K8sError("kubectl %s failed: %s" % (" ".join(args), excerpt(proc.stderr, 400)))
        return result

    def run_json(self, args: list[str], *, timeout: int | None = _DEFAULT_TIMEOUT) -> dict:
        """Run ``kubectl <args>`` and parse stdout as JSON.

        Callers pass argv that produces JSON (either an already-JSON
        subcommand like ``version`` or one with ``-o json`` appended).
        Raises :class:`K8sError` on non-zero exit or unparseable output.
        """
        result = self.run(args, timeout=timeout)
        if not result.success:
            raise K8sError("kubectl %s failed: %s" % (" ".join(args), excerpt(result.stderr, 400)))
        if "--ignore-not-found" in args and not result.stdout.strip():
            return {}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise K8sError("kubectl %s produced non-JSON output" % " ".join(args)) from exc

    def apply(
        self,
        manifest_yaml: str,
        *,
        server_side: bool = False,
        timeout: int | None = _DEFAULT_TIMEOUT,
    ) -> RemoteResult:
        """``kubectl apply -f -`` with *manifest_yaml* piped over stdin.

        *server_side* uses ``--server-side --force-conflicts`` — required for
        large CRD bundles (e.g. the Kueue release manifest) that can exceed
        the client-side apply annotation size limit.
        """
        args = ["apply", "--server-side", "--force-conflicts", "-f", "-"] if server_side else ["apply", "-f", "-"]
        return self.run(args, input_data=manifest_yaml, timeout=timeout)

    def resource_exists(self, kind: str, name: str) -> bool:
        """True iff ``kubectl get <kind> <name>`` succeeds (e.g. a CRD)."""
        return self.run(["get", kind, name, "-o", "name"]).success

    def exec(
        self,
        pod: str,
        script: str,
        *,
        container: str | None = None,
        timeout: int | None = None,
    ) -> RemoteResult:
        """Run *script* inside *pod* via ``kubectl exec`` (transport parity).

        Mirrors ``run_remote_script(host, script)``: the script is
        base64-wrapped and executed under ``bash -c`` inside the pod's
        (optionally named) container.
        """
        args = ["exec", "-i", pod]
        if container:
            args += ["-c", container]
        args += ["--", "bash", "-c", b64_wrap_bash(script)]
        return self.run(args, timeout=timeout)

    # ------------------------------------------------------------------
    # Job transport — run an orchestration payload in-cluster
    # ------------------------------------------------------------------

    def run_launcher_job(self, spec, *, timeout: int | None = _DEFAULT_TIMEOUT) -> RemoteResult:
        """Apply a launcher Job (+ ConfigMap) from a :class:`LauncherJobSpec`.

        The in-cluster analog of ``run_remote_script``: instead of piping
        the payload to ``ssh <host> bash -s``, it lands as a Job that the
        cluster runs under the sparkrun service account.  Returns the
        ``kubectl apply`` result.
        """
        from .job import render_launcher_manifests

        return self.apply(render_launcher_manifests(spec), timeout=timeout)

    def wait_for_job(
        self,
        name: str,
        *,
        condition: str = "complete",
        timeout: int = 600,
    ) -> RemoteResult:
        """Block until ``job/<name>`` reaches *condition* (``complete``/``failed``)."""
        return self.run(
            ["wait", "--for=condition=%s" % condition, "job/%s" % name, "--timeout=%ds" % timeout],
            timeout=timeout + 30,
        )


__all__ = ["KubectlClient"]
