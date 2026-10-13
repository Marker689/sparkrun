"""A release tag must run the suites skipped by the ordinary Python job."""

from pathlib import Path

import yaml


def test_publication_requires_callable_release_suites():
    root = Path(__file__).resolve().parents[1]
    # BaseLoader keeps GitHub's `on` key a string (YAML 1.1 treats it as bool).
    workflow = yaml.load((root / ".github/workflows/publish-python.yml").read_text(), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    dependencies = jobs["publish-sparkrun"]["needs"]
    required = {"test-python.yml", "test-daemon-portable.yml", "test-application-profiles.yml", "type-check.yml"}
    called = {Path(jobs[name]["uses"]).name for name in dependencies if "uses" in jobs[name]}
    assert required <= called
    assert "verify-tag" in dependencies
    for name in required:
        suite = yaml.load((root / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)
        assert "workflow_call" in suite["on"]
    policy = yaml.safe_load((root / "versions.yaml").read_text())
    assert "publish-python" in policy["ci"]["skip_workflows"]
