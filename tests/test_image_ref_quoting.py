"""Image references from recipe content never execute in the image scripts."""

import os
import shutil
import subprocess

import pytest

from sparkrun.core.images import ImagePlanError, resolve_image_plan
from sparkrun.scripts import read_script
from sparkrun.utils.shell import quote, validate_image_ref

HOSTILE = "alpine$(touch %s)"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@needs_bash
@pytest.mark.parametrize("script", ["image_sync.sh", "image_distribute.sh"])
def test_rendered_image_scripts_do_not_expand_the_reference(tmp_path, script):
    marker = tmp_path / "pwned"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("docker", "ssh"):
        stub = bin_dir / tool
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    image = HOSTILE % marker
    rendered = read_script(script).format(
        image=quote(image), force_pull="0", offline="0", targets="", ssh_opts="", ssh_user="", max_parallel="1"
    )
    env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]))
    subprocess.run(["bash", "-s"], input=rendered, capture_output=True, text=True, env=env, timeout=30)
    assert not marker.exists()


@pytest.mark.parametrize(
    "image",
    ["nvcr.io/nvidia/vllm:25.09-py3", "localhost:5000/repo/img", "img@sha256:" + "a" * 64, "[::1]:5000/img:tag", "Repo_x/img"],
)
def test_valid_image_refs_pass(image):
    assert validate_image_ref(image) == image


@pytest.mark.parametrize("image", ["", "alpine$(id)", "img;rm -rf /", "img tag", "-img", "img`id`", 'img"x', "a\nb"])
def test_invalid_image_refs_rejected(image):
    with pytest.raises(ValueError):
        validate_image_ref(image)


def test_image_plan_rejects_hostile_reference():
    class _Recipe:
        containers = [{"host": "h1", "image": "img$(id)"}]

    with pytest.raises(ImagePlanError, match="Invalid container image reference"):
        resolve_image_plan(_Recipe(), "fallback:1", ["h1"])
