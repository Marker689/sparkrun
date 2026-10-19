"""Substitute only the Hub network call inside a real supervised adapter process."""

from textwrap import dedent, indent

from sparkrun.models import acquisition


def stub_hub_download(monkeypatch, body):
    original = acquisition.hub_script
    stub = (
        "import huggingface_hub\n"
        "def download(repo, filename, **kwargs):\n" + indent(dedent(body), "    ") + "\nhuggingface_hub.hf_hub_download = download\n"
    )

    def script(*args, **kwargs):
        shell = original(*args, **kwargs)
        return shell.replace(
            "import sys\nfrom huggingface_hub.utils import httpx", stub + "import sys\nfrom huggingface_hub.utils import httpx"
        )

    monkeypatch.setattr(acquisition, "hub_script", script)
