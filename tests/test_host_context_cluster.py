"""``with_host_context`` carries the effective cluster, loaded once.

``run``, ``proxy load`` and ``benchmark`` each re-resolved "the named or the
default cluster" after the decorator had already done so, behind
``except Exception`` — so a default cluster whose file could not be read
silently became an *anonymous* host list (no SSH user, executor, transport or
offline setting).  The decorator now injects the :class:`HostContext` with the
loaded definition, and an unreadable effective cluster is an error.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.cli._common import HostContext, resolve_host_context, with_host_context
from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.core.config import SparkrunConfig, get_config_root


def _mgr() -> ClusterManager:
    return ClusterManager(get_config_root())


def _break(mgr: ClusterManager, name: str) -> None:
    mgr._cluster_path(name).write_text("hosts: [unclosed\n")


def test_default_cluster_definition_is_carried():
    mgr = _mgr()
    mgr.create("lab", ["h1", "h2"], user="lab-user", executor="docker")
    mgr.set_default("lab")

    hctx = resolve_host_context(None, None, None, SparkrunConfig())

    assert hctx.source == "default-cluster"
    assert hctx.cluster_def is not None and hctx.cluster_def.name == "lab"
    assert hctx.cluster_def.executor == "docker"
    assert hctx.cluster.name == "lab" and hctx.cluster.user == "lab-user"
    assert hctx.host_list == ["h1", "h2"]


def test_explicit_hosts_detach_from_the_default_cluster():
    mgr = _mgr()
    mgr.create("lab", ["h1", "h2"])
    mgr.set_default("lab")

    hctx = resolve_host_context("h9", None, None, SparkrunConfig())

    assert hctx.cluster_def is None
    assert hctx.cluster_name is None


def test_named_cluster_with_explicit_hosts_keeps_its_definition_without_transfer_defaults():
    mgr = _mgr()
    mgr.create("lab", ["h1", "h2"], cache_dir="/data/hf")

    hctx = resolve_host_context("h1", None, "lab", SparkrunConfig())

    assert hctx.cluster_def is not None and hctx.cluster_def.cache_dir == "/data/hf"
    assert hctx.cluster.name == "lab"
    assert hctx.cluster.cache_dir is None  # explicit hosts suppress transfer defaults


def test_unreadable_default_cluster_is_an_error():
    mgr = _mgr()
    mgr.create("lab", ["h1"])
    mgr.set_default("lab")
    _break(mgr, "lab")

    with pytest.raises(click.ClickException, match="Cannot load cluster 'lab'"):
        resolve_host_context(None, None, None, SparkrunConfig())


def test_decorator_injects_host_context_only_when_asked():
    mgr = _mgr()
    mgr.create("lab", ["h1"])
    seen: dict = {}

    @with_host_context
    def wants(hosts=None, hosts_file=None, cluster_name=None, config=None, *, host_list, cluster_mgr, host_context):
        seen["ctx"] = host_context

    @with_host_context
    def plain(hosts=None, hosts_file=None, cluster_name=None, config=None, *, host_list, cluster_mgr):
        seen["plain"] = host_list

    wants(cluster_name="lab", config=SparkrunConfig())
    plain(cluster_name="lab", config=SparkrunConfig())

    assert isinstance(seen["ctx"], HostContext)
    assert seen["ctx"].cluster_def is not None and seen["ctx"].cluster_def.name == "lab"
    assert seen["plain"] == ["h1"]


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "some-recipe", "--dry-run"],
        ["proxy", "load", "some-recipe", "--dry-run"],
        ["benchmark", "run", "some-recipe", "--dry-run"],
    ],
)
def test_launch_commands_refuse_an_unreadable_default_cluster(argv):
    """Previously each command fell back to an anonymous cluster without a word."""
    mgr = _mgr()
    mgr.create("lab", ["h1"], user="lab-user")
    mgr.set_default("lab")
    _break(mgr, "lab")

    result = CliRunner().invoke(main, argv)

    assert result.exit_code != 0
    assert "Cannot load cluster 'lab'" in result.output
