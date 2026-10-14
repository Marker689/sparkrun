"""Tests for in-tree plugin loading — the mate of the external plugin loader.

The property that matters is that in-tree and out-of-tree plugins register
*identically*: both go through ``load_and_register_plugin``, so a first-party
integration has no capability an external one lacks, and neither can drift into
a different notion of what counts as a registration.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from sparkrun.core.features import FEATURE_FLAGS, FeatureFlag, register_feature
from sparkrun.core.in_tree_plugins import IN_TREE_PLUGIN_FEATURES, load_in_tree_plugins


class _FakePlugins:
    """A throwaway plugin package plus the flags its plugins registered."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.registered_flags: list[str] = []
        self.bound_plugins: list[str] = []


@pytest.fixture
def plugin_package(tmp_path, monkeypatch):
    """Build an importable throwaway plugin package under *tmp_path*."""
    root = tmp_path / "fake_plugins"
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    package = _FakePlugins(root)
    yield package
    for flag in package.registered_flags:
        FEATURE_FLAGS.pop(flag, None)
    for plugin in package.bound_plugins:
        IN_TREE_PLUGIN_FEATURES.pop(plugin, None)
    for name in [m for m in sys.modules if m.startswith("fake_plugins")]:
        sys.modules.pop(name, None)


def _write(package: _FakePlugins, name: str, body: str, *, flag: bool | None = True) -> None:
    """Create plugin *name*, binding and registering its gate unless *flag* is None.

    Every in-tree plugin needs an ``IN_TREE_PLUGIN_FEATURES`` entry plus a
    registered flag; ``flag=None`` models the packaging mistake of shipping one
    without either.
    """
    module = package.root / name
    module.mkdir()
    (module / "__init__.py").write_text("SPARKRUN_PLUGIN_API_VERSION = 1\n" + textwrap.dedent(body))
    if flag is not None:
        flag_name = "test.%s" % name
        IN_TREE_PLUGIN_FEATURES[name] = flag_name
        register_feature(FeatureFlag(name=flag_name, description="test plugin", default=flag))
        package.registered_flags.append(flag_name)
        package.bound_plugins.append(name)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_register_hook_runs_with_the_saf_variables(plugin_package):
    _write(
        plugin_package,
        "good",
        """
        SEEN = []

        def register(v):
            SEEN.append(v)
        """,
    )
    sentinel = object()
    assert load_in_tree_plugins(sentinel, package="fake_plugins") == ["good"]

    import fake_plugins.good

    assert fake_plugins.good.SEEN == [sentinel]


def test_a_broken_plugin_does_not_stop_the_others(plugin_package):
    """Shipping a broken integration must not take the CLI down with it."""
    _write(plugin_package, "aaa_broken", "raise RuntimeError('boom')")
    _write(
        plugin_package,
        "zzz_working",
        """
        LOADED = True

        def register(v):
            pass
        """,
    )
    assert load_in_tree_plugins(None, package="fake_plugins") == ["zzz_working"]


def test_a_failing_register_hook_is_contained(plugin_package):
    _write(
        plugin_package,
        "bad_hook",
        """
        def register(v):
            raise RuntimeError('boom')
        """,
    )
    # Successful import alone does not make a partially initialized plugin loaded.
    assert load_in_tree_plugins(None, package="fake_plugins") == []


def test_private_modules_are_skipped(plugin_package):
    _write(plugin_package, "_private", "def register(v): raise AssertionError('should not load')")
    _write(plugin_package, "public", "def register(v): pass")
    assert load_in_tree_plugins(None, package="fake_plugins") == ["public"]


def test_a_missing_package_degrades_rather_than_raising():
    assert load_in_tree_plugins(None, package="sparkrun.does_not_exist") == []


# ---------------------------------------------------------------------------
# Surface flags
# ---------------------------------------------------------------------------
#
# Every in-tree plugin carries a `plugins.<name>` flag for the same reason
# `executor.docker` and `gateway.litellm` do despite shipping on: every plugin
# surface should be controllable the same way.


def test_a_disabled_plugin_is_never_even_imported(plugin_package):
    """The flag is checked before the import, so turning a plugin off costs
    nothing — no import, no commands, no registrations."""
    _write(
        plugin_package,
        "switched_off",
        """
        raise AssertionError('a disabled plugin must not be imported')
        """,
        flag=False,
    )
    assert load_in_tree_plugins(None, package="fake_plugins") == []
    assert "fake_plugins.switched_off" not in sys.modules


def test_an_enabled_plugin_can_be_switched_off_by_env(plugin_package, monkeypatch):
    _write(plugin_package, "toggleable", "def register(v): pass", flag=True)
    assert load_in_tree_plugins(None, package="fake_plugins") == ["toggleable"]

    monkeypatch.setenv("SPARKRUN_FEATURE_TEST_TOGGLEABLE", "0")
    assert load_in_tree_plugins(None, package="fake_plugins") == []


def test_an_unbound_plugin_is_skipped_loudly(plugin_package, caplog):
    """It would be skipped anyway — an unbound plugin has no flag, and unknown
    flags resolve off. The error exists so the reason isn't far from the cause."""
    _write(plugin_package, "unflagged", "def register(v): pass", flag=None)
    with caplog.at_level("ERROR"):
        assert load_in_tree_plugins(None, package="fake_plugins") == []
    assert "unflagged" in caplog.text
    assert "IN_TREE_PLUGIN_FEATURES" in caplog.text


def test_only_release_owned_plugins_declare_at_the_in_tree_tier(plugin_package):
    """Namespace membership is not ownership: a package that merely occupies the
    in-tree plugin namespace must not be able to ship a trusted registry."""
    from sparkrun.core.in_tree_plugins import in_tree_declaration_tier
    from sparkrun.core.registry_defaults import DeclarationTier, iter_declared_registries, reset_declared_registries

    assert in_tree_declaration_tier("sparkrun.plugins.lil") is DeclarationTier.IN_TREE
    _write(
        plugin_package,
        "squatter",
        """
        from sparkrun.core.registry import RegistryEntry
        from sparkrun.core.registry_defaults import register_default_registry

        def register(v):
            register_default_registry(
                RegistryEntry(name="squatter", url="https://example.com/r.git", subpath="recipes", trusted=True),
                owner="squatter",
            )
        """,
    )
    try:
        assert load_in_tree_plugins(None, package="fake_plugins") == ["squatter"]
        (declared,) = [d for d in iter_declared_registries() if d.entry.name == "squatter"]
        assert declared.tier is DeclarationTier.OUT_OF_TREE
        assert declared.effective_entry().trusted is False
    finally:
        reset_declared_registries()
