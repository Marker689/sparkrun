"""One plugin table and a generic application-profile declaration check."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import Mock

import pytest

from sparkrun.core.application_profile import APPLICATION_PROFILE_API_VERSION, ApplicationProfile, select_application_profile
from sparkrun.core.plugin_identity import PluginIdentity, normalize_plugin_id
from sparkrun.core.plugin_policy import (
    KNOWN_PLUGINS,
    LEGACY_PLUGIN_FLAGS,
    application_profile_failure,
    declared_application_profile_api,
)
from sparkrun.core.registration import ApplicationProfileCompatibilityError, PluginCompatibilityError, load_and_register_plugin


def _alternate():
    return select_application_profile(
        ApplicationProfile(id="alternate-test", display_name="Alternate test", command="alternate-test", package="alternate-test")
    )


def test_plugin_lists_derive_from_one_table():
    from sparkrun.core.in_tree_plugins import BUNDLED_PLUGIN_NAMES, IN_TREE_PLUGIN_FEATURES

    assert all(normalize_plugin_id(plugin_id) == plugin_id for plugin_id in KNOWN_PLUGINS)
    assert all(normalize_plugin_id(known.module) == plugin_id for plugin_id, known in KNOWN_PLUGINS.items())
    assert BUNDLED_PLUGIN_NAMES == {known.module for known in KNOWN_PLUGINS.values() if known.bundled}
    assert IN_TREE_PLUGIN_FEATURES == {known.module: known.legacy_flag for known in KNOWN_PLUGINS.values()}
    # A capability flag (gateway.sparkroute) never gates loading for every source.
    assert "sparkroute" not in LEGACY_PLUGIN_FLAGS
    assert LEGACY_PLUGIN_FLAGS["oci-relay"] == IN_TREE_PLUGIN_FEATURES["oci_relay"]


def test_bundled_plugins_ship_in_tree():
    from sparkrun.core.in_tree_plugins import BUNDLED_PLUGIN_NAMES, iter_in_tree_plugin_names

    assert BUNDLED_PLUGIN_NAMES <= set(iter_in_tree_plugin_names())


@pytest.mark.parametrize("plugin_id", ["coldsnap", "some-third-party", "sparkroute"])
def test_a_mismatched_declaration_is_refused_for_every_plugin(plugin_id):
    failure = application_profile_failure(plugin_id, APPLICATION_PROFILE_API_VERSION + 1, profile_id="sparkrun")
    assert failure and "declares application profile API" in failure
    assert application_profile_failure(plugin_id, APPLICATION_PROFILE_API_VERSION, profile_id="alternate-test") is None


def test_an_undeclared_plugin_is_profile_agnostic_unless_its_releases_predate_profiles():
    assert application_profile_failure("coldsnap", None, profile_id="alternate-test") is None
    assert application_profile_failure("some-third-party", None, profile_id="alternate-test") is None
    assert application_profile_failure("sparkroute", None, profile_id="sparkrun") is None
    assert "pinned SparkRoute" in application_profile_failure("sparkroute", None, profile_id="alternate-test")


def test_the_declaration_is_read_beside_the_entry_module(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    assert declared_application_profile_api(package / "__init__.py") is None
    (package / "VENDORED.toml").write_text("application_profile_api = 7\n")
    assert declared_application_profile_api(package / "__init__.py") == 7
    (package / "plugin.toml").write_text('application_profile_api = "1"\n')
    assert declared_application_profile_api(package / "__init__.py") is None  # plugin.toml wins; not an int


def _identity(plugin_id="profile-probe"):
    return PluginIdentity(plugin_id, "profile_probe_module", ("external", "profile_probe_module", None))


def test_registration_refuses_a_mismatched_declaration_before_import():
    loader = Mock(side_effect=AssertionError("must not import"))
    with pytest.raises(ApplicationProfileCompatibilityError, match="declares application profile API"):
        load_and_register_plugin(loader, None, identity=_identity(), application_profile_api=APPLICATION_PROFILE_API_VERSION + 1)
    loader.assert_not_called()
    assert issubclass(ApplicationProfileCompatibilityError, PluginCompatibilityError)


def test_registration_reads_a_directory_plugins_manifest(tmp_path, monkeypatch):
    package = tmp_path / "profile_probe_module"
    package.mkdir()
    (package / "__init__.py").write_text("SPARKRUN_PLUGIN_API_VERSION = 1\n")
    (package / "plugin.toml").write_text("application_profile_api = %d\n" % (APPLICATION_PROFILE_API_VERSION + 1))
    monkeypatch.syspath_prepend(str(tmp_path))
    loader = Mock(return_value=ModuleType("profile_probe_module"))
    with pytest.raises(ApplicationProfileCompatibilityError):
        load_and_register_plugin(loader, None, identity=_identity())
    loader.assert_not_called()


def test_registration_refuses_an_undeclared_legacy_sparkroute_under_an_alternate_profile():
    _alternate()
    loader = Mock(side_effect=AssertionError("must not import"))
    identity = PluginIdentity("sparkroute", "sparkrun.plugins.sparkroute", ("external", "sparkrun.plugins.sparkroute", "/store"))
    with pytest.raises(ApplicationProfileCompatibilityError, match="pinned SparkRoute"):
        load_and_register_plugin(loader, None, identity=identity, application_profile_api=None)
    loader.assert_not_called()


def test_a_registered_gateway_is_not_blocked_by_the_profile_lookup(monkeypatch):
    """A managed SparkRoute that declared support loaded; resolving it must not re-refuse it."""
    from sparkrun.proxy import gateway

    _alternate()
    engine = type("Engine", (), {})
    monkeypatch.setitem(gateway._GATEWAY_LOADERS, "sparkroute", lambda: engine)
    assert gateway.gateway_class("sparkroute") is engine
