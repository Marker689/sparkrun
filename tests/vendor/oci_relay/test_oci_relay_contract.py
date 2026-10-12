# SPDX-FileCopyrightText: 2026 Spark Arena
# SPDX-License-Identifier: Apache-2.0
"""Small offline contract suite exported with the bundled adapter."""

import json
from pathlib import Path

import pytest


def test_oci_relay_release_pins_and_host_progress_level():
    from sparkrun.plugins import oci_relay
    from sparkrun.plugins.oci_relay.progress import PROGRESS
    from sparkrun.core.progress import PROGRESS as HOST_PROGRESS

    pins = json.loads(Path(oci_relay.__file__).with_name('releases.json').read_text())[oci_relay.__version__]
    assert set(pins) == {'linux/amd64', 'linux/arm64', 'darwin/arm64', 'windows/amd64', 'windows/arm64'}
    for arch, pin in pins.items():
        assert pin['url'] == (f'https://github.com/spark-arena/oci-relay/releases/download/v{oci_relay.__version__}/'
                              f'oci-relay_{oci_relay.__version__}_{arch.replace("/", "_")}.tar.gz')
        assert len(pin['sha256']) == 64 and all(c in '0123456789abcdef' for c in pin['sha256'])
    assert PROGRESS == HOST_PROGRESS


def test_oci_relay_defaults_keep_native_access_and_public_fallback_available():
    from sparkrun.plugins.oci_relay.source_policy import validate

    settings = validate({})
    assert settings.get('source_mode', 'auto') == 'auto'
    assert settings['allow_native_store'] is True
    assert settings['allow_preparation_read'] is True


def test_oci_relay_digest_handoff_capability():
    from sparkrun.plugins import IMAGE_RUNTIME_API_VERSION, ImageCopyResult
    from sparkrun.plugins.oci_relay import pins
    from sparkrun.plugins.oci_relay.provider import RelayProvider

    assert IMAGE_RUNTIME_API_VERSION == 1
    image = 'registry.test/image:tag@sha256:' + 'a' * 64
    assert pins.digest(image) == 'sha256:' + 'a' * 64
    assert pins.retention_tag(image).startswith('oci-relay/pinned:')
    assert '@' not in pins.retention_tag(image)
    RelayProvider._require_pin_api()
    assert RelayProvider.supports_offline_pull is True
    assert ImageCopyResult({'host': 'complete'}, runtime_images={'host': 'sha256:' + 'b' * 64}).runtime_images['host']


@pytest.mark.parametrize('kind', ['copy', 'pull'])
def test_oci_relay_unavailable_download_defaults_to_builtin(monkeypatch, kind):
    from sparkrun.core import image_distribution as api
    from sparkrun.plugins.oci_relay.provider import RelayProvider
    from sparkrun.plugins.oci_relay.release import BinaryInvalid, BinaryUnavailable

    provider = RelayProvider()
    monkeypatch.setattr(provider, '_settings', lambda request: {})
    monkeypatch.setattr(api, '_PROVIDERS', {'oci-relay': provider})
    monkeypatch.setattr('sparkrun.plugins.oci_relay.provider.platform.system', lambda: 'Linux')

    def unavailable(*args, **kwargs):
        raise BinaryUnavailable('release download unavailable')

    monkeypatch.setattr(provider, '_copy', unavailable)
    arguments = dict(image='registry.test/image:tag', source_host=None, targets=['host'], transfer_hosts=None,
                     ssh_user=None, ssh_key=None, ssh_options=None, timeout=30, dry_run=False,
                     offline=False, session=object())
    invoke = api.try_image_copy if kind == 'copy' else api.try_image_pull
    if kind == 'pull':
        arguments['force_pull'] = True
    token = api._CONFIG.set({})
    try:
        assert invoke(**arguments) is None
        api._CONFIG.set({'container_distribution_fallback': False})
        if kind == 'copy':
            assert invoke(**arguments) == ['host']
        else:
            with pytest.raises(api.ImageDistributionFailed):
                invoke(**arguments)

        def invalid(*args, **kwargs):
            raise BinaryInvalid('release checksum mismatch')

        monkeypatch.setattr(provider, '_copy', invalid)
        api._CONFIG.set({'container_distribution_fallback': True})
        with pytest.raises(BinaryInvalid, match='checksum'):
            invoke(**arguments)
    finally:
        api._CONFIG.reset(token)


def test_relay_structured_feedback_retains_upper_bound_and_verified_outcome():
    from sparkrun.core.operation import OperationState, operation_scope
    from sparkrun.plugins.oci_relay.progress import Progress

    seen = []
    with operation_scope(state=OperationState(events=seen.append)):
        p = Progress('example/image', interval=0)
        p.bind({'receiver': 'host-a'})
        try:
            p.event({'receiver_progress': {'receiver': {
                'sequence': 1, 'phase': 'transferring', 'inventory_ready': True,
                'received_bytes': 4, 'expected_bytes': 8, 'bytes_per_second': 2,
            }}})
            p.outcome('receiver', {'progress': {'wire_bytes': 4}}, False)
        finally:
            p.close(False, False)
    event = next(v for v in seen if v.get('stage') == 'transferring')
    assert event['type'] == 'feedback' and event['source'] == 'oci-relay'
    assert event['host'] == 'host-a' and event['total_kind'] == 'upper_bound'
    assert event['current'] == 4 and event['total'] == 8
    assert seen[-1]['status'] == 'failed'


def test_unsupported_platform_preflight_obeys_auto_fallback(monkeypatch):
    from types import SimpleNamespace
    from sparkrun.core import image_distribution as api
    from sparkrun.plugins.oci_relay.provider import RelayProvider
    from sparkrun.plugins.oci_relay.host import Runner

    monkeypatch.setattr(Runner, "execute", lambda *args, **kwargs: b"Darwin arm64\n")
    monkeypatch.setattr(Runner, "close", lambda self: [])
    monkeypatch.setattr(api, "_PROVIDERS", {"oci-relay": RelayProvider()})
    config = SimpleNamespace(get=lambda key, default=None: default, plugin_settings=lambda name: {})
    token = api._CONFIG.set(config)
    try:
        assert api.try_image_copy(image="org/image:test", source_host=None, targets=["node-a"],
            transfer_hosts=None, ssh_user=None, ssh_key=None, ssh_options=None, timeout=10,
            dry_run=False, offline=True, session=SimpleNamespace(open_process=lambda *a, **k: None)) is None
    finally:
        api._CONFIG.reset(token)
