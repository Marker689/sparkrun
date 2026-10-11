# SPDX-FileCopyrightText: 2026 Spark Arena
# SPDX-License-Identifier: Apache-2.0
"""Per-host runtime connections, separate from relay and image platforms."""

from dataclasses import replace
import json
import platform

DEFAULT_DOCKER = ['docker', '--host', 'unix:///var/run/docker.sock']
STRINGS = {'runtime', 'docker_host', 'docker_context', 'docker_ca', 'docker_cert', 'docker_key',
           'wslc_executable', 'wslc_session'}
BOOLEANS = {'docker_tls', 'docker_allow_plain_http'}


def validate(settings):
    connections = settings.get('runtime_connections', {})
    if not isinstance(connections, dict):
        raise ValueError('runtime_connections must map controller/host to connection options')
    for host, value in connections.items():
        if not isinstance(host, str) or not host or not isinstance(value, dict) or set(value) - STRINGS - BOOLEANS:
            raise ValueError('invalid runtime_connections host or options')
        if any(not isinstance(v, str) or '\x00' in v for k, v in value.items() if k in STRINGS):
            raise ValueError('runtime connection strings must be NUL-free')
        if any(type(v) is not bool for k, v in value.items() if k in BOOLEANS):
            raise ValueError('runtime connection flags must be boolean')
        if value.get('runtime', 'docker') not in {'docker', 'wslc'}:
            raise ValueError('runtime must be docker or wslc')
        if value.get('docker_host') and value.get('docker_context'):
            raise ValueError('choose docker_host or docker_context')
        if value.get('runtime') == 'wslc' and any(k.startswith('docker_') for k in value):
            raise ValueError('WSLC cannot use Docker connection options')
        if value.get('runtime', 'docker') == 'docker' and any(k.startswith('wslc_') for k in value):
            raise ValueError('WSLC options require runtime: wslc')


def config(settings, host):
    selected = dict(settings.get('runtime_connections', {}).get(host if host is not None else 'controller', {}))
    # Keep remote Linux nodes independent of the controller's Docker context.
    if selected.get('runtime', 'docker') == 'docker' and not selected.get('docker_host') and not selected.get('docker_context'):
        if host is not None or platform.system() == 'Linux':
            selected['docker_host'] = 'unix:///var/run/docker.sock'
    return selected


def arguments(settings, host):
    result = []
    for key, value in config(settings, host).items():
        flag = '--' + key.replace('_', '-')
        if type(value) is bool:
            result.append(flag + '=' + str(value).lower())
        elif value:
            result.extend([flag, value])
    return result


def portable(settings, host):
    c = config(settings, host)
    return c.get('runtime') == 'wslc' or c.get('docker_host', '') not in {'unix:///var/run/docker.sock', 'unix:///run/docker.sock'}


class RuntimeSession:
    """Route existing metadata probes to the same runtime used by the Go relay.

    WSLC has inspect JSON but no Docker Go-template formatting. Translate only
    the small, explicit metadata contract used by the plugin, never payloads.
    """

    def __init__(self, session, settings, host):
        self.session, self.settings, self.host = session, settings, host

    def __getattr__(self, name):
        return getattr(self.session, name)

    def open_process(self, host, arguments):
        if arguments[:len(DEFAULT_DOCKER)] == DEFAULT_DOCKER:
            c = config(self.settings, self.host)
            if c.get('runtime') == 'wslc':
                raise RuntimeError('WSLC cannot run Docker native-store helpers')
            if portable(self.settings, self.host):
                raise RuntimeError('native-store helpers require a local Linux Docker socket')
            arguments = ['docker', '--host', c['docker_host'], *arguments[len(DEFAULT_DOCKER):]]
        return self.session.open_process(host, arguments)

    def execute(self, host, arguments, **kwargs):
        if arguments[:len(DEFAULT_DOCKER)] != DEFAULT_DOCKER:
            return self.session.execute(host, arguments, **kwargs)
        c = config(self.settings, self.host)
        tail = arguments[len(DEFAULT_DOCKER):]
        if c.get('runtime') != 'wslc':
            command = ['docker']
            for key, flag in [('docker_host', '--host'), ('docker_context', '--context'), ('docker_ca', '--tlscacert'),
                              ('docker_cert', '--tlscert'), ('docker_key', '--tlskey')]:
                if c.get(key):
                    command.extend([flag, c[key]])
            if c.get('docker_tls') or any(c.get(k) for k in ('docker_ca', 'docker_cert', 'docker_key')):
                command.append('--tlsverify')
            result = self.session.execute(host, command + tail, **kwargs)
            if tail[:1] == ['info'] and result.returncode == 0:
                facts = json.loads(result.stdout)
                facts['native_local'] = not portable(self.settings, self.host)
                result = replace(result, stdout=json.dumps(facts).encode())
            return result
        command = [c.get('wslc_executable') or 'wslc.exe']
        if c.get('wslc_session'):
            command.extend(['--session', c['wslc_session']])
        if tail[:1] == ['info']:
            from sparkrun.transports.session import HostCommandResult

            facts = dict(os='linux', driver='wslc', native_local=False)
            return HostCommandResult(host, 0, json.dumps(facts).encode())
        if tail[:2] != ['image', 'inspect']:
            raise RuntimeError('WSLC metadata adapter only supports image inspect; transfers require the Go runtime adapter')
        template = next((v.split('=', 1)[1] for v in tail if v.startswith('--format=')), None)
        if template not in {'{{.Id}}', '{{.Size}}', '{{.Id}}|{{.Os}}|{{.Architecture}}'}:
            raise RuntimeError('unsupported WSLC metadata probe')
        result = self.session.execute(host, command + ['image', 'inspect', tail[-1]], **kwargs)
        if result.returncode:
            return result
        images = json.loads(result.stdout.decode('utf-8-sig'))
        if not isinstance(images, list) or len(images) != 1:
            raise RuntimeError('invalid WSLC image inspection')
        info = images[0]
        value = {'{{.Id}}': str(info.get('Id', '')), '{{.Size}}': str(info.get('Size', '')),
                 '{{.Id}}|{{.Os}}|{{.Architecture}}': '|'.join(str(info.get(k, '')) for k in ('Id', 'Os', 'Architecture'))}[template]
        return replace(result, stdout=(value + '\n').encode())
