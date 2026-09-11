import asyncio
import json
import os
import shlex
import socket
import subprocess
import typing as t

import pytest

from phaser.web.config import (
    DEFAULT_PROFILE_NAME,
    ServerConfig,
    SlurmProfile,
    apply_overrides,
    remote_worker_host,
    remote_worker_netloc,
    resolve_profile,
    split_netloc,
)
from phaser.web.slurm import (
    SlurmError,
    SlurmManager,
    parse_slurm_version,
    parse_squeue_json,
    parse_squeue_tabular,
    render_worker_script,
)

pytestmark = pytest.mark.web


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize(('output', 'expected'), [
    ('slurm 23.02.7', (23, 2)),
    ('slurm-wlm 21.08.5', (21, 8)),
    ('slurm 20.11.8\n', (20, 11)),
    ('slurm 24.05.0-rc1', (24, 5)),
    ('', None),
    ('slurm', None),
])
def test_parse_slurm_version(output: str, expected: t.Optional[t.Tuple[int, int]]):
    assert parse_slurm_version(output) == expected


def test_parse_squeue_tabular():
    output = (
        "1234|RUNNING\n"
        "  1235 | PENDING \n"
        "1236|CANCELLED\n"
        "1237_4|RUNNING\n"   # array task, skipped
        "garbage\n"
        "1238|\n"            # no state, skipped
        "\n"
    )
    assert parse_squeue_tabular(output) == {
        1234: ['RUNNING'], 1235: ['PENDING'], 1236: ['CANCELLED'],
    }


def test_parse_squeue_json():
    # 23.02+ reports a list of states, older versions a bare string
    output = json.dumps({'jobs': [
        {'job_id': 1234, 'job_state': ['COMPLETED']},
        {'job_id': 1235, 'job_state': 'RUNNING'},
        {'job_id': 1236, 'job_state': None},
        {'job_state': ['RUNNING']},  # no id, skipped
    ]})
    assert parse_squeue_json(output) == {
        1234: ['COMPLETED'], 1235: ['RUNNING'], 1236: [],
    }


def test_resolve_profile():
    empty = ServerConfig()
    assert resolve_profile(empty) == (DEFAULT_PROFILE_NAME, SlurmProfile())
    assert resolve_profile(empty, DEFAULT_PROFILE_NAME)[0] == DEFAULT_PROFILE_NAME
    with pytest.raises(ValueError, match="Unknown slurm profile 'gpu'"):
        resolve_profile(empty, 'gpu')

    config = ServerConfig(slurm_profiles={
        'cpu': SlurmProfile(description='cpu'), 'gpu': SlurmProfile(description='gpu'),
    })
    assert resolve_profile(config)[0] == 'cpu'  # first listed
    assert resolve_profile(config, 'gpu')[0] == 'gpu'
    with pytest.raises(ValueError, match="Configured profiles: cpu, gpu"):
        resolve_profile(config, 'other')

    with_default = ServerConfig(default_slurm_profile='gpu', slurm_profiles=config.slurm_profiles)
    assert resolve_profile(with_default)[0] == 'gpu'


def test_profile_args_and_overrides():
    profile = SlurmProfile(description='d', sbatch_args='--time=4-0 --partition=gpu', preamble='module load cuda\n')
    assert profile.args() == ['--time=4-0', '--partition=gpu']
    assert SlurmProfile(sbatch_args=['--time=4-0']).args() == ['--time=4-0']

    # only the specified fields are replaced, and the description is kept
    overridden = apply_overrides(profile, sbatch_args='--time=1:00:00')
    assert overridden.args() == ['--time=1:00:00']
    assert (overridden.preamble, overridden.description) == (profile.preamble, 'd')
    assert apply_overrides(profile) == profile


def test_sbatch_options():
    """--chdir and --output come from the profile's own fields, with '%i' as the worker id"""
    profile = SlurmProfile(working_dir='~/runs/%i', output='logs/worker-%i-%j.log')

    assert profile.sbatch_options('abc123') == [
        f"--chdir={os.path.expanduser('~/runs/abc123')}",
        '--output=logs/worker-abc123-%j.log',  # slurm's own patterns are left for slurm
    ]
    # unset and cleared fields contribute nothing
    assert SlurmProfile().sbatch_options('abc123') == []
    assert SlurmProfile(working_dir='', output='').sbatch_options('abc123') == []


def test_apply_overrides_new_fields():
    profile = SlurmProfile(working_dir='/scratch', output='a.log', sbatch_args='--time=1')
    overridden = apply_overrides(profile, working_dir='/tmp')

    assert (overridden.working_dir, overridden.output) == ('/tmp', 'a.log')
    assert overridden.args() == ['--time=1']


@pytest.mark.parametrize(('netloc', 'expected'), [
    ('host:5050', ('host', 5050)),
    ('host', ('host', None)),
    ('', ('', None)),
    ('[::1]:5050', ('::1', 5050)),
    ('[2001:db8::1]', ('2001:db8::1', None)),
    ('2001:db8::1', ('2001:db8::1', None)),
])
def test_split_netloc(netloc: str, expected: t.Tuple[str, t.Optional[int]]):
    assert split_netloc(netloc) == expected


def test_split_netloc_invalid_port():
    with pytest.raises(ValueError, match="port must be a number"):
        split_netloc('host:not-a-port')


def test_remote_worker_host_override_and_named():
    assert remote_worker_host('localhost', '172.22.254.14') == '172.22.254.14'
    assert remote_worker_host('login1.cluster') == 'login1.cluster'
    assert remote_worker_host('10.0.0.5') == '10.0.0.5'


@pytest.mark.parametrize('host', ['localhost', '127.0.0.1', '127.0.1.1', '::1'])
def test_remote_worker_host_loopback(host: str):
    with pytest.raises(ValueError, match="remote workers can't reach"):
        remote_worker_host(host)


@pytest.mark.parametrize('host', ['0.0.0.0', '::', ''])
def test_remote_worker_host_wildcard(host: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(socket, 'gethostname', lambda: 'login1')
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('10.0.0.5', 0)),
    ])
    assert remote_worker_host(host) == 'login1'


def test_remote_worker_host_hostname_is_loopback(monkeypatch: pytest.MonkeyPatch):
    """A hostname mapped to loopback in /etc/hosts is an error, not a guess"""
    monkeypatch.setattr(socket, 'gethostname', lambda: 'myhost')
    monkeypatch.setattr(socket, 'getfqdn', lambda *args: 'myhost')
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.1.1', 0)),
    ])
    with pytest.raises(ValueError, match="doesn't resolve to a remotely-reachable address"):
        remote_worker_host('0.0.0.0')


def test_remote_worker_netloc_keeps_port():
    assert remote_worker_netloc('login1:5050') == 'login1:5050'
    assert remote_worker_netloc('localhost:5050', '172.22.254.14') == '172.22.254.14:5050'
    # an override naming its own port wins
    assert remote_worker_netloc('localhost:5050', 'gateway:8080') == 'gateway:8080'
    assert remote_worker_netloc('[2001:db8::1]:5050') == '[2001:db8::1]:5050'


def test_render_worker_script(tmp_path):
    profile = SlurmProfile(
        preamble='module load cuda\nconda activate phaser\n',
        python='/opt/env/bin/python',
    )
    script = render_worker_script(profile, "http://host:5050/worker/a b/update")

    assert 'module load cuda\nconda activate phaser' in script
    assert "python_exec=/opt/env/bin/python" in script
    # the url is quoted, so a space can't split it into two words
    assert "url='http://host:5050/worker/a b/update'" in script
    assert '@PREAMBLE@' not in script and '@PYTHON@' not in script and '@URL@' not in script

    path = tmp_path / 'worker.sh'
    path.write_text(script)
    subprocess.run(['bash', '-n', str(path)], check=True)


def test_render_worker_script_empty_profile(tmp_path):
    """An unconfigured profile still renders a valid script, running the server's own python"""
    import sys

    script = render_worker_script(SlurmProfile(), 'http://host:5050/w')
    assert f"python_exec={shlex.quote(sys.executable)}" in script

    path = tmp_path / 'worker.sh'
    path.write_text(script)
    subprocess.run(['bash', '-n', str(path)], check=True)


def test_python_exec_default_follows_preamble():
    """A preamble sets up an environment, so the `python` it provides is the one to run"""
    import sys

    assert SlurmProfile().python_exec() == sys.executable
    assert SlurmProfile(preamble='conda activate phaser').python_exec() == 'python'
    assert SlurmProfile(preamble='conda activate phaser', python='/opt/py').python_exec() == '/opt/py'


def test_render_worker_script_preamble_is_not_rescanned():
    """A marker inside the preamble is left alone, rather than substituted in a later pass"""
    script = render_worker_script(SlurmProfile(preamble='echo @URL@'), 'http://host/w')

    assert 'echo @URL@' in script
    assert 'url=http://host/w' in script


def _fake_sbatch(monkeypatch: pytest.MonkeyPatch, returncode: int = 0,
                 stdout: bytes = b'1234;cluster\n', stderr: bytes = b'') -> t.List[t.Tuple[str, ...]]:
    """Replace `sbatch` with a canned result, recording the argv it was called with"""
    calls: t.List[t.Tuple[str, ...]] = []

    class FakeProc:
        def __init__(self):
            self.returncode = returncode

        async def communicate(self, input: t.Optional[bytes] = None):
            return (stdout, stderr)

    async def fake_exec(*args: str, **kwargs):
        calls.append(args)
        return FakeProc()

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake_exec)
    return calls


def _test_manager() -> SlurmManager:
    manager = SlurmManager()
    manager._slurm_exists = True
    manager._poll_task = asyncio.create_task(asyncio.sleep(0.0))  # no polling in a test
    return manager


def test_make_worker_argv(monkeypatch: pytest.MonkeyPatch):
    """The profile's own options precede its `sbatch_args`, so a repeated option wins"""
    calls = _fake_sbatch(monkeypatch)

    async def scenario():
        manager = _test_manager()

        profile = SlurmProfile(
            working_dir='/scratch/%i', output='w-%i.log',
            sbatch_args='--output=elsewhere.log --gres=gpu:1',
        )
        worker = await manager.make_worker('abc123', 'http://host/w', profile, 'gpu')

        assert worker.slurm_job_id == 1234  # `--parsable` prints 'jobid;cluster'
        assert calls[0] == (
            'sbatch', '--parsable', '--job-name=phaser_abc123',
            '--chdir=/scratch/abc123', '--output=w-abc123.log',
            '--output=elsewhere.log', '--gres=gpu:1',
        )

    run(scenario())


def test_make_worker_reports_sbatch_error(monkeypatch: pytest.MonkeyPatch):
    """A refused submission carries slurm's own complaint, not just an exit code"""
    _fake_sbatch(monkeypatch, returncode=1, stdout=b'',
                 stderr=b'sbatch: error: invalid partition specified: nonesuch\n')

    async def scenario():
        with pytest.raises(SlurmError, match='invalid partition specified: nonesuch'):
            await _test_manager().make_worker('abc123', 'http://host/w', SlurmProfile())

    run(scenario())


def test_make_worker_silent_failure(monkeypatch: pytest.MonkeyPatch):
    """A failure with nothing on stderr still says something"""
    _fake_sbatch(monkeypatch, returncode=7, stdout=b'', stderr=b'')

    async def scenario():
        with pytest.raises(SlurmError, match='exited with code 7'):
            await _test_manager().make_worker('abc123', 'http://host/w', SlurmProfile())

    run(scenario())


def test_make_worker_unparseable_job_id(monkeypatch: pytest.MonkeyPatch):
    _fake_sbatch(monkeypatch, stdout=b'Submitted batch job 1234\n')

    async def scenario():
        with pytest.raises(SlurmError, match="Couldn't parse the job id"):
            await _test_manager().make_worker('abc123', 'http://host/w', SlurmProfile())

    run(scenario())


@pytest.fixture
def slurm_server(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The app, with a config directory of our own and a slurm manager which submits nothing"""
    from phaser.web.server import server
    from phaser.web.slurm import SlurmWorker

    monkeypatch.setattr('phaser.utils.config.get_config_dir', lambda: tmp_path)
    monkeypatch.setattr(server, 'host', 'login1:5050', raising=False)
    monkeypatch.setattr(server, 'root_path', None, raising=False)

    manager = SlurmManager()
    manager._slurm_exists = True
    manager.version = 'slurm 23.02.7'
    submitted: t.List[t.Dict[str, t.Any]] = []

    async def make_worker(worker_id, url, profile=None, profile_name=None):
        submitted.append({'url': url, 'profile': profile, 'profile_name': profile_name})
        return SlurmWorker(worker_id, 1234, profile_name, url)

    monkeypatch.setattr(manager, 'make_worker', make_worker)
    monkeypatch.setattr(server, 'slurm_manager', manager, raising=False)

    return (server, tmp_path / 'server.yaml', submitted)


def test_slurm_profiles_route(slurm_server):
    async def scenario():
        (server, config_path, _) = slurm_server
        config_path.write_text(
            "default_slurm_profile: gpu\n"
            "slurm_profiles:\n"
            "  cpu: {description: cpu only, sbatch_args: '--partition=debug'}\n"
            "  gpu: {sbatch_args: ['--gres=gpu:1'], preamble: 'module load cuda'}\n"
        )

        response = await server.app.test_client().get('/slurm/profiles')
        assert response.status_code == 200
        info = await response.get_json()

        assert (info['available'], info['error']) == (True, None)
        assert (info['version'], info['default_profile']) == ('slurm 23.02.7', 'gpu')
        assert [p['name'] for p in info['profiles']] == ['cpu', 'gpu']
        # args are reported as one editable string, whichever way they were written
        assert info['profiles'][0]['sbatch_args'] == '--partition=debug'
        assert info['profiles'][1]['sbatch_args'] == '--gres=gpu:1'
        assert info['profiles'][1]['preamble'] == 'module load cuda'

    run(scenario())


def test_slurm_profiles_route_unconfigured(slurm_server):
    """With no config file, the built-in default profile is offered"""
    async def scenario():
        (server, _, _) = slurm_server

        info = await (await server.app.test_client().get('/slurm/profiles')).get_json()
        assert info['available'] and info['default_profile'] == DEFAULT_PROFILE_NAME
        assert [p['name'] for p in info['profiles']] == [DEFAULT_PROFILE_NAME]
        assert info['profiles'][0]['sbatch_args'] == ''

    run(scenario())


def test_slurm_profiles_route_bad_config(slurm_server):
    async def scenario():
        (server, config_path, _) = slurm_server
        config_path.write_text("slurm_profiles: [not, a, mapping]\n")

        info = await (await server.app.test_client().get('/slurm/profiles')).get_json()
        assert info['available']
        assert 'server.yaml' in info['error']

    run(scenario())


def test_start_slurm_worker_overrides(slurm_server):
    async def scenario():
        (server, config_path, submitted) = slurm_server
        config_path.write_text("slurm_profiles:\n  gpu: {sbatch_args: '--gres=gpu:1', preamble: 'module load cuda'}\n")

        response = await server.app.test_client().post('/worker/slurm/start', json={
            'profile': 'gpu', 'sbatch_args': '--gres=gpu:2 --time=1:00:00',
        })
        assert response.status_code == 200

        # the url the worker reports to is carried on its state, for the workers table
        assert (await response.get_json())['url'] == submitted[0]['url']

        (launch,) = submitted
        assert launch['profile_name'] == 'gpu'
        assert launch['profile'].args() == ['--gres=gpu:2', '--time=1:00:00']
        assert launch['profile'].preamble == 'module load cuda'  # not overridden
        assert launch['url'].startswith('http://login1:5050/')

    run(scenario())


def test_start_slurm_worker_directory_and_output(slurm_server):
    """The profile's own options are submitted before the user's args, which can override them"""
    async def scenario():
        (server, config_path, submitted) = slurm_server
        config_path.write_text(
            "slurm_profiles:\n"
            "  gpu: {working_dir: /scratch/%i, output: 'logs/%i.log', sbatch_args: '--gres=gpu:1'}\n"
        )

        response = await server.app.test_client().post('/worker/slurm/start', json={'profile': 'gpu'})
        assert response.status_code == 200

        (launch,) = submitted
        worker_id = (await response.get_json())['worker_id']
        assert launch['profile'].sbatch_options(worker_id) == [
            f'--chdir=/scratch/{worker_id}', f'--output=logs/{worker_id}.log',
        ]

    run(scenario())


def test_start_slurm_worker_overrides_directory(slurm_server):
    async def scenario():
        (server, config_path, submitted) = slurm_server
        config_path.write_text("slurm_profiles:\n  gpu: {working_dir: /scratch, output: a.log}\n")

        response = await server.app.test_client().post('/worker/slurm/start', json={
            'profile': 'gpu', 'working_dir': '/tmp/run',
        })
        assert response.status_code == 200

        (launch,) = submitted
        assert launch['profile'].working_dir == '/tmp/run'
        assert launch['profile'].output == 'a.log'  # not overridden

    run(scenario())


def test_slurm_profiles_route_reports_directory_and_output(slurm_server):
    async def scenario():
        (server, config_path, _) = slurm_server
        config_path.write_text("slurm_profiles:\n  gpu: {working_dir: /scratch/%i, output: 'w-%i.log'}\n")

        info = await (await server.app.test_client().get('/slurm/profiles')).get_json()
        # reported unexpanded, since that's the form the config holds and the form edits
        assert (info['profiles'][0]['working_dir'], info['profiles'][0]['output']) == ('/scratch/%i', 'w-%i.log')

    run(scenario())


def test_start_slurm_worker_defaults(slurm_server):
    """An empty body launches the default profile"""
    async def scenario():
        (server, config_path, submitted) = slurm_server
        config_path.write_text("slurm_profiles:\n  gpu: {sbatch_args: '--gres=gpu:1'}\n")

        assert (await server.app.test_client().post('/worker/slurm/start')).status_code == 200
        assert submitted[0]['profile_name'] == 'gpu'
        assert submitted[0]['profile'].args() == ['--gres=gpu:1']

    run(scenario())


def test_start_slurm_worker_rejected(slurm_server, monkeypatch: pytest.MonkeyPatch):
    """A submission slurm refuses is reported to whoever asked, not raised as a 500"""
    async def scenario():
        (server, config_path, _) = slurm_server

        async def make_worker(*args, **kwargs):
            raise SlurmError('sbatch: error: invalid partition specified: nonesuch')

        monkeypatch.setattr(server.slurm_manager, 'make_worker', make_worker)

        response = await server.app.test_client().post('/worker/slurm/start')
        assert response.status_code == 400
        assert 'invalid partition specified' in (await response.get_data(as_text=True))

    run(scenario())


def test_start_slurm_worker_unknown_profile(slurm_server):
    async def scenario():
        (server, config_path, submitted) = slurm_server
        config_path.write_text("slurm_profiles:\n  gpu: {}\n")

        response = await server.app.test_client().post('/worker/slurm/start', json={'profile': 'other'})
        assert response.status_code == 400
        assert "Unknown slurm profile 'other'" in (await response.get_data(as_text=True))
        assert not submitted

    run(scenario())


def test_start_slurm_worker_unreachable_server(slurm_server, monkeypatch: pytest.MonkeyPatch):
    """A server bound to loopback is refused before anything is submitted"""
    async def scenario():
        (server, _, submitted) = slurm_server
        monkeypatch.setattr(server, 'host', 'localhost:5050')

        response = await server.app.test_client().post('/worker/slurm/start')
        assert response.status_code == 400
        assert "remote workers can't reach" in (await response.get_data(as_text=True))
        assert not submitted

    run(scenario())



def test_run_missing_command_is_not_an_error():
    """A missing command reports itself the way a shell would, rather than raising"""
    from phaser.web.slurm import _run

    async def scenario():
        (returncode, stdout, stderr) = await _run('phaser-no-such-command', '--version')
        assert returncode == 127
        assert stdout == b''
        assert b'phaser-no-such-command' in stderr

    run(scenario())


def test_slurm_unavailable(slurm_server, monkeypatch: pytest.MonkeyPatch):
    """Without slurm, the profiles route says so instead of failing"""
    (server, _, submitted) = slurm_server
    monkeypatch.setattr(server.slurm_manager, '_slurm_exists', False)

    async def scenario():
        client = server.app.test_client()
        info = await (await client.get('/slurm/profiles')).get_json()
        assert not info['available'] and 'Slurm not found' in info['error']

        response = await client.post('/worker/slurm/start')
        assert response.status_code == 400
        assert 'Slurm not available' in (await response.get_data(as_text=True))
        assert not submitted

    run(scenario())


def test_manual_worker_url_falls_back_to_local(slurm_server, monkeypatch: pytest.MonkeyPatch):
    """A manual worker may run on this machine, so loopback is a warning rather than an error"""
    (server, _, _) = slurm_server
    monkeypatch.setattr(server, 'host', 'localhost:5050')

    async def scenario():
        response = await server.app.test_client().post('/worker/manual/start')
        assert response.status_code == 200
        state = await response.get_json()
        assert state['url'] == server.get_worker_url(state['worker_id'])
        assert state['message'].endswith(state['url'])
        await server.workers.remove(state['worker_id'])

    run(scenario())
