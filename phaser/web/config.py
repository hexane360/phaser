"""
Server configuration: slurm worker profiles, and the address remote workers connect back to.

Read from `<platformdirs config dir>/server.yaml` (see `phaser.utils.config`). The server
never writes this file; it is re-read whenever a slurm worker is launched, so edits take
effect without a restart.
"""

import ipaddress
import os
import shlex
import socket
import sys
import typing as t
import urllib.parse

import pane

from ..utils.config import Config


class SlurmProfile(pane.PaneBase):
    """A named set of options for launching slurm workers."""

    description: t.Optional[str] = None
    """Human-readable description, shown in the web interface."""

    working_dir: t.Optional[str] = None
    """
    Directory the worker is started in ('sbatch --chdir'). Defaults to the directory the
    server was started in. '%i' is replaced with the worker id.
    """

    output: t.Optional[str] = None
    """
    Where slurm writes the job's output ('sbatch --output'), e.g. 'logs/worker-%i.log'.
    Relative to 'working_dir'. '%i' is replaced with the worker id, and slurm's own
    patterns ('%j' for the job id, '%N' for the node) still apply.
    """

    sbatch_args: t.Union[str, t.Sequence[str]] = ()
    """
    Arguments passed to `sbatch`, as a string or a list of strings. Applied after the
    options above, so an option repeated here wins.
    E.g. '--time=4-0 --partition=gpu --gres=gpu:volta:1'
    """

    preamble: str = ''
    """
    Shell commands run inside the batch job before the worker starts,
     e.g. activating a Python environment
    """

    python: t.Optional[str] = None
    """
    Python executable to run the worker with. Defaults to 'python' when a preamble is set
    (the environment it sets up provides one), and to the server's own interpreter otherwise.
    """

    def args(self) -> t.List[str]:
        return shlex.split(self.sbatch_args) if isinstance(self.sbatch_args, str) else list(self.sbatch_args)

    def sbatch_options(self, worker_id: str) -> t.List[str]:
        """`sbatch` options from the profile's own fields, in the order they're applied.

        These come before `args()`, so a user repeating one of them there overrides it.
        """
        options = {'chdir': self.working_dir, 'output': self.output}
        return [
            f"--{name}={_expand(value, worker_id)}"
            for (name, value) in options.items() if value
        ]

    def python_exec(self) -> str:
        if self.python:
            return self.python
        return 'python' if self.preamble.strip() else sys.executable


class ServerConfig(pane.PaneBase):
    worker_host: t.Optional[str] = None
    """
    Host (or 'host:port') remote workers should connect back to. Defaults to the address
    the server was started on, so this is only needed when that address isn't the one
    workers can reach (a NATed or multi-homed login node).
    """

    default_slurm_profile: t.Optional[str] = None
    """Slurm profile used when none is specified. Defaults to the first profile listed."""

    slurm_profiles: t.Mapping[str, SlurmProfile] = {}
    """
    Named slurm worker profiles.
    Note that 'preamble' runs as arbitrary shell, and anyone who can reach the web
    interface can launch a worker with it.
    """


EXAMPLE_CONFIG: ServerConfig = ServerConfig(
    worker_host=None,
    default_slurm_profile='gpu',
    slurm_profiles={
        'gpu': SlurmProfile(
            description="One Volta GPU, 20 cores",
            output='phaser_worker_%i.log',
            sbatch_args="--qos=high --time=4-0 --partition=xeon-g6-volta --cpus-per-task=20"
                        " --gres=gpu:volta:1 --signal=SIGINT@120",
            preamble='module load anaconda/Python-ML-2025a\n'
                     'eval "$(conda \'shell.bash\' hook)"\n'
                     'conda activate phaser\n'
                     'module load cuda/12.9\n',
        ),
        'cpu': SlurmProfile(
            description="CPU-only debug partition",
            sbatch_args="--qos=high --time=01:00:00 --partition=debug-cpu --cpus-per-task=20",
        ),
    },
)

SERVER_CONFIG: Config[ServerConfig] = Config('server', ServerConfig, example=EXAMPLE_CONFIG)

DEFAULT_PROFILE_NAME: str = 'default'
"""Name of the built-in profile used when no profiles are configured."""


def resolve_profile(config: ServerConfig, name: t.Optional[str] = None) -> t.Tuple[str, SlurmProfile]:
    """The named profile, the configured default, or the first profile listed.

    Raises `ValueError` for a name which isn't configured.
    """
    profiles = config.slurm_profiles

    if not profiles:
        if name is not None and name != DEFAULT_PROFILE_NAME:
            raise ValueError(f"Unknown slurm profile '{name}' (no profiles are configured)")
        return (DEFAULT_PROFILE_NAME, SlurmProfile())

    name = name or config.default_slurm_profile or next(iter(profiles))
    try:
        return (name, profiles[name])
    except KeyError:
        raise ValueError(
            f"Unknown slurm profile '{name}'. Configured profiles: {', '.join(profiles)}"
        ) from None


def apply_overrides(
    profile: SlurmProfile,
    sbatch_args: t.Union[str, t.Sequence[str], None] = None,
    preamble: t.Optional[str] = None,
    python: t.Optional[str] = None,
    working_dir: t.Optional[str] = None,
    output: t.Optional[str] = None,
) -> SlurmProfile:
    """`profile` with each specified field replaced, for a single launch."""
    return SlurmProfile(
        description=profile.description,
        working_dir=profile.working_dir if working_dir is None else working_dir,
        output=profile.output if output is None else output,
        sbatch_args=profile.sbatch_args if sbatch_args is None else sbatch_args,
        preamble=profile.preamble if preamble is None else preamble,
        python=profile.python if python is None else python,
    )


def _expand(path: str, worker_id: str) -> str:
    """A configured path, with '%i' replaced by the worker id and a leading '~' expanded.

    Slurm has no pattern for the worker id, and expands no '~' of its own.
    """
    return os.path.expanduser(path.replace('%i', worker_id))


def _ip_address(host: str) -> t.Optional[t.Union[ipaddress.IPv4Address, ipaddress.IPv6Address]]:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _is_loopback_name(host: str) -> bool:
    if (addr := _ip_address(host)) is not None:
        return addr.is_loopback
    return host == 'localhost' or host.endswith('.localhost')


def _resolves_remotely(host: str) -> bool:
    """Whether `host` resolves to any address which isn't loopback"""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False
    return any(
        (addr := _ip_address(str(info[4][0]))) is not None and not addr.is_loopback
        for info in infos
    )


def split_netloc(netloc: str) -> t.Tuple[str, t.Optional[int]]:
    """Split 'host', 'host:port' or '[::1]:port' into a host and an optional port.

    Raises `ValueError` for a port which isn't a number.
    """
    if isinstance(_ip_address(netloc), ipaddress.IPv6Address):
        return (netloc, None)  # an unbracketed literal, which `urlsplit` can't parse
    split = urllib.parse.urlsplit(f'//{netloc}')
    try:
        return (split.hostname or '', split.port)
    except ValueError:
        raise ValueError(f"Invalid address '{netloc}' (port must be a number)") from None


def join_netloc(host: str, port: t.Optional[int]) -> str:
    if isinstance(_ip_address(host), ipaddress.IPv6Address):
        host = f'[{host}]'
    return host if port is None else f'{host}:{port}'


def _wildcard_host() -> str:
    """A name for this machine, for a server bound to every interface.

    A cluster's login node usually has a name which resolves everywhere; guessing an
    address from the machine's interfaces is worse than saying so.
    """
    hostname = socket.gethostname()

    for candidate in (hostname, socket.getfqdn()):
        if candidate and _resolves_remotely(candidate):
            return candidate

    raise ValueError(
         "Can't determine an address for remote workers: this machine's hostname"
        f" ('{hostname}') doesn't resolve to a remotely-reachable address."
         " Restart with '--host <hostname>' or specify 'worker_host' in the"
         " server config"
    )


def remote_worker_host(bind_host: str, override: t.Optional[str] = None) -> str:
    """
    Host a worker on another machine should use to reach a server bound to `bind_host`.

    Raises `ValueError` when the server isn't reachable remotely (bound to loopback, or
    to a wildcard address on a machine whose hostname resolves to loopback).
    """
    if override:
        return override

    host = bind_host.strip()
    addr = _ip_address(host)

    if host == '' or (addr is not None and addr.is_unspecified):
        # bound to every interface, so any name for this machine will do
        return _wildcard_host()

    if _is_loopback_name(host):
        raise ValueError(
            f"Server is bound to '{host}', which remote workers can't reach."
             " Restart with '--host 0.0.0.0' (or this machine's hostname)."
             " If this address is reachable, specify 'worker_host' in the"
             " server config as the address workers should use to reach it."
        )

    return host


def remote_worker_netloc(bind_netloc: str, override: t.Optional[str] = None) -> str:
    """`remote_worker_host`, keeping the port of `bind_netloc` unless `override` names one."""
    (bind_host, port) = split_netloc(bind_netloc)
    if override:
        (override_host, override_port) = split_netloc(override)
        return join_netloc(override_host, override_port or port)
    return join_netloc(remote_worker_host(bind_host), port)


__all__ = [
    'SlurmProfile', 'ServerConfig', 'SERVER_CONFIG', 'DEFAULT_PROFILE_NAME',
    'resolve_profile', 'apply_overrides', 'remote_worker_host', 'remote_worker_netloc',
]
