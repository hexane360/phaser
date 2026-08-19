import abc
import importlib
from pathlib import Path
import sys
import typing as t

import click

class Dependency(abc.ABC):
    @abc.abstractmethod
    def check(self):
        ...

    @abc.abstractmethod
    def install_instructions(self) -> str:
        ...


class ImportDependency(Dependency):
    def __init__(self, ref: str, install: str) -> None:
        self.ref = ref
        self.install = install

    def check(self):
        importlib.import_module(self.ref)

    def install_instructions(self) -> str:
        return self.install


def check_dependencies(command: str, dependencies: t.Sequence[str]):
    for dependency in dependencies:
        if (dep := _DEPENDENCIES.get(dependency)) is None:
            raise RuntimeError(f"Unknown dependency '{dependency}'. This is likely a bug in the hook declaration.")

        try:
            dep.check()
        except Exception as e:
            raise RuntimeError(
                f"Missing dependency '{dependency}' required for subcommand '{command}'.\n"
                f"To install: {dep.install_instructions()}"
            ) from e


class MainCommand(click.MultiCommand):
    def __init__(self, commands: t.Iterable[
        t.Union[click.Command, t.Tuple[str, str], t.Tuple[str, str, t.Sequence[str]]]
    ], **kwargs):
        super().__init__(**kwargs)
        # name: command or short_help
        self.commands: t.Dict[str, t.Union[click.Command, str, None]] = {}
        self.dependencies: t.Dict[str, t.Sequence[str]] = {}
        for c in commands:
            if isinstance(c, click.Command):
                name = str(c.name)
                self.commands[name] = c
                self.dependencies[name] = ()
                continue
            self.commands[c[0]] = c[1]
            if len(c) > 2:
                self.dependencies[c[0]] = c[2]

    def list_commands(self, ctx: click.Context):
        return list(self.commands.keys())

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        from gettext import gettext

        if len(self.commands):
            limit = formatter.width - 6 - max(map(len, self.commands.keys()))

            rows = []
            for name, cmd in self.commands.items():
                help = cmd.get_short_help_str(limit) if isinstance(cmd, click.Command) else cmd
                rows.append((name, help))

            if rows:
                with formatter.section(gettext("Commands")):
                    formatter.write_dl(rows)

    def get_command(self, ctx: click.Context, cmd_name: str) -> t.Optional[click.Command]:
        name = cmd_name.lower()
        val = (self.commands.get(name) or 
               self.commands.get(name.replace('-', '_')))
        if val is None:
            return None
        if isinstance(val, click.BaseCommand):
            return val

        check_dependencies(name, self.dependencies.get(name, ()))

        # for now, assume `validate` command is `cli.validate:validate`
        module = func = name
        mod = __import__(module, globals(), fromlist=[func], level=1)
        return getattr(mod, func)


@click.command()
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
def run(path: t.Union[str, Path]):
    """Execute a reconstruction plan"""
    from phaser.plan import ReconsPlan
    from phaser.execute import execute_plan
    plans = ReconsPlan.from_yaml_all(path)

    for plan in plans:
        execute_plan(plan)


@click.command()
@click.option('--host', type=str, default='localhost', help="Host to serve on")
@click.option('--port', type=int, help="Port to serve on")
@click.option('-v', '--verbose', count=True, help="Increase verbosity")
@click.option('--debug', is_flag=True, help="Enable debug routes (/debug/*), for testing the web interface")
def serve(host: str = 'localhost', port: t.Optional[int] = None, verbose: int = 0, debug: bool = False):
    """Run phaser server"""
    from phaser.web.server import server

    if ':' in host:
        (host, port_from_host) = host.rsplit(':', maxsplit=1)
        try:
            port_from_host = int(port_from_host)
        except ValueError:
            print(f"Invalid host '{host}:{port_from_host}'", file=sys.stderr)
            sys.exit(1)

        port = port or port_from_host

    server.run(hostname=host, port=port, verbosity=verbose, debug=debug)


@click.command()
@click.argument('url', type=str, required=True)
@click.option('--quiet/--loud', default=False, help="Whether to print output to stdout")
def worker(url: str, quiet: bool = False):
    """
    Run phaser worker.

    URL is the server URL to connect to.
    """
    from phaser.web.worker import run_worker

    run_worker(url, quiet=quiet)


_DEPENDENCIES = {
    'rsciio': ImportDependency('rsciio', "'pip install rosettasciio' or 'conda install rosettasciio'"),
}

commands: t.List[t.Union[click.Command, t.Union[t.Tuple[str, str, t.Sequence[str]], t.Tuple[str, str]]]] = [
    run, serve, worker,
    # these will be looked up in the cli folder
    ('validate', "Validate reconstruction plan file"),
    ('process_empad', "Process EMPAD XML metadata"),
    ('calc_drift', "Calculate and correct linear drift"),
]


@click.command(cls=MainCommand, commands=commands)
def cli():
    pass