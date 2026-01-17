from pathlib import Path
import sys
import typing as t

import click


class MainCommand(click.MultiCommand):
    def __init__(self, commands: t.Iterable[t.Union[click.Command, t.Tuple[str, str]]], **kwargs):
        super().__init__(**kwargs)
        # name: command or short_help
        self.commands: t.Dict[str, t.Union[click.Command, str, None]]

        self.commands = dict(
            (t.cast(str, c.name), c) if isinstance(c, click.Command) else (c[0], c[1])
            for c in commands
        )

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

        # for now, assume `validate` command is `cli.validate:validate`
        # TODO, incorporate some feature checks here
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
def serve(host: str = 'localhost', port: t.Optional[int] = None, verbose: int = 0):
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

    server.run(hostname=host, port=port, verbosity=verbose)


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


commands: t.List[t.Union[click.Command, t.Tuple[str, str]]] = [
    run, serve, worker,
    # these will be looked up in the cli folder
    ('validate', "Validate reconstruction plan file"),
]


@click.command(cls=MainCommand, commands=commands)
def cli():
    pass