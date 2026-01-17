from pathlib import Path
import sys
import typing as t

import click

@click.group()
def cli():
    pass


@cli.command('run')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.option('--raise-on-warn/--no-raise-on-warn')
def run(path: t.Union[str, Path], *, raise_on_warn: bool = False):
    from .plan import ReconsPlan
    from .execute import execute_plan

    if raise_on_warn:
        import warnings
        warnings.simplefilter('error')

    plans = ReconsPlan.from_yaml_all(path)

    for plan in plans:
        execute_plan(plan)


@cli.command('serve')
@click.option('--host', type=str, default='localhost')
@click.option('--port', type=int)
@click.option('-v', '--verbose', count=True)
def serve(host: str = 'localhost', port: t.Optional[int] = None, verbose: int = 0):
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


@cli.command('validate')
@click.argument('path', type=click.Path(allow_dash=True), default='-')
@click.option('--json/--no-json', default=False)
def validate(path: t.Union[str, Path], json: bool = False):
    from contextlib import nullcontext
    from .plan import ReconsPlan

    try:
        if path == '-':
            file = nullcontext(sys.stdin)
        else:
            file = open(Path(path).expanduser(), 'r')

        with file as file:
            plans = ReconsPlan.from_yaml_all(file)
    except Exception as e:
        print(f"Validation failed:\n{e}", file=sys.stderr)

        if json:
            from json import dump
            dump({'result': 'error', 'error': str(e)}, sys.stdout)
            print()

        sys.exit(1)

    if json:
        from json import dump, dumps

        def _serialize_complex(val: t.Any) -> t.Any:
            if isinstance(val, complex):
                return {'re': val.real, 'im': val.imag}
            raise TypeError()

        try:
            s = dumps({
                'result': 'success',
                'plans': [(plan.name, plan.into_data()) for plan in plans],
            }, default=_serialize_complex)
        except Exception as e:
            print(f"Failed to serialize validated plans: {e}", file=sys.stderr)
            dump({'result': 'error', 'error': str(e)}, sys.stdout)
            print()
            sys.exit(2)

        sys.stdout.write(s)
        print()

    if len(plans) == 1:
        print("Validation of plan successful!", file=sys.stderr)
    else:
        print(f"Validation of {len(plans)} plans successful!", file=sys.stderr)


@cli.command('worker')
@click.argument('url', type=str, required=True)
@click.option('--quiet/--loud', default=False)
def worker(url: str, quiet: bool = False):
    from phaser.web.worker import run_worker

    run_worker(url, quiet=quiet)


if __name__ == '__main__':
    cli()