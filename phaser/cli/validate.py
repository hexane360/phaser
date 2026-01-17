from contextlib import nullcontext
from pathlib import Path
import sys
import typing as t

import click


@click.command()
@click.argument('path', type=click.Path(allow_dash=True), default='-')
@click.option('--json/--no-json', default=False,
              help="Output validation result in JSON format")
def validate(path: t.Union[str, Path], json: bool = False):
    """
    Validate reconstruction plan file.

    PATH is the path to a YAML reconstruction plan,
    or '-' (default) to read from stdin.
    """
    from phaser.plan import ReconsPlan

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

    if len(plans) == 1:
        print("Validation of plan successful!", file=sys.stderr)
    else:
        print(f"Validation of {len(plans)} plans successful!", file=sys.stderr)

    if json:
        from json import dump
        dump({
            'result': 'success',
            'plans': [(plan.name, plan.into_data()) for plan in plans],
        }, sys.stdout)
        print()