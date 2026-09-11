import importlib
import typing as t

import click

from phaser.utils.config import Config

CONFIGS: t.Dict[str, str] = {
    'server': 'phaser.web.config:SERVER_CONFIG',
    'process_empad': 'phaser.cli.process_empad:CONFIG',
}
"""Known configuration files, as 'module:attribute' references to `Config` objects."""


def _get_config(name: str) -> Config[t.Any]:
    (module, _, attr) = CONFIGS[name].partition(':')
    return getattr(importlib.import_module(module), attr)


_NAME = click.argument('name', type=click.Choice(list(CONFIGS)))


@click.group()
def config():
    """Manage configuration files"""


@config.command('list')
def list_configs():
    """List configuration files"""
    for name in CONFIGS:
        path = _get_config(name).path()
        click.echo(f"{name}: {path}" + ("" if path.exists() else " (not created)"))


@config.command()
@_NAME
def path(name: str):
    """Print the path to a configuration file"""
    click.echo(str(_get_config(name).path()))


@config.command()
@_NAME
def init(name: str):
    """Write a commented configuration file, if none exists"""
    config = _get_config(name)
    if not config.write_default():
        raise click.ClickException(f"Config file already exists at '{config.path()}'")
    click.echo(f"Wrote '{config.path()}'")


@config.command()
@_NAME
def check(name: str):
    """Validate a configuration file, printing the settings it produces"""
    import yaml

    import pane

    config = _get_config(name)
    try:
        value = config.get()
    except Exception as e:
        raise click.ClickException(f"Invalid config file '{config.path()}': {e}") from e

    click.echo(f"# {config.path()}")
    click.echo(yaml.safe_dump(pane.into_data(value, config.ty), sort_keys=False, default_flow_style=False).strip())
