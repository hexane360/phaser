"""`phaser config` command group."""

import importlib
import typing as t

import click

from phaser.utils.config import Config

CONFIGS: t.Dict[str, str] = {
    'server': 'phaser.web.config:SERVER_CONFIG',
    'process_empad': 'phaser.cli.process_empad:CONFIG',
}
"""Known configuration files, as 'module:attribute' references to `Config` objects."""


def _get_config(name: str, variant: t.Optional[str] = None) -> Config[t.Any]:
    (module, _, attr) = CONFIGS[name].partition(':')
    config: Config[t.Any] = getattr(importlib.import_module(module), attr)
    if variant is None:
        return config
    try:
        return config.variant(variant)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="'--name'") from e


_NAME = click.argument('name', type=click.Choice(list(CONFIGS)))
_VARIANT = click.option('--name', 'variant', type=str, default=None,
                        help="Named variant of the configuration, stored in '{NAME}_{variant}.yaml'")


@click.group()
def config():
    """Manage configuration files"""


@config.command('list')
def list_configs():
    """List configuration files"""
    for name in CONFIGS:
        config = _get_config(name)
        path = config.path()
        click.echo(f"{name}: {path}" + ("" if path.exists() else " (not created)"))
        for variant in config.variants():
            click.echo(f"{name} --name {variant}: {config.variant(variant).path()}")


@config.command()
@_NAME
@_VARIANT
def path(name: str, variant: t.Optional[str] = None):
    """Print the path to a configuration file"""
    click.echo(str(_get_config(name, variant).path()))


@config.command()
@_NAME
@_VARIANT
def init(name: str, variant: t.Optional[str] = None):
    """Write a commented configuration file, if none exists"""
    config = _get_config(name, variant)
    if not config.write_default():
        raise click.ClickException(f"Config file already exists at '{config.path()}'")
    click.echo(f"Wrote '{config.path()}'")


@config.command()
@_NAME
@_VARIANT
def check(name: str, variant: t.Optional[str] = None):
    """Validate a configuration file, printing the settings it produces"""
    import yaml

    import pane

    config = _get_config(name, variant)
    if variant is not None and not config.path().exists():
        raise click.ClickException(f"Config '{variant}' does not exist at '{config.path()}'")
    try:
        value = config.get()
    except Exception as e:
        raise click.ClickException(f"Invalid config file '{config.path()}': {e}") from e

    click.echo(f"# {config.path()}")
    click.echo(yaml.safe_dump(pane.into_data(value, config.ty), sort_keys=False, default_flow_style=False).strip())
