import typing as t

import pane
import pytest

from phaser.utils.config import Config


class Nested(pane.PaneBase):
    description: t.Optional[str] = None
    """What this is for"""

    preamble: str = ''
    """
    Shell run first.
    Arbitrary code.
    """


class ExampleConfig(pane.PaneBase):
    name: t.Optional[str] = None
    """Name of the thing"""

    scale: float = 1.0
    """Scale factor"""

    nested: t.Dict[str, Nested] = {}
    """Nested configuration"""


EXAMPLE = ExampleConfig(
    name='example', scale=2.5,
    nested={'a': Nested(description='first', preamble='module load cuda\nconda activate env\n')},
)


def _uncomment(text: str) -> str:
    """The settings of a generated config file, with the documentation ('##') dropped"""
    return '\n'.join(
        line[2:] for line in text.splitlines()
        if line.startswith('# ') or line == '#'
    ) + '\n'


@pytest.fixture
def config_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('phaser.utils.config.get_config_dir', lambda: tmp_path)
    return tmp_path


def test_write_default_commented(config_dir):
    config = Config('test', ExampleConfig)
    assert config.write_default()
    assert not config.write_default()  # doesn't overwrite

    text = config.path().read_text()
    assert all(not line.strip() or line.startswith('#') for line in text.splitlines())
    assert '## Name of the thing' in text and '## type: str | None' in text
    assert '# name: null' in text  # settings are commented with '# ', documentation with '##'

    # everything is commented out, so the file as written is the default config
    assert config.get() == ExampleConfig()


def test_write_default_documents_nested_classes(config_dir):
    """A nested pane class isn't reachable from the top-level docstrings, so it's documented too"""
    config = Config('test', ExampleConfig)
    assert config.write_default()

    text = config.path().read_text()
    assert '## Nested fields:' in text
    assert '##   description (str | None): What this is for' in text
    # a multi-line docstring is indented under its field
    assert '##   preamble (str): Shell run first.\n##     Arbitrary code.' in text


def test_write_default_example_roundtrips(config_dir):
    """The example template, uncommented, parses back to the example it was rendered from"""
    config = Config('test', ExampleConfig, example=EXAMPLE)
    assert config.write_default()

    text = config.path().read_text()
    # multi-line strings are written as block scalars rather than escaped one-liners
    assert 'preamble: |' in text
    assert '\\n' not in text

    config.path().write_text(_uncomment(text))
    assert config.get() == EXAMPLE


def test_get_missing_and_invalid(config_dir):
    config = Config('test', ExampleConfig)
    assert config.get() == ExampleConfig()  # no file

    config.path().write_text('scale: not-a-number\n')
    with pytest.raises(pane.ConvertError):
        config.get()

    config.path().write_text('---\nscale: 1.\n---\nscale: 2.\n')
    with pytest.raises(ValueError, match='multiple YAML documents'):
        config.get()
