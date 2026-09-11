"""
Utilities for configuration
"""

from pathlib import Path
import functools
import logging
import textwrap
import typing as t

import pane
from pane.field import _MISSING  # bad

PaneClassT = t.TypeVar('PaneClassT', bound=pane.PaneBase)


@functools.cache
def get_config_dir() -> Path:
    import platformdirs
    return platformdirs.user_config_path(
        'phaser', roaming=True, use_site_for_root=True,
    )


def get_class_docstrings(cls: type) -> t.Dict[str, str]:
    """Extract attribute docstrings from class `cls`"""
    # TODO: can probably replace this with griffe or something
    import ast
    import inspect

    try:
        source = inspect.getsource(cls)
    except OSError:
        return {}

    classdef = t.cast(ast.ClassDef, ast.parse(source).body[0])
    assert classdef.name == cls.__name__

    d: t.Dict[str, str] = {}
    last_field: t.Optional[str] = None

    for stmt in classdef.body:
        if isinstance(stmt, ast.AnnAssign) and stmt.simple:
            last_field = t.cast(ast.Name, stmt.target).id
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            const = stmt.value.value
            if isinstance(const, str) and last_field:
                d[last_field] = const
        last_field = None
    return d


def _format_type(ty: t.Any) -> str:
    from types import UnionType

    origin = t.get_origin(ty)

    if origin is None:
        if ty is type(None) or ty is None:
            return "None"
        if isinstance(ty, type):
            return ty.__name__
        if isinstance(ty, str):
            return ty
        return repr(ty)

    args = t.get_args(ty)

    if origin is UnionType or origin is t.Union:
        return ' | '.join(map(_format_type, args))

    args = ', '.join(map(_format_type, args))
    return f'{_format_type(origin)}[{args}]'


def _pane_classes(ty: t.Any, seen: t.Optional[t.Set[t.Any]] = None) -> t.Iterator[t.Type[pane.PaneBase]]:
    """Pane classes reachable from the type `ty`, e.g. `SlurmProfile` in `Dict[str, SlurmProfile]`"""
    seen = set() if seen is None else seen
    if isinstance(ty, type) and issubclass(ty, pane.PaneBase):
        if ty in seen:
            return
        seen.add(ty)
        yield ty
        for field in ty.__pane_info__.fields:
            yield from _pane_classes(field.type, seen)
        return
    for arg in t.get_args(ty):
        yield from _pane_classes(arg, seen)


def _nested_docs(cls: t.Type[pane.PaneBase]) -> t.Iterator[str]:
    """Documentation lines for the fields of a nested pane class"""
    docstrings = get_class_docstrings(cls)
    yield ''
    yield f'{cls.__name__} fields:'

    for field in cls.__pane_info__.fields:
        if not field.init:
            continue
        name = f'{field.in_names[0]} ({_format_type(field.type)}):'
        doc = textwrap.dedent(docstrings.get(field.name, '')).strip('\n').splitlines()
        yield f'  {name} {doc[0]}' if doc else f'  {name}'
        for line in doc[1:]:
            yield f'    {line}'


@functools.cache
def _make_dumper() -> type:
    """YAML dumper which writes multi-line strings as block scalars"""
    import yaml
    try:
        from yaml import CSafeDumper as _Dumper
    except ImportError:
        from yaml import SafeDumper as _Dumper  # type: ignore

    base: t.Any = _Dumper

    class ConfigDumper(base):
        pass

    def repr_str(dumper: yaml.Dumper, data: str):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|' if '\n' in data else None)

    ConfigDumper.add_representer(str, repr_str)
    return ConfigDumper


class Config(t.Generic[PaneClassT]):
    def __init__(self, name: str, ty: t.Type[PaneClassT], example: t.Optional[PaneClassT] = None):
        self.config_name = name
        self.ty = ty
        self.example: t.Optional[PaneClassT] = example
        """Values rendered by `write_default`, in place of the field defaults"""

        if not issubclass(ty, pane.PaneBase):
            raise TypeError(f"Config type '{self.ty.__name__}' must be a pane class.")

    def path(self) -> Path:
        return get_config_dir() / f'{self.config_name}.yaml'

    def default(self) -> PaneClassT:
        try:
            return pane.from_data({}, self.ty)
        except pane.ConvertError as e:
            raise TypeError(
                f"Config type '{self.ty.__name__}' must be default constructible."
                " This is a bug with phaser."
            ) from e

    def get(self) -> PaneClassT:
        logger = logging.getLogger()
        path = self.path()
        logger.debug(f"Configuration path: '{path}'")

        if not path.exists():
            logger.debug("Configuration file not found, using default")
            return self.default()

        try:
            config = pane.from_yaml_all(path, self.ty)
        except pane.ConvertError as e:
            try:
                e.add_note("Invalid configuration file")
            except AttributeError: # <3.11
                raise ValueError("Invalid configuration file") from e
            raise
        except Exception as e:
            try:
                e.add_note("Failed to read configuration file")
            except AttributeError: # <3.11
                raise ValueError("Failed to read configuration file") from e
            raise
        if not len(config):
            return self.default()
        if len(config) > 1:
            raise ValueError("Invalid configuration file (multiple YAML documents)")
        return config[0]

    def write_default(self) -> bool:
        """
        Write a default configuration file.

        Documentation is commented with '##', and the settings themselves with '# ', so
        uncommenting a setting is unambiguous.

        Does nothing and returns `False` if the file already exists
        """
        logger = logging.getLogger()
        path = self.path()
        if path.exists():
            return False
        logger.info(f"Creating default config file at '{path}'")

        import yaml
        dumper = _make_dumper()
        docstrings = get_class_docstrings(self.ty)
        lines: t.List[str] = []
        documented: t.Set[t.Type[pane.PaneBase]] = set()

        for field in self.ty.__pane_info__.fields:
            if not field.init:
                continue

            if self.example is not None:
                default = getattr(self.example, field.name)
            elif not field.has_default():
                continue
            elif field.default is _MISSING:
                if field.default_factory is None:
                    continue
                default = field.default_factory()
            else:
                default = field.default

            doc_lines: t.List[str] = []
            if (docstring := docstrings.get(field.name)):
                doc_lines.extend(textwrap.dedent(docstring).strip('\n').splitlines())
            doc_lines.append(f'type: {_format_type(field.type)}')

            for nested in _pane_classes(field.type):
                if nested not in documented:
                    documented.add(nested)
                    doc_lines.extend(_nested_docs(nested))

            lines.extend(f'## {line}'.rstrip() for line in doc_lines)

            expr = t.cast(str, yaml.dump(
                {field.in_names[0]: pane.into_data(default, field.type)},
                Dumper=dumper, explicit_start=False, allow_unicode=True,
                # block style, declaration order, and no line wrapping: the result is read
                # and edited as a comment block, where a rewrapped value is hard to follow
                default_flow_style=False, sort_keys=False, width=2**31 - 1,
            )).strip('\n')
            lines.extend(f'# {line}'.rstrip() for line in expr.splitlines())
            lines.append('')

        try:
            path.parent.mkdir(parents=False, exist_ok=True)
            path.write_text('\n'.join(lines))
        except Exception as e:
            e.add_note("Failed to write config file")
            raise
        return True


__all__ = [
    'get_config_dir', 'Config',
]