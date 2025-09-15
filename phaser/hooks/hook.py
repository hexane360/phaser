from __future__ import annotations

import abc
import importlib
import typing as t

from pane.convert import ConverterHandlers, DataType
from pane.converters import Converter, make_converter
from pane.errors import ErrorNode, WrongTypeError, ParseInterrupt, ProductErrorNode

T = t.TypeVar('T')
U = t.TypeVar('U')

class Hook(t.Generic[T, U], abc.ABC):
    known: t.ClassVar[t.Dict[str, t.Union[t.Tuple[str, type], t.Tuple[str, type, t.Tuple[str, ...]]]]] = {}

    def __init__(
        self, ref: str, props: t.Optional[t.Any] = None, type: t.Optional[str] = None,
        dependencies: t.Tuple[str, ...] = ()
    ):
        self.ref: str = ref
        self.type: t.Optional[str] = type
        self.f: t.Optional[t.Callable[..., U]] = None
        self.props: t.Optional[t.Any] = props
        self.dependencies: t.Tuple[str, ...] = dependencies

    def func_ref(self) -> str:
        if self.type is not None:
            return self.type
        return self.ref

    def _resolve_ref(self) -> t.Callable:
        if ':' not in self.ref:
            if self.ref in globals():
                return globals()[self.ref]
            raise ValueError(f"Can't resolve function reference '{self.ref}'.")

        if self.dependencies is not None:
            from ._dependencies import check_dependencies
            check_dependencies(self.dependencies, self.func_ref())

        (module_path, func_name) = self.ref.split(':')
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            e.add_note(f"While resolving function reference {self.ref}")
            raise

        try:
            f = getattr(module, func_name)
        except AttributeError:
            raise AttributeError(f"No function '{func_name}' found in module '{module_path}'")

        return f

    def resolve(self) -> t.Callable[..., U]:
        if self.f is None:
            self.f = self._resolve_ref()
        return self.f

    def __call__(self, args: T) -> U:
        return self.resolve()(args, props=self.props if self.props is not None else {})

    def __getattr__(self, key: t.Any) -> t.Any:
        if isinstance(self.props, dict):
            try:
                return self.props[key]
            except KeyError:
                raise AttributeError(name=key, obj=self.props)
        return getattr(self.props, key)

    def __repr__(self) -> str:
        if self.props is not None:
            return f"FuncRef({self.func_ref()!r}, {self.props!r})"
        return f"FuncRef({self.func_ref()!r})"

    @classmethod
    def _converter(cls, *args: type, handlers: ConverterHandlers) -> HookConverter[T, U]:
        return HookConverter(cls, handlers)


def _to_dict(val: t.Any) -> dict:
    import dataclasses
    import pane

    if isinstance(val, dict):
        return val
    if dataclasses.is_dataclass(val):
        return dataclasses.asdict(val)  # type: ignore
    if isinstance(val, pane.PaneBase):
        return val.into_data()         # type: ignore
    return val.__dict__


class HookConverter(t.Generic[T, U], Converter[Hook[T, U]]):
    def __init__(self, cls: t.Type[Hook[T, U]], handlers: ConverterHandlers):
        self.cls = cls
        self.inner: Converter[t.Union[str, t.Dict[str, t.Any]]] = make_converter(t.Union[str, t.Dict[str, t.Any]], handlers)

    def expected(self, plural: bool = False) -> str:
        if plural:
            return "hooks to functions"
        return "hook to function"

    def into_data(self, val: Hook[T, U]) -> DataType:
        if val.props is not None:
            return {
                'type': val.func_ref(),
                **_to_dict(val.props)
            }
        return val.func_ref()

    def try_convert(self, val: t.Any) -> Hook[T, U]:
        val = self.inner.try_convert(val)
        if isinstance(val, str):
            ref = val
            props = {}
        else:
            if 'type' not in val:
                raise ParseInterrupt()
            ref = str(val.pop('type'))
            props = val

        dependencies = ()

        if ref in self.cls.known:
            ty = ref
            (ref, props_ty, *dep) = self.cls.known[ty]
            if len(dep):
                dependencies = dep[0]

            converter = make_converter(props_ty)
            props = converter.try_convert(props)
        elif ':' not in ref:
            raise ParseInterrupt()
        else:
            ty = None

        return self.cls(ref, props, ty, dependencies=dependencies)

    def collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        try:
            val = self.inner.try_convert(val)
        except ParseInterrupt:
            return self.inner.collect_errors(val)
        if isinstance(val, str):
            ref = val
            props = {}
        else:
            if 'type' not in val:
                return ProductErrorNode(self.expected(), {}, val, set(['type']))
            ref = str(val.pop('type'))
            props = val

        if ref in self.cls.known:
            ty = ref
            (ref, props_ty, *dep) = self.cls.known[ty]

            converter = make_converter(props_ty)
            try:
                props = converter.try_convert(props)
            except ParseInterrupt:
                return converter.collect_errors(props)
        elif ':' not in ref:
            return WrongTypeError(
                self.expected(), ref,
                info=f"Known hooks: '{', '.join(self.cls.known.keys())}'"
            )

        return None