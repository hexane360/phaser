from dataclasses import dataclass
from functools import lru_cache
import typing as t

import numpy
import pane
from pane.converters import Converter, make_converter, ConverterHandlers, ErrorNode
from pane.annotations import ConvertAnnotation
from pane.errors import ParseInterrupt, WrongTypeError
from pane.util import pluralize, list_phrase
from typing_extensions import Self

if t.TYPE_CHECKING:
    from phaser.hooks.schedule import FlagArgs, FlagLike, ScheduleLike

T = t.TypeVar('T')

@t.overload
def cast_length(val: t.Iterable[T], n: t.Literal[5]) -> t.Tuple[T, T, T, T, T]:
    ...

@t.overload
def cast_length(val: t.Iterable[T], n: t.Literal[4]) -> t.Tuple[T, T, T, T]:
    ...

@t.overload
def cast_length(val: t.Iterable[T], n: t.Literal[3]) -> t.Tuple[T, T, T]:
    ...

@t.overload
def cast_length(val: t.Iterable[T], n: t.Literal[2]) -> t.Tuple[T, T]:
    ...

@t.overload
def cast_length(val: t.Iterable[T], n: t.Literal[1]) -> t.Tuple[T]:
    ...

def cast_length(val: t.Iterable[T], n: int) -> t.Tuple[T, ...]:
    return tuple(val)


class _EmptyDictAnnotation(ConvertAnnotation, Converter[t.Dict[t.NoReturn, t.NoReturn]]):
    def _converter(self, inner_type: t.Any, *, handlers: ConverterHandlers):
        return self

    def __eq__(self, other):
        return self.__class__ is other.__class__

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)

    def expected(self, plural: bool = False) -> str:
        return pluralize("empty dict", plural, article='an')

    def try_convert(self, val: t.Any) -> t.Dict[t.NoReturn, t.NoReturn]:
        if isinstance(val, (dict, t.Mapping)) and len(val) == 0:
            return {}
        raise ParseInterrupt()

    def collect_errors(self, val: t.Any) -> t.Optional[WrongTypeError]:
        if isinstance(val, dict) and len(val) == 0:
            return None
        return WrongTypeError(self.expected(), val)


class _ReconsVarsAnnotation(ConvertAnnotation):
    def _converter(self, inner_type: t.Any, *, handlers: ConverterHandlers):
        return _ReconsVarsConverter(inner_type, handlers)

    def __eq__(self, other):
        return self.__class__ is other.__class__

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)


BackendName: t.TypeAlias = t.Literal['cuda', 'cupy', 'jax', 'cpu', 'numpy']
ReconsVar: t.TypeAlias = t.Literal['object', 'probe', 'positions', 'tilt']

ReconsVars: t.TypeAlias = t.Annotated[t.FrozenSet[ReconsVar], _ReconsVarsAnnotation()]
EmptyDict: t.TypeAlias = t.Annotated[t.Dict[t.NoReturn, t.NoReturn], _EmptyDictAnnotation()]


class Cancelled(BaseException):
    ...


class Dataclass(pane.PaneBase, kw_only=True, allow_extra=False):
    ...


class SliceList(Dataclass):
    thicknesses: t.List[float]

class SliceStep(Dataclass):
    n: int
    slice_thickness: float

    @property
    def zs(self) -> t.List[float]:
        return [float(z) for z in numpy.arange(self.n) * self.slice_thickness]

    @property
    def thicknesses(self) -> t.List[float]:
        return [self.slice_thickness] * self.n

class SliceTotal(Dataclass):
    n: int
    total_thickness: float

    @property
    def zs(self) -> t.List[float]:
        return [float(z) for z in numpy.arange(self.n) * self.total_thickness / self.n]

    @property
    def thicknesses(self) -> t.List[float]:
        return [self.total_thickness / self.n] * self.n

Slices: t.TypeAlias = t.Union[SliceList, SliceStep, SliceTotal]


class Flag(Dataclass):
    after: int = 0
    every: int = 1
    before: t.Optional[int] = None

    def any_true(self, niter: int) -> bool:
        return (
            self.after < niter
            and (self.before is None or self.before > 0)
        )

    def __call__(self, args: 'FlagArgs') -> bool:
        i = args['state'].iter.engine_iter
        return (
            (self.before is None or i < self.before)
            and i > self.after
            and (i - self.after) % self.every == 0
        )

    def resolve(self) -> Self:
        return self


class _ConstFlag:
    def __init__(self, val: bool):
        self.val = val

    def __call__(self, args: 'FlagArgs') -> bool:
        return self.val


@lru_cache
def process_flag(flag: 'FlagLike') -> t.Callable[['FlagArgs'], bool]:
    if isinstance(flag, bool):
        return _ConstFlag(flag)
    return flag


@lru_cache
def process_schedule(schedule: 'ScheduleLike') -> t.Callable[['FlagArgs'], float]:
    if isinstance(schedule, (int, float)):
        return lambda _: schedule
    return schedule


def flag_any_true(flag: t.Callable[['FlagArgs'], bool], niter: int) -> bool:
    if isinstance(flag, Flag):
        return flag.any_true(niter)
    elif isinstance(flag, _ConstFlag):
        return flag.val
    # assume flag will return true
    return True


@dataclass(init=False, frozen=True)
class IsVersion(ConvertAnnotation):
    min: t.Optional[t.Tuple[int, ...]] = None
    max: t.Optional[t.Tuple[int, ...]] = None

    def __init__(self, *,
                 min: t.Union[str, t.Tuple[int, ...], None] = None,
                 max: t.Union[str, t.Tuple[int, ...], None] = None,
                 exactly: t.Union[str, t.Tuple[int, ...], None] = None):
        if exactly is not None:
            if min is not None or max is not None:
                raise TypeError("'exactly' cannot be specified with 'min' or 'max'")
            min = max = _VersionConverter.parse_version(exactly)
        if min is not None:
            min = _VersionConverter.parse_version(min)
        if max is not None:
            max = _VersionConverter.parse_version(max)
        object.__setattr__(self, 'min', min)
        object.__setattr__(self, 'max', max)

    def _converter(self, inner_type: t.Any, *,
                   handlers: ConverterHandlers):
        return _VersionConverter(inner_type, handlers, min=self.min, max=self.max)


Version: t.TypeAlias = t.Annotated[str, IsVersion()]


class _VersionConverter(Converter[t.Any]):
    def __init__(self, inner_type: t.Any, handlers: ConverterHandlers,
                 min: t.Optional[t.Tuple[int, ...]] = None,
                 max: t.Optional[t.Tuple[int, ...]] = None):
        self.inner = make_converter(inner_type, handlers)
        self.min = min
        self.max = max

    def expected(self, plural: bool = False) -> str:
        if self.min is not None and self.max is not None:
            min_s = '.'.join(map(str, self.min))
            max_s = '.'.join(map(str, self.max))
            if self.min == self.max:
                return f"{pluralize('version', plural)} {min_s}"
            return f"{pluralize('version', plural)} between {min_s} and {max_s}"
        elif self.min is not None:
            min_s = '.'.join(map(str, self.min))
            return f"{pluralize('version', plural)} at least {min_s}"
        elif self.max is not None:
            max_s = '.'.join(map(str, self.max))
            return f"{pluralize('version', plural)} at most {max_s}"
        else:
            return f"version {pluralize('string', plural)}"

    def into_data(self, val: t.Any) -> str:
        return str(val)

    @staticmethod
    def parse_version(val: t.Union[str, t.Tuple[int, ...]]) -> t.Tuple[int, ...]:
        def to_int(seg: t.Union[int, str]) -> int:
            if isinstance(seg, int):
                return seg
            seg = seg.strip()
            if not seg.isdigit():
                raise ValueError()
            return int(seg)

        return tuple(map(to_int, val.split('.') if isinstance(val, str) else val))

    def check_version(self, val: t.Tuple[int, ...]):
        if self.min == self.max:
            if self.min is not None and val != self.min:
                raise ValueError(f"Version {'.'.join(map(str, val))} is not supported version {'.'.join(map(str, self.min))}")
        elif self.min is not None and val < self.min:
            raise ValueError(f"Version {'.'.join(map(str, val))} less than minimum supported version {'.'.join(map(str, self.min))}")
        elif self.max is not None and val > self.max:
            raise ValueError(f"Version {'.'.join(map(str, val))} greater than maximum supported version {'.'.join(map(str, self.max))}")

    def try_convert(self, val: t.Any) -> str:
        s = self.inner.try_convert(val)
        try:
            version = self.parse_version(val)
            self.check_version(version)
        except ValueError:
            raise ParseInterrupt()
        return s

    def collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        try:
            s = self.inner.try_convert(val)
        except ParseInterrupt:
            return WrongTypeError(self.expected(), val)
        try:
            version = self.parse_version(s)
        except ValueError:
            return WrongTypeError(self.expected(), val, info='Invalid version string')
        try:
            self.check_version(version)
        except ValueError as e:
            return WrongTypeError(self.expected(), val, info=e.args[0])
        return None


class _ReconsVarsConverter(Converter[t.FrozenSet[ReconsVar]]):
    def __init__(self, ty: type, handlers: ConverterHandlers):
        self.inner = make_converter(ty, handlers)

    def expected(self, plural: bool = False) -> str:
        known_params = t.get_args(ReconsVar)
        return f"{pluralize('set', plural, article='a')} of comma-separated " \
            f"variables ({list_phrase(tuple(map(repr, known_params)))})"

    def into_data(self, val: t.Any) -> str:
        return ", ".join(self.inner.into_data(val))  # type: ignore

    def try_convert(self, val: t.Any) -> t.FrozenSet[ReconsVar]:
        if isinstance(val, str):
            val = tuple(v.strip() for v in val.split(","))

        return self.inner.try_convert(val)

    def collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        if isinstance(val, str):
            val = tuple(v.strip() for v in val.split(","))

        return self.inner.collect_errors(val)


__all__ = [
    'BackendName', 'Dataclass', 'Slices', 'Flag',
    'process_flag', 'flag_any_true',
]