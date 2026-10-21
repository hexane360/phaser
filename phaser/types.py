"""Shared types and plan-file annotations."""

import typing as t
from dataclasses import dataclass
from functools import lru_cache

import numpy
from typing_extensions import Self

import pane
from pane.annotations import Condition, ConvertAnnotation
from pane.converters import Converter, ConverterHandlers, ErrorNode, make_converter
from pane.errors import ParseInterrupt, WrongTypeError
from pane.util import list_phrase, pluralize

if t.TYPE_CHECKING:
    from phaser.hooks.schedule import Flag, FlagArgs, FlagLike, Schedule, ScheduleLike
    from phaser.state import ReconsState

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
    """Return `val` as a tuple, typed as having length `n`. The length is not checked."""
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


BackendName: t.TypeAlias = t.Literal['cupy', 'jax', 'torch', 'numpy']
"""Array backend names."""
ReconsVar: t.TypeAlias = t.Literal['object', 'probe', 'positions', 'tilt', 'positions_affine', 'positions_line']
"""Reconstruction variables, including synthetic variables."""

ReconsVars: t.TypeAlias = t.Annotated[t.FrozenSet[ReconsVar], _ReconsVarsAnnotation()]
"""Set of `ReconsVar`. Parsed from a comma-separated string in plan files."""
EmptyDict: t.TypeAlias = t.Annotated[t.Dict[t.NoReturn, t.NoReturn], _EmptyDictAnnotation()]
"""Empty dict."""


class EarlyTermination(BaseException):
    """Raised by an observer to end the current engine early.

    Args:
        state: State at termination.
        continue_next_engine: Whether to continue with the next engine.
    """
    def __init__(self, state: 'ReconsState', continue_next_engine: bool = False):
        self.state: ReconsState = state
        self.continue_next_engine: bool = continue_next_engine


class Dataclass(pane.PaneBase, kw_only=True, allow_extra=False):
    """Base class for plan dataclasses. Fields are keyword-only and unknown keys are rejected."""


class SliceList(Dataclass):
    """Slices given by explicit thicknesses."""
    thicknesses: t.List[float]
    """Thickness of each slice [angstrom]."""

class SliceStep(Dataclass):
    """Slices of equal thickness, given by thickness."""
    n: int
    """Number of slices."""
    slice_thickness: float
    """Thickness of each slice [angstrom]."""

    @property
    def zs(self) -> t.List[float]:
        """Position of the beginning of each slice [angstrom]."""
        return [float(z) for z in numpy.arange(self.n) * self.slice_thickness]

    @property
    def thicknesses(self) -> t.List[float]:
        """Thickness of each slice [angstrom]."""
        return [self.slice_thickness] * self.n

class SliceTotal(Dataclass):
    """Slices of equal thickness, given by total thickness."""
    n: int
    """Number of slices."""
    total_thickness: float
    """Total thickness of all slices [angstrom]."""

    @property
    def zs(self) -> t.List[float]:
        """Position of the beginning of each slice [angstrom]."""
        return [float(z) for z in numpy.arange(self.n) * self.total_thickness / self.n]

    @property
    def thicknesses(self) -> t.List[float]:
        """Thickness of each slice [angstrom]."""
        return [self.total_thickness / self.n] * self.n

Slices: t.TypeAlias = t.Union[SliceList, SliceStep, SliceTotal]
"""Specification of object slices."""


class ComplexCartesian(pane.PaneBase, kw_only=True):
    """Complex number in Cartesian form."""
    re: float
    """Real part."""
    im: float = 0.0
    """Imaginary part."""

    def __complex__(self) -> complex:
        return complex(self.re, self.im)

class ComplexPolar(pane.PaneBase, kw_only=True):
    """Complex number in polar form."""
    mag: float
    """Magnitude."""
    angle: float = 0.0
    """Phase angle [degrees]."""

    def __complex__(self) -> complex:
        theta = numpy.deg2rad(self.angle)
        return self.mag * complex(numpy.cos(theta), numpy.sin(theta))


class Krivanek(pane.PaneBase):
    """Aberration of radial order `n` and azimuthal order `m` in Krivanek notation."""
    n: int
    """Radial order."""
    m: int
    """Azimuthal order."""
    scale_factor: float = 1.0
    """Scale factor (populated when converting from Haider notation)."""

    def __post_init__(self):
        if (
            self.n < 0 or self.m < 0 or
            self.m > self.n + 1 or
            self.m % 2 + self.n % 2 != 1
        ):
            raise ValueError(f"Invalid Krivanek aberration n={self.n} m={self.m}")

    @staticmethod
    def from_known(s: str) -> 'Krivanek':
        """Return the aberration with name `s` (case-insensitive), e.g. `'c1'` or `'a2'`.

        Raises:
            ValueError: If `s` is not a known aberration.
        """
        try:
            return _KNOWN_ABERRATIONS[s.lower()]
        except (KeyError, TypeError):
            raise ValueError(f"Unknown aberration '{s}'") from None

class KrivanekComplex(Krivanek, kw_only=True):
    """Aberration with a complex coefficient."""
    val: complex
    """Coefficient [angstrom]."""

    def __complex__(self) -> complex:
        return self.val

class KrivanekCartesian(Krivanek, ComplexCartesian, kw_only=True):
    """Aberration with a coefficient in Cartesian form."""

class KrivanekPolar(Krivanek, ComplexPolar, kw_only=True):
    """Aberration with a coefficient in polar form."""


_KNOWN_ABERRATIONS: t.Dict[str, Krivanek] = {
    'c1': Krivanek.make_unchecked(1, 0),
    'a1': Krivanek.make_unchecked(1, 2),
    'b2': Krivanek.make_unchecked(2, 1, 3.0),  # C_21 = 3*B2
    'a2': Krivanek.make_unchecked(2, 3),
    'c3': Krivanek.make_unchecked(3, 0),
    's3': Krivanek.make_unchecked(3, 2, 3.0),  # C_32 = 3*S3
    'a3': Krivanek.make_unchecked(3, 4),
    'b4': Krivanek.make_unchecked(4, 1, 4.0),  # C_41 = 4*B4
    'd4': Krivanek.make_unchecked(4, 3, 4.0),  # C_43 = 4*D4
    'a4': Krivanek.make_unchecked(4, 5),
    'c5': Krivanek.make_unchecked(5, 0),
}

KnownAberration: t.TypeAlias = t.Annotated[str, Condition(
    lambda s: s.lower() in _KNOWN_ABERRATIONS,
    'known aberration',
    lambda exp, plural: pluralize('known aberration', plural)
)]
"""Name of a known aberration (e.g. `'c1'`, `'a1'`, `'c3'`)."""

Aberration: t.TypeAlias = t.Union[
    t.Dict[KnownAberration, t.Union[complex, ComplexCartesian, ComplexPolar]],
    KrivanekComplex, KrivanekCartesian, KrivanekPolar,
]
"""Aberration given by name or by orders `n` and `m`."""

def process_aberrations(aberrations: t.Iterable[Aberration]) -> t.Iterator[KrivanekComplex]:
    """Yield each aberration, with named aberrations expanded, as a `KrivanekComplex`."""
    for ab in aberrations:
        if isinstance(ab, dict):
            for known, val in ab.items():
                ty = Krivanek.from_known(known)
                yield KrivanekComplex(ty.n, ty.m, val=ty.scale_factor * complex(val))
        else:
            yield KrivanekComplex(ab.n, ab.m, val=complex(ab))


class SimpleFlag(Dataclass):
    """Flag that is true on a regular schedule of engine iterations (1-indexed)."""
    after: int = 0
    """True only for iterations after this one."""
    every: int = 1
    """True every this many iterations, counted from `after`."""
    before: t.Optional[int] = None
    """True only for iterations before this one. `None` for no limit."""
    first: bool = False
    """Also true on the first iteration."""

    def any_true(self, niter: int) -> bool:
        """Return whether the flag may be true for an engine with `niter` iterations."""
        return self.first or (
            self.after < niter
            and (self.before is None or self.before > 0)
        )

    def __call__(self, args: 'FlagArgs') -> bool:
        i = args['state'].iter.engine_iter
        if (self.first and i == 1):
            return True
        return (
            (self.before is None or i < self.before)
            and i > self.after
            and (i - self.after) % self.every == 0
        )

    def resolve(self) -> Self:
        """Return `self`."""
        return self


class _ConstFlag:
    def __init__(self, val: bool):
        self.val = val

    def __call__(self, args: 'FlagArgs') -> bool:
        return self.val


@lru_cache
def process_flag(flag: 'FlagLike') -> 'Flag':
    """Return `flag` as a callable, converting a `bool` to a constant flag."""
    if isinstance(flag, bool):
        return _ConstFlag(flag)
    return flag


@lru_cache
def process_schedule(schedule: 'ScheduleLike') -> 'Schedule':
    """Return `schedule` as a callable, converting a number to a constant schedule."""
    if isinstance(schedule, (int, float)):
        return lambda _: schedule
    return schedule


def flag_any_true(flag: t.Callable[['FlagArgs'], bool], niter: int) -> bool:
    """Return whether `flag` may be true during an engine with `niter` iterations. Arbitrary flags are assumed to be."""
    if isinstance(flag, SimpleFlag):
        return flag.any_true(niter)
    elif isinstance(flag, _ConstFlag):
        return flag.val
    # assume flag will return true
    return True


@dataclass(init=False, frozen=True)
class IsVersion(ConvertAnnotation):
    """Annotation restricting a version string `'major.minor...'` to a range.

    Args:
        min: Minimum accepted version, inclusive.
        max: Maximum accepted version, inclusive.
        exactly: Only accepted version. Cannot be combined with `min` or `max`.
    """
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
"""Version string, e.g. `'1.0'`."""


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
                raise ValueError(f"Version {'.'.join(map(str, val))} is not supported, version {'.'.join(map(str, self.min))} required")
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
    'cast_length', 'BackendName', 'ReconsVar', 'ReconsVars',
    'EmptyDict', 'EarlyTermination', 'Dataclass',
    'SliceList', 'SliceStep', 'SliceTotal', 'Slices',
    'ComplexCartesian', 'ComplexPolar',
    'Krivanek', 'KrivanekComplex', 'KrivanekCartesian', 'KrivanekPolar',
    'KnownAberration', 'Aberration', 'process_aberrations',
    'SimpleFlag', 'process_flag', 'process_schedule', 'flag_any_true',
    'IsVersion', 'Version',
]