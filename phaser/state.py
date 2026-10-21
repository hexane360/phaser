"""Reconstruction state dataclasses."""

import typing as t

import numpy
from frozendict import frozendict
from numpy.typing import NDArray
from typing_extensions import Self

from phaser.utils.num import Float, Sampling, get_array_module, to_numpy
from phaser.utils.object import ObjectSampling
from phaser.utils.tree import field, tree_dataclass

if t.TYPE_CHECKING:
    from phaser.observer import Observer, ObserverSet
    from phaser.utils.image import _InterpBoundaryMode
    from phaser.utils.io import HdfLike


@tree_dataclass
class Patterns:
    """Diffraction patterns and their mask."""

    patterns: NDArray[numpy.floating]
    """Raw diffraction patterns, with 0-frequency sample in corner"""
    pattern_mask: NDArray[numpy.floating]
    """Mask indicating which portions of the diffraction patterns contain data."""

    def to_numpy(self) -> Self:
        """Return a copy with arrays converted to NumPy."""
        return self.__class__(
            to_numpy(self.patterns), to_numpy(self.pattern_mask)
        )


@tree_dataclass
class IterState:
    """Iteration counters."""

    engine_num: int
    """Engine number. 1-indexed (0 means before any reconstruction)."""
    engine_iter: int
    """Iteration number on this engine. 1-indexed (0 means before any iterations)."""
    total_iter: int
    """Total iteration number. 1-indexed (0 means before any iterations)."""

    n_engine_iters: int | None = None
    """Total number of iterations in this engine."""
    n_total_iters: int | None = None
    """Total number of iterations in the reconstruction."""

    def to_numpy(self) -> Self:
        """Return a copy with arrays converted to NumPy."""
        return self.__class__(
            int(self.engine_num), int(self.engine_iter), int(self.total_iter),
            int(self.n_engine_iters) if self.n_engine_iters else None,
            int(self.n_total_iters) if self.n_total_iters else None,
        )

    def copy(self) -> Self:
        """Return a deep copy."""
        import copy
        return copy.deepcopy(self)

    @staticmethod
    def empty() -> 'IterState':
        """Return the state before any iterations."""
        return IterState(0, 0, 0)


@tree_dataclass(static_fields=('sampling', 'meta', 'ty'))
class PixelatedProbeState:
    """Probe state in a pixelated representation."""

    sampling: Sampling
    """Probe coordinate system. See `Sampling` for more details."""
    data: NDArray[numpy.complexfloating]
    """Probe wavefunction, in realspace. Shape (modes, y, x)"""

    meta: frozendict[str, t.Any] = field(default_factory=frozendict)
    ty: t.Literal['pixelated'] = 'pixelated'

    def resample(
        self, new_samp: Sampling,
        rotation: float = 0.0,
        order: int = 1,
        mode: '_InterpBoundaryMode' = 'grid-constant',
    ) -> Self:
        """Return the probe resampled onto `new_samp`.

        `rotation`, `order`, and `mode` are passed to `Sampling.resample`.
        """
        new_data = self.sampling.resample(
            self.data, new_samp,
            rotation=rotation,
            order=order,
            mode=mode,
        )
        return self.__class__(new_samp, new_data, self.meta)

    def to_xp(self, xp: t.Any) -> Self:
        """Return a copy with arrays converted to the array module `xp`."""
        return self.__class__(
            self.sampling, xp.asarray(self.data), self.meta
        )

    def to_numpy(self) -> Self:
        """Return a copy with arrays converted to NumPy."""
        return self.__class__(
            self.sampling, to_numpy(self.data), self.meta
        )

    def copy(self) -> Self:
        """Return a deep copy."""
        import copy
        return copy.deepcopy(self)


# discriminated union of probe state types
ProbeState: t.TypeAlias = PixelatedProbeState
"""Probe state of any type."""


@tree_dataclass(static_fields=('sampling', 'meta', 'ty'))
class PixelatedObjectState:
    """Object state in a pixelated representation."""

    sampling: ObjectSampling
    """Object coordinate system. See `ObjectSampling` for more details."""
    data: NDArray[numpy.complexfloating]
    """Object wavefunction. Shape (z, y, x)"""
    thicknesses: NDArray[numpy.floating]
    """
    Slice thicknesses (in length units).
    Length < 2 for single slice, equal to the number of slices otherwise.
    """

    meta: frozendict[str, t.Any] = field(default_factory=frozendict)
    ty: t.Literal['pixelated'] = 'pixelated'

    def to_xp(self, xp: t.Any) -> Self:
        """Return a copy with arrays converted to the array module `xp`."""
        return self.__class__(
            self.sampling, xp.asarray(self.data), xp.asarray(self.thicknesses), self.meta,
        )

    def to_numpy(self) -> Self:
        """Return a copy with arrays converted to NumPy."""
        return self.__class__(
            self.sampling, to_numpy(self.data), to_numpy(self.thicknesses), self.meta,
        )

    def zs(self) -> NDArray[numpy.floating]:
        """Return the position of the beginning of each slice [angstrom]."""
        xp = get_array_module(self.thicknesses)
        if len(self.thicknesses) < 2:
            return xp.asarray([0.], dtype=self.thicknesses.dtype)
        return xp.cumsum(self.thicknesses) - self.thicknesses

    def copy(self) -> Self:
        """Return a deep copy."""
        import copy
        return copy.deepcopy(self)


# discriminated union of object state types
ObjectState: t.TypeAlias = PixelatedObjectState
"""Object state of any type."""


@tree_dataclass(static_fields=('meta',))
class ScanState:
    """Scan positions and per-position tilts."""

    data: NDArray[numpy.floating]
    """Scan coordinates (y, x), in length units. Shape (..., 2)"""
    initial: NDArray[numpy.floating]
    """Initial scan coordinates (y, x), in length units."""
    tilt: NDArray[numpy.floating] | None = None
    """Tilt angles (y, x) per scan position, in mrad. Shape (..., 2)"""

    meta: frozendict[str, t.Any] = field(default_factory=frozendict)

    def to_xp(self, xp: t.Any) -> Self:
        """Return a copy with arrays converted to the array module `xp`."""
        return self.__class__(
            xp.asarray(self.data),
            xp.asarray(self.initial),
            None if self.tilt is None else xp.asarray(self.tilt),
            self.meta,
        )

    def to_numpy(self) -> Self:
        """Return a copy with arrays converted to NumPy."""
        return self.__class__(
            to_numpy(self.data),
            to_numpy(self.initial),
            None if self.tilt is None else to_numpy(self.tilt),
            self.meta,
        )

    def copy(self) -> Self:
        """Return a deep copy."""
        import copy
        return copy.deepcopy(self)


@tree_dataclass
class ProgressState:
    """Time series of a progress measurement."""

    iters: list[int] = field(default_factory=list)
    """Iterations error measurements were taken at."""
    values: list[float] = field(default_factory=list)
    """Detector error measurements at those iterations"""

    def copy(self) -> Self:
        """Return a deep copy."""
        import copy
        return copy.deepcopy(self)


@tree_dataclass(kw_only=True, drop_fields=('progress',))
class ReconsState:
    """Reconstruction state."""

    iter: IterState
    """Iteration counters."""
    wavelength: Float
    """Wavelength [angstrom]."""

    probe: ProbeState
    object: ObjectState
    scan: ScanState

    progress: dict[str, ProgressState] = field(default_factory=dict)
    """Progress measurements by name (e.g. `'total_loss'`)."""

    def to_xp(self, xp: t.Any) -> Self:
        """Return a copy with arrays converted to the array module `xp`."""
        return self.__class__(
            iter=self.iter,
            probe=self.probe.to_xp(xp),
            object=self.object.to_xp(xp),
            scan=self.scan.to_xp(xp),
            progress=self.progress,
            wavelength=self.wavelength,
        )

    def to_numpy(self) -> Self:
        """Return a copy with arrays converted to NumPy."""
        return self.__class__(
            iter=self.iter.to_numpy(),
            probe=self.probe.to_numpy(),
            object=self.object.to_numpy(),
            scan=self.scan.to_numpy(),
            progress=self.progress,
            wavelength=float(self.wavelength),
        )

    def copy(self) -> Self:
        """Return a deep copy."""
        import copy
        return copy.deepcopy(self)

    def write_hdf5(self, file: 'HdfLike'):
        """Write the state to an HDF5 file."""
        from phaser.utils.io import hdf5_write_state
        hdf5_write_state(self, file)

    @staticmethod
    def read_hdf5(file: 'HdfLike') -> 'ReconsState':
        """Read a complete state from an HDF5 file.

        Raises:
            ValueError: If the file has no probe, object, scan, or wavelength.
        """
        from phaser.utils.io import hdf5_read_state
        return hdf5_read_state(file).to_complete()


@tree_dataclass(kw_only=True, static_fields=('progress',))
class PartialReconsState:
    """Reconstruction state with optional components, as read from a file."""

    iter: IterState | None = None
    wavelength: Float | None = None

    probe: ProbeState | None = None
    object: ObjectState | None = None
    scan: ScanState | None = None
    progress: dict[str, ProgressState] | None = None

    def to_numpy(self) -> Self:
        """Return a copy with arrays converted to NumPy."""
        return self.__class__(
            iter=self.iter.to_numpy() if self.iter is not None else None,
            probe=self.probe.to_numpy() if self.probe is not None else None,
            object=self.object.to_numpy() if self.object is not None else None,
            scan=self.scan.to_numpy() if self.scan is not None else None,
            wavelength=float(self.wavelength) if self.wavelength is not None else None,
            progress=self.progress,
        )

    def to_complete(self) -> ReconsState:
        """Return the complete state, with `iter` and `progress` defaulted if missing.

        Raises:
            ValueError: If the probe, object, scan, or wavelength is missing.
        """
        missing = tuple(filter(lambda k: getattr(self, k) is None, ('probe', 'object', 'scan', 'wavelength')))
        if len(missing):
            raise ValueError(f"ReconsState missing {', '.join(map(repr, missing))}")

        progress = self.progress if self.progress is not None else {}
        iter = self.iter if self.iter is not None else IterState.empty()

        return ReconsState(
            wavelength=t.cast(Float, self.wavelength),
            probe=t.cast(ProbeState, self.probe),
            object=t.cast(ObjectState, self.object),
            scan=t.cast(ScanState, self.scan),
            progress=progress, iter=iter,
        )

    def write_hdf5(self, file: 'HdfLike'):
        """Write the state to an HDF5 file."""
        from phaser.utils.io import hdf5_write_state
        hdf5_write_state(self, file)

    @staticmethod
    def read_hdf5(file: 'HdfLike') -> 'PartialReconsState':
        """Read a state from an HDF5 file."""
        from phaser.utils.io import hdf5_read_state
        return hdf5_read_state(file)


@tree_dataclass(static_fields=('name', 'observer'))
class PreparedRecons:
    """Reconstruction ready to run an engine."""

    patterns: Patterns
    state: ReconsState
    name: str
    """Reconstruction name."""
    observer: 'ObserverSet'

    def to_xp(self, xp: t.Any) -> Self:
        """Return a copy with arrays converted to the array module `xp`."""
        return self.__class__(
            self.patterns,
            self.state.to_xp(xp),
            self.name, self.observer,
        )

    def to_numpy(self) -> Self:
        """Return a copy with arrays converted to NumPy."""
        return self.__class__(
            self.patterns.to_numpy(), self.state.to_numpy(), self.name, self.observer
        )

    def with_observer(self, observer: t.Union['Observer', t.Iterable['Observer']]) -> Self:
        """Return a copy with additional observer(s)."""
        from phaser.observer import Observer, ObserverSet

        observers = list(self.observer.inner)
        if isinstance(observer, Observer):
            observers.append(observer)
        else:
            observers.extend(observer)

        return self.__class__(self.patterns, self.state, self.name, ObserverSet(observers))


__all__ = [
    'IterState',
    'ObjectState',
    'PartialReconsState',
    'Patterns',
    'PixelatedObjectState',
    'PixelatedProbeState',
    'PreparedRecons',
    'ProbeState',
    'ProgressState',
    'ReconsState',
    'ScanState',
]