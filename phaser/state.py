import typing as t

import numpy
from numpy.typing import NDArray
from typing_extensions import Self

from phaser.utils.num import Sampling, to_numpy, get_array_module, Float
from phaser.utils.misc import jax_dataclass
from phaser.utils.object import ObjectSampling

if t.TYPE_CHECKING:
    from phaser.utils.io import HdfLike
    from phaser.utils.image import _BoundaryMode
    from phaser.observer import Observer, ObserverSet


@jax_dataclass
class Patterns():
    patterns: NDArray[numpy.floating]
    """Raw diffraction patterns, with 0-frequency sample in corner"""
    pattern_mask: NDArray[numpy.floating]
    """Mask indicating which portions of the diffraction patterns contain data."""

    def to_numpy(self) -> Self:
        return self.__class__(
            to_numpy(self.patterns), to_numpy(self.pattern_mask)
        )


@jax_dataclass
class IterState():
    engine_num: int
    """Engine number. 1-indexed (0 means before any reconstruction)."""
    engine_iter: int
    """Iteration number on this engine. 1-indexed (0 means before any iterations)."""
    total_iter: int
    """Total iteration number. 1-indexed (0 means before any iterations)."""

    n_engine_iters: t.Optional[int] = None
    """Total number of iterations in this engine."""
    n_total_iters: t.Optional[int] = None
    """Total number of iterations in the reconstruction."""

    def to_numpy(self) -> Self:
        return self.__class__(
            int(self.engine_num), int(self.engine_iter), int(self.total_iter),
            int(self.n_engine_iters) if self.n_engine_iters else None,
            int(self.n_total_iters) if self.n_total_iters else None,
        )

    def copy(self) -> Self:
        import copy
        return copy.deepcopy(self)

    @staticmethod
    def empty() -> 'IterState':
        return IterState(0, 0, 0)


@jax_dataclass(static_fields=('sampling',))
class ProbeState():
    sampling: Sampling
    """Probe coordinate system. See `Sampling` for more details."""
    data: NDArray[numpy.complexfloating]
    """Probe wavefunction, in realspace. Shape (modes, y, x)"""

    def resample(
        self, new_samp: Sampling,
        rotation: float = 0.0,
        order: int = 1,
        mode: '_BoundaryMode' = 'grid-constant',
    ) -> Self:
        new_data = self.sampling.resample(
            self.data, new_samp,
            rotation=rotation,
            order=order,
            mode=mode,
        )
        return self.__class__(new_samp, new_data)

    def to_xp(self, xp: t.Any) -> Self:
        return self.__class__(
            self.sampling, xp.array(self.data)
        )

    def to_numpy(self) -> Self:
        return self.__class__(
            self.sampling, to_numpy(self.data)
        )

    def copy(self) -> Self:
        import copy
        return copy.deepcopy(self)


@jax_dataclass(static_fields=('sampling',))
class ObjectState():
    sampling: ObjectSampling
    """Object coordinate system. See `ObjectSampling` for more details."""
    data: NDArray[numpy.complexfloating]
    """Object wavefunction. Shape (z, y, x)"""
    thicknesses: NDArray[numpy.floating]
    """
    Slice thicknesses (in length units).
    Length < 2 for single slice, equal to the number of slices otherwise.
    """

    def to_xp(self, xp: t.Any) -> Self:
        return self.__class__(
            self.sampling, xp.array(self.data), xp.array(self.thicknesses)
        )

    def to_numpy(self) -> Self:
        return self.__class__(
            self.sampling, to_numpy(self.data), to_numpy(self.thicknesses)
        )

    def zs(self) -> NDArray[numpy.floating]:
        xp = get_array_module(self.thicknesses)
        if len(self.thicknesses) < 2:
            return xp.array([0.], dtype=self.thicknesses.dtype)
        return xp.cumsum(self.thicknesses) - self.thicknesses

    def copy(self) -> Self:
        import copy
        return copy.deepcopy(self)


@jax_dataclass
class ProgressState:
    iters: NDArray[numpy.integer]
    """Iterations error measurements were taken at."""
    detector_errors: NDArray[numpy.floating]
    """Detector error measurements at those iterations"""

    def to_numpy(self) -> Self:
        return self.__class__(
            to_numpy(self.iters), to_numpy(self.detector_errors)
        )

    def copy(self) -> Self:
        import copy
        return copy.deepcopy(self)

    @staticmethod
    def empty() -> 'ProgressState':
        return ProgressState(
            numpy.array([], dtype=numpy.uint64),
            numpy.array([], dtype=numpy.float64),
        )

    # TODO: this is a hack to prevent JIT recompilation.
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: t.Any) -> bool:
        if type(self) is not type(other):
            return False
        xp = get_array_module(self.iters, other.iters)
        return (
            xp.array_equal(self.iters, other.iters) and
            xp.array_equal(self.detector_errors, other.detector_errors)
        )

@jax_dataclass(kw_only=True, static_fields=('progress',))
class ReconsState:
    iter: IterState
    wavelength: Float

    probe: ProbeState
    object: ObjectState
    scan: NDArray[numpy.floating]
    """Scan coordinates (y, x), in length units. Shape (..., 2)"""
    tilt: t.Optional[NDArray[numpy.floating]] = None
    """Tilt angles (y, x) per scan position, in mrad. Shape (..., 2)"""
    progress: ProgressState

    def to_xp(self, xp: t.Any) -> Self:
        return self.__class__(
            iter=self.iter,
            probe=self.probe.to_xp(xp),
            object=self.object.to_xp(xp),
            scan=xp.array(self.scan),
            tilt=None if self.tilt is None else xp.array(self.tilt),
            progress=self.progress,
            wavelength=self.wavelength,
        )

    def to_numpy(self) -> Self:
        return self.__class__(
            iter=self.iter.to_numpy(),
            probe=self.probe.to_numpy(),
            object=self.object.to_numpy(),
            scan=to_numpy(self.scan),
            tilt=None if self.tilt is None else to_numpy(self.tilt),
            progress=self.progress.to_numpy(),
            wavelength=float(self.wavelength),
        )

    def copy(self) -> Self:
        import copy
        return copy.deepcopy(self)

    def write_hdf5(self, file: 'HdfLike'):
        from phaser.utils.io import hdf5_write_state
        hdf5_write_state(self, file)

    @staticmethod
    def read_hdf5(file: 'HdfLike') -> 'ReconsState':
        from phaser.utils.io import hdf5_read_state
        return hdf5_read_state(file).to_complete()


@jax_dataclass(kw_only=True, static_fields=('progress',))
class PartialReconsState:
    iter: t.Optional[IterState] = None
    wavelength: t.Optional[Float] = None

    probe: t.Optional[ProbeState] = None
    object: t.Optional[ObjectState] = None
    scan: t.Optional[NDArray[numpy.floating]] = None
    """Scan coordinates (y, x), in length units. Shape (..., 2)"""
    tilt: t.Optional[NDArray[numpy.floating]] = None
    progress: t.Optional[ProgressState] = None

    def to_numpy(self) -> Self:
        return self.__class__(
            iter=self.iter.to_numpy() if self.iter is not None else None,
            probe=self.probe.to_numpy() if self.probe is not None else None,
            object=self.object.to_numpy() if self.object is not None else None,
            scan=to_numpy(self.scan) if self.scan is not None else None,
            tilt=to_numpy(self.tilt) if self.tilt is not None else None,
            progress=self.progress.to_numpy() if self.progress is not None else None,
            wavelength=float(self.wavelength) if self.wavelength is not None else None,
        )

    def to_complete(self) -> ReconsState:
        missing = tuple(filter(lambda k: getattr(self, k) is None, ('probe', 'object', 'scan', 'wavelength')))
        if len(missing):
            raise ValueError(f"ReconsState missing {', '.join(map(repr, missing))}")

        progress = self.progress if self.progress is not None else ProgressState.empty()
        iter = self.iter if self.iter is not None else IterState.empty()

        return ReconsState(
            wavelength=t.cast(Float, self.wavelength),
            probe=t.cast(ProbeState, self.probe),
            object=t.cast(ObjectState, self.object),
            scan=t.cast(NDArray[numpy.floating], self.scan),
            tilt=self.tilt, progress=progress, iter=iter,
        )

    def write_hdf5(self, file: 'HdfLike'):
        from phaser.utils.io import hdf5_write_state
        hdf5_write_state(self, file)

    @staticmethod
    def read_hdf5(file: 'HdfLike') -> 'PartialReconsState':
        from phaser.utils.io import hdf5_read_state
        return hdf5_read_state(file)


@jax_dataclass(static_fields=('name', 'observer'))
class PreparedRecons:
    patterns: Patterns
    state: ReconsState
    name: str
    observer: 'ObserverSet'

    def to_numpy(self) -> Self:
        return self.__class__(
            self.patterns.to_numpy(), self.state.to_numpy(), self.name, self.observer
        )

    def with_observer(self, observer: t.Union['Observer', t.Iterable['Observer']]) -> Self:
        from phaser.observer import Observer, ObserverSet

        observers = list(self.observer.inner)
        if isinstance(observer, Observer):
            observers.append(observer)
        else:
            observers.extend(observer)

        return self.__class__(self.patterns, self.state, self.name, ObserverSet(observers))