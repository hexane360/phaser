"""Hook types and the properties of the built-in hooks."""

import typing as t
from pathlib import Path

import numpy
from numpy.typing import DTypeLike, NDArray
from typing_extensions import NotRequired

from pane import annotations

from ..types import Aberration, Dataclass, Slices
from .filter import FilterHook
from .hook import Hook

if t.TYPE_CHECKING:
    from phaser.utils.num import Sampling
    from phaser.utils.object import ObjectSampling

    from ..execute import Observer
    from ..state import (  # noqa: F401
        ObjectState,
        Patterns,
        ProbeState,
        ReconsState,
        ScanState,
    )


class RawData(t.TypedDict):
    """Raw diffraction data and metadata, as returned by [`RawDataHook`][phaser.hooks.RawDataHook]."""

    patterns: NDArray[numpy.floating]
    """Diffraction patterns `(..., ky, kx)`, with the zero frequency at the array corner."""
    mask: NDArray[numpy.floating]
    """Detector mask `(ky, kx)`; nonzero where pixels are valid."""
    sampling: 'Sampling'
    """Sampling corresponding to the diffraction patterns."""
    wavelength: NotRequired[float | None]
    """Wavelength [angstrom]."""
    scan_hook: NotRequired[dict[str, t.Any] | None]
    """Raw [`ScanHook`][phaser.hooks.ScanHook] data, used if no scan is given in the plan."""
    tilt_hook: NotRequired[dict[str, t.Any] | None]
    """Raw [`TiltHook`][phaser.hooks.TiltHook] data, used if no tilt is given in the plan."""
    probe_hook: NotRequired[dict[str, t.Any] | None]
    """Raw [`ProbeHook`][phaser.hooks.ProbeHook] data, used if no probe is given in the plan."""
    seed: NotRequired[object | None]
    """Random seed."""


class LoadEmpadProps(Dataclass):
    """Properties for loading EMPAD data."""

    path: Path
    """Path to the raw file, or to a `.json` metadata file."""

    empad_version: int | None = None
    """EMPAD version (`1` or `2`). Defaults to metadata, then `1`."""
    diff_step: float | None = None
    """Detector pixel size in reciprocal space [mrad]. Defaults to metadata."""
    kv: float | None = None
    """Accelerating voltage [kV]. Defaults to metadata."""
    adu: float | None = None
    """Detector ADU, representing the single-particle signal. Used to scale patterns. Defaults to metadata."""
    det_flips: tuple[bool, bool, bool] | None = None
    """Detector flips `(flip_y, flip_x, transpose)`. Defaults to metadata, then the typical orientation for `empad_version`."""


class LoadGatanProps(Dataclass):
    """Properties for loading Gatan `.dm4` data."""

    path: Path
    """Path to the `.dm4` file."""

    diff_step: float | None = None
    """Detector pixel size in reciprocal space [mrad]. Defaults to metadata."""
    kv: float | None = None
    """Accelerating voltage [kV]. Defaults to metadata."""
    adu: float | None = None
    """Detector ADU. Unused."""

class LoadNionProps(Dataclass):
    """Properties for loading Nion data."""

    path: Path
    """Path to the Nion data file."""

    diff_step: float
    """Detector pixel size in reciprocal space [mrad]."""
    detector_rotation_offset: float | None = None
    """Rotation between scan and detector, added to the scan rotation in the metadata [degrees]."""

class LoadManualProps(Dataclass, kw_only=True):
    """Properties for loading diffraction patterns from `.npy`/`.npz`, TIFF, HDF5, or raw binary files."""

    path: Path
    """Path to the file."""

    det_shape: tuple[int, int] | None = None
    """Detector shape `(ny, nx)` (after flips are applied). Required when loading raw binary files, optional otherwise."""
    dtype: str | None = None
    """Numpy dtype to load (e.g. 'float32'). Applies only when loading raw binary files."""

    gap: int = 0
    """Gap (in bytes) between patterns in the file. Applies only when loading raw binary files."""

    offset: int = 0
    """Offset (in bytes) before start of patterns in the file. Applies only when loading raw binary files."""

    key: str | None = None
    """Key to load from HDF5 or mat file (ex. 'raw.patterns.data')"""

    diff_step: float
    """Detector pixel size in reciprocal space [mrad]."""
    # TODO: post-validate (one of kv or wavelength must be specified)
    kv: float | None = None
    """Accelerating voltage [kV]. Exactly one of `kv` or `wavelength` is required."""
    wavelength: float | None = None
    """Wavelength [angstrom]. Exactly one of `kv` or `wavelength` is required."""
    adu: float | None = None
    """Detector ADU, representing the single-particle signal. Used to scale patterns."""

    det_flips: tuple[bool, bool, bool] | None = None
    """Detector flips `(flip_y, flip_x, transpose)`."""
    fftshifted: bool = False
    """Whether patterns are fftshifted (zero-frequency in corner of array)"""

class RawDataHook(Hook[None, RawData]):
    """Hook that loads [`RawData`][phaser.hooks.RawData]."""

    known: t.ClassVar = {
        'empad': ('phaser.hooks.io.empad:load_empad', LoadEmpadProps),
        'gatan': ('phaser.hooks.io.gatan:load_gatan', LoadGatanProps, ('rsciio',)),
        'nion': ('phaser.hooks.io.nion:load_nion', LoadNionProps),
        'manual': ('phaser.hooks.io.manual:load_manual', LoadManualProps),
    }


class ProbeHookArgs(t.TypedDict):
    """Arguments passed to a [`ProbeHook`][phaser.hooks.ProbeHook]."""

    sampling: 'Sampling'
    """Probe sampling."""
    wavelength: float
    """Wavelength [angstrom]."""
    seed: object | None
    """Random seed."""
    dtype: DTypeLike
    """Floating-point dtype of the probe."""
    xp: t.Any
    """Array module."""


class FocusedProbeProps(Dataclass):
    """Properties for a focused probe with an aperture."""

    defocus: float | None = None
    """Defocus, positive is overfocus [angstrom]. Defaults to raw data metadata."""
    conv_angle: float | None = None
    """Semiconvergence angle [mrad]. Defaults to raw data metadata."""
    aberrations: t.Sequence[Aberration] = ()
    """Additional aberrations."""


class ProbeHook(Hook[ProbeHookArgs, 'ProbeState']):
    """Hook that creates the initial probe."""

    known: t.ClassVar = {
        'focused': ('phaser.hooks.probe:focused_probe', FocusedProbeProps),
    }


class ObjectHookArgs(t.TypedDict):
    """Arguments passed to an [`ObjectHook`][phaser.hooks.ObjectHook]."""

    sampling: 'ObjectSampling'
    """Object sampling."""
    wavelength: float
    """Wavelength [angstrom]."""
    slices: Slices | None
    """Multislice slicing, or `None` for a single-slice object."""
    seed: object | None
    """Random seed."""
    dtype: DTypeLike
    """Floating-point dtype of the object."""
    xp: t.Any
    """Array module."""


class RandomObjectProps(Dataclass):
    """Properties for a random-phase object."""

    sigma: float = 1e-6
    """Standard deviation of the random phase [radians]."""


class ObjectHook(Hook[ObjectHookArgs, 'ObjectState']):
    """Hook that creates the initial object."""

    known: t.ClassVar = {
        'random': ('phaser.hooks.object:random_object', RandomObjectProps),
    }


class ScanHookArgs(t.TypedDict):
    """Arguments passed to a [`ScanHook`][phaser.hooks.ScanHook]."""

    seed: object | None
    """Random seed."""
    dtype: DTypeLike
    """Floating-point dtype of the scan positions."""
    xp: t.Any
    """Array module."""


class RasterScanProps(Dataclass):
    """Properties for a raster scan."""

    shape: tuple[int, int] | None = None
    """Scan shape `(ny, nx)`. Defaults to raw data metadata."""
    step_size: None | float | tuple[float, float] = None
    """Step size, or `(y, x)` step sizes [angstrom]. Defaults to raw data metadata."""
    rotation: float | None = None
    """Scan rotation [degrees CCW]."""
    affine: t.Annotated[NDArray[numpy.floating], annotations.shape((2, 2))] | None = None
    """Affine matrix applied to the positions before `rotation`."""


class ScanHook(Hook[ScanHookArgs, 'ScanState']):
    """Hook that creates the initial scan positions."""

    known: t.ClassVar = {
        'raster': ('phaser.hooks.scan:raster_scan', RasterScanProps),
    }


class TiltHookArgs(t.TypedDict):
    """Arguments passed to a [`TiltHook`][phaser.hooks.TiltHook]."""

    dtype: DTypeLike
    """Floating-point dtype of the tilt."""
    xp: t.Any
    """Array module."""
    shape: tuple[int, ...]
    """Scan shape, matching the raster scan."""


class GlobalTiltProps(Dataclass):
    """Properties for a uniform tilt."""

    tilt: t.Annotated[
        NDArray[numpy.floating],
        annotations.shape((2,))
    ]
    """Tilt `(ty, tx)` applied to every scan position [mrad]."""


class CustomTiltProps(Dataclass):
    """Properties for a tilt loaded from file."""

    path: str
    """Path to .npy file containing tilt array matching the size of the scan"""


class TiltHook(Hook[TiltHookArgs, NDArray[numpy.floating]]):
    """Hook that creates the initial per-position tilt `(..., 2)` [mrad]."""

    known: t.ClassVar = {
        'global': ('phaser.hooks.tilt:generate_global_tilt', GlobalTiltProps),
        'custom': ('phaser.hooks.tilt:load_custom_tilt', CustomTiltProps),
    }


class PostInitArgs(t.TypedDict):
    """Arguments passed to a [`PostInitHook`][phaser.hooks.PostInitHook]."""

    data: 'Patterns'
    """Diffraction patterns."""
    state: 'ReconsState'
    """Initialized reconstruction state."""
    seed: object | None
    """Random seed."""
    dtype: DTypeLike
    """Floating-point dtype."""
    xp: t.Any
    """Array module."""


class ScaleProps(Dataclass):
    """Properties for scaling patterns."""

    scale: float
    """Factor to multiply patterns by."""

class OffsetProps(Dataclass):
    """Properties for offsetting patterns."""

    offset: float
    """Value to subtract from patterns."""

class BinProps(Dataclass):
    """Properties for binning patterns."""

    bin: int
    """Binning factor along each detector axis."""



class CropDataProps(Dataclass):
    """Properties for cropping the scan dimensions."""

    crop: tuple[
        # y_i, y_f, x_i, x_f
        int | None, int | None, int | None, int | None,
    ] 
    """Crop bounds `(y_i, y_f, x_i, x_f)` with Python slicing semantics; `None` leaves a bound open."""


class PoissonProps(Dataclass):
    """Properties for adding noise to patterns."""

    scale: float | None = None
    """Factor to scale patterns by before sampling Poisson noise."""
    gaussian: float | None = 1.0e-3
    """Standard deviation of Gaussian noise added after Poisson sampling. `None` adds none."""


class DropNanProps(Dataclass):
    """Properties for dropping NaN patterns."""

    threshold: float = 0.9
    """Fraction of NaN pixels above which a pattern is dropped."""


class DiffractionAlignProps(Dataclass):
    """Properties for diffraction alignment (no fields)."""

    ...


class ApplyMtfProps(Dataclass):
    """Properties for applying a detector MTF to patterns."""

    mtf: FilterHook
    """Hook creating the filter to apply."""
    domain: t.Literal['real', 'recip'] = 'recip'
    """Whether to apply the filter by real-space convolution (`'real'`) or Fourier-space multiplication (`'recip'`)."""


class SetDetectorMaskProps(Dataclass):
    """Properties for loading a detector mask."""

    path: Path
    """Path to detector mask"""
    invert: bool = False
    """Whether to invert detector mask"""


class PostLoadHook(Hook[RawData, RawData]):
    """Hook that modifies raw data after loading."""

    known: t.ClassVar = {
        'crop_data': ('phaser.hooks.preprocessing:crop_data', CropDataProps),
        'poisson': ('phaser.hooks.preprocessing:add_poisson_noise', PoissonProps),
        'scale': ('phaser.hooks.preprocessing:scale_patterns', ScaleProps),
        'offset': ('phaser.hooks.preprocessing:offset_patterns', OffsetProps),
        'bin': ('phaser.hooks.preprocessing:bin_patterns', BinProps),
        'apply_mtf': ('phaser.hooks.preprocessing:apply_mtf', ApplyMtfProps),
        'set_mask': ('phaser.hooks.preprocessing:set_detector_mask', SetDetectorMaskProps),
    }


class PostInitHook(Hook[PostInitArgs, tuple['Patterns', 'ReconsState']]):
    """Hook that modifies patterns and state after initialization, before reconstruction."""

    known: t.ClassVar = {
        'drop_nans': ('phaser.hooks.preprocessing:drop_nan_patterns', DropNanProps),
        'diffraction_align': ('phaser.hooks.preprocessing:diffraction_align', DiffractionAlignProps),
    }


class EngineArgs(t.TypedDict):
    """Arguments passed to an [`EngineHook`][phaser.hooks.EngineHook]."""

    data: 'Patterns'
    """Diffraction patterns."""
    state: 'ReconsState'
    """Reconstruction state at the start of the engine."""
    dtype: type[numpy.floating]
    """Floating-point dtype."""
    xp: t.Any
    """Array module."""
    recons_name: str
    """Name of the reconstruction."""
    observer: 'Observer'
    """Observer notified of progress."""
    seed: t.Any
    """Random seed."""


class EngineHook(Hook[EngineArgs, 'ReconsState']):
    """Hook that runs a reconstruction engine."""

    known: t.ClassVar = {}  # filled in by plan.py
