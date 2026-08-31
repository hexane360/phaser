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
    patterns: NDArray[numpy.floating]
    mask: NDArray[numpy.floating]
    sampling: 'Sampling'
    wavelength: NotRequired[float | None]
    scan_hook: NotRequired[dict[str, t.Any] | None]
    tilt_hook: NotRequired[dict[str, t.Any] | None]
    probe_hook: NotRequired[dict[str, t.Any] | None]
    seed: NotRequired[object | None]


class LoadEmpadProps(Dataclass):
    path: Path

    diff_step: float | None = None
    kv: float | None = None
    adu: float | None = None
    det_flips: tuple[bool, bool, bool] | None = None


class LoadGatanProps(Dataclass):
    path: Path
    
    diff_step: float | None = None
    kv: float | None = None
    adu: float | None = None

class LoadNionProps(Dataclass):
    path: Path

    diff_step: float
    detector_rotation_offset: float | None = None

class LoadManualProps(Dataclass, kw_only=True):
    path: Path

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
    # TODO: post-validate (one of kv or wavelength must be specified)
    kv: float | None = None
    wavelength: float | None = None
    adu: float | None = None
    """Detector ADU, representing the single-particle signal. Used to scale patterns."""

    det_flips: tuple[bool, bool, bool] | None = None
    fftshifted: bool = False
    """Whether patterns are fftshifted (zero-frequency in corner of array)"""

class RawDataHook(Hook[None, RawData]):
    known: t.ClassVar = {
        'empad': ('phaser.hooks.io.empad:load_empad', LoadEmpadProps),
        'gatan': ('phaser.hooks.io.gatan:load_gatan', LoadGatanProps, ('rsciio',)),
        'nion': ('phaser.hooks.io.nion:load_nion', LoadNionProps),
        'manual': ('phaser.hooks.io.manual:load_manual', LoadManualProps),
    }


class ProbeHookArgs(t.TypedDict):
    sampling: 'Sampling'
    wavelength: float
    seed: object | None
    dtype: DTypeLike
    xp: t.Any


class FocusedProbeProps(Dataclass):
    defocus: float | None = None  # defocus, + is overfocus [A]
    conv_angle: float | None = None  # semiconvergence angle [mrad]
    aberrations: t.Sequence[Aberration] = ()


class ProbeHook(Hook[ProbeHookArgs, 'ProbeState']):
    known: t.ClassVar = {
        'focused': ('phaser.hooks.probe:focused_probe', FocusedProbeProps),
    }


class ObjectHookArgs(t.TypedDict):
    sampling: 'ObjectSampling'
    wavelength: float
    slices: Slices | None
    seed: object | None
    dtype: DTypeLike
    xp: t.Any


class RandomObjectProps(Dataclass):
    sigma: float = 1e-6


class ObjectHook(Hook[ObjectHookArgs, 'ObjectState']):
    known: t.ClassVar = {
        'random': ('phaser.hooks.object:random_object', RandomObjectProps),
    }


class ScanHookArgs(t.TypedDict):
    seed: object | None
    dtype: DTypeLike
    xp: t.Any


class RasterScanProps(Dataclass):
    shape: tuple[int, int] | None = None  # ny, nx (total shape)
    step_size: None | float | tuple[float, float] = None  # A
    rotation: float | None = None     # degrees CCW
    affine: t.Annotated[NDArray[numpy.floating], annotations.shape((2, 2))] | None = None


class ScanHook(Hook[ScanHookArgs, 'ScanState']):
    known: t.ClassVar = {
        'raster': ('phaser.hooks.scan:raster_scan', RasterScanProps),
    }


class TiltHookArgs(t.TypedDict):
    dtype: DTypeLike
    xp: t.Any
    shape: tuple[int, ...]  # To match raster scan shape


class GlobalTiltProps(Dataclass):
    tilt: t.Annotated[
        NDArray[numpy.floating],
        annotations.shape((2,))
    ]
    """global [ty, tx] in mrad"""


class CustomTiltProps(Dataclass):
    path: str
    """Path to .npy file containing tilt array matching the size of the scan"""


class TiltHook(Hook[TiltHookArgs, NDArray[numpy.floating]]):
    known: t.ClassVar = {
        'global': ('phaser.hooks.tilt:generate_global_tilt', GlobalTiltProps),
        'custom': ('phaser.hooks.tilt:load_custom_tilt', CustomTiltProps),
    }


class PostInitArgs(t.TypedDict):
    data: 'Patterns'
    state: 'ReconsState'
    seed: object | None
    dtype: DTypeLike
    xp: t.Any


class ScaleProps(Dataclass):
    scale: float

class OffsetProps(Dataclass):
    offset: float

class BinProps(Dataclass):
    bin: int



class CropDataProps(Dataclass):
    crop: tuple[
        # y_i, y_f, x_i, x_f
        int | None, int | None, int | None, int | None,
    ] 


class PoissonProps(Dataclass):
    scale: float | None = None
    gaussian: float | None = 1.0e-3


class DropNanProps(Dataclass):
    threshold: float = 0.9


class DiffractionAlignProps(Dataclass):
    ...


class ApplyMtfProps(Dataclass):
    mtf: FilterHook
    domain: t.Literal['real', 'recip'] = 'recip'
    """Whether to apply the filter by direct spatial-domain convolution ('real') or by
    multiplying in the Fourier domain ('recip')."""


class PostLoadHook(Hook[RawData, RawData]):
    known: t.ClassVar = {
        'crop_data': ('phaser.hooks.preprocessing:crop_data', CropDataProps),
        'poisson': ('phaser.hooks.preprocessing:add_poisson_noise', PoissonProps),
        'scale': ('phaser.hooks.preprocessing:scale_patterns', ScaleProps),
        'offset': ('phaser.hooks.preprocessing:offset_patterns', OffsetProps),
        'bin': ('phaser.hooks.preprocessing:bin_patterns', BinProps),
        'apply_mtf': ('phaser.hooks.preprocessing:apply_mtf', ApplyMtfProps),
    }


class PostInitHook(Hook[PostInitArgs, tuple['Patterns', 'ReconsState']]):
    known: t.ClassVar = {
        'drop_nans': ('phaser.hooks.preprocessing:drop_nan_patterns', DropNanProps),
        'diffraction_align': ('phaser.hooks.preprocessing:diffraction_align', DiffractionAlignProps),
    }


class EngineArgs(t.TypedDict):
    data: 'Patterns'
    state: 'ReconsState'
    dtype: type[numpy.floating]
    xp: t.Any
    recons_name: str
    observer: 'Observer'
    seed: t.Any


class EngineHook(Hook[EngineArgs, 'ReconsState']):
    known: t.ClassVar = {}  # filled in by plan.py
