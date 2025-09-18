from pathlib import Path
import typing as t

import numpy
from numpy.typing import NDArray, DTypeLike
import pane.annotations as annotations
from typing_extensions import NotRequired

from ..types import Dataclass, Slices
from .hook import Hook

if t.TYPE_CHECKING:
    from phaser.utils.num import Sampling
    from phaser.utils.object import ObjectSampling
    from ..state import ObjectState, ProbeState, ReconsState, Patterns
    from ..execute import Observer


class RawData(t.TypedDict):
    patterns: NDArray[numpy.floating]
    mask: NDArray[numpy.floating]
    sampling: 'Sampling'
    wavelength: NotRequired[t.Optional[float]]
    scan_hook: NotRequired[t.Union[t.Dict[str, t.Any], None]]
    tilt_hook: NotRequired[t.Union[t.Dict[str, t.Any], None]]
    probe_hook: NotRequired[t.Union[t.Dict[str, t.Any], None]]
    seed: NotRequired[t.Optional[object]]


class LoadEmpadProps(Dataclass):
    path: Path

    diff_step: t.Optional[float] = None
    kv: t.Optional[float] = None
    adu: t.Optional[float] = None
    det_flips: t.Optional[t.Tuple[bool, bool, bool]] = None


class LoadGatanProps(Dataclass):
    path: Path
    
    diff_step: t.Optional[float] = None
    kv: t.Optional[float] = None
    adu: t.Optional[float] = None

class LoadNionProps(Dataclass):
    path: Path

    diff_step: float
    detector_rotation_offset: t.Optional[float] = None

class LoadManualProps(Dataclass, kw_only=True):
    path: Path

    det_shape: t.Optional[t.Tuple[int, int]] = None
    """Detector shape `(ny, nx)` (after flips are applied). Required when loading raw binary files, optional otherwise."""
    dtype: t.Optional[str] = None
    """Numpy dtype to load (e.g. 'float32'). Applies only when loading raw binary files."""
    gap: int = 0
    """Gap (in bytes) between patterns in the file. Applies only when loading raw binary files."""
    offset: int = 0
    """Offset (in bytes) before start of patterns in the file. Applies only when loading raw binary files."""

    key: t.Optional[str] = None
    """Key to load from HDF5 or mat file (ex. 'raw.patterns.data')"""

    diff_step: float
    # TODO: post-validate (one of kv or wavelength must be specified)
    kv: t.Optional[float] = None
    wavelength: t.Optional[float] = None
    adu: t.Optional[float] = None
    """Detector ADU, representing the single-particle signal. Used to scale patterns."""

    det_flips: t.Optional[t.Tuple[bool, bool, bool]] = None
    fftshifted: bool = False
    """Whether patterns are fftshifted (zero-frequency in corner of array)"""

class RawDataHook(Hook[None, RawData]):
    known = {
        'empad': ('phaser.hooks.io.empad:load_empad', LoadEmpadProps),
        'gatan': ('phaser.hooks.io.gatan:load_gatan', LoadGatanProps, ('rsciio',)),
        'nion': ('phaser.hooks.io.nion:load_nion', LoadNionProps),
        'manual': ('phaser.hooks.io.manual:load_manual', LoadManualProps),
    }




class ProbeHookArgs(t.TypedDict):
    sampling: 'Sampling'
    wavelength: float
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any


class FocusedProbeProps(Dataclass):
    defocus: t.Optional[float] = None  # defocus, + is overfocus [A]
    conv_angle: t.Optional[float] = None  # semiconvergence angle [mrad]


class ProbeHook(Hook[ProbeHookArgs, 'ProbeState']):
    known = {
        'focused': ('phaser.hooks.probe:focused_probe', FocusedProbeProps),
    }


class ObjectHookArgs(t.TypedDict):
    sampling: 'ObjectSampling'
    wavelength: float
    slices: t.Optional[Slices]
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any


class RandomObjectProps(Dataclass):
    sigma: float = 1e-6


class ObjectHook(Hook[ObjectHookArgs, 'ObjectState']):
    known = {
        'random': ('phaser.hooks.object:random_object', RandomObjectProps),
    }


class ScanHookArgs(t.TypedDict):
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any


class RasterScanProps(Dataclass):
    shape: t.Optional[t.Tuple[int, int]] = None  # ny, nx (total shape)
    step_size: t.Union[None, float, t.Tuple[float, float]] = None  # A
    rotation: t.Optional[float] = None     # degrees CCW
    affine: t.Optional[t.Annotated[NDArray[numpy.floating], annotations.shape((2, 2))]] = None


class ScanHook(Hook[ScanHookArgs, NDArray[numpy.floating]]):
    known = {
        'raster': ('phaser.hooks.scan:raster_scan', RasterScanProps),
    }


class TiltHookArgs(t.TypedDict):
    dtype: DTypeLike
    xp: t.Any
    shape: t.Tuple[int, ...]  # To match raster scan shape


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
    known = {
        'global': ('phaser.hooks.tilt:generate_global_tilt', GlobalTiltProps),
        'custom': ('phaser.hooks.tilt:load_custom_tilt', CustomTiltProps),
    }


class PostInitArgs(t.TypedDict):
    data: 'Patterns'
    state: 'ReconsState'
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any


class ScaleProps(Dataclass):
    scale: float

class OffsetProps(Dataclass):
    offset: float

class BinProps(Dataclass):
    bin: int



class CropDataProps(Dataclass):
    crop: t.Tuple[
        # y_i, y_f, x_i, x_f
        t.Optional[int], t.Optional[int], t.Optional[int], t.Optional[int],
    ] 


class PoissonProps(Dataclass):
    scale: t.Optional[float] = None
    gaussian: t.Optional[float] = 1.0e-3


class DropNanProps(Dataclass):
    threshold: float = 0.9


class DiffractionAlignProps(Dataclass):
    ...


class PostLoadHook(Hook[RawData, RawData]):
    known = {
        'crop_data': ('phaser.hooks.preprocessing:crop_data', CropDataProps),
        'poisson': ('phaser.hooks.preprocessing:add_poisson_noise', PoissonProps),
        'scale': ('phaser.hooks.preprocessing:scale_patterns', ScaleProps),
        'offset': ('phaser.hooks.preprocessing:offset_patterns', OffsetProps),
        'bin': ('phaser.hooks.preprocessing:bin_patterns', BinProps),
    }


class PostInitHook(Hook[PostInitArgs, t.Tuple['Patterns', 'ReconsState']]):
    known = {
        'drop_nans': ('phaser.hooks.preprocessing:drop_nan_patterns', DropNanProps),
        'diffraction_align': ('phaser.hooks.preprocessing:diffraction_align', DiffractionAlignProps),
    }


class EngineArgs(t.TypedDict):
    data: 'Patterns'
    state: 'ReconsState'
    dtype: DTypeLike
    xp: t.Any
    recons_name: str
    observer: 'Observer'
    seed: t.Any


class EngineHook(Hook[EngineArgs, 'ReconsState']):
    known = {}  # filled in by plan.py
