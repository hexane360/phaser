from pathlib import Path
import typing as t

import numpy
from numpy.typing import NDArray, DTypeLike
import pane.annotations as annotations
from typing_extensions import NotRequired

from ..types import Dataclass, Slices
from .hook import Hook

from ..utils.optics import ABERRATION_SPECS, ABERRATION_KEYS

if t.TYPE_CHECKING:
    from phaser.utils.num import Sampling
    from phaser.utils.object import ObjectSampling
    from ..state import ObjectState, ProbeState, ParameterizedProbeState, ReconsState, Patterns
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


class RawDataHook(Hook[None, RawData]):
    known = {
        'empad': ('phaser.hooks.io.empad:load_empad', LoadEmpadProps),
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


class ParameterizedProbeProps(Dataclass):
    defocus: t.Optional[float] = None  # defocus, + is overfocus [A]
    conv_angle: t.Optional[float] = None
    aberration_dict: t.Optional[t.Dict[str, t.Union[float, t.List[float]]]] = None  # {'C1': 1.2, 'A1': 0.3}

    def __post_init__(self):
        if self.aberration_dict is None:
            if self.defocus is not None:
                object.__setattr__(self, 'aberration_dict', {'C1': float(self.defocus)})
        else:
            c1_val = self.aberration_dict.get("C1")
            if c1_val is None or not isinstance(c1_val, (int, float)):
                if self.defocus is not None:
                    new_dict = dict(self.aberration_dict)
                    new_dict['C1'] = float(self.defocus)
                    object.__setattr__(self, 'aberration_dict', new_dict)
            else:
                if self.defocus is None:
                    object.__setattr__(self, 'defocus', float(c1_val))

    def to_aberration_array(self) -> NDArray[numpy.floating]:
        coeffs = []
        for name, _, is_complex in ABERRATION_SPECS:
            val = self.aberration_dict.get(name, 0.0) if self.aberration_dict else 0.0
            if is_complex:
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    coeffs.extend([float(val[0]), float(val[1])])
                elif isinstance(val, (int, float)):
                    coeffs.extend([float(val), float(val)])  # angle = 0
                else:
                    raise ValueError(f"Aberration {name} must be (mag, ang) tuple or real scalar.")
            else:
                if isinstance(val, (int, float)):
                    coeffs.append(float(val))
                else:
                    raise ValueError(f"Aberration {name} must be a real scalar for non-complex term.")
        return numpy.array(coeffs, dtype=numpy.floating)
    

class ProbeHook(Hook[ProbeHookArgs, t.Union['ProbeState', 'ParameterizedProbeState']]):
    known = {
        'focused': ('phaser.hooks.probe:focused_probe', FocusedProbeProps),
        'parameterized': ('phaser.hooks.probe:parameterized_probe', ParameterizedProbeProps),
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
    engine_i: int
    observer: 'Observer'
    seed: t.Any


class EngineHook(Hook[EngineArgs, 'ReconsState']):
    known = {}  # filled in by plan.py
