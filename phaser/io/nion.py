
from pathlib import Path
import typing as t

import numpy
import pane
import pane.io
from numpy.typing import NDArray
from pane.convert import IntoConverterHandlers, from_data
from typing_extensions import Self

from phaser.utils.image import apply_flips

import zipfile as zf
import io


class SpatialCalibrations(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    offset: float
    scale: float
    units: str

class DetectorConfiguration(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    pass

class CameraProcessingParameters(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    readout_area: t.Tuple[float, float, float, float]
    processing: t.List[str]


class Properties(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    detector_configuration:DetectorConfiguration
    camera_processing_parameters: CameraProcessingParameters


class InstrumentMetadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    high_tension: float
    defocus:float

class ScanMetadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    fov_nm: float
    scan_context_size: t.Tuple[float, float]
    scan_size: t.Tuple[float, float] 
    rotation_deg: t.Optional[float] = 0

class Metadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    scan: ScanMetadata
    instrument: InstrumentMetadata

class NionMetadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    file_type: t.Literal['nion_metadata'] = 'nion_metadata'

    title: str
    """Experiment name"""

    version: t.Optional[float]
    
    """Metadata version"""
    spatial_calibrations: t.List[SpatialCalibrations]

    intensity_calibration: t.Dict

    # intensity_calibration['offset']: t.Float
    # intensity_calibration['scale']: t.Float

    metadata: Metadata
    properties: Properties


def load_4d(path: t.Union[str, Path], scan_shape: t.Optional[t.Union[t.Tuple[int, int], t.Tuple[float, float]]] = None, *,
            memmap: bool = False, flips: t.Optional[t.Tuple[bool, bool, bool]] = None) -> NDArray[numpy.float32]:
    """
    Load a raw nion dataset into memory.

    The file is loaded so the dimensions are: (scan_y, scan_x, k_y, k_x), with y decreasing downwards.

    Patterns are not fftshifted or normalized upon loading.

    # Parameters

     - `path`: Path to file to load
     - `scan_shape`: Scan shape of dataset. Will be inferred from the filename if not specified.
     - `memmap`: If specified, memmap the file as opposed to loading it eagerly.
     - `flips`: Flips to apply to the diffraction patterns, `(flip_y, flip_x, transpose)`.
       Defaults to `(True, False, False)` (appears to be the most common orientation).

    Returns a numpy array (or `numpy.memmap`)
    """
    path = Path(path)

    scan_shape = tuple(map(int, scan_shape))

    n_y, n_x = scan_shape

    with zf.ZipFile(path, 'r') as data_file:

        with io.BufferedReader(data_file.open('data.npy', mode='r')) as f:
            a = numpy.load(f)
            print(f"Loaded 'data.npy'")
    
    if a.shape[0] !=  n_y:
        raise ValueError(f"Got {a.shape[0]} y probes, expected {n_y}.")
    
    if a.shape[1] !=  n_x:
        raise ValueError(f"Got {a.shape[1]} x probes, expected {n_x}.")

    return apply_flips(a, flips or (False, False, False)) 
