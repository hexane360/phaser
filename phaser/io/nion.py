"""Reader for Nion Swift 4D STEM datasets (zip archive containing `data.npy`), and their metadata."""

import typing as t
import zipfile as zf
from pathlib import Path

import numpy
from numpy.typing import NDArray

import pane
from phaser.utils.image import apply_flips


class SpatialCalibrations(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    """Calibration of one axis."""
    offset: float
    scale: float
    units: str

class DetectorConfiguration(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    """Detector configuration."""

class CameraProcessingParameters(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    """Camera processing parameters."""
    readout_area: t.Tuple[float, float, float, float]
    processing: t.List[str]


class Properties(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    """Dataset `properties` entry."""
    detector_configuration:DetectorConfiguration
    camera_processing_parameters: CameraProcessingParameters


class InstrumentMetadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    """Instrument settings."""
    high_tension: float
    defocus:float

class ScanMetadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    """Scan settings."""
    fov_nm: float
    scan_context_size: t.Tuple[float, float]
    scan_size: t.Tuple[float, float] 
    rotation_deg: t.Optional[float] = 0

class Metadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    """Dataset `metadata` entry."""
    scan: ScanMetadata
    instrument: InstrumentMetadata

class NionMetadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    """Metadata of a Nion Swift dataset."""
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


def load_4d(path: t.Union[str, Path], scan_shape: t.Tuple[int, int], *,
            flips: t.Optional[t.Tuple[bool, bool, bool]] = None) -> NDArray[numpy.float32]:
    """Load the `data.npy` array from a Nion Swift zip file.

    Patterns are not fftshifted or normalized.

    Args:
        path: Path to the zip file.
        scan_shape: Scan shape `(n_y, n_x)`.
        memmap: Unused. The array is always read into memory.
        flips: Flips `(flip_y, flip_x, transpose)` applied to the diffraction patterns.
            Defaults to no flips.

    Returns:
        Array of shape `(n_y, n_x, k_y, k_x)`, in the dtype stored in the file.

    Raises:
        ValueError: If the scan axes do not match `scan_shape`.
    """
    path = Path(path)

    scan_shape = t.cast(tuple[int, int], tuple(map(int, scan_shape)))

    n_y, n_x = scan_shape

    with zf.ZipFile(path, 'r') as data_file, data_file.open('data.npy', mode='r') as f:
        a = numpy.load(f)
        print("Loaded 'data.npy'")

    if a.shape[0] !=  n_y:
        raise ValueError(f"Got {a.shape[0]} y probes, expected {n_y}.")
    
    if a.shape[1] !=  n_x:
        raise ValueError(f"Got {a.shape[1]} x probes, expected {n_x}.")

    return apply_flips(a, flips or (False, False, False)) 
