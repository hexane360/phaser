"""Reader for Gatan DigitalMicrograph (`.dm4`) 4D STEM datasets, and their metadata."""

from pathlib import Path
import re
import typing as t

import numpy
import pane
import pane.io
from numpy.typing import NDArray
from pane.annotations import shape
from pane.convert import IntoConverterHandlers, from_data
from typing_extensions import Self
from rsciio import digitalmicrograph as dm

from numpy import flip 
import numpy as np


from phaser.utils.physics import Electron

from phaser.types import IsVersion


def _get_dir(f: pane.io.FileOrPath) -> t.Optional[Path]:
    if isinstance(f, (str, Path)):
        return Path(f).parent

    name = getattr(f, 'name', None)
    if name in (None, '<stdout>', '<stderr>'):
        return None
    path = Path(name)
    return path.parent if path.exists() else None


class GatanMetadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    """Metadata of a Gatan `.dm4` 4D STEM dataset."""
    file_type: t.Literal['gatan_file'] = 'gatan_file'

    @classmethod
    def from_dm4(cls, f: pane.io.FileOrPath, *,
                  custom: t.Optional[IntoConverterHandlers] = None) -> Self:
        """Read metadata from the first image in the `.dm4` file `f`. Requires `rsciio`.

        `diff_step` [mrad] and `scan_step` [m] are derived from the image calibrations.
        Fields with no corresponding tag (`conv_angle`, `defocus`, ...) take placeholder values.
        """
        path = _get_dir(f)
        
        file = dm.file_reader(f, lazy=True) 

        metadata = {'file_type':'gatan_file',
             'name':str(f.stem),
             'gatan_filename':str(f.stem),
             'raw':str(f.suffix.removeprefix('.')),
             'orig_path':str(f.parent.absolute()),
             'author': None,
             'voltage': 0.0,
             'conv_angle': 0.0,
             'defocus': 1.0e-10,
             'camera_length': 0.0,
             'diff_step': 0.0,
             'e_scaling': 1.0,
             'background_offset': 0.0,
             'scan_rotation': 0.0,
             'detector_shape': [0, 0],
             'scan_shape': [0, 0],
             'scan_fov': [0, 0],
             'scan_step': [0, 0],
             'exposure_time': 0.0,
             'post_exposure_time': 0.0,
             'beam_current': 0.0,
             'scan_correction': None,
             'diff_transpose': [False, False, False], #diagonal, horizontal, vertical
             'scan_transpose': [False, False, False], #diagonal, horizontal, vertical
             'notes': None,
             'crop': None
             }
        
        gatan_metadata = file[0]['original_metadata']['ImageList']['TagGroup0']['ImageTags'] # first image metadata
        imagedata_metadata = file[0]['original_metadata']['ImageList']['TagGroup0']['ImageData']
        digiscan_metadata = file[0]['original_metadata']['ImageList']['TagGroup0']['ImageTags']['DigiScan']


        e_calibration = imagedata_metadata['Calibrations']['Brightness']
        metadata['e_scaling'] = e_calibration['Scale']
        metadata['background_offset'] = e_calibration['Origin']

        calibrations = list(imagedata_metadata['Calibrations']['Dimension'].values())

        diff_calibrations = calibrations[0:2]
        real_calibrations = calibrations[2:4]
        data_dim = list(imagedata_metadata['Dimensions'].values())
        
        cam_acq = gatan_metadata['Acquisition']
        microscope = gatan_metadata['Microscope Info']

        metadata['diff_transpose'] = [bool(s) for s in cam_acq['Device']['Configuration']['Transpose'].values()]

        SI_info  = gatan_metadata['SI']

        metadata['time'] = SI_info['Acquisition']['Date']
        metadata['scan_shape'] = [int(s) for s in SI_info['Acquisition']['Spatial Sampling'].values()]
        metadata['exposure_time'] = SI_info['Acquisition']['Pixel time (s)']

        metadata['camera_length'] =  microscope['STEM Camera Length']
        metadata['voltage'] = microscope['Voltage']


        # diffraction step

        units = diff_calibrations[0]['Units']

        diff_scale = 1

        wavelength = float(Electron(metadata['voltage']).wavelength*1e-10)   


        if units == '1/nm':
            diff_scale = 1e9
        elif units == '1/um':
            diff_scale = 1e6
        elif units == '1/pm':
            diff_scale = 1e12      

        metadata['diff_step'] = diff_calibrations[0]['Scale']*diff_scale*wavelength*1e3 # to mrad
       
        scan_steps = [real_calibrations[0]['Scale'], real_calibrations[1]['Scale']]

        # Need to implement error handling, not sure what dm might record as units
        units = real_calibrations[0]['Units']
        scan_scale = 1


        if units == 'nm': 
            scan_scale = float(1e-9)
        elif units == 'um':
            scan_scale = 1e-6
        elif units == 'pm':
            scan_scale = float(1e-12)



        metadata['scan_step'] = [scan_scale*scan_step for scan_step in scan_steps]
        metadata['scan_rotation'] = digiscan_metadata['Rotation']

        metadata['detector_shape'] = [data_dim[1], data_dim[0]]
        metadata['scan_shape'] = [data_dim[2], data_dim[3]]
        metadata['scan_fov'] = [metadata['scan_shape'][0]*metadata['scan_step'][0], metadata['scan_shape'][1]*metadata['scan_step'][1]]

        self = from_data(metadata, cls, custom=custom) 

        object.__setattr__(self, 'path', path)
        return self


    def __post_init__(self):
        object.__setattr__(self, 'path', None)

    name: str
    """Experiment name"""

    version: t.Annotated[str, IsVersion(exactly="1.0")] = "1.0"
    """Gatan Metadata version"""

    gatan_filename: str
    """Gatan 4DSTEM data filename."""

    path: t.Optional[Path] = pane.field(init=False, exclude=True)
    """Current path to experimental folder (based on metadata loading)"""

    voltage: float
    """Accelerating voltage (V)."""
    conv_angle: t.Optional[float] = None
    """Convergence angle (mrad)."""
    defocus: t.Optional[float] = None
    """Defocus (m). Positive is overfocus."""
    camera_length: t.Optional[float] = None
    """Camera length (m)."""
    diff_step: t.Optional[float] = None
    """Diffraction pixel size (mrad/px)."""

    scan_rotation: float
    """Scan rotation (degrees)."""
    scan_shape: t.Tuple[int, int]
    """Scan shape (x, y)."""
    scan_fov: t.Tuple[float, float]
    """Scan field of view (m)."""
    scan_step: t.Tuple[float, float]
    """Scan step (m/px)."""

    exposure_time: t.Optional[float] = None
    """Pixel exposure time (s)."""
    # post_exposure_time: t.Optional[float] = None
    # """Pixel post-exposure time (s)."""
    beam_current: t.Optional[float] = None
    """Approx. beam current (A)."""
    e_scaling:t.Optional[float] = 1.0
    """Intensity calibration scale (`Brightness` `Scale`)."""
    background_offset:t.Optional[float] = 0.0
    """Intensity calibration offset (`Brightness` `Origin`)."""

    scan_correction: t.Optional[t.Annotated[NDArray[numpy.floating], shape((2, 2))]] = None
    """Scan correction matrix, [x', y'] = scan_correction @ [x, y]"""

    scan_positions: t.Optional[t.List[t.Tuple[float, float]]] = None
    """
    Scan position override (m).
    Should be specified as a 1d list of (x, y) positions, in scan order. `scan_correction` is applied to these positions (if present).
    """

    notes: t.Optional[str] = None
    """Free-form notes."""

    crop: t.Optional[t.Tuple[int, int, int, int]] = None
    """Region scan is valid within, (min_y, max_y, min_x, max_x). Python-style slicing."""

    def is_simulated(self) -> bool:
        """Return whether the dataset is simulated."""
        return self.file_type == "pyMultislicer_metadata"


def load_4d(path: t.Union[str, Path], scan_shape: t.Tuple[int, int],
            memmap: bool = False) -> NDArray[numpy.float32]:
    """Load the first image of a Gatan `.dm4` file as a 4D dataset. Requires `rsciio`.

    Patterns are not fftshifted or normalized, and no flips are applied.

    Args:
        path: Path to the `.dm4` file.
        scan_shape: Scan shape `(n_y, n_x)`.
        memmap: Load lazily (the result is then a lazy array, not a `numpy.ndarray`).

    Returns:
        `float32` array of shape `(n_y, n_x, k_y, k_x)`.

    Raises:
        ValueError: If the product of the first two axes is not `n_y * n_x`.
    """
    path = Path(path)

    n_y, n_x = scan_shape

    if memmap:
        a = dm.file_reader(path, lazy=True)[0]['data']
    else:
        a = dm.file_reader(path, lazy=False)[0]['data']
    
    if a.shape[0]*a.shape[1] != n_x * n_y:
        raise ValueError(f"Got {a.shape[0]*a.shape[1]} probes, expected {n_x}x{n_y} = {n_x * n_y}.")
    
    return a.astype(numpy.float32)
