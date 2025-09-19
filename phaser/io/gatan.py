
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
    file_type: t.Literal['gatan_file'] = 'gatan_file'

    @classmethod
    def from_dm4(cls, f: pane.io.FileOrPath, *,
                  custom: t.Optional[IntoConverterHandlers] = None) -> Self:
        
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
             'e-scaling': 1.0,
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
    # """Gatan 4DSTEM data filename, relative to metadata location."""

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
    
    """Single-electron scaling )."""
    background_offset:t.Optional[float] = 0.0

    scan_correction: t.Optional[t.Annotated[NDArray[numpy.floating], shape((2, 2))]] = None
    """Scan correction matrix, [x', y'] = scan_correction @ [x, y]"""

    scan_positions: t.Optional[t.List[t.Tuple[float, float]]] = None
    """
    Scan position override (m).
    Should be specified as a 1d list of (x, y) positions, in scan order. `scan_correction` is applied to these positions (if present).
    """

    notes: t.Optional[str] = None

    crop: t.Optional[t.Tuple[int, int, int, int]] = None
    """Region scan is valid within, (min_y, max_y, min_x, max_x). Python-style slicing."""

    def is_simulated(self) -> bool:
        return self.file_type == "pyMultislicer_metadata"


def load_4d(path: t.Union[str, Path], scan_shape: t.Optional[t.Tuple[int, int]] = None,
            memmap: bool = False) -> NDArray[numpy.float32]:
    """
    Load a gatan dm4 dataset into memory.

    The file is loaded so the dimensions are: (scan_y, scan_x, k_y, k_x), with y decreasing downwards.

    Patterns are not fftshifted or normalized upon loading.

    # Parameters

     - `path`: Path to file to load
     - `scan_shape`: Scan shape of dataset. Will be inferred from the filename if not specified.
     - `memmap`: If specified, memmap the file as opposed to loading it eagerly.

    Returns a numpy array (or `numpy.memmap`)
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


@t.overload
def save_4d(arr: NDArray[numpy.float32], *, path: t.Union[str, Path], folder: None = None, name: None = None):
    ...

@t.overload
def save_4d(arr: NDArray[numpy.float32], *, path: None = None, folder: t.Union[str, Path], name: t.Optional[str] = None):
    ...

def save_4d(arr: NDArray[numpy.float32], *, path: t.Union[str, Path, None] = None,
            folder: t.Union[str, Path, None] = None, name: t.Optional[str] = None): #):
    """
    Save a raw EMPAD dataset.

    Either `path` or `folder` can be specified. If `folder` is specified,
    `name` will be used as a format string to determine the filename.
    `path` and `folder` cannot be specified simultaneously.

    Patterns are not fftshifted or normalized upon saving.

    Parameters:
     - `arr`: Array to save
     - `path`: Path to save dataset to.
     - `folder`: Folder to save dataset inside.
     - `name`: When `folder` is specified, format to use to determine filename. Defaults to `"scan_x{x}_y{y}.raw"`.
       Will be formatted using the scan shape `{'x': n_x, 'y': n_y}`.
    """

    try:
        assert len(arr.shape) == 4
        assert arr.shape[2:] == (128, 128)
    except AssertionError as e:
        raise ValueError("Invalid data format") from e

    if folder is not None:
        if path is not None:
            raise ValueError("Cannot specify both 'path' and 'folder'")

        n_y, n_x = arr.shape[:2]
        path = Path(folder) / (name or "scan_x{x}_y{y}.raw").format(x=n_x, y=n_y)
    elif path is not None:
        path = Path(path)
    else:
        raise ValueError("Must specify either 'path' or 'folder'")

    out_shape = list(arr.shape)
    out_shape[2] = 130  # dead rows

    out = numpy.zeros(out_shape, dtype=numpy.float32)
    out[..., 127::-1, :] = arr.astype(numpy.float32)

    with open(path, 'wb') as f:
        out.tofile(f)
