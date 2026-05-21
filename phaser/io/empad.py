
from pathlib import Path
import re
import typing as t

import numpy
import pane
import pane.io
from numpy.typing import NDArray
from pane.annotations import shape
from pane.convert import IntoConverterHandlers
from typing_extensions import Self

from phaser.types import IsVersion
from phaser.utils.image import apply_flips
from phaser.utils.num import to_numpy


def _get_dir(f: pane.io.FileOrPath) -> t.Optional[Path]:
    if isinstance(f, (str, Path)):
        return Path(f).parent

    name = getattr(f, 'name', None)
    if name in (None, '<stdout>', '<stderr>'):
        return None
    path = Path(name)
    return path.parent if path.exists() else None


class UnscannedDataset(BaseException):
    ...


class EmpadMetadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    file_type: t.Literal['pyMultislicer_metadata', 'empad_metadata'] = 'empad_metadata'

    @classmethod
    def from_json(cls, f: pane.io.FileOrPath, *,
                  custom: t.Optional[IntoConverterHandlers] = None) -> Self:
        path = _get_dir(f)
        self = pane.io.from_json(f, cls, custom=custom)
        object.__setattr__(self, 'path', path)
        return self

    def __post_init__(self):
        object.__setattr__(self, 'path', None)

    name: str
    """Experiment name"""

    version: t.Annotated[str, IsVersion(exactly="2.0")] = "2.0"
    """Metadata version"""

    empad_version: t.Optional[int] = None
    """Empad version used. Defaults to v1 if not specified."""

    raw_filename: str
    """Raw 4DSTEM data filename, relative to metadata location."""

    det_flips: t.Optional[t.Tuple[bool, bool, bool]] = None
    """
    Flips to apply to the raw diffraction patterns, (flip_y, flip_x, transpose).
    Defaults to `(True, False, False)` (appears to be the most common orientation).
    """
    det_rotation: t.Optional[float] = None
    """Detector rotation (degrees)."""

    orig_path: t.Optional[Path] = None
    """Original path to experimental folder."""

    path: t.Optional[Path] = pane.field(init=False, exclude=True)
    """Current path to experimental folder (based on metadata loading)"""

    author: t.Optional[str] = None
    """Author of dataset"""
    time: t.Optional[str] = None
    """Image acquisition time (RFC 2822 format)"""
    time_unix: t.Optional[float] = None
    """Image acquisition time (seconds since Unix epoch)"""
    bg_unix: t.Optional[float] = None
    """Background image acquisition time (seconds since Unix epoch)"""
    has_bg: t.Optional[bool] = None
    """Whether background image is valid"""

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

    scan_rotation: float = 0.0
    """Scan rotation (degrees)."""
    scan_shape: t.Tuple[int, int]
    """Scan shape (x, y)."""
    scan_fov: t.Tuple[float, float]
    """Scan field of view (m)."""
    scan_step: t.Tuple[float, float]
    """Scan step (m/px)."""

    exposure_time: t.Optional[float] = None
    """Pixel exposure time (s)."""
    post_exposure_time: t.Optional[float] = None
    """Pixel post-exposure time (s)."""
    beam_current: t.Optional[float] = None
    """Approx. beam current (A)."""
    adu: t.Optional[float] = None
    """Single-electron intensity (data units)."""

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

    @staticmethod
    def from_xml(xml_path: t.Union[str, Path]) -> 'EmpadMetadata':
        from lxml import etree

        orig_path = Path(xml_path).parent
        xml = t.cast(etree._ElementTree, etree.parse(str(xml_path)))  # type: ignore

        def get(root, tag: str) -> etree._Element:
            elem = root.find(tag)  # type: ignore
            if elem is None:
                raise ValueError(f"Couldn't find tag '{tag}' in XML metadata.")
            return elem

        def try_get(root, tag: str) -> t.Optional[etree._Element]:
            return root.find(tag)  # type: ignore

        root: etree._Element = xml.getroot()
        timestamp = get(root, 'timestamp')
        raw_filename = get(root, 'raw_file').attrib['filename']
        time_unix = float(timestamp.attrib['timestamp'])

        scan_params = get(root, 'scan_parameters')
        if try_get(scan_params, 'series_count') is not None:
            # not a 4D stem dataset
            raise UnscannedDataset()

        background = try_get(root, 'background_image')
        exposure_time = float(str(get(root, 'exposure_time').text)) * 1e-3  # ms to s
        post_exposure_time = float(str(get(root, 'post_exposure_time').text)) * 1e-3  # ms to s

        bg_unix = None if background is None else background.attrib.get('timestamp', None)
        bg_unix = None if bg_unix is None else float(bg_unix)
        # True if background is <6 hours out of date
        has_bg = False if bg_unix is None else abs(time_unix - bg_unix) < 60*60*12

        iom = get(root, 'iom_measurements')
        scan_rotation = float(str(get(iom, 'scan_rotation').text)) * 180. / numpy.pi
        camera_length = float(str(get(iom, 'nominal_camera_length').text))  # m
        voltage = float(str(get(iom, 'high_voltage').text))                 # V

        scan_params = get(root, "scan_parameters[@mode='acquire']")
        scan_shape = (get(scan_params, 'scan_resolution_x'), get(scan_params, 'scan_resolution_y'))
        scan_shape = tuple(map(lambda elem: int(str(elem.text)), scan_shape))
        scan_size = float(str(get(scan_params, 'scan_size').text))

        if abs(scan_size - 1.) > 1e-5:
            print("Warning: scan_size != 1 has not been tested.")

        # all in m
        fov = get(iom, 'full_scan_field_of_view')
        scale_factor = float(str(get(fov, 'scale_factor').text))
        fov = tuple(map(
            lambda elem: float(str(elem.text)) * scan_size / scale_factor,
            (get(fov, 'x'), get(fov, 'y'))
        ))
        scan_step = tuple(map(lambda v: v / max(scan_shape), fov))

        return EmpadMetadata(
            name=root.attrib['name'],
            orig_path=orig_path,
            raw_filename=raw_filename,
            time=timestamp.attrib['isoformat'],
            time_unix=time_unix,
            bg_unix= bg_unix,
            has_bg=has_bg,
            voltage=voltage,
            camera_length=camera_length,
            scan_rotation=scan_rotation,
            scan_shape=t.cast(t.Tuple[int, int], scan_shape),
            scan_fov=t.cast(t.Tuple[float, float], fov),
            scan_step=t.cast(t.Tuple[float, float], scan_step),
            exposure_time=exposure_time,
            post_exposure_time=post_exposure_time,
        )


def load_4d(path: t.Union[str, Path], scan_shape: t.Optional[t.Tuple[int, int]] = None, *,
            memmap: bool = False, flips: t.Optional[t.Tuple[bool, bool, bool]] = None,
            empad_version: int = 1) -> NDArray[numpy.float32]:
    """
    Load a raw EMPAD dataset into memory.

    The file is loaded so the dimensions are: (scan_y, scan_x, k_y, k_x), with y decreasing downwards.

    Patterns are not fftshifted or normalized upon loading.

    # Parameters

     - `path`: Path to file to load
     - `scan_shape`: Scan shape of dataset. Will be inferred from the filename if not specified.
     - `memmap`: If specified, memmap the file as opposed to loading it eagerly.
     - `flips`: Flips to apply to the diffraction patterns, `(flip_y, flip_x, transpose)`.
       Defaults to `(True, False, False)` (appears to be the most common orientation).
     - `empad_version`: Version of empad dataset. Defaults to v1.

    Returns a numpy array (or `numpy.memmap`)
    """
    path = Path(path)

    if empad_version not in (1, 2):
        raise ValueError(f"Unsupported empad file version '{empad_version}'."
                         " Currently, '1' and '2' are supported.")

    if scan_shape is None:
        match = re.search(r"x(\d+)_y(\d+)", path.name)
        if match:
            n_x, n_y = map(int, (match[1], match[2]))
        else:
            raise ValueError(f"Unable to infer probe dimensions from name {path.name}")
    else:
        n_y, n_x = scan_shape

    if memmap:
        a = numpy.memmap(path, dtype=numpy.float32, mode='r')
    else:
        a = numpy.fromfile(path, dtype=numpy.float32)

    n_pad = 2 if empad_version == 1 else 0
    if not flips:
        # typical orientation on v1 and v2
        flips = (True, False, False) if empad_version == 1 else (False, False, False)

    if not a.size % ((128 + n_pad)*128) == 0:
        raise ValueError(f"File not divisible by {128 + n_pad}x128 (size={a.size}).")
    a = a.reshape(-1, 128 + n_pad, 128)

    if a.shape[0] != n_x * n_y:
        raise ValueError(f"Got {a.shape[0]} probes, expected {n_x}x{n_y} = {n_x * n_y}.")
    a = a.reshape(n_y, n_x, *a.shape[1:])

    if n_pad:
        a = a[..., :128, :]  # crop junk rows

    return apply_flips(a, flips)


@t.overload
def save_4d(arr: NDArray[numpy.float32], *, path: t.Union[str, Path], folder: None = None, name: None = None,
            flips: t.Optional[t.Tuple[bool, bool, bool]] = None, empad_version: int = 1):
    ...

@t.overload
def save_4d(arr: NDArray[numpy.float32], *, path: None = None, folder: t.Union[str, Path], name: t.Optional[str] = None,
            flips: t.Optional[t.Tuple[bool, bool, bool]] = None, empad_version: int = 1):
    ...

def save_4d(arr: NDArray[numpy.float32], *, path: t.Union[str, Path, None] = None,
            folder: t.Union[str, Path, None] = None, name: t.Optional[str] = None,
            flips: t.Optional[t.Tuple[bool, bool, bool]] = None, empad_version: int = 1):
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
     - `flips`: Flips to apply to the diffraction patterns, `(flip_y, flip_x, transpose)`.
       Defaults to `(True, False, False)` (appears to be the most common orientation).
    """
    if empad_version not in (1, 2):
        raise ValueError(f"Unsupported empad file version '{empad_version}'."
                         " Currently, '1' and '2' are supported.")

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

    n_pad = 2 if empad_version == 1 else 0
    if not flips:
        # typical orientation on v1 and v2
        flips = (True, False, False) if empad_version == 1 else (False, False, False)

    out_shape = list(arr.shape)
    out_shape[2] = 128 + n_pad  # dead rows

    out = numpy.zeros(out_shape, dtype=numpy.float32)
    out[..., :128, :] = apply_flips(to_numpy(arr.astype(numpy.float32)), flips)

    with open(path, 'wb') as f:
        out.tofile(f)
