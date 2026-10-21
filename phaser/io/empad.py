"""Reader and writer for raw EMPAD datasets, and their metadata."""

from datetime import datetime
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
    """Raised when a dataset is not a raster scan dataset."""


class EmpadMetadata(pane.PaneBase, frozen=False, kw_only=True, allow_extra=True):
    """EMPAD dataset metadata."""
    file_type: t.Literal['pyMultislicer_metadata', 'empad_metadata'] = 'empad_metadata'

    @classmethod
    def from_json(cls, f: pane.io.FileOrPath, *,
                  custom: t.Optional[IntoConverterHandlers] = None) -> Self:
        """Load metadata from a JSON file. `path` is set to the file's directory, if known."""
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
    """Free-form notes."""

    crop: t.Optional[t.Tuple[int, int, int, int]] = None
    """Region scan is valid within, (min_y, max_y, min_x, max_x). Python-style slicing."""

    def is_simulated(self) -> bool:
        """Return whether the dataset is simulated."""
        return self.file_type == "pyMultislicer_metadata"

    @staticmethod
    def from_xml(xml_path: t.Union[str, Path]) -> 'EmpadMetadata':
        """Load metadata from an EMPAD v1 XML file. Requires `lxml`.

        Converts exposure times to seconds and scan rotation to degrees.
        `scan_step` is `scan_fov / max(scan_shape)`.

        Raises:
            UnscannedDataset: If the dataset is not a raster scan dataset.
            ValueError: If a required tag is missing.
        """
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
            empad_version=1,
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

    @staticmethod
    def from_toml(toml_path: t.Union[str, Path]) -> 'EmpadMetadata':
        """Load metadata from an EMPAD v2 TOML file.

        `raw_filename` is the only `.raw` file next to the metadata, if there is exactly one.
        Otherwise it is `scan_x{x}_y{y}.raw`. `scan_rotation` is read as degrees.

        Raises:
            UnscannedDataset: If the dataset is not a raster scan dataset.
            ValueError: If a required key is missing.
        """
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # python < 3.11

        orig_path = Path(toml_path).parent
        with open(toml_path, 'rb') as f:
            toml = tomllib.load(f)

        def get(root: t.Dict[str, t.Any], key: str) -> t.Any:
            for k in key.split('.'):
                if not isinstance(root, dict) or k not in root:
                    raise ValueError(f"Couldn't find key '{key}' in TOML metadata.")
                root = root[k]
            return root

        def try_get(root: t.Dict[str, t.Any], key: str) -> t.Optional[t.Any]:
            try:
                return get(root, key)
            except ValueError:
                return None

        acquisition = get(toml, 'acquisition')
        scan = get(toml, 'scan')
        if scan.get('scan_service') != 'stem_raster':
            # not a 4D stem dataset
            raise UnscannedDataset()

        timestamp = str(get(acquisition, 'timestamp'))
        time_unix = datetime.fromisoformat(timestamp).timestamp()
        bg_time = try_get(toml, 'dark_reference.timestamp')
        bg_unix = None if bg_time is None else datetime.fromisoformat(str(bg_time)).timestamp()
        has_bg = try_get(scan, 'dark_suitable')

        optics = get(toml, 'microscope.optics')
        scan_shape = (int(get(scan, 'width')), int(get(scan, 'height')))

        # 'raw_file.filename' doesn't match the file on disk, so it isn't used
        raw_files = list(filter(lambda f: not f.name.startswith('.'), orig_path.glob('*.raw')))
        if len(raw_files) == 1:
            raw_filename = raw_files[0].name
        else:
            raw_filename = f"scan_x{scan_shape[0]}_y{scan_shape[1]}.raw"
        # all in m
        fov = (float(get(optics, 'scan_fov_x')), float(get(optics, 'scan_fov_y')))
        scan_step = (float(get(optics, 'scan_pixel_size_x')), float(get(optics, 'scan_pixel_size_y')))

        conv_angle = try_get(toml, 'microscope.illumination.convergence_angle')

        return EmpadMetadata(
            name=str(get(acquisition, 'name')),
            empad_version=2,
            orig_path=orig_path,
            raw_filename=raw_filename,
            author=acquisition.get('operator_name') or None,
            time=timestamp,
            time_unix=time_unix,
            bg_unix=bg_unix,
            has_bg=None if has_bg is None else bool(has_bg),
            voltage=float(get(toml, 'microscope.high_tension.voltage')),  # V
            conv_angle=None if conv_angle is None else float(conv_angle) * 1e3,  # rad to mrad
            camera_length=float(get(optics, 'camera_length')),  # m
            scan_rotation=float(get(optics, 'scan_rotation')) * 180. / numpy.pi,  # rad to deg
            scan_shape=scan_shape,
            scan_fov=fov,
            scan_step=scan_step,
            exposure_time=float(get(scan, 'exposure_time_s')),
            notes=acquisition.get('notes') or None,
        )


def load_4d(path: t.Union[str, Path], scan_shape: t.Optional[t.Tuple[int, int]] = None, *,
            memmap: bool = False, flips: t.Optional[t.Tuple[bool, bool, bool]] = None,
            empad_version: int = 1) -> NDArray[numpy.float32]:
    """Load a raw EMPAD dataset.

    Patterns are not fftshifted or normalized.

    Args:
        path: Path to the `.raw` file.
        scan_shape: Scan shape `(n_y, n_x)`. If `None`, parsed from `x{n_x}_y{n_y}` in the filename.
        memmap: Memory-map the file instead of reading it.
        flips: Flips `(flip_y, flip_x, transpose)` applied to the diffraction patterns.
            Defaults to `(True, False, False)` for `empad_version=1` and no flips for `empad_version=2`.
        empad_version: Raw format version, `1` or `2`.

    Returns:
        Array of shape `(n_y, n_x, 128, 128)`, `numpy.memmap` if `memmap` is set.

    Raises:
        ValueError: If the version is unsupported, `scan_shape` cannot be inferred,
            or the file size does not match the scan shape.
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
    """Save a dataset of shape `(n_y, n_x, 128, 128)` as a raw EMPAD file.

    Patterns are not fftshifted or normalized. Exactly one of `path` and `folder` must be given.

    Args:
        arr: Array to save.
        path: Output file path.
        folder: Output directory. The filename is `name` formatted with `x=n_x, y=n_y`.
        name: Filename format used with `folder`. Defaults to `"scan_x{x}_y{y}.raw"`.
        flips: Flips `(flip_y, flip_x, transpose)` applied to the diffraction patterns.
            Defaults as in `load_4d`.
        empad_version: Raw format version, `1` or `2`.

    Raises:
        ValueError: If `arr` has the wrong shape, the version is unsupported,
            or `path` and `folder` are both or neither given.
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
