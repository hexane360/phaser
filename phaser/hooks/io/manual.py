
from itertools import chain
import math
from pathlib import Path
import logging
import typing as t

import numpy

from phaser.utils.image import apply_flips
from phaser.utils.num import Sampling
from phaser.utils.physics import Electron
from phaser.types import cast_length
from .. import LoadManualProps, RawData


def load_manual(args: None, props: LoadManualProps) -> RawData:
    logger = logging.getLogger(__name__)

    path = Path(props.path).expanduser()

    if props.wavelength is not None:
        if props.kv is not None:
            raise ValueError("Cannot specify both 'kv' and 'wavelength'.")
        wavelength = props.wavelength
    elif props.kv is not None:
        wavelength = Electron(props.kv * 1e3).wavelength
    else:
        raise ValueError("Either 'kv' or 'wavelength' must be specified.")

    if not path.exists():
        raise ValueError(f"Couldn't find raw data at path {path}")

    ext = path.suffix.lower()

    if ext in ('.npy', '.npz'):
        logger.info("Loading as npy/npz...")
        patterns = numpy.load(path)
    elif ext in ('.tif', '.tiff'):
        logger.info("Loading as TIFF...")
        import tifffile
        patterns = numpy.asarray(tifffile.imread(path))
    elif ext in ('.h5', '.hdf5', '.emd'):
        logger.info("Loading as HDF5...")
        patterns = _load_hdf5(path, props.key, logger)
    elif ext in ('.mat',):
        raise NotImplementedError(".mat files not currently supported, open a Github issue!")
    else:
        logger.info("Loading as raw binary...")
        if props.det_shape is None:
            raise ValueError("det_shape is required when loading raw files")
        if props.dtype is None:
            logger.warning("dtype not specified when loading raw file, defaulting to 'float32'...")

        dtype = numpy.dtype(props.dtype or 'float32')

        file_size = path.stat().st_size - props.offset
        pattern_size = dtype.itemsize * math.prod(props.det_shape)
        total_pattern_size = pattern_size + props.gap

        if file_size % total_pattern_size != 0:
            raise ValueError(f"File size ({file_size} bytes after offset) not divisible by diffraction pattern size ({total_pattern_size} bytes)")

        patterns = numpy.fromfile(path, dtype=numpy.dtype([
            ('pat', dtype, math.prod(props.det_shape)), ('gap', numpy.bytes_, props.gap),
        ]), offset=props.offset)['pat']
        # reshape to detector (before transpose)
        patterns = patterns.reshape(-1, *(reversed(props.det_shape) if props.det_flips and props.det_flips[2] else props.det_shape))

    # normalize datatype and shape
    if not numpy.issubdtype(patterns.dtype, numpy.number) and not numpy.issubdtype(patterns.dtype, numpy.bool_):
        raise ValueError(f"Error loading raw data: Expected numeric data, got dtype {patterns.dtype} instead")

    if numpy.iscomplexobj(patterns):
        raise ValueError(f"Error loading raw data: Expected real-valued data, got dtype {patterns.dtype} instead")

    if not numpy.issubdtype(patterns.dtype, numpy.floating):
        dtype = numpy.float32 #numpy.result_type(patterns.dtype, numpy.float32)
        logger.info(f"Casting patterns to floating-point dtype {numpy.dtype(dtype).name}")
        patterns = patterns.astype(dtype)

    if patterns.ndim < 3:
        raise ValueError(f"Error loading raw data: Expected at least 3 dimensions, got shape {patterns.shape} instead")

    flips = props.det_flips or (False, False, False)
    logger.info(f"Applying detector flips: {list(map(int, flips))} [y, x, transpose]")
    patterns = apply_flips(patterns, flips)

    if props.det_shape is not None and patterns.shape[-2:] != props.det_shape:
        raise ValueError(f"Error loading raw data: Expected detector shape {props.det_shape}, got shape {patterns.shape[-2:]} instead")

    if props.adu is None:
        logger.warning("ADU not supplied for experimental dataset. This is not recommended.")
    else:
        logger.info(f"Scaling patterns by ADU ({props.adu:.1f})")
        patterns /= props.adu

    a = wavelength / (props.diff_step * 1e-3)  # recip. pixel size -> 1 / real space extent
    sampling = Sampling(cast_length(patterns.shape[-2:], 2), extent=(a, a))

    mask = numpy.ones_like(patterns, shape=patterns.shape[-2:])

    if not props.fftshifted:
        patterns = numpy.fft.ifftshift(patterns, axes=(-1, -2))
        mask = numpy.fft.ifftshift(mask, axes=(-1, -2))

    return {
        'patterns': patterns,
        'mask': mask,
        'sampling': sampling,
        'wavelength': wavelength,
        'probe_hook': None,
        'scan_hook': None,
        'tilt_hook': None,
        'seed': None,
    }


def _normalize_key(key: str) -> t.Tuple[str, ...]:
    # split on both '/' and '.'
    return tuple(chain.from_iterable(map(lambda k: k.split('/'), key.split('.'))))


_HDF5_KNOWN_KEYS: t.List[t.Tuple[str, ...]] = [
    ('dp',),
    ('data',),
]


def _load_hdf5(path: Path, key: t.Optional[str], logger: logging.Logger) -> numpy.ndarray:
    import h5py

    def _hdf5_get(g: t.Any, key: t.Tuple[str, ...], crumbs: t.Tuple[str, ...] = ()) -> h5py.Dataset:
        if len(key) == 0:
            # base case
            if not isinstance(g, h5py.Dataset):
                raise ValueError(f"Expected a dataset at path {'.'.join(crumbs)}, got type {type(g)}")
            return g

        if not isinstance(g, h5py.Group):
            raise ValueError(f"Expected a group at path {'.'.join(crumbs)}, got type {type(g)}")

        k, rest = key[0], key[1:]

        if (child := g.get(k)) is None:
            raise ValueError(f"Key '{k}' not found in group {g.name}")

        return _hdf5_get(child, rest, crumbs=crumbs + (k,))

    with h5py.File(path, 'r') as f:
        if key is not None:
            patterns = _hdf5_get(f, _normalize_key(key))
            logger.info(f"Loaded patterns from key '{key}' in HDF5 file.")
            return patterns[()]

        # otherwise, look at common keys
        for k in _HDF5_KNOWN_KEYS:
            try:
                patterns = _hdf5_get(f, k)
                if patterns.ndim < 3:
                    continue
                logger.info(f"Found patterns at key '{'.'.join(k)}' (inferred) in HDF5 file.")
                return patterns[()]
            except ValueError:
                continue

    note = "" if key is not None else " Consider specifying a 'key' with the path to the dataset."
    raise ValueError(f"Couldn't find raw data in HDF5 file {path}.{note}")