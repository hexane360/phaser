import typing as t

import numpy
from numpy.typing import NDArray
from pathlib import Path

import h5py

from phaser.utils.image import apply_flips
from phaser.utils.num import cast_array_module
from . import GlobalTiltProps, CustomTiltProps, TiltHookArgs


def generate_global_tilt(args: TiltHookArgs, props: GlobalTiltProps) -> NDArray[numpy.floating]:
    """
    Generate uniform simulated tilt array.

    Returns an array of shape (ny*nx, 2) where every row is [ty, tx].
    """
    xp = cast_array_module(args['xp'])

    ty, tx = props.tilt
    ny, nx = args['shape']

    base = xp.array([ty, tx], dtype=xp.float32)
    tilt_array = xp.broadcast_to(base, (ny, nx, 2))
    return tilt_array


def load_custom_tilt(args: TiltHookArgs, props: CustomTiltProps) -> NDArray[numpy.floating]:
    """
    Load tilt array from a .npy or .tilts file.

    A .npy file can have shape (ny, nx, 2) matching the scan, or shape (N, 2)
    where N == ny*nx. A .tilts file is an HDF5 file (as written by 4DSTEM
    Explorer's 'Sample Tilt' plugin), whose 'scan/tilt_x' and 'scan/tilt_y'
    datasets (mrad, one value per probe position) are stacked into [ty, tx] rows.

    If specified, `props.flips` (flip_y, flip_x, transpose) is applied to the
    map before it is checked against the scan shape.
    """
    xp = cast_array_module(args['xp'])
    shape = tuple(args['shape'])

    path = Path(props.path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Custom tilt file not found: {path}")

    ext = path.suffix.lower()
    if (loader := _TILT_LOADERS.get(ext)) is None:
        raise ValueError(f"Unsupported tilt file extension '{ext}'. Supported extensions: {', '.join(sorted(_TILT_LOADERS))}")

    # native shape of the map as stored in the file, such that flips (incl. transpose) yield the scan shape
    native_shape = shape[::-1] if props.flips is not None and props.flips[2] else shape
    tilt_data = loader(path, native_shape)

    if props.flips is not None:
        tilt_data = numpy.moveaxis(apply_flips(numpy.moveaxis(tilt_data, -1, 0), props.flips), 0, -1)

    if tilt_data.shape != (*shape, 2):
        extra = " (it matches the transposed scan; check scan orientation or the 'flips' prop)" if tilt_data.shape[:2] == shape[::-1] else ""
        raise ValueError(f"Tilt map shape {tilt_data.shape[:2]} doesn't match scan shape {shape}{extra}")

    return xp.array(tilt_data * props.scale, dtype=xp.float32)


def _load_tilts_npy(path: Path, native_shape: t.Tuple[int, ...]) -> NDArray[numpy.floating]:
    tilt_data = numpy.load(path)

    if tilt_data.ndim == 2:
        if tilt_data.shape != (numpy.prod(native_shape), 2):
            raise ValueError(f"Loaded tilt data shape {tilt_data.shape} is incompatible with expected 2D shape {(numpy.prod(native_shape), 2)}")
        tilt_data = tilt_data.reshape((*native_shape, 2))
    elif tilt_data.ndim != 3 or tilt_data.shape[-1] != 2:
        raise ValueError(f"Loaded tilt data must be a 2D or 3D array of [ty, tx] rows, got shape {tilt_data.shape}")

    return tilt_data


def _load_tilts_hdf5(path: Path, native_shape: t.Tuple[int, ...]) -> NDArray[numpy.floating]:
    with h5py.File(path, 'r') as f:
        try:
            tilt_x = numpy.asarray(f['scan/tilt_x'])
            tilt_y = numpy.asarray(f['scan/tilt_y'])
        except KeyError as e:
            raise ValueError(
                f"'{path}' doesn't look like a .tilts file: missing 'scan/tilt_x' or 'scan/tilt_y' dataset"
            ) from e

    if tilt_x.shape != tilt_y.shape:
        raise ValueError(f"Tilt map shape mismatch in '{path}': tilt_x {tilt_x.shape} vs tilt_y {tilt_y.shape}")

    return numpy.stack([tilt_y, tilt_x], axis=-1)


_TILT_LOADERS: t.Dict[str, t.Callable[[Path, t.Tuple[int, ...]], NDArray[numpy.floating]]] = {
    '.npy': _load_tilts_npy,
    '.tilts': _load_tilts_hdf5,
}
"""Tilt map loaders by file extension. Each takes (path, native_shape) and returns a (ny, nx, 2) array of [ty, tx] rows [mrad]."""
