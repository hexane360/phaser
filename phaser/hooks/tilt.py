import numpy
from numpy.typing import NDArray
from pathlib import Path

import h5py

from phaser.utils.image import apply_flips
from phaser.utils.num import cast_array_module
from . import GlobalTiltProps, CustomTiltProps, TiltsFileProps, TiltHookArgs


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
    Load tilt array from a .npy file.

    The loaded array can have shape (ny, nx, 2) matching props.shape,
    or shape (N, 2) where N == ny*nx, which will be reshaped accordingly.
    """
    xp = cast_array_module(args['xp'])

    path = Path(props.path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Custom tilt file not found: {path}")

    tilt_data = numpy.load(path)

    shape = args['shape']
    expected_shape_3d = (*shape, 2)
    expected_shape_2d = (numpy.prod(shape), 2)

    if tilt_data.ndim == 3:
        if tilt_data.shape != expected_shape_3d:
            raise ValueError(f"Loaded tilt data shape {tilt_data.shape} does not match expected shape {expected_shape_3d}")
        result = tilt_data
    elif tilt_data.ndim == 2:
        if tilt_data.shape != expected_shape_2d:
            raise ValueError(f"Loaded tilt data shape {tilt_data.shape} is incompatible with expected 2D shape {expected_shape_2d}")
        result = tilt_data.reshape(expected_shape_3d)
    else:
        raise ValueError(f"Loaded tilt data must be 2D or 3D array, got shape {tilt_data.shape}")

    return xp.array(result, dtype=xp.float32)


def load_tilts_file(args: TiltHookArgs, props: TiltsFileProps) -> NDArray[numpy.floating]:
    """
    Load a tilt map from a .tilts HDF5 file (as written by 4DSTEM Explorer's 'Sample Tilt' plugin).

    Reads the 'scan/tilt_x' and 'scan/tilt_y' datasets (mrad, one value per probe position)
    and returns an array of shape (ny, nx, 2) where every row is [ty, tx].

    If specified, `props.flips` (flip_y, flip_x, transpose) is applied to the maps
    before they are checked against the scan shape.
    """
    xp = cast_array_module(args['xp'])

    path = Path(props.path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Tilts file not found: {path}")

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

    tilt_x = apply_flips(tilt_x, props.flips)
    tilt_y = apply_flips(tilt_y, props.flips)

    shape = tuple(args['shape'])
    if tilt_x.shape != shape:
        extra = " (it matches the transposed scan; check scan orientation or the 'flips' prop)" if tilt_x.shape == shape[::-1] else ""
        raise ValueError(f"Tilt map shape {tilt_x.shape} doesn't match scan shape {shape}{extra}")

    tilt = numpy.stack([tilt_y, tilt_x], axis=-1) * props.scale
    return xp.array(tilt, dtype=xp.float32)
