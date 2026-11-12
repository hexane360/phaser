"""Tilt initialization hooks."""

import numpy
from numpy.typing import NDArray
from pathlib import Path

from phaser.utils.num import cast_array_module
from . import GlobalTiltProps, CustomTiltProps, TiltHookArgs


def generate_global_tilt(args: TiltHookArgs, props: GlobalTiltProps) -> NDArray[numpy.floating]:
    """Return a uniform tilt array of shape `(*scan_shape, 2)` [mrad].

    The scan need not be a raster: a flat list of positions (as from the `manual` scan hook) gives `(n, 2)`.
    """
    xp = cast_array_module(args['xp'])

    ty, tx = props.tilt

    base = xp.array([ty, tx], dtype=xp.float32)
    tilt_array = xp.broadcast_to(base, (*args['shape'], 2))
    return tilt_array


def load_custom_tilt(args: TiltHookArgs, props: CustomTiltProps) -> NDArray[numpy.floating]:
    """
    Load tilt array from a .npy file.

    The loaded array can have shape (ny, nx, 2) matching the scan shape,
    or shape (N, 2) where N is the number of scan positions, which will be reshaped accordingly.
    When the scan is a flat list of positions (as from the `manual` scan hook), a (ny, nx, 2)
    array with ny*nx == N is also accepted and flattened. In every case the tilts must be
    in the same order as the diffraction patterns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the array shape does not match the scan.
    """
    xp = cast_array_module(args['xp'])

    path = Path(props.path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Custom tilt file not found: {path}")

    tilt_data = numpy.load(path)

    shape = args['shape']
    expected_shape_3d = (*shape, 2)
    expected_shape_2d = (numpy.prod(shape), 2)

    if tilt_data.ndim == 3 and len(shape) == 1:
        # flat scan: only the number of positions can be checked
        if tilt_data.shape[-1] != 2 or numpy.prod(tilt_data.shape[:-1]) != shape[0]:
            raise ValueError(f"Loaded tilt data shape {tilt_data.shape} does not match the {shape[0]} scan positions")
        result = tilt_data.reshape(expected_shape_3d)
    elif tilt_data.ndim == 3:
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
