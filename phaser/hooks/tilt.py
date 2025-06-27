import numpy
from numpy.typing import NDArray
from pathlib import Path

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
