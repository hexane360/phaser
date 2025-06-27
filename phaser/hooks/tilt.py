import numpy
from numpy.typing import NDArray
from pathlib import Path

from phaser.utils.num import cast_array_module
from . import SimuTiltProps, CustomTiltProps, TiltHookArgs


def generate_simu_tilt(args: TiltHookArgs, props: SimuTiltProps) -> NDArray[numpy.floating]:
    """
    Generate uniform simulated tilt array.

    Returns an array of shape (ny*nx, 2) where every row is [ty, tx].
    """
    xp = cast_array_module(args['xp'])

    if props.shape is None:
        ValueError("scan 'shape' must be specified by metadata or manually")

    ty, tx = props.tilt
    ny, nx = props.shape
    total_points = ny * nx

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

    path = Path(props.path)
    if not path.exists():
        raise FileNotFoundError(f"Custom tilt file not found: {path}")

    tilt_data = numpy.load(path)

    if props.shape is None:
        ValueError("scan 'shape' must be specified by metadata or manually")

    ny, nx = props.shape
    expected_shape_3d = (ny, nx, 2)
    expected_size_2d = ny * nx

    if tilt_data.ndim == 3:
        if tilt_data.shape != expected_shape_3d:
            raise ValueError(f"Loaded tilt data shape {tilt_data.shape} does not match expected shape {expected_shape_3d}")
        result = tilt_data
    elif tilt_data.ndim == 2:
        if tilt_data.shape[0] != expected_size_2d or tilt_data.shape[1] != 2:
            raise ValueError(f"Loaded tilt data shape {tilt_data.shape} is incompatible with expected 2D shape ({expected_size_2d}, 2)")
        result = tilt_data.reshape(expected_shape_3d)
    else:
        raise ValueError(f"Loaded tilt data must be 2D or 3D array, got shape {tilt_data.shape}")

    return xp.array(result, dtype=xp.float32)
