
import numpy
from numpy.typing import NDArray
from pathlib import Path

from phaser.utils.num import cast_array_module
from phaser.utils.scan import make_raster_scan
from . import ScanHookArgs, RasterScanProps, CustomScanProps


def raster_scan(args: ScanHookArgs, props: RasterScanProps) -> NDArray[numpy.floating]:
    xp = cast_array_module(args['xp'])

    if props.shape is None:
        raise ValueError("scan 'shape' must be specified by metadata or manually")
    if props.step_size is None:
        raise ValueError("scan 'step_size' must be specified by metadata or manually")

    scan = make_raster_scan(
        props.shape, props.step_size, props.rotation or 0.0,
        dtype=args['dtype'], xp=xp,
    )

    if props.affine is not None:
        affine = xp.array(props.affine, dtype=scan.dtype)
        # equivalent to (affine @ scan.T).T (active transformation)
        scan = scan @ affine.T

    numpy.save('scan', scan)
    return scan


def load_custom_scan(args: ScanHookArgs, props: CustomScanProps) -> NDArray[numpy.floating]:
# def load_custom_tilt(args: TiltHookArgs, props: CustomTiltProps) -> NDArray[numpy.floating]:
    """
    Load scan array from a .npy file.

    The loaded array can have shape (ny, nx, 2) matching props.shape,
    or shape (N, 2) where N == ny*nx, which will be reshaped accordingly.
    """
    xp = cast_array_module(args['xp'])

    path = Path(props.path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Custom scan file not found: {path}")

    scan = numpy.load(path)

    # shape = args['shape']
    # expected_shape_3d = (*shape, 2)
    # expected_shape_2d = (numpy.prod(shape), 2)

    # if tilt_data.ndim == 3:
    #     if tilt_data.shape != expected_shape_3d:
    #         raise ValueError(f"Loaded tilt data shape {tilt_data.shape} does not match expected shape {expected_shape_3d}")
    #     result = tilt_data
    # elif tilt_data.ndim == 2:
    #     if tilt_data.shape != expected_shape_2d:
    #         raise ValueError(f"Loaded tilt data shape {tilt_data.shape} is incompatible with expected 2D shape {expected_shape_2d}")
    #     result = tilt_data.reshape(expected_shape_3d)
    # else:
    #     raise ValueError(f"Loaded tilt data must be 2D or 3D array, got shape {tilt_data.shape}")
    print("loaded scan")
    return xp.array(scan, dtype=xp.float32)
