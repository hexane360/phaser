
import numpy
from numpy.typing import NDArray

from phaser.utils.num import cast_array_module
from phaser.utils.scan import make_raster_scan
from . import ScanHookArgs, RasterScanProps


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

    return scan