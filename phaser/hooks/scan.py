
import logging
import numpy
from numpy.typing import NDArray

from phaser.utils.num import cast_array_module
from phaser.utils.scan import make_raster_scan
from . import ScanHookArgs, RasterScanProps


def raster_scan(args: ScanHookArgs, props: RasterScanProps) -> NDArray[numpy.floating]:
    xp = cast_array_module(args['xp'])
    logger = logging.getLogger(__name__)

    if props.shape is None:
        raise ValueError("scan 'shape' must be specified by metadata or manually")
    if props.step_size is None:
        raise ValueError("scan 'step_size' must be specified by metadata or manually")
    step_size = numpy.broadcast_to(props.step_size, (2,))
    rot = props.rotation or 0.0

    if props.affine is not None:
        affine = xp.asarray(props.affine, dtype=args['dtype'])
    else:
        affine = None

    logger.info(f"Making raster scan, shape {props.shape},"
                f" step size [{step_size[0]:.2f}, {step_size[1]:.2f}],"
                f" rotation {rot:.2f} deg"
                f" affine transformation {affine.ravel() if affine is not None else 'None'}")
    
    scan = make_raster_scan(
        props.shape, step_size, rot, affine,
        dtype=args['dtype'], xp=xp,
    )

    return scan
