"""
Utilities for probe positions/scan
"""

import typing as t

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .num import get_array_module, cast_array_module, NumT


@t.overload
def make_raster_scan(shape: t.Tuple[int, int], scan_step: ArrayLike,  # pyright: ignore[reportOverlappingOverload]
                     rotation: float = 0., *, dtype: NumT, xp: t.Any = None) -> NDArray[NumT]:
    ...

@t.overload
def make_raster_scan(shape: t.Tuple[int, int], scan_step: ArrayLike,
                     rotation: float = 0., *, dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> NDArray[numpy.floating]:
    ...

def make_raster_scan(shape: t.Tuple[int, int], scan_step: ArrayLike,
                     rotation: float = 0., *, dtype: t.Any = None, xp: t.Any = None) -> NDArray[numpy.number]:
    """
    Make a raster scan, centered around the origin.

    Returns an array of shape `(n_y, n_x, 2)`, with the last dimension corresponding to `(y, x)` pairs.

    # Parameters

    - `shape`: Shape `(n_y, n_x)` of scan to create
    - `scan_step`: Scan step size `(s_y, s_x)`
    - `rotation`: Scan rotation to add (degrees CCW). Rotation is applied
      around the center of the scan.
    - `dtype`: Datatype of positions to return. Defaults to `numpy.float64`.
    - `xp`: Array module
    """
    xp2 = get_array_module(shape, scan_step) if xp is None else cast_array_module(xp)

    if dtype is None:
        dtype = numpy.float64

    # TODO actually center this around (0, 0)
    yy = xp2.arange(shape[0], dtype=dtype) - xp2.asarray(shape[0] / 2., dtype=dtype)
    xx = xp2.arange(shape[1], dtype=dtype) - xp2.asarray(shape[1] / 2., dtype=dtype)
    pts = xp2.stack(xp2.meshgrid(yy, xx, indexing='ij'), axis=-1)


    if rotation != 0.:
        theta = rotation * numpy.pi/180.
        mat = xp2.asarray([[numpy.cos(theta), -numpy.sin(theta)], [numpy.sin(theta), numpy.cos(theta)]], dtype=dtype)
        pts = (pts @ mat.T)

    return pts * xp2.broadcast_to(xp2.asarray(scan_step), (2,)).astype(dtype)  # type: ignore


__all__ = [
    'make_raster_scan',
]