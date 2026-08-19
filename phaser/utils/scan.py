"""
Utilities for probe positions/scan
"""

import typing as t

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .num import FloatT, cast_array_module, get_array_module


@t.overload
def make_raster_scan(shape: tuple[int, int], scan_step: ArrayLike,
                     rotation: float = 0., affine: None | ArrayLike = None, *, dtype: FloatT, xp: t.Any = None) -> NDArray[FloatT]:
    ...

@t.overload
def make_raster_scan(shape: tuple[int, int], scan_step: ArrayLike,
                     rotation: float = 0., affine: None | ArrayLike = None, *, dtype: DTypeLike | None = None, xp: t.Any = None) -> NDArray[numpy.floating]:
    ...

def make_raster_scan(shape: tuple[int, int], scan_step: ArrayLike,
                     rotation: float = 0., affine: None | ArrayLike = None, *, dtype: t.Any = None, xp: t.Any = None) -> NDArray[numpy.floating]:
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
    pts *= xp2.broadcast_to(xp2.asarray(scan_step, dtype=dtype), (2,))

    if affine is not None:
        affine = xp2.asarray(affine, dtype=dtype)
        pts = (pts @ affine.T)

    if rotation != 0.:
        theta = rotation * numpy.pi/180.
        mat = xp2.asarray([[numpy.cos(theta), -numpy.sin(theta)], [numpy.sin(theta), numpy.cos(theta)]], dtype=dtype)
        pts = (pts @ mat.T)

    return t.cast(NDArray[numpy.floating], pts)


__all__ = [
    'make_raster_scan',
]