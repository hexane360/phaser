"""Scan creation utilities."""

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
    """Return a raster scan centered on the origin.

    Args:
        shape: Scan shape `(n_y, n_x)`.
        scan_step: Step size `(s_y, s_x)` [angstrom].
        rotation: Rotation about the scan center [degrees CCW].
        affine: Affine matrix applied to the positions before `rotation`.
        dtype: Position dtype. Defaults to `numpy.float64`.
        xp: Array module.

    Returns:
        Positions of shape `(n_y, n_x, 2)`, with the last axis in `(y, x)` order [angstrom].
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