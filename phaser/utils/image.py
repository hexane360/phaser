"""
Utilities for image processing & filtering
"""

import abc
import dataclasses
import functools
import typing as t
import warnings

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray
from typing_extensions import Self

from .num import (
    Float,
    Sampling,
    abs2,
    at,
    cast_array_module,
    dct2,
    fft2shift,
    get_array_module,
    get_scipy_module,
    idct2,
    ifft2shift,
    max_supported_float,
    pad,
    to_complex_dtype,
    to_numpy,
    to_real_dtype,
    xp_is_jax,
    xp_is_torch,
)
from .tree import tree_dataclass

NumT = t.TypeVar('NumT', bound=numpy.number)
InexactT = t.TypeVar('InexactT', bound=numpy.inexact)
ComplexT = t.TypeVar('ComplexT', bound=numpy.complexfloating)
NumT_co = t.TypeVar('NumT_co', bound=numpy.number, covariant=True)
InexactT_co = t.TypeVar('InexactT_co', bound=numpy.inexact, covariant=True)


def apply_flips(
    data: NDArray[NumT], flips: t.Optional[t.Tuple[bool, bool, bool]]
) -> NDArray[NumT]:
    """
    Applies flips to `data` along the last two axes.
    If specified, transpose is applied last (after X and Y flips).

    Parameters:
        data: Input data array.
        flips: Tuple of three booleans `(flip_y, flip_x, transpose)`.

    Returns: `data` with the specified flips applied.
    """
    if flips is None or not any(flips):
        return data

    xp = get_array_module(data)

    if flips[0]:
        data = xp.flip(data, axis=-2)
    if flips[1]:
        data = xp.flip(data, axis=-1)
    if flips[2]:
        data = xp.moveaxis(data, -2, -1)

    return data


@t.overload
def remove_linear_ramp(  # pyright: ignore[reportOverlappingOverload]
    data: NDArray[NumT], mask: t.Optional[NDArray[numpy.bool_]] = None
) -> NDArray[NumT]:
    ...

@t.overload
def remove_linear_ramp(
    data: ArrayLike, mask: t.Optional[NDArray[numpy.bool_]] = None
) -> NDArray[numpy.float64]:
    ...

def remove_linear_ramp(
    data: ArrayLike, mask: t.Optional[NDArray[numpy.bool_]] = None
) -> NDArray[numpy.number]:
    """
    Removes a linear 'ramp' from an image or stack of images.
    """
    xp = get_array_module(data)
    float_dtype = max_supported_float(xp)
    output = xp.empty_like(data)

    data = xp.array(data)

    (yy, xx) = (arr.flatten() for arr in xp.indices(data.shape[-2:], dtype=float_dtype))
    pts = xp.stack((xp.ones_like(xx), xx, yy), axis=-1)

    if mask is None:
        mask = xp.ones(len(yy), dtype=numpy.bool_)
    else:
        mask = mask.flatten()

    for idx in numpy.ndindex(data.shape[:-2]):
        layer = data[tuple(idx)].astype(float_dtype)
        p, _residues, _rank, _singular = xp.linalg.lstsq(pts[mask], layer.flatten()[mask], rcond=None)
        output = at(output, idx).set((layer - (p @ pts.T).reshape(layer.shape)).astype(output.dtype))

    return output


def colorize_complex(vals: ArrayLike, amp: bool = False, rescale: bool = True) -> NDArray[numpy.floating]:
    """Colorize a ndarray of complex values as rgb."""
    from matplotlib.colors import hsv_to_rgb
    xp = get_array_module(vals)

    vals = xp.asarray(vals)
    # promote to complex
    vals = vals.astype(numpy.promote_types(vals.dtype, numpy.complex64))

    v = xp.abs(vals) if amp else abs2(vals)
    if rescale:
        v /= xp.max(v)
    arg = xp.angle(vals) 

    h = (arg + numpy.pi) / (2*numpy.pi)
    s = 0.85 * xp.ones_like(v)
    return xp.clip(hsv_to_rgb(to_numpy(xp.stack((h, s, v), axis=-1))), 0.0, 1.0)


def scale_to_integral_type(
    arr: NDArray[numpy.floating],
    ty: t.Literal['8bit', '16bit', '32bit', '64bit'],
    mask: t.Optional[NDArray[numpy.bool_]] = None,
    min_range: t.Optional[float] = None,
) -> NDArray[numpy.unsignedinteger]:
    xp = get_array_module(arr)

    dtype = {
        '8bit': numpy.uint8,
        '16bit': numpy.uint16,
        '32bit': numpy.uint32,
        '64bit': numpy.uint64,
    }[ty]

    imax = numpy.iinfo(dtype).max

    arr_crop = arr[..., mask] if mask is not None else arr
    # TODO: cupy doesn't support nanquantile
    vmax = xp.nanquantile(arr_crop, 0.999)
    vmin = xp.nanquantile(arr_crop, 0.001)

    if min_range is not None and (delta := min_range - (vmax - vmin)) > 0:
        # expand max and min to cover min_range
        vmax += delta/2
        vmin -= delta/2

    return (xp.clip((imax + 1) / (vmax - vmin) * (arr - vmin), 0, imax)).astype(dtype)


_InterpBoundaryMode: t.TypeAlias = t.Literal['constant', 'nearest', 'mirror', 'reflect', 'wrap', 'grid-mirror', 'grid-wrap', 'grid-constant']
_FilterBoundaryMode: t.TypeAlias = t.Literal['reflect', 'grid-wrap']
_RecipBoundaryMode: t.TypeAlias = t.Union[_FilterBoundaryMode, t.Literal['reflect_dct', 'reflect_adjoint']]


def to_affine_matrix(arr: ArrayLike, ndim: int = 2) -> NDArray[numpy.floating]:
    arr = numpy.asarray(arr)

    if arr.shape == (ndim, ndim):
        arr = numpy.block([[arr, numpy.zeros((ndim, 1))], [numpy.zeros((1, ndim)), 1.]])
    elif arr.shape == (ndim,):
        arr = numpy.diag([*arr, 1.])
    elif arr.shape == (ndim+1,):
        arr = numpy.diag(arr)
    elif arr.shape != (ndim+1, ndim+1):
        raise ValueError(f"Expected an affine matrix of shape ({ndim}, {ndim}), ({ndim+1}, {ndim+1}),"
                         f" ({ndim+1},), or ({ndim},), instead got shape: {arr.shape}")

    assert arr.shape == (ndim+1, ndim+1)
    return arr.astype(numpy.promote_types(arr.dtype, numpy.float32)) #arr.astype(numpy.floating)


def scale_matrix(scale: ArrayLike) -> NDArray[numpy.floating]:
    scale = numpy.asarray(scale)
    assert scale.ndim == 1
    a = numpy.diag([*scale, 1.0])
    return a.astype(numpy.promote_types(a.dtype, numpy.float32))


def translation_matrix(vec: ArrayLike) -> NDArray[numpy.floating]:
    vec = numpy.asarray(vec)
    assert vec.ndim == 1
    a = numpy.eye(vec.size + 1, dtype=vec.dtype)
    a[:vec.size, vec.size] = vec
    return a.astype(numpy.promote_types(a.dtype, numpy.float32)) #a.astype(numpy.floating)


def rotation_matrix(theta: float) -> NDArray[numpy.floating]:
    t = theta * numpy.pi/180.

    return numpy.array([
        [numpy.cos(t), numpy.sin(t), 0.,],
        [-numpy.sin(t), numpy.cos(t), 0.],
        [0., 0., 1.],
    ])


def affine_transform(
    input: NDArray[NumT],
    matrix: ArrayLike,
    offset: t.Optional[ArrayLike] = None,
    output_shape: t.Optional[t.Tuple[int, ...]] = None,
    order: int = 1,
    mode: _InterpBoundaryMode = 'grid-constant',
    cval: t.Union[NumT, float] = 0.0,
) -> NDArray[NumT]:
    if mode in ('constant', 'wrap'):
        # these modes aren't supported by jax
        raise ValueError(f"Resampling mode '{mode}' not supported (try 'grid-constant' or 'grid-wrap' instead)")

    xp = get_array_module(input, matrix, offset)

    if xp_is_torch(xp):
        from ._torch_kernels import affine_transform, torch
        return t.cast(NDArray[NumT], affine_transform(
            t.cast(torch.Tensor, input), matrix, offset,
            output_shape, order, mode, cval 
        ))

    if xp_is_jax(xp):
        if order not in (0, 1):
            raise ValueError(f"Interpolation order {order} not supported (jax currently only supports order=0, 1)")
        from ._jax_kernels import affine_transform, jax
        return t.cast(NDArray[NumT], affine_transform(
            t.cast(jax.Array, input), matrix, offset,
            output_shape, order, mode, cval
        ))

    scipy = get_scipy_module(input, matrix, offset)

    if offset is None:
        offset = 0.
    if output_shape is None:
        output_shape = t.cast(t.Tuple[int, ...], input.shape)
    n_axes = len(output_shape)  # num axes to transform over

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message="The behavior of affine_transform with a 1-D array")

        output = xp.empty((*input.shape[:-n_axes], *output_shape), dtype=input.dtype)

        for idx in numpy.ndindex(input.shape[:-n_axes]):  # TODO: parallelize this on CUDA?
            scipy.ndimage.affine_transform(
                input[tuple(idx)], xp.array(matrix), offset=offset,
                output_shape=output_shape, output=output[tuple(idx)],
                order=order, mode=mode, cval=cval,
            )

        return output


def _split_pair(val: t.Union[Float, t.Tuple[Float, Float]]) -> t.Tuple[float, float]:
    return (float(val[0]), float(val[1])) if isinstance(val, (tuple, list)) else (float(val), float(val))


def _canonicalize_axis(axis: int, num_dims: int) -> int:
  """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
  axis = axis.__index__()
  if not -num_dims <= axis < num_dims:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {num_dims}")
  if axis < 0:
        axis = axis + num_dims
  return axis


def convolve1d(
    arr: NDArray[NumT], weights: ArrayLike, axis: int = -1, *,
    mode: _InterpBoundaryMode = 'reflect', cval: t.Union[NumT, float] = 0.
) -> NDArray[NumT]:
    """
    Convolve `arr` with the 1D filter `weights` along `axis`.

    Parameters:
        arr: Array to filter.
        weights: 1D filter to convolve with. May only be complex if `arr` is complex.
        axis: Axis of `arr` to filter along.
        mode: How to extend `arr` past its boundaries.
        cval: Fill value, for `mode='constant'` and `mode='grid-constant'`.

    Returns: Array of the same shape as `arr`.
    """
    xp = get_array_module(arr, weights)

    arr = xp.asarray(arr)
    weights = xp.asarray(weights)
    if weights.ndim != 1:
        raise ValueError("convolve1d: Expected 'weights' to be 1D")
    _check_real_psf(weights, xp.iscomplexobj(arr), xp, 'convolve1d')
    axis = _canonicalize_axis(axis, arr.ndim)

    if xp_is_torch(xp):
        import torch

        from ._torch_kernels import convolve1d

        return t.cast(NDArray[NumT], convolve1d(
            t.cast(torch.Tensor, arr),
            t.cast(torch.Tensor, weights),
            axis=axis, mode=mode, cval=cval
        ))

    if xp_is_jax(xp):
        from ._jax_kernels import convolve1d

        return t.cast(NDArray[NumT], convolve1d(
            arr, weights, axis,  # type: ignore
            mode=mode, cval=cval
        ))

    scipy = get_scipy_module(arr, weights)

    return scipy.ndimage.convolve1d(
        arr, weights, axis, mode=mode, cval=cval
    )


def convolve2d_separable(
    arr: NDArray[NumT], y_weights: ArrayLike, x_weights: t.Optional[ArrayLike] = None, *,
    mode: _InterpBoundaryMode = 'reflect', cval: t.Union[NumT, float] = 0.
) -> NDArray[NumT]:
    """
    Convolve the last two axes of `arr` with a separable 2D filter.

    Equivalent to [`convolve2d`][phaser.utils.image.convolve2d] with the outer
    product `y_weights[:, None] * x_weights[None, :]`.

    Parameters:
        arr: Array to filter.
        y_weights: 1D filter to convolve along the second-to-last axis. May only be
                   complex if `arr` is complex.
        x_weights: 1D filter to convolve along the last axis. Defaults to `y_weights`.
                   May only be complex if `arr` is complex.
        mode: How to extend `arr` past its boundaries.
        cval: Fill value, for `mode='constant'` and `mode='grid-constant'`.

    Returns: Array of the same shape as `arr`.
    """
    xp = get_array_module(arr, y_weights, x_weights)

    y_weights = xp.asarray(y_weights)
    if y_weights.ndim != 1:
        raise ValueError("convolve2d_separable: Expected 'y_weights' to be 1D")

    if x_weights is None:
        x_weights = y_weights
    else:
        x_weights = xp.asarray(x_weights)
        if x_weights.ndim != 1:
            raise ValueError("convolve2d_separable: Expected 'x_weights' to be 1D")

    # the second pass fills against a boundary the first pass already filtered,
    # scaling its fill value by the dc gain of `y_weights`
    x_cval = cval * xp.sum(y_weights) if mode in ('constant', 'grid-constant') and cval != 0. else cval

    return convolve1d(
        convolve1d(arr, y_weights, axis=-2, mode=mode, cval=cval),
        x_weights, axis=-1, mode=mode, cval=x_cval,
    )


@t.overload
def convolve2d(
    arr: NDArray[NumT], filte: t.Union['Filter', ArrayLike], /, *,
    mode: _InterpBoundaryMode = 'reflect', cval: t.Union[NumT, float] = 0.
) -> NDArray[NumT]:
    ...

@t.overload
def convolve2d(
    arr: ArrayLike, weights: t.Union['Filter', ArrayLike], /, *,
    mode: _InterpBoundaryMode = 'reflect', cval: float = 0.
) -> numpy.ndarray:
    ...

def convolve2d(
    arr: ArrayLike, weights: t.Union['Filter', ArrayLike], /, *,
    mode: _InterpBoundaryMode = 'reflect', cval: t.Union[numpy.number, float] = 0.
) -> numpy.ndarray:
    """
    Convolve the last two axes of `arr` with a filter or 2D filter weights,
    performing the convolution in real space.

    Parameters:
        arr: Array to filter.
        weights: 2D filter to convolve with, or a `Filter` to evaluate at unit sampling.
                 May only be complex if `arr` is complex.
        mode: How to extend `arr` past its boundaries.
        cval: Fill value, for `mode='constant'` and `mode='grid-constant'`.

    Returns: Array of the same shape as `arr`.
    """
    if isinstance(weights, Filter):
        xp = get_array_module(arr)
        arr = xp.asarray(arr)
        samp = Sampling(_canonicalize_shape(arr, 'convolve2d'), sampling=(1., 1.))
        dtype = to_complex_dtype(arr.dtype) if xp.iscomplexobj(arr) else to_real_dtype(arr.dtype)
        return prepare_convolve2d(weights, samp, mode=mode, cval=cval, xp=xp, dtype=dtype)(arr)

    xp = get_array_module(arr, weights)
    arr = xp.asarray(arr)
    weights = xp.asarray(weights)
    if weights.ndim != 2:
        raise ValueError("convolve2d: Expected 'weights' to be 2D")
    _check_real_psf(weights, xp.iscomplexobj(arr), xp, 'convolve2d')

    if xp_is_torch(xp):
        from ._torch_kernels import convolve2d

        return convolve2d(
            arr, weights,  # type: ignore
            mode=mode, cval=cval,
        )
    if xp_is_jax(xp):
        from ._jax_kernels import convolve2d

        return convolve2d(
            arr, weights,  # type: ignore
            mode=mode, cval=cval,
        )

    scipy = get_scipy_module(arr, weights)

    output = xp.empty_like(arr)
    # we can't use the axes parameter, we support too old of scipy
    scipy.ndimage.convolve(
        arr, weights[(None,) * (arr.ndim - weights.ndim)], output=output,
        mode=mode, cval=cval,
    )
    return output


def _canonicalize_shape(arr: NDArray[t.Any], name: str) -> t.Tuple[int, int]:
    if arr.ndim < 2:
        raise ValueError(f"{name}: Expected 'arr' to be at least 2D, instead got shape {tuple(arr.shape)}")
    return (int(arr.shape[-2]), int(arr.shape[-1]))


def _cast_filter(arr: NDArray[numpy.number], dtype: DTypeLike) -> NDArray[numpy.inexact]:
    xp = get_array_module(arr)
    return arr.astype(to_complex_dtype(dtype) if xp.iscomplexobj(arr) else dtype)


def _cast_transfer(arr: NDArray[numpy.number], dtype: DTypeLike, symmetric: bool) -> NDArray[numpy.inexact]:
    """
    Cast a transfer function to `dtype`, forcing it complex unless `symmetric`.
    `dtype` must be resolved (non-`None`).
    """
    xp = get_array_module(arr)
    return arr.astype(to_complex_dtype(dtype)) if not symmetric else xp.real(arr).astype(dtype)


def _check_real_psf(psf: NDArray[numpy.number], complex_target: bool, xp: t.Any, name: str) -> None:
    """A complex point spread function can only be used to filter a complex array."""
    if xp.iscomplexobj(psf) and not complex_target:
        raise ValueError(f"{name}: Expected a real point spread function for a real array")


def _sampling_shape(samp: Sampling) -> t.Tuple[int, int]:
    return (int(samp.shape[0]), int(samp.shape[1]))


def _resolve_xp(xp: t.Any, *arrs: t.Any) -> t.Any:
    """Resolve an explicit `xp`, or infer it from `arrs` (falling back to numpy)."""
    if xp is not None:
        return cast_array_module(xp)
    return get_array_module(*arrs) if arrs else numpy


def _resolve_xp_dtype(
    xp: t.Any, dtype: t.Optional[DTypeLike], *arrs: t.Any
) -> t.Tuple[t.Any, DTypeLike]:
    """Resolve `xp` (as [`_resolve_xp`][phaser.utils.image._resolve_xp]) and `dtype`,
    defaulting the latter to `xp`'s max supported float."""
    xp = _resolve_xp(xp, *arrs)
    return xp, (dtype or max_supported_float(xp))


class Filter(abc.ABC):
    """
    A 2D image filter, specified in real space (a point spread function) or in
    reciprocal space (a transfer function).

    Subclasses implement whichever of [`psf`][phaser.utils.image.Filter.psf] and
    [`transfer_function`][phaser.utils.image.Filter.transfer_function] they natively
    have; the other is derived automatically.

    LSI filters form a commutative ring under [`__add__`][phaser.utils.image.Filter.__add__]
    (pointwise sum of point spread functions/transfer functions, identity
    [`SumFilter(())`][phaser.utils.image.SumFilter]) and
    [`__mul__`][phaser.utils.image.Filter.__mul__] (composition, identity
    [`ProductFilter(())`][phaser.utils.image.ProductFilter]).
    """

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.psf is Filter.psf and cls.transfer_function is Filter.transfer_function:
            raise TypeError(f"{cls.__name__} must implement 'psf' or 'transfer_function'")

    symmetric: bool = False
    """
    Whether the point spread function of the filter is real and even.
    """

    def psf(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        """
        Return the point spread function of the filter, evaluated for data sampled on `samp`.

        The kernel is centered at `tuple(n // 2 for n in kernel.shape)`, matching the
        convention of [`PsfFilter`][phaser.utils.image.PsfFilter] and
        [`convolve2d`][phaser.utils.image.convolve2d]. When the filter has a natural
        compact representation (e.g. [`SeparableFilter`][phaser.utils.image.SeparableFilter],
        [`PsfFilter`][phaser.utils.image.PsfFilter], and composites of these), the returned
        kernel may be smaller than `samp.shape`; use
        [`_embed_psf`][phaser.utils.image._embed_psf] to place it on the full grid.
        This default implementation has no such compact form to fall back on (a transfer
        function alone carries no locality information), so it always returns an array of
        shape `samp.shape`.

        Parameters:
            samp: Sampling of the data to filter.
            xp: Array module to return an array of.
            dtype: Floating point precision to work in.

        Returns: Array of shape `samp.shape` (or smaller, for a filter with a compact
                 representation). If `dtype` is real (the default), the transfer function
                 is assumed to be Hermitian and only the real part is kept.
        """
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        psf = fft2shift(xp.fft.ifft2(self.transfer_function(samp, xp=xp, dtype=dtype)))
        complex_dtype = numpy.issubdtype(dtype, numpy.complexfloating)
        return (psf if complex_dtype else xp.real(psf)).astype(dtype)

    def transfer_function(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        """
        Return the transfer function of the filter, sampled at the frequencies
        `samp.recip_grid()` (cycles/length, in fft order).

        Parameters:
            samp: Sampling of the data to filter.
            xp: Array module to return an array of.
            dtype: Floating point precision to work in.

        Returns: Array of shape `samp.shape`.
        """
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        kernel = self.psf(samp, xp=xp, dtype=dtype)
        psf = _embed_psf(kernel, _sampling_shape(samp), xp, kernel.dtype)
        transfer = xp.fft.fft2(ifft2shift(psf))
        return _cast_transfer(transfer, dtype, self.symmetric)

    def transfer_function_sym(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        """
        Return the transfer function of the filter on the doubled grid used for
        symmetric (`'reflect'`) boundaries, i.e. the transfer function of a sampling
        of twice the extent.

        Parameters:
            samp: Sampling of the data to filter (*not* of the returned array).
            xp: Array module to return an array of.
            dtype: Floating point precision to work in.

        Returns: Array of shape `2 * samp.shape`.
        """
        (n, m) = _sampling_shape(samp)
        doubled = Sampling((2 * n, 2 * m), sampling=samp.sampling)
        return self.transfer_function(doubled, xp=xp, dtype=dtype)

    def transfer_function_dct(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.floating]:
        """
        Return the transfer function of the filter on the grid diagonalized by a type-II
        DCT, i.e. the frequencies `j / (2 n s)`, `j` in `[0, n)`. This is the non-negative
        quadrant of [`transfer_function_sym`][phaser.utils.image.Filter.transfer_function_sym].

        Only meaningful for a [`symmetric`][phaser.utils.image.Filter.symmetric] filter
        (calling it otherwise is a logic error); always returns real.

        Parameters:
            samp: Sampling of the data to filter.
            xp: Array module to return an array of.
            dtype: Floating point precision to work in.

        Returns: Array of shape `samp.shape`.
        """
        (n, m) = _sampling_shape(samp)
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        transfer = self.transfer_function_sym(samp, xp=xp, dtype=dtype)[..., :n, :m]
        return xp.real(transfer).astype(to_real_dtype(dtype))

    def __mul__(self, other: 't.Union[Filter, float]') -> 'Filter':
        """
        Compose with another filter, or scale by a real amplitude factor.

        Composing two filters (`filt1 * filt2`) is the filter obtained by applying
        `self` then `other` (or in either order, since LSI filters commute): point
        spread functions convolve, transfer functions multiply. Scaling by a scalar
        (`filt * 2.`) multiplies both the point spread function and transfer function
        by that scalar.
        """
        if isinstance(other, Filter):
            return _multiply_filters(self, other)
        if isinstance(other, (int, float)):
            return _scale_filter(self, float(other))
        return NotImplemented

    def __rmul__(self, other: float) -> 'Filter':
        if isinstance(other, (int, float)):
            return _scale_filter(self, float(other))
        return NotImplemented

    def __add__(self, other: 'Filter') -> 'Filter':
        """
        Add another filter (`filt1 + filt2`): point spread functions and transfer
        functions add pointwise.
        """
        if isinstance(other, Filter):
            return _add_filters(self, other)
        return NotImplemented

    def __radd__(self, other: 'Filter') -> 'Filter':
        if isinstance(other, Filter):
            return _add_filters(other, self)
        return NotImplemented


def _embed_psf(
    kernel: NDArray[numpy.number], shape: t.Tuple[int, int], xp: t.Any, dtype: DTypeLike
) -> NDArray[numpy.inexact]:
    """Place `kernel` (centered at `kernel.shape // 2`) into a centered array of `shape`."""
    (a, b), (n, m) = kernel.shape, shape
    if a > n or b > m:
        raise ValueError(f"Filter kernel of shape {(a, b)} doesn't fit in a grid of shape {shape}")

    kernel = _cast_filter(kernel, dtype)
    iy = xp.arange(a) - a // 2 + n // 2
    ix = xp.arange(b) - b // 2 + m // 2
    return at(xp.zeros(shape, dtype=kernel.dtype), (iy[:, None], ix[None, :])).set(kernel)


def _pad_centered_1d(
    kernel: NDArray[numpy.number], n: int, xp: t.Any, dtype: DTypeLike
) -> NDArray[numpy.inexact]:
    """Place 1D `kernel` (centered at `kernel.shape[-1] // 2`) into a centered array of length `n`."""
    a = kernel.shape[-1]
    if a > n:
        raise ValueError(f"Filter kernel of length {a} doesn't fit in a grid of length {n}")

    kernel = _cast_filter(kernel, dtype)
    i = xp.arange(a) - a // 2 + n // 2
    return at(xp.zeros(n, dtype=kernel.dtype), i).set(kernel)


def _convolve_kernels_1d(
    a: NDArray[numpy.inexact], b: NDArray[numpy.inexact], xp: t.Any, dtype: DTypeLike
) -> NDArray[numpy.inexact]:
    """
    Full linear convolution of two centered 1D kernels (`numpy.convolve(a, b, 'full')`),
    keeping the same 'centered at `n // 2`' convention as [`_embed_psf`][phaser.utils.image._embed_psf].
    """
    if a.shape[-1] == 1:
        return (b * a[..., 0]).astype(dtype)
    if b.shape[-1] == 1:
        return (a * b[..., 0]).astype(dtype)
    n = a.shape[-1] + b.shape[-1] - 1
    work_dtype = to_complex_dtype(dtype)
    fa = xp.fft.fft(xp.fft.ifftshift(_pad_centered_1d(a, n, xp, work_dtype)))
    fb = xp.fft.fft(xp.fft.ifftshift(_pad_centered_1d(b, n, xp, work_dtype)))
    out = xp.fft.fftshift(xp.fft.ifft(fa * fb))
    complex_dtype = numpy.issubdtype(dtype, numpy.complexfloating)
    return (out if complex_dtype else xp.real(out)).astype(dtype)


def _convolve_kernels_2d(
    a: NDArray[numpy.inexact], b: NDArray[numpy.inexact], xp: t.Any, dtype: DTypeLike
) -> NDArray[numpy.inexact]:
    """
    Full linear convolution of two centered 2D kernels, keeping the same
    'centered at `shape // 2`' convention as [`_embed_psf`][phaser.utils.image._embed_psf].
    """
    shape = (a.shape[-2] + b.shape[-2] - 1, a.shape[-1] + b.shape[-1] - 1)
    work_dtype = to_complex_dtype(dtype)
    fa = xp.fft.fft2(ifft2shift(_embed_psf(a, shape, xp, work_dtype)))
    fb = xp.fft.fft2(ifft2shift(_embed_psf(b, shape, xp, work_dtype)))
    out = fft2shift(xp.fft.ifft2(fa * fb))
    complex_dtype = numpy.issubdtype(dtype, numpy.complexfloating)
    return (out if complex_dtype else xp.real(out)).astype(dtype)


@dataclasses.dataclass(frozen=True)
class PsfFilter(Filter):
    """
    A filter specified by a numeric point spread function, centered at `kernel.shape // 2`.

    `kernel` may be complex, but a complex `kernel` can only filter a complex array.
    """

    kernel: NDArray[numpy.inexact]
    symmetric: bool = False
    """Whether `kernel` is even."""

    def psf(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        xp, dtype = _resolve_xp_dtype(xp, dtype, self.kernel)
        kernel = xp.asarray(self.kernel)
        if kernel.ndim != 2:
            raise ValueError("PsfFilter: Expected 'kernel' to be 2D")
        _check_real_psf(kernel, numpy.issubdtype(dtype, numpy.complexfloating), xp, 'PsfFilter')
        return _cast_filter(kernel, dtype)


class SeparableFilter(Filter):
    """
    A filter whose point spread function is the outer product of two 1D kernels,
    and can therefore be applied one axis at a time (see
    [`convolve2d_separable`][phaser.utils.image.convolve2d_separable]).
    """

    @t.overload
    def __mul__(self, other: 'SeparableFilter') -> 'ProductSeparableFilter': ...
    @t.overload
    def __mul__(self, other: 't.Union[Filter, float]') -> 'Filter': ...

    def __mul__(self, other: 't.Union[Filter, float]') -> 'Filter':
        if isinstance(other, Filter):
            return _multiply_filters(self, other)
        if isinstance(other, (int, float)):
            return _scale_filter(self, float(other))
        return NotImplemented

    @abc.abstractmethod
    def psf_separable(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> t.Tuple[NDArray[numpy.inexact], NDArray[numpy.inexact]]:
        """
        Return the 1D kernels `(y_kernel, x_kernel)` whose outer product is the
        point spread function of the filter. Each kernel is centered at `n // 2`.

        The kernels are compact, so only the sample spacing of `samp` is used.

        Parameters:
            samp: Sampling of the data to filter.
            xp: Array module to return arrays of.
            dtype: Floating point precision to work in.
        """
        ...

    def psf(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        y_kernel, x_kernel = self.psf_separable(samp, xp=xp, dtype=dtype)
        return y_kernel[:, None] * x_kernel[None, :]


class AnalyticFilter(Filter):
    """
    A filter with a closed-form transfer function, which can be evaluated at any frequency.
    """

    @abc.abstractmethod
    def transfer_at(
        self, kyy: NDArray[numpy.floating], kxx: NDArray[numpy.floating], samp: Sampling
    ) -> NDArray[numpy.number]:
        """
        Return the transfer function of the filter, evaluated at the frequencies
        `(kyy, kxx)` (units of 1/length, matching the sample spacing of `samp`).

        The grid is not necessarily an fft grid of `samp`.
        """
        ...

    def transfer_function(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        kyy, kxx = samp.recip_grid(dtype=dtype, xp=xp)
        return _cast_transfer(xp.asarray(self.transfer_at(kyy, kxx, samp)), dtype, self.symmetric)

    def transfer_function_dct(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.floating]:
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        real_dtype = to_real_dtype(dtype)
        # evaluate directly on the dct grid
        # (samp.sampling is a numpy.float64 array, so dividing by an element of it
        # is a 'strong' scalar under NEP 50 and would silently upcast a float32 'dtype')
        ks = tuple(
            xp.arange(n, dtype=real_dtype) / xp.asarray(2. * n * s, dtype=real_dtype)
            for (n, s) in zip(_sampling_shape(samp), samp.sampling)
        )
        kyy, kxx = xp.meshgrid(*ks, indexing='ij')
        return xp.real(xp.asarray(self.transfer_at(kyy, kxx, samp))).astype(real_dtype)


@dataclasses.dataclass(frozen=True)
class SeparablePsfFilter(SeparableFilter):
    """
    A filter specified by the outer product `y_kernel[:, None] * x_kernel[None, :]`.

    `y_kernel`/`x_kernel` may be complex, but can only filter a complex array.
    """

    y_kernel: NDArray[numpy.inexact]
    x_kernel: NDArray[numpy.inexact]
    symmetric: bool = False
    """Whether both kernels are even (asserted, not checked)."""

    def psf_separable(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> t.Tuple[NDArray[numpy.inexact], NDArray[numpy.inexact]]:
        xp, dtype = _resolve_xp_dtype(xp, dtype, self.y_kernel, self.x_kernel)
        y_kernel, x_kernel = xp.asarray(self.y_kernel), xp.asarray(self.x_kernel)
        if y_kernel.ndim != 1 or x_kernel.ndim != 1:
            raise ValueError("SeparablePsfFilter: Expected 'y_kernel' and 'x_kernel' to be 1D")
        _check_real_psf(y_kernel, numpy.issubdtype(dtype, numpy.complexfloating), xp, 'SeparablePsfFilter')
        _check_real_psf(x_kernel, numpy.issubdtype(dtype, numpy.complexfloating), xp, 'SeparablePsfFilter')
        return (_cast_filter(y_kernel, dtype), _cast_filter(x_kernel, dtype))


def _upsample_axis(psf: NDArray[numpy.inexact], axis: int, xp: t.Any) -> NDArray[numpy.inexact]:
    """Zero-pad an fft-ordered `psf` from length `n` to `2n` along `axis`."""
    n = psf.shape[axis]
    c = n // 2
    pre = (slice(None),) * (axis % psf.ndim)

    def take(sl: slice) -> NDArray[numpy.inexact]:
        return psf[(*pre, sl)]

    if n % 2:
        pieces = (take(slice(None, c + 1)), xp.zeros_like(take(slice(None, n))), take(slice(c + 1, None)))
    else:
        # the nyquist tap is equally +n/2 and -n/2; splitting it keeps the psf symmetric
        nyq = take(slice(c, c + 1)) * 0.5
        pieces = (take(slice(None, c)), nyq, xp.zeros_like(take(slice(None, n - 1))), nyq, take(slice(c + 1, None)))
    return xp.concatenate(pieces, axis=axis)


@dataclasses.dataclass(frozen=True)
class TransferFilter(Filter):
    """A filter specified by a numeric transfer function, sampled on the fft grid of its own shape."""

    transfer: NDArray[numpy.number]
    symmetric: bool = False
    """Whether `transfer` is real and even."""

    def _check(self, shape: t.Tuple[int, int], xp: t.Any) -> NDArray[numpy.number]:
        transfer = xp.asarray(self.transfer)
        _check_transfer(transfer, shape, 'TransferFilter')
        return transfer

    def transfer_function(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        xp, dtype = _resolve_xp_dtype(xp, dtype, self.transfer)
        return _cast_transfer(self._check(_sampling_shape(samp), xp), dtype, self.symmetric)

    def transfer_function_sym(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        xp, dtype = _resolve_xp_dtype(xp, dtype, self.transfer)
        transfer = _cast_transfer(self._check(_sampling_shape(samp), xp), dtype, self.symmetric)
        psf = _upsample_axis(_upsample_axis(xp.fft.ifft2(transfer), -2, xp), -1, xp)
        upsampled = xp.fft.fft2(psf)
        return _cast_transfer(upsampled, dtype, self.symmetric)


@dataclasses.dataclass(frozen=True)
class GaussianFilter(SeparableFilter, AnalyticFilter):
    """
    A Gaussian blur of standard deviation `sigma`, in the length units of the
    sampling it is evaluated on.

    The point spread function is truncated at `psf_sigma` standard deviations
    (and renormalized). It is also bandwidth limited to match a smooth Gaussian
    transfer function. This leads to some ringing for small `sigma` values.
    """

    sigma: t.Union[Float, t.Tuple[Float, Float]]
    """Standard deviation in (y, x), in units of length."""
    psf_sigma: float = 3.
    """Half-width of the point spread function, in standard deviations."""

    symmetric = True

    def transfer_at(
        self, kyy: NDArray[numpy.floating], kxx: NDArray[numpy.floating], samp: Sampling
    ) -> NDArray[numpy.number]:
        xp = get_array_module(kyy, kxx)
        sigma_y, sigma_x = _split_pair(self.sigma)
        return xp.exp(-2 * numpy.pi**2 * ((kyy * sigma_y)**2 + (kxx * sigma_x)**2))

    def psf_separable(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> t.Tuple[NDArray[numpy.inexact], NDArray[numpy.inexact]]:
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        real_dtype = to_real_dtype(dtype)

        def kernel_1d(sigma: float, n: int) -> NDArray[numpy.floating]:
            # built in recip. space, for consistency with the reciprocal
            # version. For small values of `sigma`, these diverge.
            # This means the transfer function is truly a Gaussian, but
            # the PSF is a bandwidth-limited Gaussian, which rings slightly.
            k = xp.fft.fftfreq(n).astype(real_dtype)
            transfer = xp.exp(-2 * numpy.pi**2 * sigma**2 * k**2).astype(real_dtype)
            full = xp.fft.fftshift(xp.fft.ifft(transfer).real)

            center = n // 2
            r = int(numpy.ceil(self.psf_sigma * sigma))
            kernel = full[center-r : center+r+1]
            return kernel / xp.sum(kernel)

        sigmas = numpy.array(_split_pair(self.sigma), dtype=numpy.float64) / samp.sampling
        return t.cast(t.Tuple[NDArray[numpy.inexact], NDArray[numpy.inexact]], tuple(
            kernel_1d(sigma, n) for (sigma, n) in zip(sigmas, _sampling_shape(samp))
        ))


@dataclasses.dataclass(frozen=True)
class SquarePixelFilter(SeparableFilter, AnalyticFilter):
    """
    The transfer function of an ideal, square-pixeled detector, i.e. integration over
    one detector pixel. Note that this contrast transfer is not isotropic; the MTF is
    better along the detector axes than the diagonals.

    The point spread function is truncated at `psf_radius` detector pixels (and renormalized).
    """

    pixel_sampling: t.Optional[t.Union[Float, t.Tuple[Float, Float]]] = None
    """
    Size of one detector pixel, in (y, x), in the same units as `sampling`.
    Defaults to that sampling's own `sampling` (i.e. one detector pixel per sample).
    """
    psf_radius: int = 10
    """Half-width of the point spread function, in detector pixels."""

    symmetric = True

    def _pixel_sampling(self, samp: Sampling) -> t.Tuple[float, float]:
        return _split_pair(self.pixel_sampling) if self.pixel_sampling is not None \
            else (float(samp.sampling[0]), float(samp.sampling[1]))

    def transfer_at(
        self, kyy: NDArray[numpy.floating], kxx: NDArray[numpy.floating], samp: Sampling
    ) -> NDArray[numpy.number]:
        xp = get_array_module(kyy, kxx)
        size_y, size_x = self._pixel_sampling(samp)
        return xp.sinc(kyy * size_y) * xp.sinc(kxx * size_x)

    def psf_separable(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> t.Tuple[NDArray[numpy.inexact], NDArray[numpy.inexact]]:
        xp, dtype = _resolve_xp_dtype(xp, dtype)

        import scipy.special

        pixel_y, pixel_x = self._pixel_sampling(samp)
        # width of one detector pixel, in samples of `samp`
        widths = (pixel_y / samp.sampling[0], pixel_x / samp.sampling[1])

        def kernel_1d(width: float) -> NDArray[numpy.float64]:
            # truncated, normalized 1D pixel-integration kernel, the inverse DTFT of
            # sinc(k*width): h[n] = (Si(pi*(n+width/2)) - Si(pi*(n-width/2))) / pi,
            # truncated at `psf_radius` detector pixels (i.e. `psf_radius * width` samples)
            radius = int(numpy.ceil(self.psf_radius * width))
            ns = numpy.arange(-radius, radius + 1, dtype=numpy.float64)
            si = lambda x: scipy.special.sici(numpy.pi * x)[0]
            kernel = (si(ns + width / 2.) - si(ns - width / 2.)) / numpy.pi
            return kernel / numpy.sum(kernel)

        return t.cast(t.Tuple[NDArray[numpy.inexact], NDArray[numpy.inexact]], tuple(
            _cast_filter(xp.asarray(kernel_1d(width)), dtype)
            for width in widths
        ))


@dataclasses.dataclass(frozen=True)
class ProductFilter(Filter):
    """
    The composition (product) of several filters, applied in sequence (order doesn't
    matter, since LSI filters commute): point spread functions convolve, transfer
    functions multiply.

    Construct via [`Filter.__mul__`][phaser.utils.image.Filter.__mul__] (`filt1 * filt2`)
    rather than directly. The empty product, `ProductFilter(())`, is the multiplicative
    identity filter.
    """

    filters: t.Tuple[Filter, ...] = ()
    symmetric: bool = dataclasses.field(init=False)
    """Whether every component filter is symmetric."""

    def __post_init__(self) -> None:
        object.__setattr__(self, 'symmetric', all(f.symmetric for f in self.filters))

    def transfer_function(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        transfer = xp.ones(_sampling_shape(samp), dtype=dtype)
        for f in self.filters:
            transfer = transfer * f.transfer_function(samp, xp=xp, dtype=dtype)
        return transfer

    def transfer_function_dct(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.floating]:
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        transfer = xp.ones(_sampling_shape(samp), dtype=dtype)
        for f in self.filters:
            transfer = transfer * f.transfer_function_dct(samp, xp=xp, dtype=dtype)
        return xp.real(transfer).astype(to_real_dtype(dtype))

    def psf(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        """
        The product point spread function, computed by directly convolving the
        components' own (compact) point spread functions, rather than going through
        [`Filter`][phaser.utils.image.Filter]'s default derivation, which would take an
        FFT over the full `samp.shape` grid regardless of how compact the components are.
        """
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        if not self.filters:
            return xp.ones((1, 1), dtype=dtype)
        return functools.reduce(
            lambda a, b: _convolve_kernels_2d(a, b, xp, dtype),
            (f.psf(samp, xp=xp, dtype=dtype) for f in self.filters),
        )


@dataclasses.dataclass(frozen=True)
class ProductSeparableFilter(SeparableFilter, ProductFilter):
    """
    A [`ProductFilter`][phaser.utils.image.ProductFilter] all of whose components are
    [`SeparableFilter`][phaser.utils.image.SeparableFilter]s, so the product point spread
    function can be built (and convolved) one axis at a time, rather than going through
    [`ProductFilter`][phaser.utils.image.ProductFilter]'s 2D compact convolution.
    """

    filters: t.Tuple[SeparableFilter, ...] = ()

    def psf_separable(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> t.Tuple[NDArray[numpy.inexact], NDArray[numpy.inexact]]:
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        if not self.filters:
            one = xp.ones((1,), dtype=dtype)
            return (one, one)
        y_kernels, x_kernels = zip(*(
            f.psf_separable(samp, xp=xp, dtype=dtype) for f in self.filters
        ))
        y_kernel = functools.reduce(lambda a, b: _convolve_kernels_1d(a, b, xp, dtype), y_kernels)
        x_kernel = functools.reduce(lambda a, b: _convolve_kernels_1d(a, b, xp, dtype), x_kernels)
        return (y_kernel, x_kernel)


def _multiply_filters(a: Filter, b: Filter) -> Filter:
    a_filters = a.filters if isinstance(a, ProductFilter) else (a,)
    b_filters = b.filters if isinstance(b, ProductFilter) else (b,)
    filters = a_filters + b_filters
    if all(isinstance(f, SeparableFilter) for f in filters):
        return ProductSeparableFilter(t.cast(t.Tuple[SeparableFilter, ...], filters))
    return ProductFilter(filters)


def _scale_filter(filt: Filter, scale: float) -> Filter:
    one = numpy.array([scale])
    return _multiply_filters(filt, SeparablePsfFilter(one, numpy.array([1.0]), symmetric=True))


def _add_kernels_2d(
    a: NDArray[numpy.inexact], b: NDArray[numpy.inexact], xp: t.Any, dtype: DTypeLike
) -> NDArray[numpy.inexact]:
    """Add two centered compact kernels, embedding both into their union bounding box."""
    shape = (max(a.shape[-2], b.shape[-2]), max(a.shape[-1], b.shape[-1]))
    return _embed_psf(a, shape, xp, dtype) + _embed_psf(b, shape, xp, dtype)


@dataclasses.dataclass(frozen=True)
class SumFilter(Filter):
    """
    The sum of several filters: point spread functions and transfer functions add
    pointwise.

    Construct via [`Filter.__add__`][phaser.utils.image.Filter.__add__] (`filt1 + filt2`)
    rather than directly. The empty sum, `SumFilter(())`, is the additive identity
    (zero) filter.
    """

    filters: t.Tuple[Filter, ...] = ()
    symmetric: bool = dataclasses.field(init=False)
    """Whether every component filter is symmetric."""

    def __post_init__(self) -> None:
        object.__setattr__(self, 'symmetric', all(f.symmetric for f in self.filters))

    def transfer_function(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        transfer = xp.zeros(_sampling_shape(samp), dtype=dtype)
        for f in self.filters:
            transfer = transfer + f.transfer_function(samp, xp=xp, dtype=dtype)
        return transfer

    def transfer_function_dct(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.floating]:
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        transfer = xp.zeros(_sampling_shape(samp), dtype=dtype)
        for f in self.filters:
            transfer = transfer + f.transfer_function_dct(samp, xp=xp, dtype=dtype)
        return xp.real(transfer).astype(to_real_dtype(dtype))

    def psf(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        """
        The sum point spread function, computed by directly adding the components'
        own (compact) point spread functions, rather than going through
        [`Filter`][phaser.utils.image.Filter]'s default derivation, which would take an
        FFT over the full `samp.shape` grid regardless of how compact the components are.
        """
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        if not self.filters:
            return xp.zeros((1, 1), dtype=dtype)
        return functools.reduce(
            lambda a, b: _add_kernels_2d(a, b, xp, dtype),
            (f.psf(samp, xp=xp, dtype=dtype) for f in self.filters),
        )


@dataclasses.dataclass(frozen=True)
class SumSeparableFilter(SumFilter):
    """
    A [`SumFilter`][phaser.utils.image.SumFilter] all of whose components are
    [`SeparableFilter`][phaser.utils.image.SeparableFilter]s.

    Unlike [`ProductSeparableFilter`][phaser.utils.image.ProductSeparableFilter], the
    sum itself is *not* generally separable (a sum of outer products isn't generally
    a single outer product), so this doesn't implement
    [`SeparableFilter`][phaser.utils.image.SeparableFilter]. It exists so that
    [`psf`][phaser.utils.image.SumFilter.psf] can build the result directly from the
    components' compact `psf_separable` kernels, rather than paying for a full
    `samp.shape` FFT per component.
    """

    filters: t.Tuple[SeparableFilter, ...] = ()


def _add_filters(a: Filter, b: Filter) -> Filter:
    a_filters = a.filters if isinstance(a, SumFilter) else (a,)
    b_filters = b.filters if isinstance(b, SumFilter) else (b,)
    filters = a_filters + b_filters
    if all(isinstance(f, SeparableFilter) for f in filters):
        return SumSeparableFilter(t.cast(t.Tuple[SeparableFilter, ...], filters))
    return SumFilter(filters)


def _check_transfer(
    transfer: NDArray[numpy.number], shape: t.Tuple[int, int], name: str
) -> None:
    if tuple(transfer.shape) != shape:
        raise ValueError(f"{name}: Expected a transfer function of shape {shape},"
                         f" instead got shape {tuple(transfer.shape)}")


def _cast_output(
    out: NDArray[numpy.inexact], arr: NDArray[InexactT]
) -> NDArray[InexactT]:
    """Cast `out` to `arr`'s dtype. If `arr` is real, the transfer function is
    assumed to be Hermitian, so only the real part of `out` is kept (up to roundoff)."""
    xp = get_array_module(arr)
    complex_out = bool(xp.iscomplexobj(arr))
    return t.cast(NDArray[InexactT], (out if complex_out else out.real).astype(arr.dtype))


def convolve2d_recip_wrap(arr: NDArray[InexactT], transfer: ArrayLike) -> NDArray[InexactT]:
    """
    Convolve the last two axes of `arr` with the transfer function `transfer`
    and with periodic boundary conditions.

    Parameters:
        arr: Array to filter. Must be floating point or complex.
        transfer: Transfer function, of the same shape as the last two axes of `arr`,
                  sampled on that grid's fft frequencies.

    Returns: Array of the same shape and dtype as `arr`. If `arr` is real, `transfer`
             is assumed to be Hermitian and only the real part of the result is kept.
    """
    xp = get_array_module(arr, transfer)
    arr, transfer = xp.asarray(arr), xp.asarray(transfer)
    shape = _canonicalize_shape(arr, 'convolve2d_recip_wrap')
    _check_transfer(transfer, shape, 'convolve2d_recip_wrap')

    return _cast_output(xp.fft.ifft2(xp.fft.fft2(arr) * transfer), arr)


def convolve2d_recip_reflect(arr: NDArray[InexactT], transfer: ArrayLike) -> NDArray[InexactT]:
    """
    Convolve the last two axes of `arr` with the transfer function `transfer`
    and with reflecting boundary conditions.

    Parameters:
        arr: Array to filter. Must be floating point or complex.
        transfer: Transfer function, of twice the shape of the last two axes of `arr`,
                  sampled on the doubled grid's fft frequencies.

    Returns: Array of the same shape and dtype as `arr`. If `arr` is real, `transfer`
             is assumed to be Hermitian and only the real part of the result is kept.
    """
    xp = get_array_module(arr, transfer)
    arr, transfer = xp.asarray(arr), xp.asarray(transfer)
    (n, m) = _canonicalize_shape(arr, 'convolve2d_recip_reflect')
    _check_transfer(transfer, (2 * n, 2 * m), 'convolve2d_recip_reflect')

    (ly, lx) = (n // 2, m // 2)
    pad_width = ((0, 0),) * (arr.ndim - 2) + ((ly, n - ly), (lx, m - lx))
    # circular convolution of the symmetrically extended array, cropped back to size
    out = xp.fft.ifft2(xp.fft.fft2(pad(arr, pad_width, mode='symmetric')) * transfer)
    return _cast_output(out[..., ly:ly + n, lx:lx + m], arr)


def _zero_embed(
    arr: NDArray[numpy.inexact], shape: t.Tuple[int, int], offset: t.Tuple[int, int], xp: t.Any
) -> NDArray[numpy.inexact]:
    """Embed `arr` into a zeros array of `shape`, at `offset` along the last two axes.
    The adjoint of cropping the same region back out."""
    (n, m) = arr.shape[-2:]
    (ly, lx) = offset
    out = xp.zeros(arr.shape[:-2] + shape, dtype=arr.dtype)
    idx = (..., slice(ly, ly + n), slice(lx, lx + m))
    return at(out, idx).set(arr)


def _fold_symmetric_axis(arr: NDArray[numpy.inexact], axis: int, n: int, xp: t.Any) -> NDArray[numpy.inexact]:
    """Fold a length-`2n` axis back down to length `n`, the adjoint of the half-sample
    symmetric padding used by [`pad(..., mode='symmetric')`][phaser.utils.image.pad]
    with left width `n // 2`."""
    (l, r) = (n // 2, n - n // 2)

    idx_direct = [slice(None)] * arr.ndim
    idx_direct[axis] = slice(l, l + n)
    out = arr[tuple(idx_direct)]

    if l > 0:
        idx_left = [slice(None)] * arr.ndim
        idx_left[axis] = slice(0, l)
        left = xp.flip(arr[tuple(idx_left)], axis=axis)
        idx_out = [slice(None)] * out.ndim
        idx_out[axis] = slice(0, l)
        out = at(out, tuple(idx_out)).add(left)

    if r > 0:
        idx_right = [slice(None)] * arr.ndim
        idx_right[axis] = slice(l + n, 2 * n)
        right = xp.flip(arr[tuple(idx_right)], axis=axis)
        idx_out = [slice(None)] * out.ndim
        idx_out[axis] = slice(l, n)
        out = at(out, tuple(idx_out)).add(right)

    return out


def convolve2d_recip_reflect_adjoint(arr: NDArray[InexactT], transfer: ArrayLike) -> NDArray[InexactT]:
    """
    Apply the adjoint (transpose) of
    [`convolve2d_recip_reflect`][phaser.utils.image.convolve2d_recip_reflect] with the
    same (already-conjugated) `transfer`.

    `convolve2d_recip_reflect` is `Crop . IFFT2 . diag(transfer) . FFT2 . Pad`.
    Therefore, this computes the adjoint `Pad^T . IFFT2 . diag(transfer) . FFT2 . Crop^T`,
    wher `Crop^T` is zero-embedding and `Pad^T` is folding.

    Parameters:
        arr: Array to filter. Must be floating point or complex.
        transfer: Transfer function, of twice the shape of the last two axes of `arr`,
                  sampled on the doubled grid's fft frequencies.

    Returns: Array of the same shape and dtype as `arr`. If `arr` is real, `transfer`
             is assumed to be Hermitian and only the real part of the result is kept.
    """
    xp = get_array_module(arr, transfer)
    arr, transfer = xp.asarray(arr), xp.asarray(transfer)
    (n, m) = _canonicalize_shape(arr, 'convolve2d_recip_reflect_adjoint')
    _check_transfer(transfer, (2 * n, 2 * m), 'convolve2d_recip_reflect_adjoint')

    (ly, lx) = (n // 2, m // 2)
    embedded = _zero_embed(arr, (2 * n, 2 * m), (ly, lx), xp)
    out = xp.fft.ifft2(xp.fft.fft2(embedded) * transfer)
    out = _fold_symmetric_axis(out, -2, n, xp)
    out = _fold_symmetric_axis(out, -1, m, xp)
    return _cast_output(out, arr)


def convolve2d_recip_reflect_dct(arr: NDArray[InexactT], transfer: ArrayLike) -> NDArray[InexactT]:
    """
    Convolve the last two axes of `arr` with the transfer function `transfer`
    and reflecting boundary conditions, utilizing a DCT.

    Parameters:
        arr: Array to filter. Must be floating point or complex.
        transfer: Transfer function, of the same shape as the last two axes of `arr`,
                  sampled on the dct frequencies `j / (2 n*s)`.

    Returns: Array of the same shape and dtype as `arr` (a DCT-based transfer
             function must be real).
    """
    xp = get_array_module(arr, transfer)
    arr, transfer = xp.asarray(arr), xp.asarray(transfer)
    shape = _canonicalize_shape(arr, 'convolve2d_recip_reflect_dct')
    _check_transfer(transfer, shape, 'convolve2d_recip_reflect_dct')
    if xp.iscomplexobj(transfer):
        raise ValueError("convolve2d_recip_reflect_dct: Expected a real transfer function")

    return _cast_output(idct2(dct2(arr) * transfer), arr)


_RECIP_CONVOLVERS: t.Dict[_RecipBoundaryMode, t.Callable[[t.Any, ArrayLike], t.Any]] = {
    'grid-wrap': convolve2d_recip_wrap,
    'reflect': convolve2d_recip_reflect,
    'reflect_dct': convolve2d_recip_reflect_dct,
    'reflect_adjoint': convolve2d_recip_reflect_adjoint,
}


@tree_dataclass(frozen=True, static_fields=('mode', 'shape', 'symmetric'))
class PreparedOTF(t.Generic[InexactT_co]):
    """
    A [`Filter`][phaser.utils.image.Filter] evaluated for a given sampling and boundary
    mode, ready to be applied by calling it.
    """

    transfer: NDArray[InexactT_co]
    mode: _RecipBoundaryMode
    shape: t.Tuple[int, int]
    symmetric: bool = False
    """Whether the source filter was [`symmetric`][phaser.utils.image.Filter.symmetric]."""

    def __call__(self, arr: NDArray[InexactT]) -> NDArray[InexactT]:
        """
        Convolve the last two axes of `arr` with this filter.

        Parameters:
            arr: Array to filter. Must be floating point or complex.

        Returns: Array of the same shape and dtype as `arr`. If `arr` is real, the
                 filter's transfer function is assumed to be Hermitian and only the
                 real part of the result is kept.
        """
        shape = _canonicalize_shape(get_array_module(arr).asarray(arr), 'PreparedOTF')
        if shape != self.shape:
            raise ValueError(f"PreparedOTF: Filter was prepared for shape {self.shape},"
                             f" instead got shape {shape}")

        return t.cast(NDArray[InexactT], _RECIP_CONVOLVERS[self.mode](arr, self.transfer))

    def adjoint(self) -> Self:
        """
        Return the transpose (adjoint) of this filter, i.e. the filter `g` such that
        `<self(x), y> == <x, g(y)>` for all `x`, `y`.

        A [`symmetric`][phaser.utils.image.Filter.symmetric] filter is self-adjoint
        (the transfer function is already real-valued), so this simply returns `self`
        in that case.
        """
        if self.symmetric:
            return self
        if self.mode == 'reflect':
            return dataclasses.replace(self, transfer=self.transfer.conj(), mode='reflect_adjoint')
        if self.mode == 'reflect_adjoint':
            return dataclasses.replace(self, transfer=self.transfer.conj(), mode='reflect')
        return dataclasses.replace(self, transfer=self.transfer.conj())

@t.overload
def prepare_convolve2d_recip(
    filt: Filter, samp: Sampling, *,
    mode: _FilterBoundaryMode = 'reflect', xp: t.Any = None,
    dtype: type[ComplexT],
) -> PreparedOTF[ComplexT]: ...
@t.overload
def prepare_convolve2d_recip(
    filt: Filter, samp: Sampling, *,
    mode: _FilterBoundaryMode = 'reflect', xp: t.Any = None,
    dtype: type[numpy.float32],
) -> PreparedOTF[numpy.float32] | PreparedOTF[numpy.complex64]: ...
@t.overload
def prepare_convolve2d_recip(
    filt: Filter, samp: Sampling, *,
    mode: _FilterBoundaryMode = 'reflect', xp: t.Any = None,
    dtype: type[numpy.float64],
) -> PreparedOTF[numpy.float64] | PreparedOTF[numpy.complex128]: ...
@t.overload
def prepare_convolve2d_recip(
    filt: Filter, samp: Sampling, *,
    mode: _FilterBoundaryMode = 'reflect', xp: t.Any = None,
    dtype: t.Optional[DTypeLike] = None,
) -> PreparedOTF[numpy.floating] | PreparedOTF[numpy.complexfloating]: ...
def prepare_convolve2d_recip(
    filt: Filter, samp: Sampling, *,
    mode: _FilterBoundaryMode = 'reflect', xp: t.Any = None,
    dtype: t.Optional[DTypeLike] = None
) -> PreparedOTF[numpy.floating] | PreparedOTF[numpy.complexfloating]:
    """
    Evaluate `filt` for data sampled on `samp`.

    Parameters:
        filt: Filter to evaluate.
        samp: Sampling of the last two axes of the arrays to be filtered.
        mode: How to extend arrays past their boundaries. `'reflect'` is half-sample
              symmetric, `'grid-wrap'` is periodic.
        xp: Array module to evaluate on.
        dtype: Floating point precision to work in.

    Returns: A [`PreparedOTF`][phaser.utils.image.PreparedOTF], ready to be called on an array.
             Real iff `dtype` is real (or unspecified) and `filt` is
             [`symmetric`][phaser.utils.image.Filter.symmetric]; complex otherwise.
    """
    xp, dtype = _resolve_xp_dtype(xp, dtype)

    # a symmetric filter can be applied under symmetric boundaries by a smaller dct
    if mode == 'reflect' and filt.symmetric:
        transfer = filt.transfer_function_dct(samp, xp=xp, dtype=dtype)
        if numpy.issubdtype(dtype, numpy.complexfloating):
            transfer = transfer.astype(to_complex_dtype(dtype))
        # this cast is needed to take the union outside the typevar
        return t.cast(
            PreparedOTF[numpy.floating] | PreparedOTF[numpy.complexfloating],
            PreparedOTF(transfer, 'reflect_dct', _sampling_shape(samp), symmetric=True),
        )

    f = filt.transfer_function_sym if mode == 'reflect' else filt.transfer_function
    # this cast is needed to take the union outside the typevar
    return t.cast(
        PreparedOTF[numpy.floating] | PreparedOTF[numpy.complexfloating],
        PreparedOTF(f(samp, xp=xp, dtype=dtype), mode, _sampling_shape(samp), symmetric=filt.symmetric)
    )


def convolve2d_recip(
    arr: NDArray[InexactT], filt: Filter, samp: Sampling, *, mode: _FilterBoundaryMode = 'reflect',
) -> NDArray[InexactT]:
    """
    Convolve `arr` with `filt`, evaluated at `samp`.

    Convenience wrapper around [`prepare_convolve2d_recip`][phaser.utils.image.prepare_convolve2d_recip];
    prefer preparing once and reusing it to filter more than one array.

    The result always has the dtype of `arr`. If `arr` is real, `filt`'s transfer
    function is assumed to be Hermitian and only the real part of the result is kept.
    """
    xp = get_array_module(arr)
    return prepare_convolve2d_recip(filt, samp, mode=mode, xp=xp, dtype=to_real_dtype(arr.dtype))(arr)


@tree_dataclass(frozen=True, static_fields=('mode', 'symmetric'))
class PreparedPSF(t.Generic[NumT_co]):
    """
    A [`Filter`][phaser.utils.image.Filter]'s point spread function, evaluated for a
    given sampling and boundary mode, ready to be applied by calling it.
    """

    psf: t.Union[NDArray[NumT_co], t.Tuple[NDArray[NumT_co], NDArray[NumT_co]]]
    mode: _InterpBoundaryMode
    cval: t.Union[NumT_co, float] = 0.
    """Fill value, for `mode='constant'` and `mode='grid-constant'`."""
    symmetric: bool = False
    """Whether the source filter was [`symmetric`][phaser.utils.image.Filter.symmetric]."""

    def __call__(self, arr: NDArray[NumT]) -> NDArray[NumT]:
        """
        Convolve the last two axes of `arr` with this filter.

        Parameters:
            arr: Array to filter.

        Returns: Array of the same shape and dtype as `arr`.
        """
        if isinstance(self.psf, tuple):
            return convolve2d_separable(arr, *self.psf, mode=self.mode, cval=t.cast(float, self.cval))
        return convolve2d(arr, self.psf, mode=self.mode, cval=t.cast(float, self.cval))

    def adjoint(self) -> Self:
        """
        Return the transpose (adjoint) of this filter, i.e. the filter `g` such that
        `<self(x), y> == <x, g(y)>` for all `x`, `y`.

        A [`symmetric`][phaser.utils.image.Filter.symmetric] filter's PSF is already
        even about the origin, so this simply returns `self` in that case. Otherwise,
        for `mode='grid-wrap'` (periodic boundaries), the adjoint of correlation is
        correlation with the kernel reversed about its center. Other boundary modes
        pad the array before correlating, and transposing that padding exactly isn't
        implemented.
        """
        if self.symmetric:
            return self
        if self.mode != 'grid-wrap':
            raise NotImplementedError(
                f"PreparedPSF.adjoint() is not implemented for an asymmetric filter"
                f" under mode={self.mode!r}"
            )
        xp = get_array_module(*(self.psf if isinstance(self.psf, tuple) else (self.psf,)))
        psf = (
            tuple(xp.flip(p, axis=-1) for p in self.psf)
            if isinstance(self.psf, tuple) else xp.flip(self.psf, axis=(-2, -1))
        )
        return dataclasses.replace(self, psf=psf)


@t.overload
def prepare_convolve2d(
    filt: Filter, samp: Sampling, *,
    mode: _InterpBoundaryMode = 'reflect', cval: t.Union[NumT, float] = 0.,
    xp: t.Any = None, dtype: type[NumT],
) -> PreparedPSF[NumT]: ...
@t.overload
def prepare_convolve2d(
    filt: Filter, samp: Sampling, *,
    mode: _InterpBoundaryMode = 'reflect', cval: t.Union[numpy.number, float] = 0.,
    xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
) -> PreparedPSF[numpy.number]: ...
def prepare_convolve2d(
    filt: Filter, samp: Sampling, *,
    mode: _InterpBoundaryMode = 'reflect', cval: t.Union[numpy.number, float] = 0.,
    xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
) -> PreparedPSF[numpy.number]:
    """
    Evaluate `filt`'s point spread function for data sampled on `samp`, ready for
    direct spatial-domain convolution.

    Parameters:
        filt: Filter to evaluate.
        samp: Sampling of the last two axes of the arrays to be filtered.
        mode: How to extend arrays past their boundaries.
        cval: Fill value, for `mode='constant'` and `mode='grid-constant'`.
        xp: Array module to evaluate on.
        dtype: Floating point precision to work in.

    Returns: A [`PreparedPSF`][phaser.utils.image.PreparedPSF], ready to be called on an array.
    """
    if isinstance(filt, SeparableFilter):
        psf = filt.psf_separable(samp, xp=xp, dtype=dtype)
    else:
        psf = filt.psf(samp, xp=xp, dtype=dtype)
    return PreparedPSF(psf, mode, cval, symmetric=filt.symmetric)


__all__ = [
    'apply_flips',
    'remove_linear_ramp', 'colorize_complex', 'scale_to_integral_type',
    'affine_transform', 'to_affine_matrix',
    'convolve1d', 'convolve2d', 'convolve2d_separable',
    'Filter', 'SeparableFilter', 'AnalyticFilter',
    'PsfFilter', 'SeparablePsfFilter', 'TransferFilter',
    'GaussianFilter', 'SquarePixelFilter',
    'ProductFilter', 'ProductSeparableFilter',
    'SumFilter', 'SumSeparableFilter',
    'convolve2d_recip_wrap', 'convolve2d_recip_reflect', 'convolve2d_recip_reflect_dct',
    'convolve2d_recip',
    'PreparedOTF', 'prepare_convolve2d_recip',
    'PreparedPSF', 'prepare_convolve2d',
    'scale_matrix', 'rotation_matrix', 'translation_matrix',
]
