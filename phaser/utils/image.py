"""Utilities for image processing and filtering."""

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
    """Flip and transpose the last two axes of `data`.

    Args:
        data: Input array.
        flips: Booleans `(flip_y, flip_x, transpose)`. The transpose is applied last.
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
    """Remove a best-fit linear ramp from each image of the stack `data`.

    Args:
        data: Images of shape `(..., ny, nx)`.
        mask: Boolean mask `(ny, nx)` of pixels used for the fit.
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
    """Return complex `vals` as RGB, with hue from phase and value from intensity.

    Args:
        vals: Complex array.
        amp: Use amplitude instead of intensity for the value.
        rescale: Normalize the value by its maximum.

    Returns:
        Array of shape `(*vals.shape, 3)`.
    """
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
    min_range: t.Optional[float] = 1e-8,
    vmin: t.Optional[float] = None,
    vmax: t.Optional[float] = None,
) -> NDArray[numpy.unsignedinteger]:
    """Scale `arr` to the full range of an unsigned integer type.

    Args:
        arr: Array to scale.
        ty: Output integer width.
        mask: Boolean mask of the pixels used to determine the range.
        min_range: Minimum value range. A smaller range is expanded by raising the maximum.
        vmin: Value mapped to zero. Defaults to the 0.1% quantile.
        vmax: Value mapped to the integer maximum. Defaults to the 99.9% quantile.
    """
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
    vmin_a = arr_crop.dtype.type(vmin) if vmin is not None else xp.nanquantile(arr_crop, 0.001)
    vmax_a = arr_crop.dtype.type(vmax) if vmax is not None else xp.nanquantile(arr_crop, 0.999)

    if min_range is not None and (delta := min_range - (vmax_a - vmin_a)) > 0:
        # expand max to cover min_range (this keeps actual values near black level)
        vmax_a += delta

    return (xp.clip((imax + 1) / (vmax_a - vmin_a) * (arr - vmin_a), 0, imax)).astype(dtype)


_InterpBoundaryMode: t.TypeAlias = t.Literal['constant', 'nearest', 'mirror', 'reflect', 'wrap', 'grid-mirror', 'grid-wrap', 'grid-constant']
_FilterBoundaryMode: t.TypeAlias = t.Literal['reflect', 'grid-wrap']
_RecipBoundaryMode: t.TypeAlias = t.Union[_FilterBoundaryMode, t.Literal['reflect_dct', 'reflect_adjoint']]

_SCIPY_MODE_REWRITES: t.Dict[str, _InterpBoundaryMode] = {
    'grid-mirror': 'reflect', 'grid-constant': 'constant', 'grid-wrap': 'wrap',
}


def to_affine_matrix(arr: ArrayLike, ndim: int = 2) -> NDArray[numpy.floating]:
    """Return `arr` as an affine matrix of shape `(ndim+1, ndim+1)`.

    Accepts a linear matrix `(ndim, ndim)`, a diagonal scaling `(ndim,)` or `(ndim+1,)`, or a full affine matrix.

    Raises:
        ValueError: If `arr` has another shape.
    """
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
    """Return the affine matrix scaling each axis by `scale`."""
    scale = numpy.asarray(scale)
    assert scale.ndim == 1
    a = numpy.diag([*scale, 1.0])
    return a.astype(numpy.promote_types(a.dtype, numpy.float32))


def translation_matrix(vec: ArrayLike) -> NDArray[numpy.floating]:
    """Return the affine matrix translating by `vec`."""
    vec = numpy.asarray(vec)
    assert vec.ndim == 1
    a = numpy.eye(vec.size + 1, dtype=vec.dtype)
    a[:vec.size, vec.size] = vec
    return a.astype(numpy.promote_types(a.dtype, numpy.float32)) #a.astype(numpy.floating)


def rotation_matrix(theta: float) -> NDArray[numpy.floating]:
    """Return the 2D affine rotation matrix for `theta` [degrees]."""
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
    """Apply an affine transform to the last axes of `input`, as `scipy.ndimage.affine_transform`.

    Leading axes beyond `len(output_shape)` are batch axes.

    Args:
        input: Array to transform.
        matrix: Matrix mapping output to input coordinates.
        offset: Offset added to the output coordinates.
        output_shape: Shape of the transformed axes. Defaults to the shape of `input`.
        order: Spline interpolation order. JAX supports only `0` and `1`.
        mode: Boundary handling. `'constant'` and `'wrap'` are not supported.
        cval: Fill value for `'grid-constant'`.

    Raises:
        ValueError: If `mode` or `order` is unsupported.
    """
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
    """Convolve `arr` with the 1D filter `weights` along `axis`.

    Args:
        arr: Array to filter.
        weights: 1D filter. Complex only if `arr` is complex.
        axis: Axis to filter along.
        mode: Boundary extension mode.
        cval: Fill value for `'constant'` and `'grid-constant'`.
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
    mode = _SCIPY_MODE_REWRITES.get(mode, mode)

    return scipy.ndimage.convolve1d(
        arr, weights, axis, mode=mode, cval=cval if isinstance(cval, (complex, float, int)) else cval.tolist()
    )


def convolve2d_separable(
    arr: NDArray[NumT], y_weights: ArrayLike, x_weights: t.Optional[ArrayLike] = None, *,
    mode: _InterpBoundaryMode = 'reflect', cval: t.Union[NumT, float] = 0.
) -> NDArray[NumT]:
    """Convolve the last two axes of `arr` with a separable 2D filter.

    Equivalent to [`convolve2d`][phaser.utils.image.convolve2d] with the outer
    product `y_weights[:, None] * x_weights[None, :]`.

    Args:
        arr: Array to filter.
        y_weights: 1D filter along the second-to-last axis. Complex only if `arr` is complex.
        x_weights: 1D filter along the last axis. Defaults to `y_weights`.
        mode: Boundary extension mode.
        cval: Fill value for `'constant'` and `'grid-constant'`.
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
    """Convolve the last two axes of `arr` with a 2D filter, in real space.

    Args:
        arr: Array to filter.
        weights: 2D filter, or a `Filter` evaluated at unit sampling. Complex only if `arr` is complex.
        mode: Boundary extension mode.
        cval: Fill value for `'constant'` and `'grid-constant'`.
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
    mode = _SCIPY_MODE_REWRITES.get(mode, mode)

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
    """2D image filter, defined by a point spread function or a transfer function.

    Subclasses implement [`psf`][phaser.utils.image.Filter.psf] or
    [`transfer_function`][phaser.utils.image.Filter.transfer_function]; the other is derived.

    Filters add with `+` (pointwise sum), and compose or scale with `*`.
    """

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.psf is Filter.psf and cls.transfer_function is Filter.transfer_function:
            raise TypeError(f"{cls.__name__} must implement 'psf' or 'transfer_function'")

    symmetric: bool = False
    """Whether the point spread function is real and even."""

    def psf(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        """Return the point spread function for data sampled on `samp`.

        Args:
            samp: Sampling of the data to filter.
            xp: Array module.
            dtype: Floating point precision.

        Returns:
            Kernel centered at `n // 2` along each axis. Compact filters may return a kernel
            smaller than `samp.shape`. If `dtype` is real (the default), only the real part
            is kept.
        """
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        psf = fft2shift(xp.fft.ifft2(self.transfer_function(samp, xp=xp, dtype=dtype)))
        complex_dtype = numpy.issubdtype(dtype, numpy.complexfloating)
        return (psf if complex_dtype else xp.real(psf)).astype(dtype)

    def transfer_function(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        """Return the transfer function sampled at `samp.recip_grid()` (fft order).

        Args:
            samp: Sampling of the data to filter.
            xp: Array module.
            dtype: Floating point precision.
        """
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        kernel = self.psf(samp, xp=xp, dtype=dtype)
        psf = _embed_psf(kernel, _sampling_shape(samp), xp, kernel.dtype)
        transfer = xp.fft.fft2(ifft2shift(psf))
        return _cast_transfer(transfer, dtype, self.symmetric)

    def transfer_function_sym(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        """Return the transfer function on the doubled grid used for `'reflect'` boundaries.

        Args:
            samp: Sampling of the data to filter, not of the returned array.
            xp: Array module.
            dtype: Floating point precision.

        Returns:
            Array of shape `2 * samp.shape`.
        """
        (n, m) = _sampling_shape(samp)
        doubled = Sampling((2 * n, 2 * m), sampling=samp.sampling)
        return self.transfer_function(doubled, xp=xp, dtype=dtype)

    def transfer_function_dct(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.floating]:
        """Return the real transfer function on the type-II DCT frequencies `j / (2 n s)`, `j` in `[0, n)`.

        This is the non-negative quadrant of
        [`transfer_function_sym`][phaser.utils.image.Filter.transfer_function_sym].
        Only meaningful for a [`symmetric`][phaser.utils.image.Filter.symmetric] filter.

        Args:
            samp: Sampling of the data to filter.
            xp: Array module.
            dtype: Floating point precision.
        """
        (n, m) = _sampling_shape(samp)
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        transfer = self.transfer_function_sym(samp, xp=xp, dtype=dtype)[..., :n, :m]
        return xp.real(transfer).astype(to_real_dtype(dtype))

    def __mul__(self, other: 't.Union[Filter, float]') -> 'Filter':
        """Compose with another filter, or scale by a real factor.

        Composing (`filt1 * filt2`) convolves the point spread functions and multiplies the
        transfer functions. Scaling (`filt * 2.`) multiplies both by the factor.
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
        """Add another filter (`filt1 + filt2`), summing point spread functions and transfer functions."""
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
    """Filter defined by a numeric point spread function.

    A complex `kernel` can only filter a complex array.
    """

    kernel: NDArray[numpy.inexact]
    """2D kernel, centered at `kernel.shape // 2`."""
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
    """Filter whose point spread function is the outer product of two 1D kernels.

    See [`convolve2d_separable`][phaser.utils.image.convolve2d_separable].
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
        """Return the 1D kernels `(y_kernel, x_kernel)`, each centered at `n // 2`.

        Only the sample spacing of `samp` is used.

        Args:
            samp: Sampling of the data to filter.
            xp: Array module.
            dtype: Floating point precision.
        """
        ...

    def psf(
        self, samp: Sampling, *, xp: t.Any = None, dtype: t.Optional[DTypeLike] = None
    ) -> NDArray[numpy.inexact]:
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        y_kernel, x_kernel = self.psf_separable(samp, xp=xp, dtype=dtype)
        return y_kernel[:, None] * x_kernel[None, :]


class AnalyticFilter(Filter):
    """Filter with a closed-form transfer function."""

    @abc.abstractmethod
    def transfer_at(
        self, kyy: NDArray[numpy.floating], kxx: NDArray[numpy.floating], samp: Sampling
    ) -> NDArray[numpy.number]:
        """Return the transfer function evaluated at frequencies `(kyy, kxx)` [1/length].

        The frequencies are not necessarily an fft grid of `samp`.
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
    """Filter defined by the outer product `y_kernel[:, None] * x_kernel[None, :]`.

    Complex kernels can only filter a complex array.
    """

    y_kernel: NDArray[numpy.inexact]
    """1D kernel along y, centered at `n // 2`."""
    x_kernel: NDArray[numpy.inexact]
    """1D kernel along x, centered at `n // 2`."""
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
    """Filter defined by a numeric transfer function."""

    transfer: NDArray[numpy.number]
    """Transfer function on the fft grid. Must match the shape of the data."""
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
    """Gaussian blur.

    The point spread function is truncated at `psf_sigma` standard deviations and renormalized.
    It is bandwidth limited to match the Gaussian transfer function, which rings for small `sigma`.
    """

    sigma: t.Union[Float, t.Tuple[Float, Float]]
    """Standard deviation `(y, x)` [length]. A scalar applies to both axes."""
    psf_sigma: float = 3.
    """Half-width of the point spread function [standard deviations]."""

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
    """Transfer function of an ideal square-pixel detector (integration over one pixel).

    The transfer is not isotropic: the MTF is better along the detector axes than the diagonals.
    The point spread function is truncated at `psf_radius` detector pixels and renormalized.
    """

    pixel_sampling: t.Optional[t.Union[Float, t.Tuple[Float, Float]]] = None
    """Detector pixel size `(y, x)` [length]. Defaults to the sample spacing of the data."""
    psf_radius: int = 10
    """Half-width of the point spread function [detector pixels]."""

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
    """Composition of several filters: point spread functions convolve, transfer functions multiply.

    Normally constructed with `filt1 * filt2`. The empty product is the identity filter.
    """

    filters: t.Tuple[Filter, ...] = ()
    """Component filters."""
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
        """Return the product point spread function, convolving the compact component kernels."""
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        if not self.filters:
            return xp.ones((1, 1), dtype=dtype)
        return functools.reduce(
            lambda a, b: _convolve_kernels_2d(a, b, xp, dtype),
            (f.psf(samp, xp=xp, dtype=dtype) for f in self.filters),
        )


@dataclasses.dataclass(frozen=True)
class ProductSeparableFilter(SeparableFilter, ProductFilter):
    """[`ProductFilter`][phaser.utils.image.ProductFilter] of [`SeparableFilter`][phaser.utils.image.SeparableFilter]s."""

    filters: t.Tuple[SeparableFilter, ...] = ()
    """Component filters."""

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
    """Sum of several filters: point spread functions and transfer functions add pointwise.

    Normally constructed with `filt1 + filt2`. The empty sum is the zero filter.
    """

    filters: t.Tuple[Filter, ...] = ()
    """Component filters."""
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
        """Return the sum point spread function, adding the compact component kernels."""
        xp, dtype = _resolve_xp_dtype(xp, dtype)
        if not self.filters:
            return xp.zeros((1, 1), dtype=dtype)
        return functools.reduce(
            lambda a, b: _add_kernels_2d(a, b, xp, dtype),
            (f.psf(samp, xp=xp, dtype=dtype) for f in self.filters),
        )


@dataclasses.dataclass(frozen=True)
class SumSeparableFilter(SumFilter):
    """[`SumFilter`][phaser.utils.image.SumFilter] of [`SeparableFilter`][phaser.utils.image.SeparableFilter]s.

    The sum is not itself separable.
    """

    filters: t.Tuple[SeparableFilter, ...] = ()
    """Component filters."""


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
    """Convolve the last two axes of `arr` with `transfer`, using periodic boundaries.

    Args:
        arr: Floating point or complex array to filter.
        transfer: Transfer function on the fft frequencies of the last two axes of `arr`.

    Returns:
        Array of the same shape and dtype as `arr`. If `arr` is real, `transfer` is assumed
        Hermitian and only the real part is kept.
    """
    xp = get_array_module(arr, transfer)
    arr, transfer = xp.asarray(arr), xp.asarray(transfer)
    shape = _canonicalize_shape(arr, 'convolve2d_recip_wrap')
    _check_transfer(transfer, shape, 'convolve2d_recip_wrap')

    return _cast_output(xp.fft.ifft2(xp.fft.fft2(arr) * transfer), arr)


def convolve2d_recip_reflect(arr: NDArray[InexactT], transfer: ArrayLike) -> NDArray[InexactT]:
    """Convolve the last two axes of `arr` with `transfer`, using reflecting boundaries.

    Args:
        arr: Floating point or complex array to filter.
        transfer: Transfer function on the fft frequencies of the doubled grid, with twice
            the shape of the last two axes of `arr`.

    Returns:
        Array of the same shape and dtype as `arr`. If `arr` is real, `transfer` is assumed
        Hermitian and only the real part is kept.
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
    """Apply the adjoint of [`convolve2d_recip_reflect`][phaser.utils.image.convolve2d_recip_reflect].

    `transfer` is the same (already conjugated) transfer function passed to the forward operation.

    Args:
        arr: Floating point or complex array to filter.
        transfer: Transfer function on the fft frequencies of the doubled grid, with twice
            the shape of the last two axes of `arr`.

    Returns:
        Array of the same shape and dtype as `arr`. If `arr` is real, `transfer` is assumed
        Hermitian and only the real part is kept.
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
    """Convolve the last two axes of `arr` with `transfer`, using reflecting boundaries and a DCT.

    Args:
        arr: Floating point or complex array to filter.
        transfer: Real transfer function on the DCT frequencies `j / (2 n s)`, with the shape
            of the last two axes of `arr`.

    Raises:
        ValueError: If `transfer` is complex.
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
    """[`Filter`][phaser.utils.image.Filter] evaluated in reciprocal space, applied by calling it."""

    transfer: NDArray[InexactT_co]
    """Transfer function."""
    mode: _RecipBoundaryMode
    """Boundary mode."""
    shape: t.Tuple[int, int]
    """Shape of the arrays the filter applies to."""
    symmetric: bool = False
    """Whether the source filter was [`symmetric`][phaser.utils.image.Filter.symmetric]."""

    def __call__(self, arr: NDArray[InexactT]) -> NDArray[InexactT]:
        """Convolve the last two axes of `arr` with the filter.

        Raises:
            ValueError: If `arr` does not have the prepared `shape`.
        """
        shape = _canonicalize_shape(get_array_module(arr).asarray(arr), 'PreparedOTF')
        if shape != self.shape:
            raise ValueError(f"PreparedOTF: Filter was prepared for shape {self.shape},"
                             f" instead got shape {shape}")

        return t.cast(NDArray[InexactT], _RECIP_CONVOLVERS[self.mode](arr, self.transfer))

    def adjoint(self) -> Self:
        """Return the adjoint filter `g`, where `<self(x), y> == <x, g(y)>`.

        A [`symmetric`][phaser.utils.image.Filter.symmetric] filter is self-adjoint and returns `self`.
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
    """Evaluate the transfer function of `filt` for data sampled on `samp`.

    Args:
        filt: Filter to evaluate.
        samp: Sampling of the last two axes of the arrays to filter.
        mode: Boundary mode. `'reflect'` is half-sample symmetric, `'grid-wrap'` is periodic.
        xp: Array module.
        dtype: Floating point precision.

    Returns:
        The prepared filter. Its transfer function is real iff `dtype` is real (or unspecified)
        and `filt` is [`symmetric`][phaser.utils.image.Filter.symmetric].
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
    """Convolve `arr` with `filt` evaluated at `samp`, in reciprocal space.

    The result has the dtype of `arr`. If `arr` is real, only the real part is kept.
    Use [`prepare_convolve2d_recip`][phaser.utils.image.prepare_convolve2d_recip] to filter multiple arrays.
    """
    xp = get_array_module(arr)
    return prepare_convolve2d_recip(filt, samp, mode=mode, xp=xp, dtype=to_real_dtype(arr.dtype))(arr)


@tree_dataclass(frozen=True, static_fields=('mode', 'symmetric'))
class PreparedPSF(t.Generic[NumT_co]):
    """[`Filter`][phaser.utils.image.Filter] evaluated in real space, applied by calling it."""

    psf: t.Union[NDArray[NumT_co], t.Tuple[NDArray[NumT_co], NDArray[NumT_co]]]
    """2D point spread function, or the 1D kernels `(y_kernel, x_kernel)` of a separable filter."""
    mode: _InterpBoundaryMode
    """Boundary mode."""
    cval: t.Union[NumT_co, float] = 0.
    """Fill value for `'constant'` and `'grid-constant'`."""
    symmetric: bool = False
    """Whether the source filter was [`symmetric`][phaser.utils.image.Filter.symmetric]."""

    def __call__(self, arr: NDArray[NumT]) -> NDArray[NumT]:
        """Convolve the last two axes of `arr` with the filter."""
        if isinstance(self.psf, tuple):
            return convolve2d_separable(arr, *self.psf, mode=self.mode, cval=t.cast(float, self.cval))
        return convolve2d(arr, self.psf, mode=self.mode, cval=t.cast(float, self.cval))

    def adjoint(self) -> Self:
        """Return the adjoint filter `g`, where `<self(x), y> == <x, g(y)>`.

        A [`symmetric`][phaser.utils.image.Filter.symmetric] filter is self-adjoint and returns `self`.

        Raises:
            NotImplementedError: If the filter is asymmetric and `mode` is not `'grid-wrap'`.
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
    """Evaluate the point spread function of `filt` for data sampled on `samp`.

    Args:
        filt: Filter to evaluate.
        samp: Sampling of the last two axes of the arrays to filter.
        mode: Boundary mode.
        cval: Fill value for `'constant'` and `'grid-constant'`.
        xp: Array module.
        dtype: Floating point precision.
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
