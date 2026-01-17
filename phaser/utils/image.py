"""
Utilities for image processing & filtering
"""

import warnings
import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray

from .num import (
    Sampling, cast_array_module, get_array_module, get_scipy_module, max_supported_float, to_numpy, at, abs2,
    xp_is_jax, xp_is_torch, Float
)


NumT = t.TypeVar('NumT', bound=numpy.number)


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
        p, residues, rank, singular = xp.linalg.lstsq(pts[mask], layer.flatten()[mask], rcond=None)
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


def gaussian_transfer(
    ky: NDArray[numpy.floating],
    kx: NDArray[numpy.floating],
    sigma: t.Union[Float, t.Tuple[Float, Float]],
) -> NDArray[numpy.floating]:
    """
    Construct a 2D Gaussian transfer function.

    Can be used to apply a Gaussian blur in Fourier space.
    The units of `sigma` should cancel the units of `ky`/`kx`.

    Parameters:
        ky: y frequency components, units of 1/length (not rad/length)
        ky: x frequency components, units of 1/length
        sigma: Standard deviation in (y, x).

    Returns: Array of same shape as `ky`/`kx`.
    """
    xp = get_array_module(ky, kx)

    if isinstance(sigma, (tuple, list)):
        sigma_y, sigma_x = sigma
    else:
        sigma_y = sigma_x = sigma

    pre = -2 * numpy.pi**2
    return xp.exp(pre * ((ky * sigma_y)**2 + (kx * sigma_x)**2))


def square_pixel_transfer(shape: t.Tuple[int, int], *, xp: t.Any = None) -> NDArray[numpy.floating]:
    """
    Construct the transfer function for an ideal, square-pixeled detector.
    Note that this contrast transfer is not isotropic; the MTF is better
    along the detector axes than it is along the diagonals.

    Parameters:
        shape: Shape of transfer function to generate.

    Returns: Array of shape `shape`
    """
    xp = numpy if xp is None else cast_array_module(xp)
    ky, kx = Sampling(shape, sampling=[1., 1.]).recip_grid(xp=xp)
    return xp.sinc(ky) * xp.sinc(kx)


# convert scipy boundary mode to numpy.pad mode
_INTERP_TO_PAD: t.Dict[_InterpBoundaryMode, str] = {
    'reflect': 'symmetric',
    'mirror': 'reflect',
    'nearest': 'edge',
    'grid-mirror': 'reflect',
    'grid-wrap': 'wrap',
    'grid-constant': 'constant',
}


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
    mode: _InterpBoundaryMode = 'reflect', cval: float = 0.
) -> NDArray[NumT]:
    xp = get_array_module(arr, weights)

    arr = xp.asarray(arr)
    weights = xp.asarray(weights)
    if weights.ndim != 1:
        raise ValueError("Expected 'weights' to be 1D")
    axis = _canonicalize_axis(axis, arr.ndim)

    if xp_is_torch(xp):
        from ._torch_kernels import _convolve1d, _MockTensor

        return t.cast(NDArray[NumT], _convolve1d(
            t.cast(_MockTensor, arr),
            t.cast(_MockTensor, weights),
            axis=axis, mode=mode, cval=cval
        ))

    scipy = get_scipy_module(arr, weights)

    if xp_is_jax(xp):
        r = len(weights) // 2
        pad_mode = _INTERP_TO_PAD.get(mode, mode)
        pad_kwargs = {'constant_values': cval} if pad_mode == 'constant' else {}

        pad = tuple(
            (len(weights) - r - 1, r) if i == axis else (0, 0)
            for i in range(arr.ndim)
        )
        weights = weights[tuple(
            slice(None) if i == axis else None
            for i in range(arr.ndim)
        )]
        # TODO: use jax.lax.conv_general_dilated directly
        return scipy.signal.convolve(
            xp.pad(arr, pad, mode=pad_mode, **pad_kwargs),  # type: ignore
            weights, mode='valid', method='direct'
        ).astype(arr.dtype)

    return scipy.ndimage.convolve1d(
        arr, weights, axis, mode=mode, cval=cval
    )




__all__ = [
    'apply_flips',
    'remove_linear_ramp', 'colorize_complex', 'scale_to_integral_type',
    'affine_transform', 'to_affine_matrix',
    'convolve1d',
    'scale_matrix', 'rotation_matrix', 'translation_matrix',
    'gaussian_transfer', 'square_pixel_transfer',
]