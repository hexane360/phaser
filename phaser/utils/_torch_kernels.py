import functools
import itertools
import operator
import typing as t

import numpy
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike

from phaser.utils.image import _InterpBoundaryMode
from phaser.utils.misc import _MockModule
from phaser.utils.num import _PadMode


def get_cutouts(obj: torch.Tensor, start_idxs: torch.Tensor, cutout_shape: t.Tuple[int, int]) -> torch.Tensor:
    #out_shape = (*start_idxs.shape[:-1], *obj.shape[:-2], *cutout_shape)
    ys, xs = torch.arange(cutout_shape[0], device=obj.device), torch.arange(cutout_shape[1], device=obj.device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    yy = start_idxs[..., 0][..., None, None] + yy
    xx = start_idxs[..., 1][..., None, None] + xx

    out = obj[..., yy, xx]
    if obj.ndim > 2:
        # oof
        out = torch.permute(out, (*(i + obj.ndim - 2 for i in range(start_idxs.ndim - 1)), *range(obj.ndim - 2), -2, -1))
        #assert out.shape == out_shape
    return out


_TORCH_TO_NUMPY_DTYPE: t.Dict[torch.dtype, t.Type[numpy.generic]] = {
    torch.bool       : numpy.bool_,
    torch.uint8      : numpy.uint8,
    torch.int8       : numpy.int8,
    torch.int16      : numpy.int16,
    torch.int32      : numpy.int32,
    torch.int64      : numpy.int64,
    torch.float16    : numpy.float16,
    torch.float32    : numpy.float32,
    torch.float64    : numpy.float64,
    torch.complex64  : numpy.complex64,
    torch.complex128 : numpy.complex128,
}

_NUMPY_TO_TORCH_DTYPE: t.Dict[t.Type[numpy.generic], torch.dtype] = {
    numpy.bool_      : torch.bool,
    numpy.uint8      : torch.uint8,
    numpy.int8       : torch.int8,
    numpy.int16      : torch.int16,
    numpy.int32      : torch.int32,
    numpy.int64      : torch.int64,
    numpy.float16    : torch.float16,
    numpy.float32    : torch.float32,
    numpy.float64    : torch.float64,
    numpy.complex64  : torch.complex64,
    numpy.complex128 : torch.complex128,
}


def to_torch_dtype(dtype: t.Union[str, torch.dtype, numpy.dtype, t.Type[numpy.generic]]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, numpy.dtype):
        dtype = dtype.type
    elif not isinstance(dtype, type) or not issubclass(dtype, numpy.generic):
        dtype = numpy.dtype(dtype).type

    try:
        return _NUMPY_TO_TORCH_DTYPE[dtype]
    except KeyError:
        raise ValueError(f"Can't convert dtype '{dtype}' to a PyTorch dtype")


def to_numpy_dtype(dtype: t.Union[str, torch.dtype, numpy.dtype, t.Type[numpy.generic]]) -> t.Type[numpy.generic]:
    if isinstance(dtype, str):
        return numpy.dtype(dtype).type
    if isinstance(dtype, numpy.dtype):
        return dtype.type
    if isinstance(dtype, torch.dtype):
        return _TORCH_TO_NUMPY_DTYPE[dtype]
    return dtype


def _tensor_astype(self: torch.Tensor, dtype: t.Union[str, torch.dtype, t.Type[numpy.generic]]) -> torch.Tensor:
    return self.to(to_torch_dtype(dtype))


def _tensor_T(self: torch.Tensor) -> torch.Tensor:
    # torch's native `.T` is 2D-only and deprecated for other ranks - though it still
    # computes the same full-axis-reversal permutation today, just with a UserWarning
    # (which itself suggests this exact replacement). Doing it explicitly gives numpy's
    # `.T` semantics without the warning, and without depending on deprecated behavior.
    return self.permute(tuple(range(self.ndim - 1, -1, -1)))


# Patch numpy-style `.astype`/`.T` directly onto `torch.Tensor` (rather than wrapping
# results in a subclass, as a previous `_MockTensor` design did) so generic
# backend-agnostic code can call them uniformly, without ever introducing a tensor
# subclass for `torch.compile`'s Dynamo tracer (or FakeTensor) to trip over.
torch.Tensor.astype = _tensor_astype  # pyright: ignore[reportAttributeAccessIssue]
torch.Tensor.T = property(_tensor_T)  # pyright: ignore[reportAttributeAccessIssue]


def _mirror(idx: torch.Tensor, size: int) -> torch.Tensor:
    s = size -1
    return torch.abs((idx + s) % (2 * s) - s)


_BOUNDARY_FNS: t.Dict[str, t.Callable[[torch.Tensor, int], torch.Tensor]] = {
    'nearest': lambda idx, size: torch.clip(idx, 0, size - 1),
    'grid-wrap': lambda idx, size: idx % size,
    'reflect': lambda idx, size: torch.floor_divide(_mirror(2*idx+1, 2*size+1), 2),
    'mirror': _mirror,
}

_PAD_MODE_MAP: t.Dict[_PadMode, str] = {
    'constant': 'constant',
    'edge': 'replicate',
    'reflect': 'reflect',
    'wrap': 'circular',
}


def nan_to_num(arr: torch.Tensor, **kwargs: t.Any) -> torch.Tensor:
    if torch.is_complex(arr):
        return torch.view_as_complex(
            torch.nan_to_num(torch.view_as_real(arr), **kwargs)
        )

    return torch.nan_to_num(arr, **kwargs) 


def min(
    arr: torch.Tensor, axis: t.Union[int, t.Tuple[int, ...], None] = None, *,
    keepdims: bool = False
) -> torch.Tensor:
    if axis is None:
        if keepdims:
            return torch.min(arr).reshape((1,) * arr.ndim)
        return torch.min(arr)
    return torch.amin(arr, axis, keepdim=keepdims)


def max(
    arr: torch.Tensor, axis: t.Union[int, t.Tuple[int, ...], None] = None, *,
    keepdims: bool = False
) -> torch.Tensor:
    if axis is None:
        if keepdims:
            return torch.max(arr).reshape((1,) * arr.ndim)
        return torch.max(arr)
    return torch.amax(arr, axis, keepdim=keepdims)


def nanmin(
    arr: torch.Tensor, axis: t.Union[int, t.Tuple[int, ...], None] = None, *,
    keepdims: bool = False
) -> torch.Tensor:
    return min(torch.nan_to_num(arr, nan=torch.inf), axis, keepdims=keepdims)


def nanmax(
    arr: torch.Tensor, axis: t.Union[int, t.Tuple[int, ...], None] = None, *,
    keepdims: bool = False
) -> torch.Tensor:
    return max(torch.nan_to_num(arr, nan=-torch.inf), axis, keepdims=keepdims)


def minimum(
    x1: ArrayLike, x2: ArrayLike
) -> torch.Tensor:
    if not isinstance(x1, torch.Tensor):
        x1 = torch.asarray(x1)
    if not isinstance(x2, torch.Tensor):
        x2 = torch.asarray(x2)

    return torch.minimum(x1, x2)


def maximum(
    x1: ArrayLike, x2: ArrayLike
) -> torch.Tensor:
    if not isinstance(x1, torch.Tensor):
        x1 = torch.asarray(x1)
    if not isinstance(x2, torch.Tensor):
        x2 = torch.asarray(x2)

    return torch.maximum(x1, x2)


def flip(
    arr: torch.Tensor, axis: t.Union[int, t.Tuple[int, ...], None] = None,
) -> torch.Tensor:
    # torch.flip's second parameter is `dims`, not `dim` like most other torch
    # reductions, so it can't go through `_wrap_call`'s generic axis -> dim rename
    if axis is None:
        axis = tuple(range(arr.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    return torch.flip(arr, axis)


def cumsum(
    arr: torch.Tensor, axis: t.Optional[int] = None,
) -> torch.Tensor:
    if axis is None:
        return torch.cumsum(arr.ravel(), 0)
    return torch.cumsum(arr, axis)


def split(
    arr: torch.Tensor, sections: int, *, axis: int = 0 
) -> t.Tuple[torch.Tensor, ...]:
    if arr.shape[axis] % sections != 0:
        raise ValueError("array split does not result in an equal division")
    return torch.split(arr, arr.shape[axis] // sections, axis)


def _pad_idxs_edge(idxs: torch.Tensor, left: int, right: int, size: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
    return (idxs.new_zeros(left), idxs.new_full((right,), size - 1))


def _pad_idxs_wrap(idxs: torch.Tensor, left: int, right: int, size: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
    left_idx = torch.arange(-left, 0, dtype=idxs.dtype, device=idxs.device) % size
    right_idx = torch.arange(size, size + right, dtype=idxs.dtype, device=idxs.device) % size
    return (left_idx, right_idx)


def _pad_idxs_reflect(idxs: torch.Tensor, left: int, right: int, size: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
    if size == 1:
        return (idxs.new_zeros(left), idxs.new_zeros(right))
    period = 2 * (size - 1)
    def fold(i: torch.Tensor) -> torch.Tensor:
        i = i % period
        return (size - 1) - ((i - (size - 1)).abs())
    left_idx = fold(torch.arange(left, 0, -1, dtype=idxs.dtype, device=idxs.device))
    right_idx = fold(torch.arange(size, size + right, dtype=idxs.dtype, device=idxs.device))
    return (left_idx, right_idx)


def _pad_idxs_symmetric(idxs: torch.Tensor, left: int, right: int, size: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
    period = 2 * size
    def fold(i: torch.Tensor) -> torch.Tensor:
        i = i % period
        return torch.where(i < size, i, period - 1 - i)
    left_idx = fold(torch.arange(-left, 0, dtype=idxs.dtype, device=idxs.device) % period)
    right_idx = fold(torch.arange(size, size + right, dtype=idxs.dtype, device=idxs.device))
    return (left_idx, right_idx)


_FAST_PAD_MODES: t.FrozenSet[str] = frozenset(('constant', 'edge', 'reflect', 'wrap'))
_PAD_MODES: t.FrozenSet[str] = _FAST_PAD_MODES | frozenset(('symmetric',))
_MAKE_PAD_IDXS: t.Dict[_PadMode, t.Callable[[torch.Tensor, int, int, int], t.Tuple[torch.Tensor, torch.Tensor]]] = {
    'edge':      _pad_idxs_edge,
    'wrap':      _pad_idxs_wrap,
    'reflect':   _pad_idxs_reflect,
    'symmetric': _pad_idxs_symmetric,
}


def pad(
    arr: torch.Tensor, pad_width: t.Union[int, t.Tuple[int, int], t.Sequence[t.Tuple[int, int]]], /, *,
    mode: _PadMode = 'constant', cval: complex | numpy.number = 0.
) -> torch.Tensor:
    if mode not in _PAD_MODES:
        raise ValueError(f"Unsupported padding mode '{mode}'")

    if mode == 'constant':
        cval_arr = cval.numpy(force=True) if isinstance(cval, torch.Tensor) else numpy.asarray(cval)
        if numpy.iscomplexobj(cval_arr) and torch.is_complex(arr):
            cval = complex(cval_arr)
            return torch.complex(
                pad(arr.real, pad_width, mode=mode, cval=cval.real),
                pad(arr.imag, pad_width, mode=mode, cval=cval.imag),
            )
        cval = complex(cval_arr).real

    widths = (pad_width, pad_width) if isinstance(pad_width, int) else pad_width

    if isinstance(widths[0], int):
        widths = (t.cast(t.Tuple[int, int], widths),)

    if len(widths) == 1:
        widths = tuple(widths) * arr.ndim
    elif len(widths) != arr.ndim:
        raise ValueError(f"Invalid `pad_width` '{pad_width}'.")

    # check for fast path (F.pad)
    # checks supported mode, dim <= 3, pad lengths all less than array size
    # constant padding has no restrictions
    if mode == 'constant' or (
        mode in _FAST_PAD_MODES
        and arr.ndim <= 3
        and all(p <= s - 1 if isinstance(p, int) else all(p1 <= s - 1 for p1 in p) for (p, s) in zip(widths, arr.shape))
    ):
        widths = tuple(itertools.chain.from_iterable(t.cast(t.Sequence[t.Tuple[int, int]], reversed(widths))))
        kwargs = {'value': float(t.cast(float | numpy.number, cval))} if mode == 'constant' else {}
        return F.pad(arr.reshape(1, *arr.shape), widths, mode=_PAD_MODE_MAP[mode], **kwargs)[0]

    # slow path
    for dim, (p, size) in enumerate(zip(widths, arr.shape)):
        (left, right) = (p, p) if isinstance(p, int) else p
        if left == 0 and right == 0:
            continue

        idxs = torch.arange(size, dtype=torch.int64, device=arr.device)
        (left_idx, right_idx) = _MAKE_PAD_IDXS[mode](idxs, left, right, size)
        idxs = torch.cat([left_idx, idxs, right_idx]).to(arr.device)
        arr = arr.index_select(dim, idxs)

    return arr


def unwrap(arr: torch.Tensor, discont: t.Optional[float] = None, axis: int = -1, *,
           period: float = 2.*numpy.pi) -> torch.Tensor:
    if discont is None:
        discont = period / 2

    diff = torch.diff(arr, dim=axis)
    dtype = torch.result_type(diff, period)

    if dtype.is_floating_point:
        interval_high = period / 2
        boundary_ambiguous = True
    else:
        interval_high, rem = divmod(period, 2)
        boundary_ambiguous = rem == 0

    interval_low = -interval_high
    diffmod = torch.remainder(diff - interval_low, period) + interval_low
    if boundary_ambiguous:
        diffmod[(diffmod == interval_low) & (diff > 0)] = interval_high

    phase_correct = diffmod - diff
    phase_correct[abs(diff) < discont] = 0.

    prepend_shape = list(arr.shape)
    prepend_shape[axis] = 1
    return arr + torch.cat([torch.zeros(prepend_shape, dtype=dtype, device=arr.device), torch.cumsum(phase_correct, axis)], dim=axis)


def indices(
    shape: t.Tuple[int, ...],
    dtype: t.Union[str, None, t.Type[numpy.generic], torch.dtype] = None,
    sparse: bool = False,
    device: t.Optional[torch.device] = None,
) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, ...]]:
    dtype = to_torch_dtype(dtype) if dtype is not None else torch.int32

    n = len(shape)

    if sparse:
        return tuple(
            torch.arange(s, dtype=dtype, device=device).reshape((1,) * i + (s,) + (1,) * (n - i - 1))
            for (i, s) in enumerate(shape)
        )

    arrs = tuple(torch.arange(s, dtype=dtype, device=device) for s in shape)
    return torch.stack(torch.meshgrid(*arrs, indexing='ij'), dim=0)


def size(arr: torch.Tensor, axis: t.Optional[int]) -> int:
    return arr.size(axis) if axis is not None else arr.numel()


def asarray(
    arr: t.Any, dtype: t.Union[str, torch.dtype, numpy.dtype, t.Type[numpy.generic], None] = None, *,
    copy: t.Optional[bool] = None,
) -> torch.Tensor:
    dtype = to_torch_dtype(dtype) if dtype is not None else None
    requires_grad = arr.requires_grad if isinstance(arr, torch.Tensor) else False

    if isinstance(arr, torch.Tensor):
        return torch.asarray(arr, dtype=dtype, requires_grad=requires_grad, copy=copy)

    # Neither a numpy buffer nor a bare Python sequence/scalar has an existing
    # device to preserve, so target the default device explicitly - regardless of
    # `copy`, which only decides *how* (see below), not *whether*. Plain
    # `torch.asarray(...)` on a numpy buffer would otherwise always land on CPU,
    # since it just wraps the existing (CPU) buffer.
    device = torch.get_default_device()
    if device.type == 'cpu':
        return torch.asarray(arr, dtype=dtype, requires_grad=requires_grad, copy=copy)

    if isinstance(arr, numpy.ndarray) and arr.flags['WRITEABLE'] and not copy:
        # `non_blocking` async transfers are only a proven-safe optimization on
        # CUDA; MPS has been observed to hand back a still-uninitialized buffer
        # (garbage values) when the source numpy array's lifetime isn't
        # externally pinned, so transfer synchronously there instead.
        return torch.from_numpy(arr).to(device=device, dtype=dtype, non_blocking=(device.type == 'cuda'))

    return torch.asarray(arr, dtype=dtype, requires_grad=requires_grad, copy=copy, device=device)


def affine_transform(
    input: torch.Tensor, matrix: ArrayLike,
    offset: t.Optional[ArrayLike] = None,
    output_shape: t.Optional[t.Tuple[int, ...]] = None,
    order: int = 1, mode: _InterpBoundaryMode = 'grid-constant',
    cval: ArrayLike = 0.0,
) -> torch.Tensor:
    float_dtype = max_supported_float(input.device)

    if output_shape is None:
        output_shape = input.shape
    n_axes = len(output_shape)  # num axes to transform over

    idxs = t.cast(torch.Tensor, indices(output_shape, dtype=float_dtype, device=input.device))

    matrix = asarray(matrix, dtype=float_dtype)
    if matrix.size() == (n_axes + 1, n_axes + 1):
        # homogenous transform matrix
        coords = torch.tensordot(
            matrix, torch.stack((*idxs, torch.ones_like(idxs[0])), dim=0), dims=1
        )[:-1]
    elif matrix.size() == (n_axes,):
        coords = (idxs.T * matrix + asarray(offset, dtype=float_dtype)).T
    else:
        raise ValueError(f"Expected matrix of shape ({n_axes + 1}, {n_axes + 1}) or ({n_axes},), instead got shape {matrix.shape}")

    cval = torch.asarray(cval, dtype=input.dtype)

    return torch.vmap(
        lambda a: map_coordinates(a, coords, order=order, mode=mode, cval=cval)
    )(input.reshape(-1, *input.shape[-n_axes:])).reshape((*input.shape[:-n_axes], *output_shape))


def map_coordinates(
    arr: torch.Tensor, coordinates: torch.Tensor,
    order: int = 1, mode: _InterpBoundaryMode = 'grid-constant',
    cval: ArrayLike = 0.0
) -> torch.Tensor:
    from phaser.utils.num import to_real_dtype
    if arr.ndim != coordinates.shape[0]:
        raise ValueError("invalid shape for coordinate array")

    if order not in (0, 1):
        raise ValueError(f"Interpolation order {order} not supported (torch currently only supports order=0, 1)")

    if mode == 'grid-constant':
        return _map_coordinates_constant(
            arr, coordinates, order=order, cval=cval
        )

    remap_fn = _BOUNDARY_FNS.get(mode)
    if remap_fn is None:
        raise ValueError(f"Interpolation mode '{mode}' not supported (torch supports one of "
                         "('constant', 'nearest', 'reflect', 'mirror', 'grid-wrap'))")

    weight_dtype = to_torch_dtype(to_real_dtype(to_numpy_dtype(arr.dtype)))

    ax_nodes: t.List[t.Tuple[t.Tuple[torch.Tensor, torch.Tensor], ...]] = []

    for ax_coords, size in zip(coordinates, arr.shape):
        if order == 1:
            lower = torch.floor(ax_coords)
            upper_weight = ax_coords - lower
            lower_idx = lower.type(torch.int32)
            ax_nodes.append((
                (remap_fn(lower_idx, size), 1.0 - upper_weight),
                (remap_fn(lower_idx + 1, size), upper_weight),
            ))
        else:
            idx = torch.round(ax_coords).type(torch.int32)
            ax_nodes.append(((remap_fn(idx, size), torch.ones((), dtype=weight_dtype, device=arr.device)),))

    outputs = []
    for corner in itertools.product(*ax_nodes):
        idxs, weights = zip(*corner)
        outputs.append(arr[idxs] * functools.reduce(operator.mul, weights))

    result = functools.reduce(operator.add, outputs)
    return result.type(arr.dtype)


def _map_coordinates_constant(
    arr: torch.Tensor, coordinates: torch.Tensor,
    order: int = 1, cval: ArrayLike = 0.0
) -> torch.Tensor:
    from phaser.utils.num import to_real_dtype
    weight_dtype = to_torch_dtype(to_real_dtype(to_numpy_dtype(arr.dtype)))
    cval = torch.asarray(cval, device=arr.device)

    is_valid = lambda idx, size: (0 <= idx) & (idx < size)  # noqa: E731
    clip = lambda idx, size: torch.clip(idx, 0, size - 1)   # noqa: E731

    ax_nodes: t.List[t.Tuple[t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]] = []

    for ax_coords, size in zip(coordinates, arr.shape):
        if order == 1:
            lower = torch.floor(ax_coords)
            upper_weight = ax_coords - lower
            lower_idx = lower.type(torch.int32)
            ax_nodes.append((
                (clip(lower_idx, size), is_valid(lower_idx, size), 1.0 - upper_weight),
                (clip(lower_idx + 1, size), is_valid(lower_idx + 1, size), upper_weight),
            ))
        else:
            idx = torch.round(ax_coords).type(torch.int32)
            ax_nodes.append(((clip(idx, size), is_valid(idx, size), torch.ones((), dtype=weight_dtype, device=arr.device)),))

    outputs = []
    for corner in itertools.product(*ax_nodes):
        idxs, valids, weights = zip(*corner)
        val = torch.where(functools.reduce(operator.and_, valids), arr[idxs], cval)
        outputs.append(val * functools.reduce(operator.mul, weights))

    result = functools.reduce(operator.add, outputs)
    return result.type(arr.dtype)


# convert scipy boundary mode to numpy.pad mode
_INTERP_TO_PAD: t.Dict[_InterpBoundaryMode, str] = {
    'reflect': 'symmetric',
    'mirror': 'reflect',
    'nearest': 'edge',
    'grid-mirror': 'reflect',
    'grid-wrap': 'wrap',
    'grid-constant': 'constant',
}


def convolve1d(
    arr: torch.Tensor, weights: torch.Tensor, axis: int, *,
    mode: _InterpBoundaryMode, cval: t.Union[complex, numpy.number] = 0.
) -> torch.Tensor:
    r = len(weights) // 2
    pad_mode = t.cast(_PadMode, _INTERP_TO_PAD.get(mode, mode))

    # reorder to last axis, pad
    arr = torch.moveaxis(arr, axis, -1)
    out_shape_t = arr.shape
    # pad
    arr = pad(
        arr,
        ((0, 0),) * (arr.ndim-1) + ((len(weights) - r - 1, r),),
        mode=pad_mode, cval=cval
    )

    # convolve
    arr = F.conv1d(
        arr.reshape((-1, 1, arr.shape[-1])),
        weights.flip(0).to(arr.dtype)[None, None]
    ).reshape(out_shape_t)

    return torch.moveaxis(arr, -1, axis)


def convolve2d(
    arr: torch.Tensor, weights: torch.Tensor, *,
    mode: _InterpBoundaryMode, cval: t.Union[complex, numpy.number] = 0.
) -> torch.Tensor:
    pad_mode = t.cast(_PadMode, _INTERP_TO_PAD.get(mode, mode))

    # pad array
    out_shape_t = arr.shape
    arr = pad(
        arr,
        ((0, 0),) * (arr.ndim - weights.ndim) + tuple(
            (w - w // 2 - 1, w // 2) for w in weights.shape
        ),
        mode=pad_mode, cval=cval
    )

    # convolve
    return F.conv2d(
        arr.reshape(-1, 1, arr.shape[-2], arr.shape[-1]),
        weights.flip(0, 1)[None, None].to(arr.dtype),
    ).reshape(out_shape_t)


def _dct_weights(arr: torch.Tensor, axis: int, inverse: bool) -> torch.Tensor:
    """Twiddle factors & orthonormalization for a type-II/III DCT along `axis` of `arr`."""
    n = arr.shape[axis]
    sign = 1. if inverse else -1.
    w = numpy.exp(sign * 1.j * numpy.pi * numpy.arange(n) / (2. * n))
    w *= (2. * n) ** 0.5 if inverse else (0.5 / n) ** 0.5
    w[0] *= 2. ** 0.5 if inverse else 0.5 ** 0.5

    shape = tuple(n if i == axis % arr.ndim else 1 for i in range(arr.ndim))
    dtype = torch.promote_types(arr.dtype, torch.complex64)
    return torch.as_tensor(w, dtype=dtype, device=arr.device).reshape(shape)


def _dct(arr: torch.Tensor, axis: int) -> torch.Tensor:
    """Type-II DCT along `axis`, orthonormalized."""
    n = arr.shape[axis]
    # under half-sample symmetric extension the real FFT is the DCT, up to twiddle factors
    v = torch.fft.rfft(torch.cat((arr, arr.flip(axis)), dim=axis), dim=axis)
    return (torch.narrow(v, axis, 0, n) * _dct_weights(arr, axis, False)).real


def _idct(arr: torch.Tensor, axis: int) -> torch.Tensor:
    """Type-III DCT along `axis`, orthonormalized (the inverse of `_dct`)."""
    n = arr.shape[axis]
    v = arr * _dct_weights(arr, axis, True)
    # the Nyquist term of the symmetric extension is always zero
    v = torch.cat((v, torch.zeros_like(torch.narrow(v, axis, 0, 1))), dim=axis)
    return torch.narrow(torch.fft.irfft(v, n=2 * n, dim=axis), axis, 0, n)


def dct2(arr: torch.Tensor) -> torch.Tensor:
    return _dct(_dct(arr, -2), -1)


def idct2(arr: torch.Tensor) -> torch.Tensor:
    return _idct(_idct(arr, -1), -2)


def get_devices() -> t.Tuple[torch.device, ...]:
    devices = []
    devices.extend(f'cuda:{i}' for i in range(torch.cuda.device_count()))

    if torch.backends.mps.is_available():
        devices.append('mps')

    return tuple(map(torch.device, devices))


def to_device(device: t.Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def set_default_device(device: torch.device):
    if not isinstance(device, torch.device):
        raise TypeError(f"Invalid device '{device}' for backend torch")
    torch.set_default_device(device)

    default_dtype = to_torch_dtype(max_supported_float(device))
    torch.set_default_dtype(default_dtype)


def max_supported_float(
    device: t.Optional[torch.device] = None
) -> t.Union[t.Type[numpy.float32], t.Type[numpy.float64]]:
    if device is None:
        device = torch.get_default_device()
    return numpy.float32 if device.type in ('mps', 'xpu') else numpy.float64


def _wrap_call(f, *args: t.Any, **kwargs: t.Any) -> t.Any:
    try:
        kwargs['dtype'] = to_torch_dtype(kwargs['dtype'])
    except KeyError:
        pass

    try:
        kwargs['dim'] = kwargs.pop('axes')
    except KeyError:
        try:
            kwargs['dim'] = kwargs.pop('axis')
        except KeyError:
            pass

    if f is torch.asarray and isinstance(args[0], numpy.ndarray):
        if not args[0].flags['W']:
            raise ValueError()

    # TODO: deal with tuples of output, pytrees, etc. here
    # this will result in some nasty bugs
    return f(*args, **kwargs)


mock_torch = _MockModule(torch, {
    'torch.array': functools.update_wrapper(lambda *args, **kwargs: _wrap_call(torch.asarray, *args, **kwargs), torch.asarray),  # type: ignore
    'torch.asarray': asarray,
    'torch.mod': functools.update_wrapper(lambda *args, **kwargs: _wrap_call(torch.remainder, *args, **kwargs), torch.remainder),  # type: ignore
    'torch.split': split,
    'torch.pad': pad,
    'torch.nan_to_num': nan_to_num,
    'torch.min': min, 'torch.max': max,
    'torch.nanmin': nanmin, 'torch.nanmax': nanmax,
    'torch.minimum': minimum, 'torch.maximum': maximum,
    'torch.flip': flip,
    'torch.cumsum': cumsum,
    'torch.unwrap': unwrap,
    'torch.indices': indices,
    'torch.size': size,
    'torch.iscomplexobj': lambda arr: torch.is_complex(arr),
    'torch.isrealobj': lambda arr: not torch.is_complex(arr),
}, _wrap_call)
