
import typing as t
from functools import partial

import jax  # pyright: ignore[reportMissingImports]
import jax.numpy as jnp  # pyright: ignore[reportMissingImports]
import numpy
from numpy.typing import ArrayLike

from phaser.utils.image import _InterpBoundaryMode

Device: t.TypeAlias = t.Any


def to_nd(arr: jax.Array, n: int) -> jax.Array:
    if arr.ndim > n:
        arr = arr.reshape(-1, *arr.shape[arr.ndim - n + 1:])
    elif arr.ndim < n:
        arr = jax.lax.expand_dims(arr, tuple(range(n - arr.ndim)))

    return arr

to_2d = partial(to_nd, n=2)
to_3d = partial(to_nd, n=3)


def _flatten_ndim(arr: jax.Array, n: int) -> jax.Array:
    return arr.reshape(-1, *arr.shape[n+1:])


@jax.jit
def set_cutouts(obj: jax.Array, cutouts: jax.Array, start_idxs: jax.Array) -> jax.Array:
    def inner(obj: jax.Array, v: t.Tuple[jax.Array, jax.Array]) -> t.Tuple[jax.Array, None]:
        (cutouts, start_idxs) = v
        obj = jax.vmap(lambda slice, cutouts: jax.lax.dynamic_update_slice(slice, cutouts, start_idxs))(obj, to_3d(cutouts))
        return (obj, None)

    start_idxs = jnp.atleast_2d(start_idxs)
    (cutouts, start_idxs) = (_flatten_ndim(a, start_idxs.ndim - 2) for a in (cutouts, start_idxs))
    return jax.lax.scan(inner, to_3d(obj), (cutouts, start_idxs))[0].reshape(obj.shape)


@jax.jit
def add_cutouts(obj: jax.Array, cutouts: jax.Array, start_idxs: jax.Array) -> jax.Array:
    zero_obj = jax.numpy.zeros_like(obj, shape=obj.shape[-2:])

    def inner(obj: jax.Array, v: t.Tuple[jax.Array, jax.Array]) -> t.Tuple[jax.Array, None]:
        (cutouts, start_idxs) = v
        obj += jax.vmap(lambda cutouts: jax.lax.dynamic_update_slice(zero_obj, cutouts, start_idxs))(to_3d(cutouts))
        return (obj, None)

    start_idxs = jnp.atleast_2d(start_idxs)
    (cutouts, start_idxs) = (_flatten_ndim(a, start_idxs.ndim - 2) for a in (cutouts, start_idxs))
    return jax.lax.scan(inner, to_3d(obj), (cutouts, start_idxs))[0].reshape(obj.shape)


@partial(jax.jit, static_argnums=2)
def get_cutouts(obj: jax.Array, start_idxs: jax.Array, cutout_shape: t.Tuple[int, int]) -> jax.Array:
    idxs = to_2d(start_idxs)
    ys = idxs[:, 0:1] + jax.numpy.arange(cutout_shape[0])
    xs = idxs[:, 1:2] + jax.numpy.arange(cutout_shape[1])
    return jax.numpy.swapaxes(
        to_3d(obj)[..., ys[:, :, None], xs[:, None, :]], 0, 1
    ).reshape((*start_idxs.shape[:-1], *obj.shape[:-2], *cutout_shape))


@partial(jax.jit, static_argnums=0)
def outer(ufunc: t.Any, x: jax.Array, y: jax.Array) -> jax.Array:
    if x.ndim == 0 or y.ndim == 0:
        return ufunc(x, y)

    out_shape = (*x.shape, *y.shape)
    return jax.vmap(jax.vmap(ufunc, (None, 0)), (0, None))(x.ravel(), y.ravel()).reshape(out_shape)


@partial(jax.jit, static_argnames=('output_shape', 'order', 'mode', 'cval'))
def affine_transform(
    input: jax.Array,
    matrix: ArrayLike,
    offset: t.Optional[ArrayLike] = None,
    output_shape: t.Optional[t.Tuple[int, ...]] = None,
    order: int = 1,
    mode: str = 'constant',
    cval: t.Any = 0.0,
) -> jax.Array:
    import jax.scipy.ndimage
    jax_mode = {'grid-constant': 'constant', 'grid-wrap': 'wrap'}.get(mode, mode)

    if output_shape is None:
        output_shape = input.shape
    n_axes = len(output_shape)  # num axes to transform over

    indices = jnp.indices(output_shape, dtype=float)

    matrix = jnp.array(matrix)
    offset = jnp.zeros((), matrix.dtype) if offset is None else jnp.array(offset)

    if matrix.shape == (n_axes, n_axes + 1):
        # add bottom row to transformation matrix
        matrix = jnp.stack((matrix, jnp.array([0.] * n_axes + [1.], matrix.dtype)), axis=0)
    if matrix.shape == (n_axes + 1, n_axes + 1):
        # homogenous transform matrix
        coords = jnp.tensordot(
            matrix, jnp.stack((*indices, jnp.ones_like(indices[0])), axis=0), axes=1
        )[:-1]
    elif matrix.shape == (n_axes, n_axes):
        coords = (indices.T @ matrix + jnp.array(offset)).T
    elif matrix.shape == (n_axes,):
        coords = (indices.T * matrix + jnp.array(offset)).T
    else:
        raise ValueError(f"Expected matrix of shape ({n_axes}, {n_axes}), ({n_axes},), ({n_axes + 1}, {n_axes + 1}), or ({n_axes}, {n_axes + 1}), instead got shape {matrix.shape}")

    coords += jnp.finfo(coords.dtype).eps

    return jax.vmap(
        lambda a: jax.scipy.ndimage.map_coordinates(a, tuple(coords), order=order, mode=jax_mode, cval=cval),
    )(to_nd(input, n_axes + 1)).reshape((*input.shape[:-n_axes], *output_shape))


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
    arr: jnp.ndarray, weights: jnp.ndarray, axis: int = -1, *,
    mode: _InterpBoundaryMode, cval: t.Union[complex, numpy.number] = 0.
) -> jnp.ndarray:
    r = len(weights) // 2
    pad_mode = _INTERP_TO_PAD.get(mode, mode)
    pad_kwargs = {'constant_values': cval} if pad_mode == 'constant' else {}

    # transpose, pad
    arr = jnp.moveaxis(arr, axis, -1)
    out_shape_t = arr.shape
    arr = jnp.pad(
        arr,
        ((0, 0),) * (arr.ndim-1) + ((len(weights) - r - 1, r),),
        mode=pad_mode, **pad_kwargs
    )

    # convolve
    arr = jax.lax.conv_general_dilated(
        arr.reshape(-1, 1, arr.shape[-1]),
        to_3d(jnp.flip(weights)).astype(arr.dtype),
        window_strides=(1,), padding='VALID',
        dimension_numbers=('NCW', 'OIW', 'NCW'),
    )

    # unflatten, untranspose
    return jnp.moveaxis(arr.reshape(out_shape_t), -1, axis)


def convolve2d(
    arr: jnp.ndarray, weights: jnp.ndarray, *,
    mode: _InterpBoundaryMode, cval: t.Union[complex, numpy.number] = 0.
) -> jnp.ndarray:
    pad_mode = _INTERP_TO_PAD.get(mode, mode)
    pad_kwargs = {'constant_values': cval} if pad_mode == 'constant' else {}

    # pad array
    out_shape_t = arr.shape
    arr = jnp.pad(
        arr,
        ((0, 0),) * (arr.ndim - weights.ndim) + tuple(
            (w - w // 2 - 1, w // 2) for w in weights.shape
        ),
        mode=pad_mode, **pad_kwargs
    )

    # convolve
    arr = jax.lax.conv_general_dilated(
        arr.reshape(-1, 1, arr.shape[-2], arr.shape[-1]),
        jnp.flip(weights)[None, None].astype(arr.dtype),
        window_strides=(1,1), padding='VALID',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
    )

    # unflatten
    return arr.reshape(out_shape_t)


def get_devices() -> t.Tuple[Device, ...]:
    devices = []

    for backend in ('gpu', 'tpu', 'cpu'):
        try:
            devices.extend(jax.devices(backend))
        except RuntimeError:
            pass

    return tuple(devices)


def to_device(device: t.Union[str, Device]) -> Device:
    if isinstance(device, jax.Device):
        return device

    split = device.rsplit(':', maxsplit=1)
    if len(split) == 1:
        [backend] = split
        index = 0
    else:
        [backend, index] = split
        index = int(index)

    try:
        backend_devices = jax.devices(backend)
    except RuntimeError:
        raise RuntimeError(f"Can't use device '{device}': jax backend '{backend}' is unavailable")

    try:
        return backend_devices[index]
    except IndexError:
        pass
    if len(backend_devices) == 0:
        raise RuntimeError(f"Can't use device '{device}': No available devices on jax backend '{backend}'")
    raise RuntimeError(f"Can't use device '{device}': Device index {index} not available"
                       f" ({len(backend_devices)} device(s) on jax backend '{backend}')")


def set_default_device(device: Device):
    if not isinstance(device, jax.Device):
        raise TypeError(f"Invalid device '{device}' for backend jax")
    jax.config.update('jax_default_device', device)