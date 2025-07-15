"""
General numeric utilities.
"""

import functools
import logging
import warnings
import typing as t

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray

from phaser.types import BackendName
from .misc import jax_dataclass


if t.TYPE_CHECKING:
    from phaser.utils.image import _BoundaryMode


Float: t.TypeAlias = t.Union[float, numpy.floating]
NumT = t.TypeVar('NumT', bound=numpy.number)
FloatT = t.TypeVar('FloatT', bound=numpy.floating)
ComplexT = t.TypeVar('ComplexT', bound=numpy.complexfloating)
DTypeT = t.TypeVar('DTypeT', bound=numpy.generic)
T = t.TypeVar('T')
P = t.ParamSpec('P')

IndexLike: t.TypeAlias = t.Union[
    int,
    NDArray[numpy.integer[t.Any]],
    NDArray[numpy.bool_],
    t.Tuple[t.Union[int, NDArray[numpy.integer[t.Any]], NDArray[numpy.bool_]], ...],
]


logger = logging.getLogger(__name__)

try:
    import jax
    jax.config.update('jax_enable_x64', jax.default_backend() != 'METAL')
    #jax.config.update('jax_log_compiles', True)
    #jax.config.update('jax_debug_nans', True)
except ImportError:
    pass


def get_backend_module(backend: t.Optional[BackendName] = None):
    """Get the module `xp` associated with a compute backend"""
    if backend is None:
        return get_default_backend_module()

    backend = t.cast(BackendName, backend.lower())
    if backend not in ('cuda', 'cupy', 'jax', 'cpu', 'numpy'):
        raise ValueError(f"Unknown backend '{backend}'")

    if not t.TYPE_CHECKING:
        try:
            if backend == 'jax':
                import jax.numpy
                return jax.numpy
            if backend in ('cupy', 'cuda'):
                import cupy
                return cupy
        except ImportError:
            raise ValueError(f"Backend '{backend}' is not available")

    return numpy


def get_default_backend_module():
    if not t.TYPE_CHECKING:
        try:
            import jax.numpy
            return jax.numpy
        except ImportError:
            pass

        try:
            import cupy
            return cupy
        except ImportError:
            pass

    return numpy


def get_array_module(*arrs: t.Optional[ArrayLike]):
    try:
        import jax
        if any(isinstance(arr, jax.Array) for arr in arrs) \
           and not t.TYPE_CHECKING:
            return jax.numpy
    except ImportError:
        pass
    try:
        from cupy import get_array_module as f  # type: ignore
        if not t.TYPE_CHECKING:
            return f(*arrs)
    except ImportError:
        pass
    return numpy


def cast_array_module(xp: t.Any):
    if t.TYPE_CHECKING:
        return numpy
    return xp


def get_scipy_module(*arrs: t.Optional[ArrayLike]):
    # pyright: ignore[reportMissingImports,reportUnusedImport]

    import scipy

    try:
        import jax
        if any(isinstance(arr, jax.Array) for arr in arrs) \
           and not t.TYPE_CHECKING:
            return jax.scipy
    except ImportError:
        pass
    try:
        with warnings.catch_warnings():
            # https://github.com/cupy/cupy/issues/8718
            warnings.filterwarnings(action='ignore', message=r"cupyx\.jit\.rawkernel is experimental", category=FutureWarning)

            import cupyx.scipy.signal  # pyright: ignore[reportMissingImports]
            import cupyx.scipy.ndimage  # pyright: ignore[reportMissingImports]  # noqa: F401
            from cupyx.scipy import get_array_module as f  # pyright: ignore[reportMissingImports]

        if not t.TYPE_CHECKING:
            return f(*arrs)
    except ImportError:
        pass

    return scipy


def to_numpy(arr: t.Union[DTypeT, NDArray[DTypeT]], stream=None) -> NDArray[DTypeT]:
    """
    Convert an array to numpy.
    For cupy backend, this is equivalent to `cupy.asnumpy`.
    """
    if not t.TYPE_CHECKING:
        if is_jax(arr):
            return numpy.array(arr)

        if is_cupy(arr):
            return arr.get(stream)

    return numpy.array(arr)


def as_numpy(arr: ArrayLike, stream=None) -> NDArray:
    """
    Convert an ArrayLike to a numpy array.
    For cupy backend, this is equivalent to `cupy.asnumpy`.
    """
    if not t.TYPE_CHECKING:
        if is_jax(arr):
            return numpy.array(arr)

        if is_cupy(arr):
            return arr.get(stream)

    return numpy.asarray(arr)


def as_array(arr: ArrayLike, xp: t.Any = None) -> numpy.ndarray:
    """
    Convert an ArrayLike to an array, but not necessarily
    a numpy array.
    """
    if not t.TYPE_CHECKING:
        if xp is not None:
            return xp.asarray(arr)
        xp = get_array_module(arr)
        if xp is not numpy:
            return arr
    return numpy.asarray(arr)


def is_cupy(arr: NDArray[DTypeT]) -> bool:
    try:
        import cupy  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return isinstance(arr, cupy.ndarray)


def is_jax(arr: t.Any) -> bool:
    try:
        import jax  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return any(
        isinstance(arr, jax.Array) for arr in jax.tree_util.tree_leaves(arr)
    )


def xp_is_cupy(xp: t.Any) -> bool:
    try:
        import cupy  # pyright: ignore[reportMissingImports]
        return xp is cupy
    except ImportError:
        return False


def xp_is_jax(xp: t.Any) -> bool:
    try:
        import jax.numpy  # pyright: ignore[reportMissingImports]
        return xp is jax.numpy
    except ImportError:
        return False


def block_until_ready(arr: NDArray[DTypeT]) -> NDArray[DTypeT]:
    if hasattr(arr, 'block_until_ready'):  # jax
        return arr.block_until_ready()  # type: ignore

    if is_cupy(arr):
        import cupy  # pyright: ignore[reportMissingImports]
        stream = cupy.cuda.get_current_stream()
        stream.synchronize()

    return arr


class _JitKernel(t.Generic[P, T]):
    def __init__(
        self, f: t.Callable[P, T], *,
        static_argnums: t.Union[int, t.Sequence[int], None] = None,
        static_argnames: t.Union[str, t.Iterable[str], None] = None,
        donate_argnums: t.Union[int, t.Sequence[int], None] = None,
        donate_argnames: t.Union[str, t.Iterable[str], None] = None,
        inline: bool = False,
        #compiler_options: t.Optional[t.Dict[str, t.Any]] = None,
        cupy_fuse: bool = False
    ):
        self.inner = f
        functools.update_wrapper(self, f)

        if cupy_fuse:
            try:
                import cupy  # pyright: ignore[reportMissingImports]
                self.inner = cupy.fuse()(self.inner)
            except ImportError:
                pass

        # in jax: self.__call__ -> jax.jit -> jax_f -> f
        # otherwise: self.__call__ -> f
        try:
            import jax
        except ImportError:
            self.jax_jit = None
        else:
            @functools.wraps(f)
            def jax_f(*args: P.args, **kwargs: P.kwargs) -> T:
                logger.info(f"JIT-compiling kernel '{self.__qualname__}'...")
                return self.inner(*args, **kwargs)

            self.jax_jit = jax.jit(
                jax_f, static_argnums=static_argnums, static_argnames=static_argnames,
                donate_argnums=donate_argnums, donate_argnames=donate_argnames,
                inline=inline, #compiler_options=compiler_options
            )


    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if self.jax_jit and (any(map(is_jax, args)) or any(map(is_jax, kwargs.values()))):  # type: ignore
            return self.jax_jit(*args, **kwargs)

        return self.inner(*args, **kwargs)


def jit(
        f: t.Callable[P, T], *,
        static_argnums: t.Union[int, t.Sequence[int], None] = None,
        static_argnames: t.Union[str, t.Iterable[str], None] = None,
        donate_argnums: t.Union[int, t.Sequence[int], None] = None,
        donate_argnames: t.Union[str, t.Iterable[str], None] = None,
        inline: bool = False,
        #compiler_options: t.Optional[t.Dict[str, t.Any]] = None,
        cupy_fuse: bool = False,
) -> t.Callable[P, T]:
    return _JitKernel(
        f, static_argnums=static_argnums, static_argnames=static_argnames,
        donate_argnums=donate_argnums, donate_argnames=donate_argnames,
        inline=inline, #compiler_options=compiler_options,
        cupy_fuse=cupy_fuse
    )


def fuse(*args, **kwargs) -> t.Callable[[T], T]:
    """
    Equivalent to `cupy.fuse`, if supported.
    """
    try:
        import cupy  # pyright: ignore[reportMissingImports]
        if not t.TYPE_CHECKING:
            return cupy.fuse(*args, **kwargs)
    except ImportError:
        pass
    return lambda x: x


def debug_callback(callback: t.Callable[P, None], *args: P.args, **kwargs: P.kwargs):
    try:
        import jax.debug
        return jax.debug.callback(callback, *args, **kwargs)
    except ImportError:
        callback(*args, **kwargs)


_COMPLEX_MAP: t.Dict[t.Type[numpy.floating], t.Type[numpy.complexfloating]] = {
    numpy.floating: numpy.complexfloating,
    numpy.float32: numpy.complex64,
    numpy.float64: numpy.complex128,
    #numpy.float80: numpy.complex160,
    #numpy.float96: numpy.complex192,
    #numpy.float128: numpy.complex256,
    #numpy.float256: numpy.complex512,
}

try:  # numpy 1.x
    _COMPLEX_MAP[numpy.float_] = numpy.complex_  # type: ignore
except AttributeError:
    pass

_REAL_MAP: t.Dict[t.Type[numpy.complexfloating], t.Type[numpy.floating]] = dict((v, k) for (k, v) in _COMPLEX_MAP.items())


@t.overload
def to_complex_dtype(dtype: t.Type[numpy.float32]) -> t.Type[numpy.complex64]:
    ...

@t.overload
def to_complex_dtype(dtype: t.Type[numpy.float64]) -> t.Type[numpy.complex128]:
    ...

@t.overload
def to_complex_dtype(dtype: DTypeLike) -> t.Type[numpy.complexfloating]:
    ...

def to_complex_dtype(dtype: DTypeLike) -> t.Type[numpy.complexfloating]:
    """
    Convert a floating point dtype to a complex version.
    """

    if not (isinstance(dtype, type) and issubclass(dtype, numpy.generic)):
        dtype = numpy.dtype(dtype).type

    if not isinstance(dtype, type) or not issubclass(dtype, (numpy.floating, numpy.complexfloating)):
        raise TypeError("Non-floating point datatype")

    if issubclass(dtype, numpy.complexfloating):
        return dtype

    try:
        return _COMPLEX_MAP[dtype]
    except KeyError:
        raise TypeError(f"Unsupported datatype '{dtype}'") from None


@t.overload
def to_real_dtype(dtype: t.Type[numpy.complex64]) -> t.Type[numpy.float32]:
    ...

@t.overload
def to_real_dtype(dtype: t.Type[numpy.complex128]) -> t.Type[numpy.float64]:
    ...

@t.overload
def to_real_dtype(dtype: DTypeLike) -> t.Type[numpy.floating]:
    ...

def to_real_dtype(dtype: DTypeLike) -> t.Type[numpy.floating]:
    """
    Convert a complex dtype to a plain float version.
    """

    if not (isinstance(dtype, type) and issubclass(dtype, numpy.generic)):
        dtype = numpy.dtype(dtype).type

    if not isinstance(dtype, type) or not issubclass(dtype, (numpy.floating, numpy.complexfloating)):
        raise TypeError("Non-floating point datatype")

    if issubclass(dtype, numpy.floating):
        return dtype

    try:
        return _REAL_MAP[dtype]
    except KeyError:
        raise TypeError(f"Unsupported datatype '{dtype}'") from None


@t.overload
def ifft2(a: t.Union[NDArray[numpy.float64], NDArray[numpy.complex128]]) -> NDArray[numpy.complex128]:
    ...

@t.overload
def ifft2(a: t.Union[NDArray[numpy.float32], NDArray[numpy.complex64]]) -> NDArray[numpy.complex64]:
    ...

@t.overload
def ifft2(a: NDArray[NumT]) -> NDArray[numpy.complexfloating]:
    ...

@t.overload
def ifft2(a: ArrayLike) -> NDArray[numpy.complexfloating]:
    ...

def ifft2(a: ArrayLike) -> NDArray[numpy.complexfloating]:
    """
    Perform an inverse FFT on the last two axes of `a`.
    
    Follows our convention of centering real space and normalizing intensities.
    """

    xp = get_array_module(a)
    return xp.fft.fftshift(xp.fft.ifft2(a, norm='ortho'), axes=(-2, -1))

@t.overload
def fft2(a: t.Union[NDArray[numpy.float64], NDArray[numpy.complex128]]) -> NDArray[numpy.complex128]:
    ...

@t.overload
def fft2(a: t.Union[NDArray[numpy.float32], NDArray[numpy.complex64]]) -> NDArray[numpy.complex64]:
    ...

@t.overload
def fft2(a: NDArray[NumT]) -> NDArray[numpy.complexfloating]:
    ...

@t.overload
def fft2(a: ArrayLike) -> NDArray[numpy.complexfloating]:
    ...

def fft2(a: ArrayLike) -> NDArray[numpy.complexfloating]:
    """
    Perform a forward FFT on the last two axes of `a`.

    Follows our convention of centering real space and normalizing intensities.
    """

    xp = get_array_module(a)
    return xp.fft.fft2(xp.fft.ifftshift(a, axes=(-2, -1)), norm='ortho')


def split_array(arr: NDArray[DTypeT], axis: int = 0, *, keepdims: bool = False) -> t.Tuple[NDArray[DTypeT], ...]:
    """
    Split an array along `axis`, returning a tuple of subarrays.

    # Parameters

    - `arr`: Input array to split
    - `axis`: Axis to split array on. Default: 0
    - `keepdims`: Whether to keep the split dimension in the returned arrays.
    """
    xp = get_array_module(arr)
    arrs = xp.split(arr, arr.shape[axis], axis=axis)
    return tuple(arr) if keepdims else tuple(xp.squeeze(arr, axis) for arr in arrs)


unstack = split_array


@t.overload
def abs2(x: t.Union[NDArray[numpy.float64], NDArray[numpy.complex128]]) -> NDArray[numpy.float64]:
    ...

@t.overload
def abs2(x: t.Union[NDArray[numpy.float32], NDArray[numpy.complex64]]) -> NDArray[numpy.float32]:
    ...

@t.overload
def abs2(x: t.Union[NDArray[numpy.complexfloating], NDArray[numpy.floating]]) -> NDArray[numpy.floating]:
    ...

@t.overload
def abs2(x: ArrayLike) -> NDArray[numpy.floating]:
    ...

def abs2(x: ArrayLike) -> NDArray[numpy.floating]:
    """
    Return the squared amplitude of a complex array.

    This is cheaper than `abs(x)**2.`
    """
    x = get_array_module(x).array(x)
    return x.real**2. + x.imag**2.  # type: ignore


@t.overload
def ufunc_outer(ufunc: numpy.ufunc, x: NDArray[DTypeT], y: ArrayLike) -> NDArray[DTypeT]:
    ...

@t.overload
def ufunc_outer(ufunc: numpy.ufunc, x: ArrayLike, y: NDArray[DTypeT]) -> NDArray[DTypeT]:
    ...

@t.overload
def ufunc_outer(ufunc: numpy.ufunc, x: ArrayLike, y: ArrayLike) -> numpy.ndarray:
    ...

def ufunc_outer(ufunc: numpy.ufunc, x: ArrayLike, y: ArrayLike) -> numpy.ndarray:
    if not t.TYPE_CHECKING and is_jax(x):
        from ._jax_kernels import outer
        return outer(ufunc, x, y)

    return ufunc.outer(x, y)


def check_finite(*arrs: NDArray[numpy.inexact], context: t.Optional[str] = None):
    xp = get_array_module(*arrs)

    if not all(xp.all(xp.isfinite(arr)) for arr in arrs):
        if context:
            raise ValueError(f"NaN or inf encountered in {context}")
        raise ValueError("NaN or inf encountered")


@jax_dataclass(frozen=True, init=False, drop_fields=('extent',))
class Sampling:
    shape: NDArray[numpy.int_]
    """Sampling shape (n_y, n_x)"""
    extent: NDArray[numpy.float64]
    """Sampling diameter (b, a)"""
    sampling: NDArray[numpy.float64]
    """Sample spacing (s_y, s_x)"""

    def __eq__(self, other: t.Any) -> bool:
        if type(self) is not type(other):
            return False
        xp = get_array_module(self.sampling, other.sampling)
        return (
            xp.array_equal(self.shape, other.shape) and
            xp.array_equal(self.extent, other.extent)
        )

    @property
    def k_max(self) -> NDArray[numpy.float64]:
        """
        Return maximum frequency (radius) of reciprocal space (1/(2s_y), 1/(2s_x))
        """
        return (1/(2 * self.sampling)).astype(numpy.float64)

    @property
    def corner(self) -> NDArray[numpy.float64]:
        return ((-self.extent + self.sampling) / 2.).astype(numpy.float64)

    @t.overload
    def __init__(self,
                 shape: t.Tuple[int, int], *,
                 extent: t.Union[ArrayLike, t.Tuple[Float, Float]],
                 sampling: None = None):
        ...

    @t.overload
    def __init__(self,
                 shape: t.Tuple[int, int], *,
                 extent: None = None,
                 sampling: t.Union[ArrayLike, t.Tuple[Float, Float]]):
        ...

    def __init__(self,
                 shape: ArrayLike, *,
                 extent: t.Union[ArrayLike, t.Tuple[Float, Float], None] = None,
                 sampling: t.Union[ArrayLike, t.Tuple[Float, Float], None] = None):
        try:
            object.__setattr__(self, 'shape', numpy.broadcast_to(as_numpy(shape).astype(numpy.int_), (2,)))
        except ValueError as e:
            raise ValueError(f"Expected a shape (n_y, n_x), instead got: {shape}") from e

        if extent is not None:
            try:
                object.__setattr__(self, 'extent', numpy.broadcast_to(
                    as_numpy(t.cast(ArrayLike, extent)).astype(numpy.float64), (2,)
                ))
            except ValueError as e:
                raise ValueError(f"Expected an extent (b, a), instead got: {extent}") from e
            object.__setattr__(self, 'sampling', self.extent / self.shape)
        elif sampling is not None:
            try:
                object.__setattr__(self, 'sampling', numpy.broadcast_to(
                    as_numpy(t.cast(ArrayLike, sampling)).astype(numpy.float64), (2,)
                ))
            except ValueError as e:
                raise ValueError(f"Expected a sampling (s_y, s_x), instead got: {sampling}") from e
            object.__setattr__(self, 'extent', self.sampling * self.shape)
        else:
            raise ValueError("Either 'extent' or 'sampling' must be specified")

    @t.overload
    def real_grid(  # pyright: ignore[reportOverlappingOverload]
        self, *, dtype: t.Type[NumT], xp: t.Any = None
    ) -> t.Tuple[NDArray[NumT], NDArray[NumT]]:
        ...

    @t.overload
    def real_grid(
        self, *, dtype: t.Optional[DTypeLike] = None, xp: t.Any = None
    ) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
        ...

    def real_grid(self, *, dtype: t.Any = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.number], NDArray[numpy.number]]:
        """Return the realspace sampling grid `(yy, xx)`. Grid is centered around `(0, 0)` (but may not contain it!)"""
        xp2 = get_array_module(self.shape, self.extent, self.sampling) if xp is None else cast_array_module(xp)

        if dtype is None:
            dtype = numpy.common_type(self.extent, self.sampling)

        corner = self.corner
        ys = xp2.linspace(corner[0], corner[0] + self.extent[0], self.shape[0], endpoint=False, dtype=dtype)
        xs = xp2.linspace(corner[1], corner[1] + self.extent[1], self.shape[1], endpoint=False, dtype=dtype)
        return tuple(xp2.meshgrid(ys, xs, indexing='ij'))  # type: ignore

    @t.overload
    def recip_grid(  # pyright: ignore[reportOverlappingOverload]
        self, *, centered: bool = False, dtype: t.Type[NumT], xp: t.Any = None
    ) -> t.Tuple[NDArray[NumT], NDArray[NumT]]:
        ...

    @t.overload
    def recip_grid(
        self, *, centered: bool = False, dtype: t.Optional[DTypeLike] = None, xp: t.Any = None
    ) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
        ...

    def recip_grid(
        self, *, centered: bool = False, dtype: t.Any = None, xp: t.Any = None
    ) -> t.Tuple[NDArray[numpy.number], NDArray[numpy.number]]:
        """
        Return the reciprocal space sampling grid `(kyy, kxx)`.

        Unless `centered` is specified, the grid is fftshifted so the zero-frequency component is in the top left.
        """
        xp2 = get_array_module(self.shape, self.extent, self.sampling) if xp is None else cast_array_module(xp)

        if dtype is None:
            dtype = numpy.common_type(self.extent, self.sampling)

        ky: NDArray[numpy.number] = xp2.fft.fftfreq(self.shape[0], self.sampling[0]).astype(dtype)
        kx: NDArray[numpy.number] = xp2.fft.fftfreq(self.shape[1], self.sampling[1]).astype(dtype)

        if centered:
            ky = xp2.fft.fftshift(ky)
            kx = xp2.fft.fftshift(kx)
        return tuple(xp2.meshgrid(ky, kx, indexing='ij'))  # type: ignore

    def bwlim(self) -> float:
        """Return the bandwidth limit (in inverse length units) for this sampling grid."""
        return float(numpy.min(self.k_max) * 2./3.)

    def mpl_real_extent(self, center: bool = True) -> t.Tuple[float, float, float, float]:
        """
        Return the extent of real space, for use in matplotlib.

        Extent is returned as `(left, right, bottom, top)`.
        If `center` is specified (the default), samples correspond to the center of pixels.
        Otherwise, they correspond to the corners of pixels.
        """
        # shift pixel corners to centers
        shift = -self.extent/2. + self.sampling/2. * int(not center)
        return (shift[1], self.extent[1] + shift[1], self.extent[0] + shift[0], shift[0])

    def mpl_recip_extent(self, center: bool = True) -> t.Tuple[float, float, float, float]:
        """
        Return the extent of reciprocal space, for use in matplotlib.

        Extent is returned as `(left, right, bottom, top)`.
        If `center` is specified (the default), samples correspond to the center of pixels.
        Otherwise, they correspond to the corners of pixels.
        """
        kmax = self.k_max
        hp = 1/(2. * self.extent)
        # for odd sampling, grid is shifted by 1/2 pixel
        # also, shift pixel corners to centers
        shift = hp * (self.shape % 2) - hp * int(center)
        return (-kmax[1] + shift[1], kmax[1] + shift[1], kmax[0] + shift[0], -kmax[0] + shift[0])

    def _coord_to_real(self, center: bool = True) -> NDArray[numpy.floating]:
        a = numpy.diag([*self.sampling, 1.])
        a[:2, 2] = -self.extent/2. + self.sampling/2. * int(center)
        return a

    def _real_to_coord(self, center: bool = True) -> NDArray[numpy.floating]:
        a = numpy.diag([*1/self.sampling, 1.])
        a[:2, 2] = (self.extent/2. - self.sampling/2. * int(center)) / self.sampling
        return a

    def resample(
        self, arr: NDArray[NumT], new_samp: 'Sampling', *,
        rotation: float = 0.0,
        order: int = 1,
        mode: '_BoundaryMode' = 'grid-constant',
        cval: t.Union[NumT, float] = 0.0,
    ) -> NDArray[NumT]:
        from .image import affine_transform, rotation_matrix

        if arr.shape[-2:] != tuple(self.shape):
            raise ValueError("Image dimension don't match sampling dimensions")

        if rotation != 0.0:
            matrix = self._real_to_coord(True) @ rotation_matrix(rotation) @ new_samp._coord_to_real(True)

            return affine_transform(arr, matrix, output_shape=tuple(new_samp.shape), order=order, mode=mode, cval=cval)

        matrix = new_samp.sampling / self.sampling
        offset = ((self.shape - 1) - matrix * (new_samp.shape - 1)) / 2.

        return affine_transform(arr, matrix, offset, output_shape=tuple(new_samp.shape), order=order, mode=mode, cval=cval)

    def resample_recip(
        self, arr: NDArray[NumT], new_samp: 'Sampling', *,
        rotation: float = 0.0,
        order: int = 1,
        mode: '_BoundaryMode' = 'grid-constant',
        cval: t.Union[NumT, float] = 0.0,
        fftshift: bool = True,
    ) -> NDArray[NumT]:
        xp = get_array_module(arr)
        # reciprocal space sampling
        old_samp = Sampling(self.shape, extent=tuple(self.k_max)) # type: ignore
        new_samp = Sampling(new_samp.shape, extent=tuple(new_samp.k_max))  # type: ignore

        if fftshift:
            arr = xp.fft.fftshift(arr, axes=(-1, -2))

        # and resample like it's in realspace
        result = old_samp.resample(arr, new_samp, rotation=rotation, order=order, mode=mode, cval=cval)

        return xp.fft.ifftshift(result, axes=(-1, -2)) if fftshift else result

#_IndexingMode: t.TypeAlias = t.Literal['promise_in_bounds', 'clip', 'drop', 'fill']


class _AtImpl(t.Generic[DTypeT]):
    def __init__(self, arr: NDArray[DTypeT], idx: IndexLike):
        self.arr: NDArray[DTypeT] = arr
        self.idx: IndexLike = idx

    def set(self, values: t.Union[NDArray[DTypeT], DTypeT]) -> NDArray[DTypeT]:
        self.arr[self.idx] = values
        return self.arr

    def add(self, values: t.Union[NDArray[DTypeT], DTypeT]) -> NDArray[DTypeT]:
        self.arr[self.idx] += values  # type: ignore
        return self.arr

    def subtract(self, values: t.Union[NDArray[DTypeT], DTypeT]) -> NDArray[DTypeT]:
        self.arr[self.idx] -= values  # type: ignore
        return self.arr

    def multiply(self, values: t.Union[NDArray[DTypeT], DTypeT]) -> NDArray[DTypeT]:
        self.arr[self.idx] *= values  # type: ignore
        return self.arr

    def divide(self, values: t.Union[NDArray[DTypeT], DTypeT]) -> NDArray[DTypeT]:
        self.arr[self.idx] /= values  # type: ignore
        return self.arr

    def power(self, values: t.Union[NDArray[DTypeT], DTypeT]) -> NDArray[DTypeT]:
        self.arr[self.idx] **= values  # type: ignore
        return self.arr

    def min(self, values: t.Union[NDArray[DTypeT], DTypeT]) -> NDArray[DTypeT]:
        xp = get_array_module(self.arr, values)
        self.arr[self.idx] = xp.minimum(self.arr[self.idx], values)
        return self.arr

    def max(self, values: t.Union[NDArray[DTypeT], DTypeT]) -> NDArray[DTypeT]:
        xp = get_array_module(self.arr, values)
        self.arr[self.idx] = xp.maximum(self.arr[self.idx], values)
        return self.arr

    #def apply(self, ufunc):
    #    ...

    def get(self) -> NDArray[DTypeT]:
        self.arr = self.arr[self.idx]
        return self.arr


def at(arr: NDArray[DTypeT], idx: IndexLike) -> _AtImpl[DTypeT]:
    if is_jax(arr) and not t.TYPE_CHECKING:
        return arr.at[idx]

    return _AtImpl(arr, idx)


__all__ = [
    'get_backend_module', 'get_default_backend_module',
    'get_array_module', 'cast_array_module', 'get_scipy_module',
    'to_numpy', 'as_numpy', 'as_array',
    'is_cupy', 'is_jax', 'xp_is_cupy', 'xp_is_jax',
    'jit', 'fuse', 'debug_callback',
    'to_complex_dtype', 'to_real_dtype',
    'fft2', 'ifft2', 'abs2', 'split_array', 'unstack',
    'at', 'ufunc_outer', 'check_finite',
    'Sampling', 'IndexLike',
]