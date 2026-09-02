"""
General numeric utilities.
"""

import functools
from itertools import chain
import logging
import warnings
from types import ModuleType, EllipsisType
import typing as t
import sys

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray

from phaser.types import BackendName
from .tree import tree_dataclass


if t.TYPE_CHECKING:
    from phaser.utils.image import _InterpBoundaryMode


Device: t.TypeAlias = t.Any
Float: t.TypeAlias = t.Union[float, numpy.floating]
NumT = t.TypeVar('NumT', bound=numpy.number)
FloatT = t.TypeVar('FloatT', bound=numpy.floating)
InexactT = t.TypeVar('InexactT', bound=numpy.inexact)
ComplexT = t.TypeVar('ComplexT', bound=numpy.complexfloating)
DTypeT = t.TypeVar('DTypeT', bound=numpy.generic)
T = t.TypeVar('T')
P = t.ParamSpec('P')

IndexLike: t.TypeAlias = t.Union[
    int, slice, EllipsisType,
    NDArray[numpy.integer[t.Any]],
    NDArray[numpy.bool_],
    t.Tuple[t.Union[int, slice, EllipsisType, NDArray[numpy.integer[t.Any]], NDArray[numpy.bool_]], ...],
]

logger = logging.getLogger(__name__)


def _load_cupy() -> ModuleType:
    from ._cuda_kernels import mock_cupy

    with warnings.catch_warnings():
        # https://github.com/cupy/cupy/issues/8718
        warnings.filterwarnings(action='ignore', message=r"cupyx\.jit\.rawkernel is experimental", category=FutureWarning)
        import cupyx.scipy.signal   # pyright: ignore[reportMissingImports,reportUnusedImport]
        import cupyx.scipy.ndimage  # pyright: ignore[reportMissingImports,reportUnusedImport] # noqa: F401

    return t.cast(ModuleType, mock_cupy)

def _load_jax() -> ModuleType:
    import jax
    jax.config.update('jax_enable_x64', jax.default_backend() != 'METAL')
    import jax.scipy

    return jax.numpy

def _load_torch() -> ModuleType:
    from ._torch_kernels import mock_torch
    return t.cast(ModuleType, mock_torch)


_NAME_REMAP: t.Dict[BackendName, BackendName] = {}

_LOAD_FNS: t.Dict[BackendName, t.Callable[[], ModuleType]] = {
    'cupy': _load_cupy,
    'jax': _load_jax,
    'torch': _load_torch,
}


class _BackendLoader:
    def __init__(self):
        self.inner: t.Dict[BackendName, t.Optional[ModuleType]] = {}
        self.cbs: t.Dict[BackendName, t.List[t.Callable[[], t.Any]]] = {}

    def _normalize(self, backend: BackendName) -> BackendName:
        name = t.cast(BackendName, backend.lower())
        name = _NAME_REMAP.get(name, name)

        if name not in ('cupy', 'jax', 'numpy', 'torch'):
            raise ValueError(f"Unknown backend '{backend}'")
        return name

    def _load(self, name: BackendName):
        try:
            self.inner[name] = _LOAD_FNS[name]()
            for cb in self.cbs.pop(name, ()):
                cb()
        except ImportError:
            self.inner[name] = None

    def _schedule_on_load(self, backend: BackendName, fn: t.Callable[[], t.Any]):
        name = self._normalize(backend)
        if self.inner.get(name):
            # already loaded, run immediately
            fn()
        else:
            # otherwise schedule for when (if) we load
            cbs = self.cbs.setdefault(name, list())
            cbs.append(fn)

    def get(self, name: BackendName):
        name = self._normalize(name)
        if name == 'numpy':
            return numpy

        if name not in self.inner:
            self._load(name)

        return None if t.TYPE_CHECKING else self.inner[name]

    def try_get(self, name: BackendName):
        name = self._normalize(name)
        if name == 'numpy':
            return numpy

        return self.inner.get(name)

    def __getitem__(self, name: BackendName):
        if (backend := self.get(name)) is not None:
            return backend

        raise ValueError(f"Backend '{name}' is not available")

_BACKEND_LOADER = _BackendLoader()


def get_backend_module(backend: t.Optional[BackendName] = None):
    """Get the module `xp` associated with a compute backend"""
    if backend is None:
        backend = get_default_backend()

    return _BACKEND_LOADER[backend]


def get_backend_scipy(backend: BackendName):
    """Get the scipy module associated with a compute backend"""

    name = _BACKEND_LOADER._normalize(backend)
    # ensure backend is loadable
    _BACKEND_LOADER[backend]

    if not t.TYPE_CHECKING:
        if name == 'torch':
            raise ValueError("`get_backend_scipy` is not supported for the PyTorch backend")
        if name == 'jax':
            return sys.modules['jax.scipy']
        if name == 'cupy':
            return sys.modules['cupyx.scipy']

    import scipy
    return scipy


def get_default_backend() -> BackendName:
    # check for jax or torch GPUs first
    if _BACKEND_LOADER.get('jax') is not None:
        import jax
        try:
            if len(jax.devices('gpu')):
                return 'jax'
        except RuntimeError:
            pass
        try:
            if len(jax.devices('tpu')):
                return 'jax'
        except RuntimeError:
            pass
    if _BACKEND_LOADER.get('torch') is not None:
        import torch
        if torch.get_default_device().type != 'cpu':
            return 'torch'

    for backend in ('jax', 'torch', 'cupy'):
        if _BACKEND_LOADER.get(backend) is not None:
            return backend
    return 'numpy'


def get_devices() -> t.Tuple[t.Tuple[BackendName, Device], ...]:
    devices: t.List[t.Tuple[BackendName, Device]] = []

    if _BACKEND_LOADER.get('jax') is not None:
        from ._jax_kernels import get_devices
        devices.extend(('jax', device) for device in get_devices())
    if _BACKEND_LOADER.get('torch') is not None:
        from ._torch_kernels import get_devices
        devices.extend(('torch', device) for device in get_devices())
    if _BACKEND_LOADER.get('cupy') is not None:
        from ._cuda_kernels import get_devices
        devices.extend(('cupy', device) for device in get_devices())
    devices.append(('numpy', 'cpu'))

    return tuple(devices)


def repr_device(device: Device) -> str:
    s = str(device)

    return {
        'TFRT_CPU_0': 'cpu',
    }.get(s, s)


def to_device(device: t.Union[str, Device], xp: t.Any) -> Device:
    if xp_is_torch(xp):
        from ._torch_kernels import to_device
        return to_device(device)
    if xp_is_cupy(xp):
        from ._cuda_kernels import to_device
        return to_device(device)
    if xp_is_jax(xp):
        from ._jax_kernels import to_device
        return to_device(device)
    if xp is not numpy:
        raise TypeError(f"Expected an array backend, got '{xp}'")
    if device != 'cpu':
        raise ValueError(f"Invalid device '{device}' for backend 'numpy'")
    return device


def get_backend_devices(xp: t.Any) -> t.Tuple[Device, ...]:
    if xp_is_torch(xp):
        from ._torch_kernels import get_devices
        return get_devices()
    if xp_is_cupy(xp):
        from ._cuda_kernels import get_devices
        return get_devices()
    if xp_is_jax(xp):
        from ._jax_kernels import get_devices
        return get_devices()
    if xp is not numpy:
        raise TypeError(f"Expected an array backend, got '{xp}'")

    return ('cpu',)


def set_default_device(device: Device, xp: t.Any):
    if xp_is_torch(xp):
        from ._torch_kernels import set_default_device
        set_default_device(device)
    elif xp_is_cupy(xp):
        from ._cuda_kernels import set_default_device
        set_default_device(device)
    elif xp_is_jax(xp):
        from ._jax_kernels import set_default_device
        set_default_device(device)
    elif xp is not numpy:
        raise TypeError(f"Expected an array backend, got '{xp}'")
    elif device != 'cpu':
        raise ValueError(f"Invalid device '{device}' for backend 'numpy'")


def max_supported_float(
    xp: t.Any,
    device: t.Optional[Device] = None,
) -> t.Union[t.Type[numpy.float32], t.Type[numpy.float64]]:
    if xp_is_torch(xp):
        from ._torch_kernels import max_supported_float
        return max_supported_float(device)
    return numpy.float64


def get_array_module(*arrs: t.Optional[ArrayLike]):
    if (xp := _BACKEND_LOADER.get('jax')) is not None:
        import jax.tree
        if any(
            isinstance(arr, xp.ndarray)
            for arr in chain.from_iterable(map(jax.tree.leaves, arrs))
        ):
            return xp
    if (xp := _BACKEND_LOADER.get('torch')) is not None:
        from torch.utils._pytree import tree_leaves
        if any(
            isinstance(arr, xp._C.TensorBase)  # type: ignore
            for arr in chain.from_iterable(map(tree_leaves, arrs))
        ):
            return xp
    if (xp := _BACKEND_LOADER.get('cupy')) is not None:
        if any(isinstance(arr, xp.ndarray) for arr in arrs):
            return xp
    return numpy


def cast_array_module(xp: t.Any):
    if t.TYPE_CHECKING:
        return numpy
    return xp


def get_scipy_module(*arrs: t.Optional[ArrayLike]):
    # pyright: ignore[reportMissingImports,reportUnusedImport]

    if not t.TYPE_CHECKING:
        if (xp := _BACKEND_LOADER.get('jax')) is not None:
            if any(isinstance(arr, xp.ndarray) for arr in arrs):
                return sys.modules['jax.scipy']
        if (xp := _BACKEND_LOADER.get('torch')) is not None:
            if any(isinstance(arr, xp._C.TensorBase) for arr in arrs):  # type: ignore
                raise ValueError("`get_scipy_module` is not supported for the PyTorch backend")
        if (xp := _BACKEND_LOADER.get('cupy')) is not None:
            f = sys.modules['cupyx.scipy'].get_array_module
            return f(*arrs)

    import scipy
    return scipy


def to_numpy(arr: t.Union[DTypeT, NDArray[DTypeT], float, DTypeT], stream=None) -> NDArray[DTypeT]:
    """
    Convert an array to numpy.
    For cupy backend, this is equivalent to `cupy.asnumpy`.
    """
    if not t.TYPE_CHECKING:
        if is_jax(arr):
            return numpy.array(arr)
        if is_torch(arr):
            return arr.numpy(force=True)
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
        if is_torch(arr):
            return arr.numpy(force=True)
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


def is_cupy(arr: NDArray[numpy.generic]) -> bool:
    if (cupy := _BACKEND_LOADER.get('cupy')) is None:
        return False
    return isinstance(arr, cupy.ndarray)


def is_jax(arr: t.Any) -> bool:
    if (jnp := _BACKEND_LOADER.get('jax')) is None:
        return False
    import jax  # pyright[ignoreMissingImports]

    return any(
        isinstance(arr, jnp.ndarray)
        for arr in jax.tree_util.tree_leaves(arr)
    )


def is_torch(arr: t.Any) -> bool:
    if (torch := t.cast(ModuleType, _BACKEND_LOADER.get('torch'))) is None:
        return False

    return any(
        isinstance(arr, torch._C.TensorBase)
        for arr in torch.utils._pytree.tree_leaves(arr)
    )


def xp_is_cupy(xp: t.Any) -> bool:
    if (cupy := _BACKEND_LOADER.try_get('cupy')) is None:
        return False
    return xp is cupy

def xp_is_jax(xp: t.Any) -> bool:
    return xp is sys.modules.get('jax.numpy')

def xp_is_torch(xp: t.Any) -> bool:
    if (torch := _BACKEND_LOADER.try_get('torch')) is None:
        return False
    return xp is torch


@t.overload
def block_until_ready(arr: DTypeT) -> DTypeT:
    ...

@t.overload
def block_until_ready(arr: NDArray[DTypeT]) -> NDArray[DTypeT]:
    ...

def block_until_ready(arr: t.Any) -> t.Any:
    if hasattr(arr, 'block_until_ready'):  # jax
        return arr.block_until_ready()  # type: ignore

    if is_torch(arr):
        import torch
        device = torch.get_default_device()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    if is_cupy(arr):
        cupy = sys.modules['cupy']
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

        if cupy_fuse and (cupy := _BACKEND_LOADER.get('cupy')):
            self.inner = cupy.fuse()(self.inner)  # type: ignore

        # in jax: self.__call__ -> jax.jit -> jax_f -> f
        # otherwise: self.__call__ -> f
        if _BACKEND_LOADER.get('jax') is not None:
            if t.TYPE_CHECKING:
                import jax
            else:
                jax = sys.modules['jax']

            @functools.wraps(f)
            def jax_f(*args: P.args, **kwargs: P.kwargs) -> T:
                logger.info(f"JIT-compiling kernel '{self.__qualname__}'...")  # type: ignore
                return self.inner(*args, **kwargs)

            self.jax_jit = jax.jit(
                jax_f, static_argnums=static_argnums, static_argnames=static_argnames,
                donate_argnums=donate_argnums, donate_argnames=donate_argnames,
                inline=inline, #compiler_options=compiler_options
            )
        else:
            self.jax_jit = None


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
    if (xp := _BACKEND_LOADER.get('cupy')):
        return xp.fuse(*args, **kwargs)  # type: ignore
    return lambda x: x


def debug_callback(callback: t.Callable[P, None], *args: P.args, **kwargs: P.kwargs):
    try:
        import jax.debug
        return jax.debug.callback(callback, *args, **kwargs)
    except ImportError:
        callback(*args, **kwargs)


def assert_dtype(arr: numpy.ndarray, dtype: t.Type[numpy.generic]):
    if is_torch(arr):
        from ._torch_kernels import to_torch_dtype, to_numpy_dtype

        if arr.dtype != to_torch_dtype(dtype):
            raise TypeError(f"Expected array to be dtype {dtype}, got dtype {to_numpy_dtype(arr.dtype)} instead")
    else:
        if arr.dtype != dtype:
            raise TypeError(f"Expected array to be dtype {dtype}, got dtype {arr.dtype} instead")


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
    if _BACKEND_LOADER.get('torch') is not None:
        from ._torch_kernels import to_numpy_dtype
        dtype = to_numpy_dtype(dtype)  # type: ignore

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
    if _BACKEND_LOADER.get('torch') is not None:
        from ._torch_kernels import to_numpy_dtype
        dtype = to_numpy_dtype(dtype)  # type: ignore

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
def ifft2(a: t.Union[NDArray[numpy.float64], NDArray[numpy.complex128]], *, shift: bool = True) -> NDArray[numpy.complex128]:
    ...

@t.overload
def ifft2(a: t.Union[NDArray[numpy.float32], NDArray[numpy.complex64]], *, shift: bool = True) -> NDArray[numpy.complex64]:
    ...

@t.overload
def ifft2(a: NDArray[numpy.number], *, shift: bool = True) -> NDArray[numpy.complexfloating]:
    ...

@t.overload
def ifft2(a: ArrayLike, *, shift: bool = True) -> NDArray[numpy.complexfloating]:
    ...

def ifft2(a: ArrayLike, *, shift: bool = True) -> NDArray[numpy.complexfloating]:
    """
    Perform an inverse FFT on the last two axes of `a`.

    Follows our convention of centering real space and normalizing intensities (when `shift` is `True`).
    """
    xp = get_array_module(a)
    if shift:
        if xp_is_torch(xp):
            return xp.fft.fftshift(xp.fft.ifft2(a, norm='ortho'), dim=(-2, -1))  # type: ignore
        return xp.fft.fftshift(xp.fft.ifft2(a, norm='ortho'), axes=(-2, -1))
    return xp.fft.ifft2(a, norm='ortho')


@t.overload
def fft2(a: t.Union[NDArray[numpy.float64], NDArray[numpy.complex128]], *, shift: bool = True) -> NDArray[numpy.complex128]:
    ...

@t.overload
def fft2(a: t.Union[NDArray[numpy.float32], NDArray[numpy.complex64]], *, shift: bool = True) -> NDArray[numpy.complex64]:
    ...

@t.overload
def fft2(a: NDArray[numpy.number], *, shift: bool = True) -> NDArray[numpy.complexfloating]:
    ...

@t.overload
def fft2(a: ArrayLike, *, shift: bool = True) -> NDArray[numpy.complexfloating]:
    ...

def fft2(a: ArrayLike, *, shift: bool = True) -> NDArray[numpy.complexfloating]:
    """
    Perform a forward FFT on the last two axes of `a`.

    Follows our convention of centering real space and normalizing intensities (when `shift` is `True`)..
    """
    xp = get_array_module(a)
    if shift:
        if xp_is_torch(xp):
            return xp.fft.fft2(xp.fft.ifftshift(a, dim=(-2, -1)), norm='ortho')  # type: ignore
        return xp.fft.fft2(xp.fft.ifftshift(a, axes=(-2, -1)), norm='ortho')
    return xp.fft.fft2(a, norm='ortho')


@t.overload
def fft2shift(a: NDArray[NumT]) -> NDArray[NumT]:
    ...

@t.overload
def fft2shift(a: ArrayLike) -> NDArray[t.Any]:
    ...

def fft2shift(a: ArrayLike) -> NDArray[t.Any]:
    """
    FFT-shift the last two axes of `a`.
    
    Shifts the zero-frequency component to the center of the image
    (i.e. this is needed to center realspace after an inverse transform).
    """
    xp = get_array_module(a)
    return xp.fft.fftshift(a, axes=(-2, -1))


@t.overload
def ifft2shift(a: NDArray[NumT]) -> NDArray[NumT]:
    ...

@t.overload
def ifft2shift(a: ArrayLike) -> NDArray[t.Any]:
    ...

def ifft2shift(a: ArrayLike) -> NDArray[t.Any]:
    """
    Inverse FFT-shift the last two axes of `a`.

    Shifts the zero-frequency component to the corner of the image
    (i.e. this is needed to corner realspace before a forward transform).
    """
    xp = get_array_module(a)
    return xp.fft.ifftshift(a, axes=(-2, -1))


@t.overload
def dct2(a: NDArray[InexactT]) -> NDArray[InexactT]:
    ...

@t.overload
def dct2(a: ArrayLike) -> NDArray[numpy.inexact]:
    ...

def dct2(a: ArrayLike) -> NDArray[numpy.inexact]:
    """
    Perform a type-II DCT on the last two axes of `a`, orthonormalized.
    """
    return _dct2(a, inverse=False)


@t.overload
def idct2(a: NDArray[InexactT]) -> NDArray[InexactT]:
    ...

@t.overload
def idct2(a: ArrayLike) -> NDArray[numpy.inexact]:
    ...

def idct2(a: ArrayLike) -> NDArray[numpy.inexact]:
    """
    Invert [`dct2`][phaser.utils.num.dct2] (i.e. a type-III DCT, orthonormalized).
    """
    return _dct2(a, inverse=True)


def _dct2(a: ArrayLike, inverse: bool) -> NDArray[numpy.inexact]:
    xp = get_array_module(a)
    a = xp.asarray(a)

    # not every backend supports complex input natively
    if xp.iscomplexobj(a):
        return _dct2(a.real, inverse) + 1.j * _dct2(a.imag, inverse)  # type: ignore

    if xp_is_torch(xp):
        from ._torch_kernels import dct2, idct2

        return t.cast(NDArray[numpy.inexact], (idct2 if inverse else dct2)(a))  # type: ignore

    scipy = get_scipy_module(a)
    f = scipy.fft.idctn if inverse else scipy.fft.dctn
    return f(a, type=2, axes=(-2, -1), norm='ortho')


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

    This is cheaper than `abs(x)**2`
    """
    xp = get_array_module(x)
    x = xp.asarray(x)

    if xp_is_torch(xp):
        if not xp.is_complex(x):  # type: ignore
            return x**2  # type: ignore
    else:
        if not xp.iscomplexobj(x):
            return x**2  # type: ignore

    return x.real**2 + x.imag**2  # type: ignore


_PadMode: t.TypeAlias = t.Literal['constant', 'edge', 'reflect', 'wrap', 'symmetric']


@t.overload
def pad(
    arr: NDArray[DTypeT], pad_width: t.Union[int, t.Tuple[int, int], t.Sequence[t.Tuple[int, int]]], /, *,
    mode: _PadMode = 'constant', cval: float = 0.,
) -> NDArray[DTypeT]:
    ...

@t.overload
def pad(
    arr: ArrayLike, pad_width: t.Union[int, t.Tuple[int, int], t.Sequence[t.Tuple[int, int]]], /, *,
    mode: _PadMode = 'constant', cval: float = 0.,
) -> numpy.ndarray:
    ...

def pad(
    arr: ArrayLike, pad_width: t.Union[int, t.Tuple[int, int], t.Sequence[t.Tuple[int, int]]], /, *,
    mode: _PadMode = 'constant', cval: float = 0.,
) -> numpy.ndarray:
    xp = get_array_module(arr)

    if xp_is_torch(xp):
        from ._torch_kernels import pad
        return pad(arr, pad_width, mode=mode, cval=cval)  # type: ignore

    if mode == 'constant':
        return xp.pad(arr, pad_width, mode=mode, constant_values=cval)
    return xp.pad(arr, pad_width, mode=mode)


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

    if not t.TYPE_CHECKING and (is_torch(x) or is_cupy(x)):
        return ufunc(x[(..., *((None,) * y.ndim))], y[(*((None,) * x.ndim), ...)])

    return ufunc.outer(x, y)


def check_finite(*arrs: NDArray[numpy.inexact], context: t.Optional[str] = None):
    xp = get_array_module(*arrs)

    if not all(xp.all(xp.isfinite(arr)) for arr in arrs):
        if context:
            raise ValueError(f"NaN or inf encountered in {context}")
        raise ValueError("NaN or inf encountered")


@tree_dataclass(frozen=True, init=False, static_fields=('shape',), drop_fields=('extent', 'shape_tup'))
class Sampling:
    shape: NDArray[numpy.int_]
    """Sampling shape (n_y, n_x)"""
    extent: NDArray[numpy.float64]
    """Sampling diameter (b, a)"""
    sampling: NDArray[numpy.float64]
    """Sample spacing (s_y, s_x)"""
    shape_tup: t.Tuple[int, int]
    """Sampling shape (n_y, n_x), as plain ints"""

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

        # Dynamo proxies any numpy array it touches while tracing, so reading
        # `self.shape[i]` as a size int inside a compiled region taints it as
        # data-dependent. Construction always happens outside one.
        object.__setattr__(self, 'shape_tup', (int(self.shape[0]), int(self.shape[1])))

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
        ys = xp2.linspace(corner[0], corner[0] + self.extent[0], self.shape_tup[0], endpoint=False, dtype=dtype)
        xs = xp2.linspace(corner[1], corner[1] + self.extent[1], self.shape_tup[1], endpoint=False, dtype=dtype)
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
            dtype = max_supported_float(xp2)

        # divide by sampling afterward rather than passing it as `fftfreq`'s `d`, so
        # `sampling` can stay a genuine dynamic (e.g. tensor) value: `d` is `.item()`
        # -extracted internally, which isn't traceable under `fullgraph=True`.
        ky: NDArray[numpy.number] = (xp2.fft.fftfreq(self.shape_tup[0], 1.) / self.sampling[0]).astype(dtype)
        kx: NDArray[numpy.number] = (xp2.fft.fftfreq(self.shape_tup[1], 1.) / self.sampling[1]).astype(dtype)

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
        mode: '_InterpBoundaryMode' = 'grid-constant',
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
        mode: '_InterpBoundaryMode' = 'grid-constant',
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
    'get_backend_module', 'get_default_backend',
    'get_devices', 'repr_device', 'to_device', 'get_backend_devices', 'set_default_device',
    'get_array_module', 'cast_array_module', 'get_scipy_module',
    'to_numpy', 'as_numpy', 'as_array',
    'is_cupy', 'is_jax', 'xp_is_cupy', 'xp_is_jax',
    'jit', 'fuse', 'debug_callback',
    'to_complex_dtype', 'to_real_dtype',
    'fft2', 'ifft2', 'fft2shift', 'ifft2shift',
    'dct2', 'idct2',
    'abs2', 'split_array', 'unstack',
    'at', 'ufunc_outer', 'check_finite',
    'Sampling', 'IndexLike',
]
