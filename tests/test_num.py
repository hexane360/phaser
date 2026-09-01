import typing as t

import numpy
import scipy.fft
import scipy.ndimage
from numpy.typing import ArrayLike
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
import pytest

from .utils import with_backends # mock_importerror

from phaser.utils.num import (
    BackendName,
    get_backend_module,
    to_real_dtype, to_complex_dtype,
    fft2, ifft2, abs2,
    dct2, idct2,
    to_numpy, as_array,
    ufunc_outer,
    pad, _PadMode
)


# TODO: this is broken, probably needs to run in a separate process
# the problem is we need to clear the _BackendLoader() cache to get
# proper behavior, but torch can only be imported once
"""
@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_get_array_module(backend: BackendName, monkeypatch: pytest.MonkeyPatch):
    expected = get_backend_module(backend)

    mocked_imports = {
        # on numpy, pretend cupy and jax don't exist
        'numpy': {'cupy', 'jax', 'torch'},
        'jax': {},
        'cupy': {},
        'torch': {},
    }[backend]

    # re-load backend loader so the effect of mocking takes place
    monkeypatch.setattr(phaser.utils.num, '_BACKEND_LOADER', _BackendLoader())

    with mock_importerror(mocked_imports):
        assert get_array_module() is numpy

        assert get_array_module(
            numpy.array([1., 2., 3.]),
            expected.asarray([1, 2, 3]),
            None,
            numpy.array([1., 2., 3.]),
        ) is expected


@with_backends('numpy', 'jax', 'cupy')
def test_get_scipy_module(backend: BackendName, monkeypatch: pytest.MonkeyPatch):
    import scipy

    xp = get_backend_module(backend)
    expected = get_backend_scipy(backend)

    mocked_imports = {
        # on cpu, pretend cupyx doesn't exist
        'numpy': {'cupyx'},
        'jax': {},
        'cupy': {},
    }[backend]

    # re-load backend loader so the effect of mocking takes place
    # monkeypatch.setattr(phaser.utils.num, '_BACKEND_LOADER', _BackendLoader())

    with mock_importerror(mocked_imports):
        assert get_scipy_module() is scipy

        assert get_scipy_module(
            numpy.array([1., 2., 3.]),
            xp.asarray([1, 2, 3]),
            None,
            numpy.array([1., 2., 3.]),
        ) is expected
"""


@pytest.mark.parametrize(('input', 'expected'), [
    (numpy.complex64, numpy.float32),
    (numpy.complex128, numpy.float64),
    (numpy.complexfloating, numpy.floating),
    ('complex128', numpy.float64),
    (float, numpy.float64),
    (complex, numpy.float64),
])
def test_to_real_dtype(input, expected):
    assert to_real_dtype(input) is expected
    # test idempotence
    assert to_real_dtype(expected) is expected


@pytest.mark.parametrize(('input', 'expected'), [
    (numpy.float32, numpy.complex64),
    (numpy.float64, numpy.complex128),
    (numpy.floating, numpy.complexfloating),
    ('float32', numpy.complex64),
    (float, numpy.complex128),
    (complex, numpy.complex128),
])
def test_to_complex_dtype(input, expected):
    assert to_complex_dtype(input) is expected
    # test idempotence
    assert to_complex_dtype(expected) is expected


def test_to_real_dtype_invalid():
    with pytest.raises(TypeError, match="Non-floating point datatype"):
        to_real_dtype(numpy.int_)


def test_to_complex_dtype_invalid():
    with pytest.raises(TypeError, match="Non-floating point datatype"):
        to_complex_dtype(numpy.int_)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_fft2(backend: BackendName):
    xp = get_backend_module(backend)

    # point input, f = 5 delta(x) delta(y)
    a = xp.asarray(numpy.pad([[5.]], (2, 2)).astype(numpy.float32))

    # even input, so output is real
    # delta function input, so output is constant
    # normalized so intensity in = intensity out
    assert_array_almost_equal(
        to_numpy(fft2(a)),
        numpy.full((5, 5), 1., dtype=numpy.complex64)
    )

    # plane input, f = 1 + 1i
    a = xp.full((5, 5), 1+1.j, dtype=numpy.complex64)

    # constant input, so output is delta function at k=0
    # normalized so intensity in = intensity out
    # zero frequency is cornered
    assert_array_almost_equal(
        to_numpy(fft2(a)),
        numpy.pad([[5.+5.j]], (0, 4)).astype(numpy.complex64)
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_ifft2(backend: BackendName):
    xp = get_backend_module(backend)

    # point input, F = delta(k_x) delta(k_y)
    a = xp.asarray(numpy.pad([[5.]], (0, 4)).astype(numpy.float32))

    # even input, so output is real
    # delta function input, so output is constant
    # normalized so intensity in = intensity out
    assert_array_almost_equal(
        to_numpy(ifft2(a)),
        numpy.full((5, 5), 1., dtype=numpy.complex64),
        decimal=5
    )

    # plane input, F = 1 + 1i
    a = xp.full((5, 5), 1+1.j, dtype=numpy.complex64)

    # constant input, so output is delta function at k=0
    # normalized so intensity in = intensity out
    # zero position is centered
    assert_array_almost_equal(
        to_numpy(ifft2(a)),
        numpy.pad([[5.+5.j]], ((2, 2), (2, 2))).astype(numpy.complex64),
        decimal=5
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_abs2(backend: BackendName):
    xp = get_backend_module(backend)

    if backend == 'numpy':
        assert_array_almost_equal(abs2([1.+1.j, 1.-1.j]), numpy.array([2., 2.]))

    assert_array_almost_equal(
        to_numpy(abs2(xp.asarray([1.+1.j, 1.-1.j]))),
        numpy.array([2., 2.]),
    )

    assert_array_almost_equal(
        to_numpy(abs2(xp.asarray([1., -2., 5.], dtype=numpy.float32))),
        numpy.array([1, 4., 25.], dtype=numpy.float32),
        decimal=5  # this is pretty poor performance
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_to_numpy(backend: BackendName):
    xp = get_backend_module(backend)

    arr = xp.asarray([1., 2., 3., 4.])

    assert_array_almost_equal(
        to_numpy(arr),
        numpy.array([1., 2., 3., 4.])
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_to_array(backend: BackendName):
    xp = get_backend_module(backend)

    arr = xp.asarray([1., 2., 3., 4.])
    assert as_array(arr) is arr

    arr = as_array([1., 2., 3., 4.])
    assert isinstance(arr, numpy.ndarray)
    assert_array_almost_equal(
        arr,
        numpy.array([1., 2., 3., 4.])
    )


@with_backends('numpy', 'jax', 'cupy')
def test_ufunc_outer(backend: BackendName):
    xp = get_backend_module(backend)

    xs = numpy.arange(12).reshape(4, 3)
    ys = numpy.arange(30).reshape(5, 6)

    expected = numpy.multiply.outer(xs, ys)
    actual = to_numpy(ufunc_outer(xp.multiply, xp.array(xs), xp.array(ys)))
    assert_array_equal(expected, actual)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize(('mode', 'pad_width', 'expected'), [
    ('constant', 0,       [1, 2, 3, 4, 5]),
    ('constant', 2,       [3, 3, 1, 2, 3, 4, 5, 3, 3]),
    ('constant', (2, 1),  [3, 3, 1, 2, 3, 4, 5, 3]),
    ('constant', (0, 3),  [1, 2, 3, 4, 5, 3, 3, 3]),
    ('constant', (3, 0),  [3, 3, 3, 1, 2, 3, 4, 5]),
    ('edge', 2,           [1, 1, 1, 2, 3, 4, 5, 5, 5]),
    ('edge', (2, 1),      [1, 1, 1, 2, 3, 4, 5, 5]),
    ('edge', (0, 3),      [1, 2, 3, 4, 5, 5, 5, 5]),
    ('edge', (3, 0),      [1, 1, 1, 1, 2, 3, 4, 5]),
    ('edge', 6,           [1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5]),
    ('reflect', 2,        [3, 2, 1, 2, 3, 4, 5, 4, 3]),
    ('reflect', (2, 1),   [3, 2, 1, 2, 3, 4, 5, 4]),
    ('reflect', (1, 3),   [2, 1, 2, 3, 4, 5, 4, 3, 2]),
    ('reflect', (0, 3),   [1, 2, 3, 4, 5, 4, 3, 2]),
    ('reflect', 6,        [3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]),
    ('symmetric', 2,      [2, 1, 1, 2, 3, 4, 5, 5, 4]),
    ('symmetric', (2, 1), [2, 1, 1, 2, 3, 4, 5, 5]),
    ('symmetric', (1, 3), [1, 1, 2, 3, 4, 5, 5, 4, 3]),
    ('symmetric', (0, 3), [1, 2, 3, 4, 5, 5, 4, 3]),
    ('symmetric', 6,      [5, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 1]),
    ('wrap', 2,           [4, 5, 1, 2, 3, 4, 5, 1, 2]),
    ('wrap', (2, 1),      [4, 5, 1, 2, 3, 4, 5, 1]),
    ('wrap', (1, 3),      [5, 1, 2, 3, 4, 5, 1, 2, 3]),
    ('wrap', (0, 3),      [1, 2, 3, 4, 5, 1, 2, 3]),
    ('wrap', 6,           [5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1]),
])
def test_pad_1d(
    mode: _PadMode,
    pad_width: t.Union[int, t.Tuple[int, int], t.Tuple[t.Tuple[int, int]]],
    expected: ArrayLike,
    backend: BackendName,
):
    xp = get_backend_module(backend)
    in_arr = xp.array([1, 2, 3, 4, 5])

    assert_array_equal(
        to_numpy(pad(in_arr, pad_width, mode=mode, cval=3)),
        numpy.array(expected),
    )


@with_backends('jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ('constant', 'edge', 'reflect', 'symmetric', 'wrap'))
@pytest.mark.parametrize(('shape', 'pad_width'), [
    ((4, 6), 2),
    ((4, 6), (2, 3)),
    ((4, 6), ((1, 2), (3, 1))),
    ((4, 6), ((0, 3), (2, 0))),
    ((3, 3), 4),
    ((3, 5), ((4, 4), (6, 6))),
    ((2, 4, 6), 2),
    ((2, 4, 6), ((1, 1), (2, 3), (0, 2))),
    ((2, 3, 4), 5),
    ((2, 3, 4, 5), 2),
    ((2, 3, 4, 5), ((1, 2), (0, 3), (2, 1), (1, 1))),
    ((5,), 3),
    ((1, 4), 2),
    ((1, 5), ((0, 0), (2, 2))),
])
def test_pad_nd(
    mode: _PadMode,
    shape: t.Sequence[int],
    pad_width: t.Union[int, t.Tuple[int, int], t.Tuple[t.Tuple[int, int]]],
    backend: BackendName,
):
    xp = get_backend_module(backend)

    kwargs = {'constant_values': 3} if mode == 'constant' else {}

    rng = numpy.random.default_rng()
    in_arr = rng.integers(1024, size=shape)

    assert_array_equal(
        to_numpy(pad(xp.array(in_arr), pad_width, mode=mode, cval=3)),
        numpy.pad(in_arr, pad_width, mode=mode, **kwargs)  # type: ignore
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('shape', [(4, 4), (5, 7), (3, 8, 8), (1, 1), (2, 1, 6, 5)])
@pytest.mark.parametrize('dtype', [
    numpy.float32, numpy.float64, numpy.complex64, numpy.complex128,
])
@pytest.mark.parametrize(('f', 'ref'), [(dct2, 'dctn'), (idct2, 'idctn')])
def test_dct2(shape, dtype, f, ref, backend: BackendName):
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=shape).astype(dtype)
    if numpy.iscomplexobj(arr):
        arr = arr + 1.j * rng.normal(size=shape).astype(dtype)

    xp = get_backend_module(backend)

    expected = getattr(scipy.fft, ref)(arr, type=2, axes=(-2, -1), norm='ortho')
    actual = to_numpy(f(xp.array(arr)))
    assert actual.dtype == arr.dtype

    tol = 1e-5 if arr.dtype.itemsize <= 8 else 1e-12
    assert_allclose(actual, expected, rtol=tol, atol=tol)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('shape', [(4, 4), (5, 7), (3, 8, 8)])
def test_dct2_roundtrip(shape, backend: BackendName):
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=shape) + 1.j * rng.normal(size=shape)

    xp = get_backend_module(backend)
    assert_allclose(to_numpy(idct2(dct2(xp.array(arr)))), arr, rtol=1e-12, atol=1e-12)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_dct2_symmetric_convolution(backend: BackendName):
    """`dct2` diagonalizes convolution under half-sample symmetric boundary conditions."""
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=(9, 11))
    weights = numpy.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.

    # transfer function of `weights`, sampled on the dct frequency grid
    ks = tuple(numpy.arange(n) / (2. * n) for n in arr.shape)
    ms = tuple(numpy.arange(n) - n // 2 for n in weights.shape)
    transfer = numpy.einsum(
        'mn,ym,xn->yx', weights,
        *(numpy.cos(2. * numpy.pi * numpy.outer(k, m)) for (k, m) in zip(ks, ms))
    )

    xp = get_backend_module(backend)
    actual = to_numpy(idct2(dct2(xp.array(arr)) * xp.array(transfer)))
    expected = scipy.ndimage.convolve(arr, weights, mode='reflect')

    assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
