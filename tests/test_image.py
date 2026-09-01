import typing as t

import numpy
import pytest
import scipy.ndimage as osp
from numpy.testing import assert_allclose, assert_array_almost_equal
from numpy.typing import ArrayLike, NDArray

from phaser.utils.image import (
    Filter,
    GaussianFilter,
    PreparedPSF,
    ProductFilter,
    ProductSeparableFilter,
    PsfFilter,
    SeparablePsfFilter,
    SquarePixelFilter,
    TransferFilter,
    _FilterBoundaryMode,
    _InterpBoundaryMode,
    affine_transform,
    convolve1d,
    convolve2d,
    convolve2d_recip,
    convolve2d_recip_wrap,
    convolve2d_separable,
    prepare_convolve2d,
    prepare_convolve2d_recip,
)
from phaser.utils.num import BackendName, Sampling, get_backend_module, to_numpy

from .utils import check_array_equals_file, with_backends


@pytest.fixture
def checkerboard() -> t.Tuple[NDArray[numpy.float32], Sampling]:
    yy, xx = numpy.indices((16, 16))
    checker = ((yy % 2) ^ (xx % 2)).astype(numpy.float32)

    return (checker, Sampling(checker.shape, sampling=(1.0, 1.0)))


def _unit_sampling(shape: t.Tuple[int, int]) -> Sampling:
    return Sampling(shape, sampling=(1., 1.))


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize(('mode', 'order', 'expected'), [
    ('grid-constant', 0, [ 1.0,  1.0,  1.0,  1.0, -2.0, -2.0, -2.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  2.0,  2.0,  2.0,  1.0,  1.0,  1.0,  1.0]),
    ('nearest'      , 0, [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0]),
    ('mirror'       , 0, [ 0.0,  0.0, -1.0, -1.0, -2.0, -2.0, -2.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  2.0,  2.0,  2.0,  1.0,  1.0,  0.0,  0.0]),
    ('reflect'      , 0, [-1.0, -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  2.0,  2.0,  2.0,  2.0,  2.0,  1.0,  1.0]),
    ('grid-wrap'    , 0, [ 1.0,  1.0,  2.0,  2.0, -2.0, -2.0, -2.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  2.0,  2.0,  2.0, -2.0, -2.0, -1.0, -1.0]),
    ('grid-constant', 1, [ 1.0,  1.0,  1.0,  0.4, -0.8, -2.0, -1.6, -1.2, -0.8, -0.4, -0.0,  0.4,  0.8,  1.2,  1.6,  2.0,  1.6,  1.2,  1.0,  1.0,  1.0]),
    ('nearest'      , 1, [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.6, -1.2, -0.8, -0.4, -0.0,  0.4,  0.8,  1.2,  1.6,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0]),
    ('mirror'       , 1, [ 0.0, -0.4, -0.8, -1.2, -1.6, -2.0, -1.6, -1.2, -0.8, -0.4, -0.0,  0.4,  0.8,  1.2,  1.6,  2.0,  1.6,  1.2,  0.8,  0.4,  0.0]),
    ('reflect'      , 1, [-1.0, -1.4, -1.8, -2.0, -2.0, -2.0, -1.6, -1.2, -0.8, -0.4, -0.0,  0.4,  0.8,  1.2,  1.6,  2.0,  2.0,  2.0,  1.8,  1.4,  1.0]),
    ('grid-wrap'    , 1, [ 1.0,  1.4,  1.8,  1.2, -0.4, -2.0, -1.6, -1.2, -0.8, -0.4, -0.0,  0.4,  0.8,  1.2,  1.6,  2.0,  0.4, -1.2, -1.8, -1.4, -1.0]),
])
def test_affine_transform_1d(mode: _InterpBoundaryMode, order: int, expected: ArrayLike, backend: BackendName):
    xp = get_backend_module(backend)

    in_ys = numpy.array([-2., -1., 0., 1., 2.])

    # interpolates at coords `numpy.linspace(-2., 6., 21, endpoint=True)`
    assert_array_almost_equal(numpy.array(expected), to_numpy(affine_transform(
        xp.asarray(in_ys), [0.4], -2.0,
        mode=mode, order=order, cval=1.0, output_shape=(21,)
    )), decimal=6)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize(('name', 'order', 'rotation', 'sampling'), [
    ('identity',   1,  0.0, Sampling((16, 16), sampling=(1.0, 1.0))),
    ('pad',        0,  0.0, Sampling((32, 32), sampling=(1.0, 1.0))),
    ('upsample',   0,  0.0, Sampling((250, 250), extent=(20.0, 20.0))),
    ('upsample',   1,  0.0, Sampling((250, 250), extent=(20.0, 20.0))),
    ('upsample',   0, 30.0, Sampling((250, 250), extent=(20.0, 20.0))),
    ('upsample',   1, 30.0, Sampling((250, 250), extent=(20.0, 20.0))),
    ('downsample', 0,  0.0, Sampling((16, 16), sampling=(2.0, 2.0))),
    ('downsample', 1,  0.0, Sampling((16, 16), sampling=(2.0, 2.0))),
])
@check_array_equals_file('resample_{name}_order{order}_rot{rotation:03.1f}.tiff', out_name='resample_{name}_order{order}_rot{rotation:03.1f}_{backend}.tiff')
def test_resample(
    backend: BackendName,
    checkerboard: t.Tuple[NDArray[numpy.float32], Sampling],
    name: str,
    order: int,
    rotation: float,
    sampling: Sampling,
):
    if (name, order, rotation) == ('upsample', 0, 0.0) and backend in ('jax', 'torch'):
        # TODO: check intermediate dtypes here?
        pytest.xfail("Rounding bug?")

    xp = get_backend_module(backend)

    (checker, old_samp) = checkerboard

    return to_numpy(old_samp.resample(xp.array(checker), sampling, rotation=rotation, order=order))


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize(('arr', 'weights', 'axis'), [
    ([1, 2, 3, 4, 5], [1, 2], 0),
    ([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0], 0),
    ([[[1, 2], [3, 4]], [[2, 3], [4, 5]], [[3, 4], [5, 6]]], [1, 2, 3], 0),
    ([[[1, 2], [3, 4]], [[2, 3], [4, 5]], [[3, 4], [5, 6]]], [1, 2, 3], -1),
    ([1+1.j, 2+2.j, 3+3.j], [1-1.j, 2-1.j], 0),
    # casting of weights
    ([1+1.j, 2+2.j, 3+3.j], [1.0, 2.0], 0),
    ([[[1, 2], [3, 4]], [[2, 3], [4, 5]], [[3, 4], [5, 6]]], [1, 2, 3], 1),
    ([1, 2, 3, 4, 5], [2], 0),
    # kernel longer than array
    ([1, 2, 3], [1, 2, 3, 4, 5, 6, 7], 0),
    # length-1 along conv axis
    ([[[1, 2], [3, 4]]], [1, 2, 3], 0),
])
@pytest.mark.parametrize(('mode', 'cval'), [
    ('constant', 1.0), ('nearest', 0.0), ('mirror', 0.0),
    ('reflect', 0.0), ('wrap', 0.0),
])
def test_convolve1d(
    arr, weights, axis, mode, cval,
    backend: BackendName,
):
    arr = numpy.asarray(arr)
    weights = numpy.asarray(weights)

    xp = get_backend_module(backend)

    expected = osp.convolve1d(
        arr, weights, axis=axis, mode=mode, cval=cval
    )
    actual = to_numpy(convolve1d(
        xp.array(arr), xp.array(weights), axis=axis, mode=mode, cval=cval
    ))
    assert actual.dtype == expected.dtype

    assert_array_almost_equal(actual, expected, decimal=6)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('weights', [
    [[1., 2., 3., 4., 5.]],
    [[1.], [2.], [3.], [4.], [5.]],
    [[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]],
])
@pytest.mark.parametrize(('mode', 'cval'), [
    ('constant', 1.0), ('nearest', 0.0), ('mirror', 0.0),
    ('reflect', 0.0), ('wrap', 0.0),
])
@pytest.mark.parametrize('dtype', [
    numpy.float32, numpy.float64, numpy.complex64, numpy.complex128,
    numpy.int32, numpy.int64,
])
@pytest.mark.parametrize('leading_shape', [(), (3,), (1,1,1,1,1,1)])
def test_convolve2d(
    weights: ArrayLike, mode, cval, dtype, leading_shape,
    backend: BackendName,
):
    arr = numpy.random.normal(size=leading_shape + (256, 256)).astype(dtype)
    weights = numpy.asarray(weights)

    if numpy.iscomplexobj(arr):
        # test complex weights as well
        weights = weights * 1.j

    xp = get_backend_module(backend)

    expected = osp.convolve(
        arr, weights, mode=mode, cval=cval, axes=(-2, -1)
    )

    actual = to_numpy(convolve2d(
        xp.array(arr), xp.array(weights), mode=mode, cval=cval
    ))
    assert actual.dtype == arr.dtype

    assert_array_almost_equal(
        actual, expected, decimal=4 if dtype in (numpy.float32, numpy.complex64) else 6
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize(('y_weights', 'x_weights'), [
    ([1., 2., 3., 4., 5.], None),
    # asymmetric, and different lengths along each axis
    ([1., 2.], [1., 0., -1.]),
    ([1.], [1., 2., 3.]),
    # kernel longer than the array
    ([1., 2., 3., 4., 5., 6., 7., 8., 9.], None),
])
@pytest.mark.parametrize(('mode', 'cval'), [
    ('constant', 1.0), ('nearest', 0.0), ('mirror', 0.0),
    ('reflect', 0.0), ('wrap', 0.0),
])
@pytest.mark.parametrize('dtype', [
    numpy.float32, numpy.float64, numpy.complex64, numpy.complex128,
])
@pytest.mark.parametrize('leading_shape', [(), (3,), (1, 1, 1, 1, 1, 1)])
def test_convolve2d_separable(
    y_weights: ArrayLike, x_weights: t.Optional[ArrayLike], mode, cval, dtype, leading_shape,
    backend: BackendName,
):
    arr = numpy.random.normal(size=leading_shape + (8, 6)).astype(dtype)
    y_weights = numpy.asarray(y_weights)
    x_weights = y_weights if x_weights is None else numpy.asarray(x_weights)

    if numpy.iscomplexobj(arr):
        # test complex weights as well
        (y_weights, x_weights) = (y_weights * 1.j, x_weights * 1.j)

    xp = get_backend_module(backend)

    # `y_weights` runs along the second-to-last axis, `x_weights` along the last.
    # the second pass fills against an already-filtered boundary
    expected = osp.convolve1d(
        osp.convolve1d(arr, y_weights, axis=-2, mode=mode, cval=cval),
        x_weights, axis=-1, mode=mode, cval=cval * numpy.sum(y_weights),
    )
    actual = to_numpy(convolve2d_separable(
        xp.array(arr), xp.array(y_weights), xp.array(x_weights), mode=mode, cval=cval
    ))
    assert actual.dtype == arr.dtype

    # relative tolerance: the longest kernels amplify by ~2000x
    tol = 1e-4 if dtype in (numpy.float32, numpy.complex64) else 1e-10
    assert_allclose(actual, expected, rtol=tol, atol=tol * float(numpy.max(numpy.abs(expected))))


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize(('mode', 'cval'), [
    ('nearest', 0.0), ('mirror', 0.0), ('reflect', 0.0), ('wrap', 0.0),
    # a nonzero fill value is what distinguishes the two in the corners
    ('constant', 0.0), ('constant', 1.0), ('constant', -3.5), ('grid-constant', 2.5),
])
@pytest.mark.parametrize(('y_weights', 'x_weights'), [
    ([1., 2., 3.], [1., 0., -1., 2.]),
    # dc gain of zero, and kernels longer than the array
    ([1., -1.], [1., 2., 1.]),
    (list(numpy.arange(1., 10.)), list(numpy.arange(1., 8.))),
])
def test_convolve2d_separable_matches_2d(
    mode, cval, y_weights, x_weights, backend: BackendName
):
    """A separable convolution is a 2D convolution by the outer product of its filters."""
    arr = numpy.random.normal(size=(3, 8, 6))
    (y_weights, x_weights) = (numpy.asarray(y_weights), numpy.asarray(x_weights))

    xp = get_backend_module(backend)

    expected = to_numpy(convolve2d(
        xp.array(arr), xp.array(numpy.outer(y_weights, x_weights)), mode=mode, cval=cval
    ))
    actual = to_numpy(convolve2d_separable(
        xp.array(arr), xp.array(y_weights), xp.array(x_weights), mode=mode, cval=cval
    ))

    assert_array_almost_equal(actual, expected, decimal=6)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('weights', [[1., 2., 3.], [1., 2.], [1.]])
def test_convolve2d_separable_defaults_x_to_y(weights: ArrayLike, backend: BackendName):
    arr = numpy.random.normal(size=(8, 6))
    xp = get_backend_module(backend)

    assert_array_almost_equal(
        to_numpy(convolve2d_separable(xp.array(arr), xp.array(weights))),
        to_numpy(convolve2d_separable(xp.array(arr), xp.array(weights), xp.array(weights))),
        decimal=6,
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_convolve2d_separable_validates_weights(backend: BackendName):
    xp = get_backend_module(backend)
    arr = xp.array(numpy.zeros((4, 4)))

    with pytest.raises(ValueError, match="Expected 'y_weights' to be 1D"):
        convolve2d_separable(arr, xp.array([[1., 2.]]))

    with pytest.raises(ValueError, match="Expected 'x_weights' to be 1D"):
        convolve2d_separable(arr, xp.array([1., 2.]), xp.array([[1., 2.]]))


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ['reflect', 'nearest', 'mirror', 'grid-wrap', 'grid-constant'])
@pytest.mark.parametrize('kernel', [
    numpy.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.,
    # asymmetric, to pin down convolution (rather than correlation)
    numpy.array([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
])
def test_convolve2d_filter_overload(mode: _InterpBoundaryMode, kernel: NDArray[numpy.floating], backend: BackendName):
    """`convolve2d`'s `Filter` overload matches convolving with the filter's own kernel directly."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=(9, 11))

    expected = to_numpy(convolve2d(xp.array(arr), xp.array(kernel), mode=mode))
    actual = to_numpy(convolve2d(xp.array(arr), PsfFilter(kernel), mode=mode))

    assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ['reflect', 'nearest', 'grid-wrap'])
def test_convolve2d_complex_psf(mode: _InterpBoundaryMode, backend: BackendName):
    """
    Real-space convolution supports a complex `arr` (with either a real or complex
    PSF), but a complex PSF can't filter a real `arr`.
    """
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(42)
    real_kernel = _KERNELS[1]
    complex_kernel = real_kernel * numpy.exp(1.j * rng.normal(size=real_kernel.shape))
    real_arr = rng.normal(size=(9, 11))
    complex_arr = real_arr + 1.j * rng.normal(size=(9, 11))

    for kernel in (real_kernel, complex_kernel):
        expected = to_numpy(convolve2d(xp.array(complex_arr), xp.array(kernel), mode=mode))
        actual = to_numpy(convolve2d(xp.array(complex_arr), PsfFilter(kernel), mode=mode))
        assert actual.dtype == numpy.complex128
        assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    for weights in (xp.array(real_kernel[0]), xp.array(complex_kernel[0])):
        assert to_numpy(convolve1d(xp.array(complex_arr), weights)).dtype == numpy.complex128

    with pytest.raises(ValueError, match="Expected a real point spread function"):
        convolve2d(xp.array(real_arr), xp.array(complex_kernel), mode=mode)
    with pytest.raises(ValueError, match="Expected a real point spread function"):
        convolve1d(xp.array(real_arr), xp.array(complex_kernel[0]))
    with pytest.raises(ValueError, match="Expected a real point spread function"):
        convolve2d(xp.array(real_arr), PsfFilter(complex_kernel), mode=mode)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ['reflect', 'nearest', 'grid-wrap'])
def test_prepare_convolve2d(mode: _InterpBoundaryMode, backend: BackendName):
    """A `PreparedPSF` applies exactly what `convolve2d`'s `Filter` overload does."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=(3, 9, 11))
    samp = _unit_sampling((9, 11))

    filt = PsfFilter(_KERNELS[1])
    prepared = prepare_convolve2d(filt, samp, mode=mode, xp=xp)
    assert isinstance(prepared, PreparedPSF)

    assert_allclose(
        to_numpy(prepared(xp.array(arr))),
        to_numpy(convolve2d(xp.array(arr), filt, mode=mode)),
        rtol=1e-10, atol=1e-10,
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_prepare_convolve2d_separable(backend: BackendName):
    """`prepare_convolve2d` on a `SeparableFilter` matches `convolve2d_separable` on its own kernels."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=(9, 11))
    samp = _unit_sampling((9, 11))

    filt = GaussianFilter(1.5)
    prepared = prepare_convolve2d(filt, samp, xp=xp)
    y_kernel, x_kernel = filt.psf_separable(samp, xp=xp)

    assert_allclose(
        to_numpy(prepared(xp.array(arr))),
        to_numpy(convolve2d_separable(xp.array(arr), y_kernel, x_kernel)),
        rtol=1e-10, atol=1e-10,
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ['reflect', 'nearest', 'grid-wrap'])
def test_prepared_psf_adjoint_symmetric(mode: _InterpBoundaryMode, backend: BackendName):
    """`PreparedPSF.adjoint()` is a no-op for a symmetric filter."""
    xp = get_backend_module(backend)
    samp = _unit_sampling((9, 11))

    prepared = prepare_convolve2d(GaussianFilter(1.5), samp, mode=mode, xp=xp)
    assert prepared.adjoint() is prepared


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_prepared_psf_adjoint_inner_product(backend: BackendName):
    """For an asymmetric filter under `mode='grid-wrap'`, `PreparedPSF.adjoint()`
    satisfies `<Mx, y> == <x, M^T y>`."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(1234)
    samp = _unit_sampling((9, 11))
    x = rng.normal(size=(9, 11))
    y = rng.normal(size=(9, 11))

    filt = PsfFilter(_KERNELS[2], symmetric=False)
    prepared = prepare_convolve2d(filt, samp, mode='grid-wrap', xp=xp)
    assert not prepared.symmetric

    lhs = numpy.sum(to_numpy(prepared(xp.array(x))) * y)
    rhs = numpy.sum(x * to_numpy(prepared.adjoint()(xp.array(y))))
    assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-8)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ['reflect', 'nearest'])
def test_prepared_psf_adjoint_not_implemented(mode: _InterpBoundaryMode, backend: BackendName):
    """The transpose of a padded (non-periodic) boundary mode isn't implemented."""
    xp = get_backend_module(backend)
    samp = _unit_sampling((9, 11))
    filt = PsfFilter(_KERNELS[2], symmetric=False)
    prepared = prepare_convolve2d(filt, samp, mode=mode, xp=xp)

    with pytest.raises(NotImplementedError):
        prepared.adjoint()


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_prepared_otf_adjoint_symmetric(backend: BackendName):
    """`PreparedOTF.adjoint()` is a no-op for a symmetric filter."""
    xp = get_backend_module(backend)
    samp = _unit_sampling((9, 11))

    prepared = prepare_convolve2d_recip(GaussianFilter(1.5), samp, mode='grid-wrap', xp=xp)
    assert prepared.adjoint() is prepared


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_prepared_otf_adjoint_inner_product(backend: BackendName):
    """For an asymmetric filter, `PreparedOTF.adjoint()` satisfies `<Mx, y> == <x, M^T y>`."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(1234)
    samp = _unit_sampling((9, 11))
    x = rng.normal(size=(9, 11))
    y = rng.normal(size=(9, 11))

    filt = PsfFilter(_KERNELS[2], symmetric=False)
    prepared = prepare_convolve2d_recip(filt, samp, mode='grid-wrap', xp=xp)
    assert not prepared.symmetric

    lhs = numpy.sum(to_numpy(prepared(xp.array(x))) * y)
    rhs = numpy.sum(x * to_numpy(prepared.adjoint()(xp.array(y))))
    assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-8)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_prepared_otf_adjoint_reflect_inner_product(backend: BackendName):
    """For an asymmetric filter under `mode='reflect'`, `PreparedOTF.adjoint()`
    satisfies `<Mx, y> == <x, M^T y>`."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(1234)
    samp = _unit_sampling((9, 11))
    x = rng.normal(size=(9, 11))
    y = rng.normal(size=(9, 11))

    filt = PsfFilter(_KERNELS[2], symmetric=False)
    prepared = prepare_convolve2d_recip(filt, samp, mode='reflect', xp=xp)
    assert not prepared.symmetric

    lhs = numpy.sum(to_numpy(prepared(xp.array(x))) * y)
    rhs = numpy.sum(x * to_numpy(prepared.adjoint()(xp.array(y))))
    assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-8)

    # adjoint of the adjoint recovers the original filter
    lhs2 = numpy.sum(to_numpy(prepared.adjoint()(xp.array(x))) * y)
    rhs2 = numpy.sum(x * to_numpy(prepared.adjoint().adjoint()(xp.array(y))))
    assert_allclose(lhs2, rhs2, rtol=1e-8, atol=1e-8)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_prepare_convolve2d_cval(backend: BackendName):
    """`PreparedPSF.cval` is used as the fill value under `'constant'`/`'grid-constant'` modes."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=(9, 11))
    samp = _unit_sampling((9, 11))

    filt = PsfFilter(_KERNELS[1])
    prepared = prepare_convolve2d(filt, samp, mode='grid-constant', cval=2.5, xp=xp)

    assert_allclose(
        to_numpy(prepared(xp.array(arr))),
        to_numpy(convolve2d(xp.array(arr), filt, mode='grid-constant', cval=2.5)),
        rtol=1e-10, atol=1e-10,
    )


_KERNELS: t.Sequence[NDArray[numpy.floating]] = [
    numpy.array([[1.]]),
    numpy.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.,
    # asymmetric, to pin down convolution (rather than correlation)
    numpy.array([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
    numpy.arange(15.).reshape((5, 3)),
]


def _recip_convolve(
    arr: NDArray[numpy.inexact], filt: Filter, mode: _FilterBoundaryMode,
    samp: t.Optional[Sampling] = None,
) -> NDArray[numpy.inexact]:
    return convolve2d_recip(arr, filt, samp if samp is not None else _unit_sampling(arr.shape[-2:]), mode=mode)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ['reflect', 'grid-wrap'])
@pytest.mark.parametrize('kernel', _KERNELS)
@pytest.mark.parametrize('shape', [(8, 8), (9, 11), (3, 6, 5)])
def test_convolve2d_recip_matches_spatial(
    mode: _FilterBoundaryMode, kernel: NDArray[numpy.floating],
    shape: t.Tuple[int, ...], backend: BackendName
):
    """Reciprocal space filtering by a `PsfFilter` matches spatial convolution by its kernel."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=shape)

    actual = to_numpy(_recip_convolve(xp.array(arr), PsfFilter(kernel), mode, _unit_sampling(shape[-2:])))
    expected = osp.convolve(arr, kernel[None] if len(shape) > 2 else kernel, mode=mode)

    assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ['reflect', 'grid-wrap'])
def test_convolve2d_recip_separable(mode: _FilterBoundaryMode, backend: BackendName):
    """`SeparablePsfFilter` matches the `PsfFilter` of its outer product."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=(9, 11))
    y_kernel, x_kernel = numpy.array([1., 4., 6., 4., 1.]) / 16., numpy.array([1., 2., 1.]) / 4.

    separable = to_numpy(_recip_convolve(xp.array(arr), SeparablePsfFilter(y_kernel, x_kernel), mode))
    outer = to_numpy(_recip_convolve(xp.array(arr), PsfFilter(y_kernel[:, None] * x_kernel[None, :]), mode))

    assert_allclose(separable, outer, rtol=1e-10, atol=1e-10)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ['reflect', 'grid-wrap'])
@pytest.mark.parametrize('kernel', _KERNELS)
@pytest.mark.parametrize('shape', [(8, 8), (9, 11)])
def test_transfer_filter_matches_psf_filter(
    mode: _FilterBoundaryMode, kernel: NDArray[numpy.floating],
    shape: t.Tuple[int, int], backend: BackendName
):
    """
    A `TransferFilter` sampled on the fft grid applies the same filter as the
    `PsfFilter` it came from, under both boundary modes.

    Under `'reflect'` this exercises `TransferFilter.transfer_function_sym`, which
    must Fourier upsample onto the doubled grid.
    """
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=shape)

    psf_filt = PsfFilter(kernel)
    transfer_filt = TransferFilter(psf_filt.transfer_function(_unit_sampling(shape)))

    assert_allclose(
        to_numpy(_recip_convolve(xp.array(arr), transfer_filt, mode)),
        to_numpy(_recip_convolve(xp.array(arr), psf_filt, mode)),
        rtol=1e-10, atol=1e-10,
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ['reflect', 'grid-wrap'])
@pytest.mark.parametrize('shape', [(8, 8), (9, 11)])
def test_gaussian_filter(mode: _FilterBoundaryMode, shape: t.Tuple[int, int], backend: BackendName):
    """`GaussianFilter` samples its transfer function on whichever grid it's asked for."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=shape)

    filt = GaussianFilter(1.5)
    actual = to_numpy(_recip_convolve(xp.array(arr), filt, mode))

    # `transfer_function_sym` is the transfer function of the doubled shape
    doubled = (2 * shape[0], 2 * shape[1])
    assert_allclose(to_numpy(filt.transfer_function_sym(_unit_sampling(shape))),
                    to_numpy(filt.transfer_function(_unit_sampling(doubled))), rtol=1e-12, atol=1e-12)

    # a gaussian blur is close to (but not exactly) its truncated spatial kernel
    ms = numpy.arange(-8, 9)
    kernel = numpy.exp(-0.5 * (ms[:, None]**2 + ms[None, :]**2) / 1.5**2)
    kernel /= numpy.sum(kernel)
    assert_allclose(actual, osp.convolve(arr, kernel, mode=mode), rtol=1e-3, atol=1e-3)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_square_pixel_filter_pixel_sampling(backend: BackendName):
    """`SquarePixelFilter.pixel_sampling` decouples the detector pixel size from the
    `Sampling` it's evaluated on, defaulting to that sampling's own spacing."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(1)
    shape = (16, 16)
    arr = rng.normal(size=shape)

    # a pixel exactly matching the sampling spacing (the old, fixed behavior) should
    # be the same whether that spacing is 1.0 or something else
    unit_samp = _unit_sampling(shape)
    coarse_samp = Sampling(shape, sampling=(2.0, 2.0))

    filt_unit = SquarePixelFilter()
    filt_matched = SquarePixelFilter(pixel_sampling=2.0)

    kernel_unit = to_numpy(filt_unit.psf_separable(unit_samp, xp=xp)[0])
    kernel_matched = to_numpy(filt_matched.psf_separable(coarse_samp, xp=xp)[0])
    assert_allclose(kernel_unit, kernel_matched, rtol=1e-10, atol=1e-10)

    actual_unit = to_numpy(_recip_convolve(xp.array(arr), filt_unit, 'grid-wrap', unit_samp))
    actual_matched = to_numpy(_recip_convolve(xp.array(arr), filt_matched, 'grid-wrap', coarse_samp))
    assert_allclose(actual_unit, actual_matched, rtol=1e-10, atol=1e-10)

    # a detector pixel twice the sampling's spacing should produce a kernel spanning
    # roughly twice as many samples, and should differ from the unscaled case
    filt_wide = SquarePixelFilter(pixel_sampling=4.0)
    kernel_wide = to_numpy(filt_wide.psf_separable(coarse_samp, xp=xp)[0])
    assert len(kernel_wide) > len(kernel_matched)
    assert not numpy.allclose(kernel_wide, numpy.pad(
        kernel_matched, (len(kernel_wide) - len(kernel_matched)) // 2,
    ), rtol=1e-6, atol=1e-6)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ['reflect', 'grid-wrap'])
def test_prepare_convolve2d_recip(mode: _FilterBoundaryMode, backend: BackendName):
    """A prepared filter applies exactly what the matching eager function does."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=(3, 9, 11))
    samp = _unit_sampling((9, 11))

    filt = PsfFilter(_KERNELS[1])
    prepared = prepare_convolve2d_recip(filt, samp, mode=mode, xp=xp)

    assert_allclose(
        to_numpy(prepared(xp.array(arr))),
        to_numpy(_recip_convolve(xp.array(arr), filt, mode, samp)),
        rtol=1e-10, atol=1e-10,
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ['reflect', 'grid-wrap'])
@pytest.mark.parametrize('shape', [(8, 8), (9, 11)])
def test_convolve2d_recip_identity(mode: _FilterBoundaryMode, shape: t.Tuple[int, int], backend: BackendName):
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=shape)

    ones = TransferFilter(numpy.ones(shape))
    assert_allclose(to_numpy(_recip_convolve(xp.array(arr), ones, mode)), arr, rtol=1e-10, atol=1e-10)
    assert_allclose(to_numpy(_recip_convolve(xp.array(arr), PsfFilter(numpy.array([[1.]])), mode)),
                    arr, rtol=1e-10, atol=1e-10)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('mode', ['reflect', 'grid-wrap'])
@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
def test_convolve2d_recip_dtypes(mode: _FilterBoundaryMode, dtype, backend: BackendName):
    """The result always has the same dtype as `arr`, even if the filter's transfer function is complex."""
    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(42)
    arr = rng.normal(size=(8, 8)).astype(dtype)
    kernel = _KERNELS[1].astype(numpy.float64)

    for filt in (PsfFilter(kernel), TransferFilter(numpy.ones((8, 8))), GaussianFilter(1.5)):
        assert to_numpy(_recip_convolve(xp.array(arr), filt, mode)).dtype == numpy.dtype(dtype)

    # a complex `PsfFilter` can't filter a real array (no explicit `dtype` -> defaults to real)
    with pytest.raises(ValueError, match="Expected a real point spread function"):
        PsfFilter(kernel * 1.j).psf(_unit_sampling((8, 8)), xp=xp)

    # a complex *transfer function* (e.g. a propagator or phase plate) is a normal filter;
    # applying one to a real `arr` assumes the transfer function is Hermitian, and keeps
    # the result real (matching `arr`'s dtype)
    complex_filt = TransferFilter(xp.asarray(numpy.exp(1.j * rng.normal(size=(8, 8)))))
    assert to_numpy(_recip_convolve(xp.array(arr), complex_filt, mode)).dtype == numpy.dtype(dtype)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_convolve2d_recip_validates(backend: BackendName):
    xp = get_backend_module(backend)
    arr = xp.array(numpy.zeros((4, 5)))
    samp = _unit_sampling((4, 5))

    with pytest.raises(TypeError, match="must implement 'psf' or 'transfer_function'"):
        class BadFilter(Filter):
            pass

    with pytest.raises(ValueError, match="Expected 'kernel' to be 2D"):
        convolve2d_recip(arr, PsfFilter(numpy.ones(3)), samp)

    with pytest.raises(ValueError, match="Expected 'y_kernel' and 'x_kernel' to be 1D"):
        convolve2d_recip(arr, SeparablePsfFilter(numpy.ones((2, 2)), numpy.ones(2)), samp)

    # under 'reflect', the kernel is embedded on the doubled grid (needed for a
    # correct whole-sample-symmetric extension), so it must be checked under
    # 'grid-wrap' (no doubling) to fail against the base data shape
    with pytest.raises(ValueError, match=r"doesn't fit in a grid of shape \(4, 5\)"):
        convolve2d_recip(arr, PsfFilter(numpy.ones((7, 3))), samp, mode='grid-wrap')

    with pytest.raises(ValueError, match=r"Expected a transfer function of shape \(4, 5\), instead got shape \(4,\)"):
        convolve2d_recip(arr, TransferFilter(numpy.ones(4)), samp)

    with pytest.raises(ValueError, match=r"Expected a transfer function of shape \(4, 5\)"):
        convolve2d_recip(arr, TransferFilter(numpy.ones((5, 4))), samp)

    with pytest.raises(ValueError, match="Expected 'arr' to be at least 2D"):
        convolve2d_recip_wrap(xp.array(numpy.zeros(4)), numpy.ones((4,)))

    with pytest.raises(ValueError, match=r"Filter was prepared for shape \(8, 8\)"):
        prepared = prepare_convolve2d_recip(PsfFilter(numpy.ones((1, 1))), _unit_sampling((8, 8)), xp=xp)
        prepared(arr)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_composite_filter_transfer_function(backend: BackendName):
    """Composing two non-separable filters multiplies their transfer functions."""
    xp = get_backend_module(backend)
    samp = _unit_sampling((8, 8))

    f1 = TransferFilter(numpy.exp(-1.j * numpy.arange(64).reshape(8, 8) / 64.))
    f2 = TransferFilter(numpy.linspace(0.1, 1., 64).reshape(8, 8))
    composite = f1 * f2

    assert isinstance(composite, ProductFilter)
    assert not isinstance(composite, ProductSeparableFilter)
    assert composite.filters == (f1, f2)

    expected = to_numpy(f1.transfer_function(samp, xp=xp)) * to_numpy(f2.transfer_function(samp, xp=xp))
    assert_allclose(to_numpy(composite.transfer_function(samp, xp=xp)), expected, rtol=1e-10, atol=1e-10)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_composite_filter_psf_matches_transfer_function(backend: BackendName):
    """
    `ProductFilter.psf`'s direct compact-kernel convolution agrees with the
    (independent) FFT-derived `ifft2(transfer_function(samp))`, once both are
    embedded onto the same full grid.
    """
    from phaser.utils.image import _embed_psf

    xp = get_backend_module(backend)
    samp = _unit_sampling((16, 16))

    f1 = PsfFilter(_KERNELS[1])
    f2 = PsfFilter(_KERNELS[2])
    composite = ProductFilter((f1, f2))

    compact = composite.psf(samp, xp=xp)
    actual = to_numpy(_embed_psf(compact, (16, 16), xp, compact.dtype))
    expected = to_numpy(Filter.psf(composite, samp, xp=xp))
    assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_composite_separable_filter(backend: BackendName):
    """
    `SeparableFilter * SeparableFilter` produces a `ProductSeparableFilter` whose
    1D kernels are the full convolution of the components' own 1D kernels, and whose
    (outer-product) `psf` agrees with the generic `ProductFilter` path.
    """
    xp = get_backend_module(backend)
    samp = _unit_sampling((64, 64))

    f1 = GaussianFilter(1.5)
    f2 = SquarePixelFilter()
    composite = f1 * f2

    assert isinstance(composite, ProductSeparableFilter)

    y1, x1 = (to_numpy(k) for k in f1.psf_separable(samp, xp=xp))
    y2, x2 = (to_numpy(k) for k in f2.psf_separable(samp, xp=xp))
    y, x = (to_numpy(k) for k in composite.psf_separable(samp, xp=xp))
    assert_allclose(y, numpy.convolve(y1, y2, mode='full'), rtol=1e-8, atol=1e-8)
    assert_allclose(x, numpy.convolve(x1, x2, mode='full'), rtol=1e-8, atol=1e-8)

    generic = ProductFilter((f1, f2))
    assert_allclose(
        to_numpy(composite.psf(samp, xp=xp)), to_numpy(generic.psf(samp, xp=xp)), rtol=1e-8, atol=1e-8,
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_composite_filter_flattens(backend: BackendName):
    """Composing composites flattens into one `.filters` tuple, rather than nesting."""
    f1, f2, f3 = GaussianFilter(1.), SquarePixelFilter(), GaussianFilter(2.)

    left = (f1 * f2) * f3
    right = f1 * (f2 * f3)
    assert left.filters == (f1, f2, f3)
    assert right.filters == (f1, f2, f3)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_composite_filter_identity(backend: BackendName):
    """The empty `ProductFilter` is the identity filter: an all-ones transfer
    function, and a single-tap (compact) point spread function."""
    xp = get_backend_module(backend)
    samp = _unit_sampling((8, 8))

    identity = ProductFilter(())
    assert_allclose(to_numpy(identity.transfer_function(samp, xp=xp)), numpy.ones((8, 8)))

    psf = to_numpy(identity.psf(samp, xp=xp))
    assert_allclose(psf, numpy.array([[1.]]), atol=1e-10)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_composite_filter_symmetric(backend: BackendName):
    """A composite is symmetric iff every component is."""
    symmetric = GaussianFilter(1.5)
    asymmetric = PsfFilter(_KERNELS[2], symmetric=False)

    assert (symmetric * GaussianFilter(2.)).symmetric
    assert not (symmetric * asymmetric).symmetric


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_convolve_kernels_1d(backend: BackendName):
    from phaser.utils.image import _convolve_kernels_1d

    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(3)
    a = rng.normal(size=7)
    b = rng.normal(size=4)

    actual = to_numpy(_convolve_kernels_1d(xp.array(a), xp.array(b), xp, numpy.float64))
    assert_allclose(actual, numpy.convolve(a, b, mode='full'), rtol=1e-10, atol=1e-10)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_convolve_kernels_2d(backend: BackendName):
    from phaser.utils.image import _convolve_kernels_2d

    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(4)
    a = rng.normal(size=(5, 3))
    b = rng.normal(size=(3, 4))

    actual = to_numpy(_convolve_kernels_2d(xp.array(a), xp.array(b), xp, numpy.float64))

    import scipy.signal
    expected = scipy.signal.convolve2d(a, b, mode='full')
    assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_convolve_kernels_1d_short_circuit(backend: BackendName):
    """A length-1 kernel on either side short-circuits to a plain scale, agreeing
    with the general FFT-based convolution."""
    from phaser.utils.image import _convolve_kernels_1d

    xp = get_backend_module(backend)
    rng = numpy.random.default_rng(5)
    a = rng.normal(size=6)
    scale = numpy.array([2.5])

    left = to_numpy(_convolve_kernels_1d(xp.array(scale), xp.array(a), xp, numpy.float64))
    right = to_numpy(_convolve_kernels_1d(xp.array(a), xp.array(scale), xp, numpy.float64))
    expected = numpy.convolve(a, scale, mode='full')
    assert_allclose(left, expected, rtol=1e-10, atol=1e-10)
    assert_allclose(right, expected, rtol=1e-10, atol=1e-10)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_filter_scalar_multiply(backend: BackendName):
    """Scaling a filter by a real factor scales both its psf and transfer function."""
    xp = get_backend_module(backend)
    samp = _unit_sampling((16, 16))

    filt = GaussianFilter(1.5)
    scaled = filt * 2.0

    expected_transfer = to_numpy(filt.transfer_function(samp, xp=xp)) * 2.0
    expected_psf = to_numpy(filt.psf(samp, xp=xp)) * 2.0
    assert_allclose(to_numpy(scaled.transfer_function(samp, xp=xp)), expected_transfer, rtol=1e-8, atol=1e-8)
    assert_allclose(to_numpy(scaled.psf(samp, xp=xp)), expected_psf, rtol=1e-8, atol=1e-8)

    rscaled = 2.0 * filt
    assert_allclose(to_numpy(rscaled.transfer_function(samp, xp=xp)), expected_transfer, rtol=1e-8, atol=1e-8)
    assert_allclose(to_numpy(rscaled.psf(samp, xp=xp)), expected_psf, rtol=1e-8, atol=1e-8)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_filter_scalar_multiply_preserves_separable(backend: BackendName):
    """Scaling a `SeparableFilter` stays separable, including under further composition."""
    scaled = GaussianFilter(1.5) * 2.0
    assert isinstance(scaled, ProductSeparableFilter)

    further = scaled * SquarePixelFilter()
    assert isinstance(further, ProductSeparableFilter)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_filter_scalar_multiply_preserves_symmetric(backend: BackendName):
    """Scaling a symmetric filter by a real factor keeps it symmetric."""
    assert (GaussianFilter(1.5) * 2.0).symmetric


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_filter_scalar_multiply_type_error(backend: BackendName):
    with pytest.raises(TypeError):
        GaussianFilter(1.5) * 'x'  # type: ignore
