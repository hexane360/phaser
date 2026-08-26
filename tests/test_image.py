import typing as t

import numpy
import pytest
import scipy.ndimage as osp
from numpy.testing import assert_allclose, assert_array_almost_equal
from numpy.typing import ArrayLike, NDArray

from phaser.utils.image import (
    _InterpBoundaryMode,
    affine_transform,
    convolve1d,
    convolve2d,
    convolve2d_separable,
)
from phaser.utils.num import BackendName, Sampling, get_backend_module, to_numpy

from .utils import check_array_equals_file, with_backends


@pytest.fixture
def checkerboard() -> t.Tuple[NDArray[numpy.float32], Sampling]:
    yy, xx = numpy.indices((16, 16))
    checker = ((yy % 2) ^ (xx % 2)).astype(numpy.float32)

    return (checker, Sampling(checker.shape, sampling=(1.0, 1.0)))


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
