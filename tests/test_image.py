import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray
from numpy.testing import assert_array_almost_equal
import scipy.ndimage as osp
import pytest

from .utils import with_backends, check_array_equals_file

from phaser.utils.num import get_backend_module, BackendName, to_numpy, Sampling
from phaser.utils.image import (
    affine_transform, _InterpBoundaryMode, convolve1d
)


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

    # TODO these fail on torch
    # kernel longer than array
    #([1, 2, 3], [1, 2, 3, 4, 5, 6, 7], 0),
    # length-1 along conv axis
    #([[[1, 2], [3, 4]]], [1, 2, 3], 0),
])
@pytest.mark.parametrize(('mode', 'cval'), [
    ('constant', 1.0), ('nearest', 0.0), ('mirror', 0.0),
    ('reflect', 0.0), ('wrap', 0.0),
])
def test_convolve1d(
    arr, weights, axis, mode, cval,
    backend: BackendName,
):
    if mode == 'reflect' and backend == 'torch':
        pytest.xfail("'reflect' not supported on torch")

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