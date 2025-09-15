
import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from .utils import with_backends, check_array_equals_file

from phaser.utils.num import get_backend_module, BackendName, to_numpy, abs2
from phaser.utils.object import random_phase_object, ObjectSampling


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_random_phase_object(backend: BackendName):
    xp = get_backend_module(backend)

    obj = random_phase_object((8, 8), 1e-4, seed=2620771887, dtype=numpy.complex64, xp=xp)

    assert obj.dtype == xp.complex64
    assert_array_almost_equal(to_numpy(obj), numpy.array([
        [1.-1.5272086e-05j, 1.+1.0225522e-04j, 1.-8.0865902e-05j, 1.-1.7328106e-05j, 1.-1.2898073e-04j, 1.+2.2908196e-05j, 1.+8.1173976e-06j, 1.+2.1377344e-05j],
        [1.+7.4363430e-05j, 1.-9.1323782e-05j, 1.-2.0272582e-04j, 1.-4.8823396e-05j, 1.+9.3021641e-05j, 1.+1.0718761e-04j, 1.+5.0221975e-06j, 1.-5.5743083e-05j],
        [1.+9.5888179e-05j, 1.+7.0838556e-05j, 1.-1.1567964e-04j, 1.+1.3202346e-04j, 1.+1.3625837e-04j, 1.-5.2489726e-05j, 1.-1.3756646e-04j, 1.+2.8579381e-05j],
        [1.-6.3651263e-05j, 1.-4.5127890e-05j, 1.+5.5954431e-05j, 1.-1.8197308e-04j, 1.+6.3579530e-05j, 1.-4.6506138e-05j, 1.+5.1510222e-05j, 1.+1.0700211e-04j],
        [0.99999994-2.6713090e-04j, 1.+1.8953861e-04j, 1.+1.1097628e-04j, 1.-5.9648257e-05j, 1.+4.2086729e-05j, 1.+6.0222395e-05j, 1.-8.4926840e-05j, 0.99999994-2.6520275e-04j],
        [1.-4.1160638e-05j, 1.+8.4538617e-05j, 1.+4.1620955e-05j, 1.+1.6012797e-05j, 1.-1.3888512e-05j, 1.+9.1871625e-06j, 1.+5.0595980e-05j, 1.+2.3048995e-04j],
        [1.-2.8506602e-05j, 1.+2.4769653e-05j, 1.-2.3920753e-05j, 1.+8.0796681e-06j, 0.99999994-2.5373933e-04j, 1.+6.9838488e-06j, 1.+3.8624425e-05j, 1.+1.1229565e-04j],
        [1.-4.8519720e-05j, 1.+9.4494520e-05j, 1.+4.9148810e-05j, 1.-1.3229759e-04j, 1.-2.6898948e-05j, 1.-9.8376579e-05j, 1.+6.9485272e-05j, 1.-9.8597156e-05j]
    ], dtype=numpy.complex64))


def test_object_sampling_grid():
    # asymmetric, mixed even & odd sampling grid
    samp = ObjectSampling((16, 15), (0.5, 2.0))

    assert_array_equal(samp.shape, [16, 15])
    assert_array_almost_equal(samp.sampling, [0.5, 2.0])
    # centered around (0, 0)
    assert_array_almost_equal(samp.corner, [-3.75, -14.0])
    assert_array_almost_equal(samp.min, [-3.75, -14.0])
    assert_array_almost_equal(samp.max, [3.75, 14.0])

    assert samp.region_min is None
    assert samp.region_max is None

    assert samp.mpl_extent() == pytest.approx((-15.0, 15.0, 4.0, -4.0))
    assert samp.mpl_extent(False) == pytest.approx((-14.0, 16.0, 4.25, -3.75))

    yy, xx = samp.grid()

    assert_array_almost_equal(xx[0, :], [
        -14., -12., -10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10., 12., 14.
    ])
    assert_array_almost_equal(yy[:, 0], [
        -3.75, -3.25, -2.75, -2.25, -1.75, -1.25, -0.75, -0.25,  0.25, 0.75,  1.25,  1.75,  2.25,  2.75,  3.25,  3.75
    ])

    assert samp.get_region_crop() == (slice(None, None), slice(None, None))


def test_object_from_scan():
    scan = numpy.array([
        [-4.8, -1.],
        [2.1, 1.],
        [4.19, 9.9],
        [0., numpy.nan],
    ])

    samp = ObjectSampling.from_scan(scan, sampling=[0.5, 0.5], pad=2.)

    assert_array_almost_equal(samp.min, [-6.8, -3.])
    assert_array_almost_equal(samp.max, [6.2, 12.0])
    assert_array_almost_equal(samp.shape, (27, 31))

    assert samp.region_min is not None
    assert_array_almost_equal(samp.region_min, [-4.8, -1.])
    assert samp.region_max is not None
    assert_array_almost_equal(samp.region_max, [4.19, 9.9])

    assert samp.get_region_crop() == (slice(4, 22), slice(4, 26))


def test_object_expand_to_scan():
    samp = ObjectSampling(
        shape=(10, 10),
        sampling=(10.0, 20.0),
        corner=(-5.0, -10.0),
        region_min=None,
        region_max=(20.0, 500.0),
    )
    scan = numpy.array([
        [5.0, -10.0],
        [75.0, 50.0],
    ])

    new_samp = samp.expand_to_scan(scan, pad=15.0)

    assert_array_equal(new_samp.shape, [12, 11])
    assert_array_equal(new_samp.sampling, samp.sampling)
    assert_array_almost_equal(new_samp.min, [-15.0, -30.0])
    assert_array_almost_equal(new_samp.max, [95.0, 170.0])
    assert new_samp.region_min is None
    assert new_samp.region_max is not None
    assert_array_almost_equal(new_samp.region_max, [75.0, 500.0])


def test_object_slicing():
    samp = ObjectSampling((40, 31), (0.5, 2.0))

    (yy, xx) = samp.grid()

    # y pixels on 0.25 + 0.5 * i
    # x pixels on 2 * j

    # when exact pixel coord, should round-trip
    yp, xp = numpy.round(samp._pos_to_object_idx(numpy.array((-1.25, 12.0)), (1, 1))).astype(numpy.int_)
    assert yy[yp, xp] == pytest.approx(-1.25)
    assert xx[yp, xp] == pytest.approx(12.0)

    # should get the same pixel when perturbed by <1/2 sample
    assert_array_equal(numpy.round(samp._pos_to_object_idx(numpy.array((-1.25 + 0.24, 12.0 - 0.99)), (1, 1))).astype(numpy.int_), (yp, xp))

    # odd cutout n*2 + 1, n samples away, same result
    n = 2
    pos = numpy.array((-1.25 - 0.24 + 0.5 * n, 12.0 - 0.99 + 2.0 * n))
    cutout_shape = (1 + 2*n, 1 + 2*n)
    assert_array_equal(numpy.round(samp._pos_to_object_idx(pos, cutout_shape)).astype(numpy.int_), (yp, xp))

    # check slicing as well
    assert samp.slice_at_pos(pos, cutout_shape) == (
        slice(yp, yp + 1 + 2*n),
        slice(xp, xp + 1 + 2*n)
    )

    n = 6
    # even cutout n*2, n - 1/2 samples away, same result
    pos = numpy.array((-1.25 - 0.24 + 0.5 * (n - 1/2.), 12.0 - 0.99 + 2.0 * (n - 1/2.)))
    cutout_shape = (2*n, 2*n)
    assert_array_equal(numpy.round(samp._pos_to_object_idx(pos, cutout_shape)).astype(numpy.int_), (yp, xp))

    # check slicing as well
    assert samp.slice_at_pos(pos, cutout_shape) == (
        slice(yp, yp + 2*n),
        slice(xp, xp + 2*n)
    )


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('dtype', ('float', 'complex', 'uint8'))
@check_array_equals_file('object_get_views_{dtype}.npy', out_name='object_get_views_{dtype}_{backend}.npy')
def test_get_cutouts(backend: BackendName, dtype: str) -> numpy.ndarray:
    samp = ObjectSampling((200, 200), (1.0, 1.0))
    cutout_shape = (64, 64)

    xp = get_backend_module(backend)

    yy, xx = xp.arange(samp.shape[-2]), xp.arange(samp.shape[-1])
    yy, xx = xp.meshgrid(yy, xx, indexing='ij')

    if dtype == 'uint8':
        obj = ((yy + 2.*xx) % 256).astype(numpy.uint8)
    elif dtype == 'float':
        obj = (yy + 2.*xx).astype(numpy.float32)
    elif dtype == 'complex':
        obj = xp.exp((yy + 2.*xx).astype(numpy.complex64) * 1.j/100)
    else:
        raise ValueError()

    pos = numpy.array([
        [0., 0.],
        [-50., -50.],
        [-50., 50.],
        [50., 50.],
    ])

    return to_numpy(samp.cutout(obj, pos, cutout_shape).get())

@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('dtype', ('float', 'complex', 'uint8'))
@check_array_equals_file('object_add_views_{dtype}.tiff', out_name='object_add_views_{dtype}_{backend}.tiff', decimal=5)
def test_add_view_at_pos(backend: BackendName, dtype: str) -> numpy.ndarray:
    samp = ObjectSampling((200, 200), (1.0, 1.0))
    cutout_shape = (64, 64)

    xp = get_backend_module(backend)

    if dtype == 'uint8':
        obj = xp.zeros(tuple(samp.shape), dtype=numpy.uint8)
        cutouts = xp.full((30, *cutout_shape), 15, dtype=numpy.uint8)
        mag = 15
    elif dtype == 'float':
        obj = xp.zeros(tuple(samp.shape), dtype=numpy.float32)
        cutouts = xp.full((30, *cutout_shape), 10., dtype=numpy.float32)
        mag = 10.
    elif dtype == 'complex':
        obj = xp.zeros(tuple(samp.shape), dtype=numpy.complex64)
        phases = xp.array([
            4.30015617, 5.15367214, 6.13496658, 4.9268498 , 3.60960355,
            0.42680191, 5.12820671, 1.3260991 , 2.2065813 , 5.1417133 ,
            4.44775022, 3.78443706, 1.39529171, 4.85271223, 5.29254269,
            0.16160692, 3.93409553, 3.30689481, 2.65883774, 5.58807904,
            2.97826432, 5.74576854, 1.22940843, 2.79499453, 2.83189806,
            5.41544805, 1.44714743, 4.48405743, 2.50836483, 3.31088288
        ], dtype=numpy.float32)
        cutouts = xp.full((30, *cutout_shape), 10., dtype=numpy.complex64) * xp.exp(1.j * phases[:, None, None])
        mag = 1.  # needed for type checking
    else:
        raise ValueError()

    pos = numpy.array([
       [ 50.60303087,  65.79992213],
       [ 55.29351178, -23.20535679],
       [-24.55795797,  61.03108143],
       [ 31.11937136, -15.76195155],
       [ 28.01925522,  45.1035061 ],
       [ 10.36488598,   2.55497549],
       [-57.42025   ,  48.55254953],
       [  4.08776065,  50.33652909],
       [-15.73631518, -36.91599871],
       [-25.16043481,  28.76977663],
       [ 31.42525951,  10.31409966],
       [-19.79372857,  55.79668508],
       [-60.01548957,  26.95982136],
       [-31.2433725 , -54.82789204],
       [ 59.78674718,  40.56994336],
       [-46.26502396, -38.71300405],
       [-66.83498502,  -6.33424509],
       [ 62.7652186 , -12.48322478],
       [ -2.31684356,  -6.0558    ],
       [-32.38632762,   5.55021944],
       [-52.72306553, -19.39256228],
       [ 13.48322753,  60.19354572],
       [-30.44280284,  52.91061857],
       [-43.27221074,  51.47703124],
       [-24.6619304 ,  66.08921008],
       [ 59.68745301,  43.99038992],
       [-60.82051963, -42.56576538],
       [-38.95303419, -38.40890994],
       [-55.76325197, -55.0998433 ],
       [ 48.4540186 ,  52.91460643]
    ])

    obj = samp.add_view_at_pos(obj, pos, cutouts)

    if dtype != 'complex':
        assert numpy.sum(to_numpy(obj)) == mag * numpy.prod(cutouts.shape)

    return to_numpy(obj)


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_cutout_2d(backend: BackendName):
    samp = ObjectSampling((200, 200), (1.0, 1.0))
    cutout_shape = (64, 64)

    xp = get_backend_module(backend)
    obj = xp.zeros(tuple(samp.shape), dtype=numpy.float32)

    cutouts = samp.cutout(obj, [[0., 0.], [2., 2.], [4., 4.], [-2., -2.]], cutout_shape)
    assert cutouts.get().shape == (4, *cutout_shape)

    # also test that addition and assignment work
    cutouts.add(cutouts.get())
    cutouts.set(cutouts.get())


@with_backends('numpy', 'jax', 'cupy', 'torch')
def test_cutout_multidim(backend: BackendName):
    samp = ObjectSampling((200, 200), (1.0, 1.0))
    cutout_shape = (80, 100)

    xp = get_backend_module(backend)

    (zz, yy, xx) = xp.indices((3, *samp.shape), dtype=numpy.float32)

    obj = 1000.*zz + (2.*yy + xx)

    cutouts = samp.cutout(obj, [
        [[-50., -50.], [-50., 50.]],
        [[50., -50.], [50., 50.]],
    ], cutout_shape)
    cutout_arr = cutouts.get()
    assert cutout_arr.shape == (2, 2, 3, *cutout_shape)

    # check top left corner of each cutout
    assert_array_almost_equal(to_numpy(cutout_arr[..., 0, 0]), [
        [[20., 1020., 2020.], [120., 1120., 2120.]],
        [[220., 1220., 2220.], [320., 1320., 2320.]],
    ])

    # check that each cutout is a ramp
    assert_array_almost_equal(*numpy.broadcast_arrays(numpy.diff(to_numpy(cutout_arr), axis=-1), 1.))  # type: ignore
    assert_array_almost_equal(*numpy.broadcast_arrays(numpy.diff(to_numpy(cutout_arr), axis=-2), 2.))  # type: ignore

    # also test that addition and assignment work
    cutouts.add(cutouts.get())
    cutouts.set(cutouts.get())


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('dtype', ('float', 'complex', 'uint8'))
@check_array_equals_file('object_set_views_{dtype}.tiff', out_name='object_set_views_{dtype}_{backend}.tiff')
def test_set_view_at_pos(backend: BackendName, dtype: str) -> numpy.ndarray:
    samp = ObjectSampling((200, 200), (1.0, 1.0))
    cutout_shape = (64, 64)

    xp = get_backend_module(backend)

    if dtype == 'uint8':
        obj = xp.zeros(tuple(samp.shape), dtype=numpy.uint8)
        cutouts = xp.full((30, *cutout_shape), 15, dtype=numpy.uint8)
        mag = 15
    elif dtype == 'float':
        obj = xp.zeros(tuple(samp.shape), dtype=numpy.float32)
        cutouts = xp.full((30, *cutout_shape), 10., dtype=numpy.float32)
        mag = 10.
    elif dtype == 'complex':
        obj = xp.zeros(tuple(samp.shape), dtype=numpy.complex64)
        cutouts = xp.full((30, *cutout_shape), 10. + 15.j, dtype=numpy.complex64)
        mag = abs2(10. + 15.j)
    else:
        raise ValueError()

    pos = numpy.array([
       [ 50.60303087,  65.79992213],
       [ 55.29351178, -23.20535679],
       [-24.55795797,  61.03108143],
       [ 31.11937136, -15.76195155],
       [ 28.01925522,  45.1035061 ],
       [ 10.36488598,   2.55497549],
       [-57.42025   ,  48.55254953],
       [  4.08776065,  50.33652909],
       [-15.73631518, -36.91599871],
       [-25.16043481,  28.76977663],
       [ 31.42525951,  10.31409966],
       [-19.79372857,  55.79668508],
       [-60.01548957,  26.95982136],
       [-31.2433725 , -54.82789204],
       [ 59.78674718,  40.56994336],
       [-46.26502396, -38.71300405],
       [-66.83498502,  -6.33424509],
       [ 62.7652186 , -12.48322478],
       [ -2.31684356,  -6.0558    ],
       [-32.38632762,   5.55021944],
       [-52.72306553, -19.39256228],
       [ 13.48322753,  60.19354572],
       [-30.44280284,  52.91061857],
       [-43.27221074,  51.47703124],
       [-24.6619304 ,  66.08921008],
       [ 59.68745301,  43.99038992],
       [-60.82051963, -42.56576538],
       [-38.95303419, -38.40890994],
       [-55.76325197, -55.0998433 ],
       [ 48.4540186 ,  52.91460643]
    ])

    obj = samp.set_view_at_pos(obj, pos, cutouts)

    if dtype == 'complex':
        assert numpy.max(to_numpy(abs2(obj))) == mag
    else:
        assert numpy.max(to_numpy(obj)) == mag

    return to_numpy(obj)