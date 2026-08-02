import numpy
import pytest

from phaser.web.pubsub import Cache
from phaser.web.util import decode_obj, encode_obj
from phaser.web.views import VIEWS, project_phase, slice_view

pytestmark = pytest.mark.web


def _wire_object(data: numpy.ndarray, thicknesses: numpy.ndarray, sampling: dict) -> dict:
    return encode_obj({'sampling': sampling, 'data': data, 'thicknesses': thicknesses})


def test_project_phase_matches_numpy_reference():
    rng = numpy.random.default_rng(0)
    data = (rng.normal(size=(4, 6, 7)) + 1j * rng.normal(size=(4, 6, 7))).astype(numpy.complex64)
    data[0, 0, 0] = complex(numpy.nan, numpy.nan)  # a NaN pixel shouldn't poison the sum
    sampling = {'shape': [6, 7], 'sampling': [1.0, 1.0], 'corner': [0.0, 0.0], 'region_min': None, 'region_max': None}
    thicknesses = numpy.array([1.0, 1.0, 1.0, 1.0], dtype=numpy.float32)

    cache = Cache()
    cache.update_raw({'object': _wire_object(data, thicknesses, sampling)})

    out = decode_obj(project_phase(cache, {}))
    ref = numpy.nansum(numpy.angle(data), axis=0)

    assert out['data'].shape == (6, 7)
    assert out['data'].dtype == numpy.float32
    numpy.testing.assert_allclose(out['data'], ref, equal_nan=True)
    assert out['sampling'] == sampling


def test_project_phase_single_slice():
    # z == 1 (single-slice object): leading-axis reduction should be a no-op, not an error
    rng = numpy.random.default_rng(1)
    data = (rng.normal(size=(1, 3, 3)) + 1j * rng.normal(size=(1, 3, 3))).astype(numpy.complex64)
    sampling = {'shape': [3, 3], 'sampling': [1.0, 1.0], 'corner': [0.0, 0.0], 'region_min': None, 'region_max': None}
    cache = Cache()
    cache.update_raw({'object': _wire_object(data, numpy.array([1.0], dtype=numpy.float32), sampling)})

    out = decode_obj(project_phase(cache, {}))
    numpy.testing.assert_allclose(out['data'], numpy.angle(data[0]))


def test_slice_view_selects_correct_index():
    rng = numpy.random.default_rng(2)
    data = (rng.normal(size=(3, 4, 5)) + 1j * rng.normal(size=(3, 4, 5))).astype(numpy.complex64)
    sampling = {'shape': [4, 5], 'sampling': [1.0, 1.0], 'corner': [0.0, 0.0], 'region_min': None, 'region_max': None}
    thicknesses = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
    cache = Cache()
    cache.update_raw({'object': _wire_object(data, thicknesses, sampling)})

    for idx in range(3):
        out = decode_obj(slice_view(cache, {'slice': idx}))
        numpy.testing.assert_array_equal(out['data'], data[idx])
        assert out['thickness'] == float(thicknesses[idx])


def test_cache_array_decodes_once_per_generation():
    calls = []
    from phaser.web import util as _util
    real_decode = _util.decode_obj

    def counting_decode(obj):
        calls.append(1)
        return real_decode(obj)

    cache = Cache()
    arr = numpy.arange(6, dtype='<f8').reshape(2, 3)
    cache.update_raw({'x': encode_obj(arr)})

    import phaser.web.pubsub as pubsub_mod
    orig = pubsub_mod.decode_obj
    pubsub_mod.decode_obj = counting_decode
    try:
        v1 = cache.array('x')
        v2 = cache.array('x')  # same generation -> memoized, no re-decode
        assert len(calls) == 1
        numpy.testing.assert_array_equal(v1, arr)
        numpy.testing.assert_array_equal(v2, arr)

        cache.update_raw({'x': encode_obj(arr + 1)})  # bumps generation
        v3 = cache.array('x')
        assert len(calls) == 2
        numpy.testing.assert_array_equal(v3, arr + 1)
    finally:
        pubsub_mod.decode_obj = orig


def test_status_progress_probes_are_raw_passthrough():
    cache = Cache()
    cache.update_raw({'status': 'running', 'progress': {'total_loss': {'iters': [1], 'values': [0.5]}}, 'probe': {'sampling': {}, 'data': 'x'}})

    assert VIEWS['status'].compute(cache, {}) == 'running'
    assert VIEWS['progress'].compute(cache, {}) == {'total_loss': {'iters': [1], 'values': [0.5]}}
    assert VIEWS['probes'].compute(cache, {}) == {'sampling': {}, 'data': 'x'}


def test_views_deps_are_disjoint_from_unrelated_fields():
    # sanity check on the registry: 'obj_phase_sum'/'obj' only fire on 'object' changes,
    # not on unrelated fields like 'iter' or 'progress' (laziness/dirty-set correctness).
    assert VIEWS['obj_phase_sum'].deps == frozenset({'object'})
    assert VIEWS['obj'].deps == frozenset({'object'})
    assert VIEWS['status'].deps == frozenset({'status'})
    assert VIEWS['progress'].deps == frozenset({'progress'})
    assert VIEWS['probes'].deps == frozenset({'probe'})
