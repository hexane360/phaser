import numpy
import pytest

from phaser.web.pubsub import Cache
from phaser.web.util import decode_obj, encode_obj
from phaser.web.views import (
    VIEWS, obj_meta_view, probe_meta_view, probes_recip_view, project_amp_mean, project_phase,
    slice_view,
)

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

    # bulk views carry the bare array; sampling lives on `obj_meta`
    assert out.shape == (6, 7)
    assert out.dtype == numpy.float32
    numpy.testing.assert_allclose(out, ref, equal_nan=True)
    assert obj_meta_view(cache, {})['sampling'] == sampling


def test_project_phase_single_slice():
    # z == 1 (single-slice object): leading-axis reduction should be a no-op, not an error
    rng = numpy.random.default_rng(1)
    data = (rng.normal(size=(1, 3, 3)) + 1j * rng.normal(size=(1, 3, 3))).astype(numpy.complex64)
    sampling = {'shape': [3, 3], 'sampling': [1.0, 1.0], 'corner': [0.0, 0.0], 'region_min': None, 'region_max': None}
    cache = Cache()
    cache.update_raw({'object': _wire_object(data, numpy.array([1.0], dtype=numpy.float32), sampling)})

    out = decode_obj(project_phase(cache, {}))
    numpy.testing.assert_allclose(out, numpy.angle(data[0]))


def test_project_amp_mean_is_geometric():
    rng = numpy.random.default_rng(3)
    amps = rng.uniform(0.5, 2.0, size=(4, 6, 7))
    phases = rng.uniform(-numpy.pi, numpy.pi, size=(4, 6, 7))
    data = (amps * numpy.exp(1j * phases)).astype(numpy.complex64)
    sampling = {'shape': [6, 7], 'sampling': [1.0, 1.0], 'corner': [0.0, 0.0], 'region_min': None, 'region_max': None}

    cache = Cache()
    cache.update_raw({'object': _wire_object(data, numpy.ones(4, dtype=numpy.float32), sampling)})

    out = decode_obj(project_amp_mean(cache, {}))
    # geometric, not arithmetic: the n'th root of the product
    numpy.testing.assert_allclose(out, numpy.prod(amps, axis=0) ** (1 / 4), rtol=1e-5)
    assert out.shape == (6, 7)


def test_project_amp_mean_single_slice_and_zeros():
    sampling = {'shape': [2, 2], 'sampling': [1.0, 1.0], 'corner': [0.0, 0.0], 'region_min': None, 'region_max': None}
    # a single slice: the geometric mean of one value is that value
    data = numpy.array([[[3.0 + 4.0j, 1.0 + 0.0j], [0.0 + 2.0j, 1.0 + 1.0j]]], dtype=numpy.complex64)
    cache = Cache()
    cache.update_raw({'object': _wire_object(data, numpy.array([], dtype=numpy.float32), sampling)})
    numpy.testing.assert_allclose(decode_obj(project_amp_mean(cache, {})), numpy.abs(data[0]), rtol=1e-6)

    # a zero pixel drives the geometric mean to zero rather than raising or returning NaN
    data = numpy.array([[[2.0 + 0j, 1.0 + 0j]], [[0.0 + 0j, 1.0 + 0j]]], dtype=numpy.complex64)
    cache.update_raw({'object': _wire_object(data, numpy.ones(2, dtype=numpy.float32), sampling)})
    numpy.testing.assert_allclose(decode_obj(project_amp_mean(cache, {})), [[0.0, 1.0]], atol=1e-7)


def test_slice_view_selects_correct_index():
    rng = numpy.random.default_rng(2)
    data = (rng.normal(size=(3, 4, 5)) + 1j * rng.normal(size=(3, 4, 5))).astype(numpy.complex64)
    sampling = {'shape': [4, 5], 'sampling': [1.0, 1.0], 'corner': [0.0, 0.0], 'region_min': None, 'region_max': None}
    thicknesses = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
    cache = Cache()
    cache.update_raw({'object': _wire_object(data, thicknesses, sampling)})

    for idx in range(3):
        numpy.testing.assert_array_equal(decode_obj(slice_view(cache, {'slice': idx})), data[idx])

    # params outlive the run that set them, so an out-of-range slice clamps rather than
    # raising -- an IndexError here would take down the whole tick's publish
    numpy.testing.assert_array_equal(decode_obj(slice_view(cache, {'slice': 99})), data[2])
    numpy.testing.assert_array_equal(decode_obj(slice_view(cache, {'slice': -5})), data[0])
    numpy.testing.assert_array_equal(decode_obj(slice_view(cache, {})), data[0])


@pytest.mark.parametrize(('shape', 'thicknesses', 'n_slices', 'expected'), (
    # multislice: one thickness per slice
    ((3, 4, 5), [1.0, 2.0, 3.0], 3, [1.0, 2.0, 3.0]),
    # `execute`'s 2D normalization: leading axis of 1, empty thicknesses
    ((1, 4, 5), [], 1, None),
    # a re-used init state can carry a single thickness ("length < 2 for single slice")
    ((1, 4, 5), [7.0], 1, None),
    # a bare (y, x) object, never normalized
    ((4, 5), [], 1, None),
))
def test_obj_meta_slice_count_and_thicknesses(shape, thicknesses, n_slices, expected):
    sampling = {'shape': [4, 5], 'sampling': [1.0, 1.0], 'corner': [0.0, 0.0], 'region_min': None, 'region_max': None}
    cache = Cache()
    cache.update_raw({'object': _wire_object(
        numpy.zeros(shape, dtype=numpy.complex64),
        numpy.array(thicknesses, dtype=numpy.float32), sampling,
    )})

    out = obj_meta_view(cache, {})
    assert out['n_slices'] == n_slices
    assert out['thicknesses'] == expected
    assert out['sampling'] == sampling
    # the client's slice bound must agree with `slice_view`'s clamp
    numpy.testing.assert_array_equal(
        decode_obj(slice_view(cache, {'slice': out['n_slices'] - 1})),
        decode_obj(slice_view(cache, {'slice': 10_000})),
    )


def test_obj_meta_does_not_decode_the_bulk_array():
    # the whole point of the view: shape comes from the `__array_interface__` the wire form
    # already carries, so a corrupt payload is never even looked at
    sampling = {'shape': [4, 5], 'sampling': [1.0, 1.0], 'corner': [0.0, 0.0], 'region_min': None, 'region_max': None}
    wire = _wire_object(numpy.zeros((3, 4, 5), dtype=numpy.complex64),
                        numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32), sampling)
    wire['data']['data'] = 'not base64 at all!!'

    cache = Cache()
    cache.update_raw({'object': wire})

    assert obj_meta_view(cache, {})['n_slices'] == 3


def _wire_probe(data: numpy.ndarray, sampling: dict) -> dict:
    return encode_obj({'sampling': sampling, 'data': data})


def test_probe_meta_reports_mode_count():
    sampling = {'shape': [4, 5], 'extent': [4.0, 5.0], 'sampling': [1.0, 1.0]}
    cache = Cache()
    cache.update_raw({
        'probe': _wire_probe(numpy.zeros((3, 4, 5), dtype=numpy.complex64), sampling),
        'wavelength': 0.0251,
    })

    assert probe_meta_view(cache, {}) == {'sampling': sampling, 'nprobes': 3, 'wavelength': 0.0251}


def test_meta_views_absent_field_is_none():
    # a job with no state yet: `has_deps()` gates the snapshot, but don't blow up regardless
    assert obj_meta_view(Cache(), {}) is None
    assert probe_meta_view(Cache(), {}) is None


def test_probe_meta_without_wavelength_is_none():
    # the reciprocal view's scales need it, so a probe without one isn't yet displayable
    sampling = {'shape': [4, 5], 'extent': [4.0, 5.0], 'sampling': [1.0, 1.0]}
    cache = Cache()
    cache.update_raw({'probe': _wire_probe(numpy.zeros((3, 4, 5), dtype=numpy.complex64), sampling)})

    assert probe_meta_view(cache, {}) is None


def test_probes_recip_matches_fft_reference():
    from phaser.utils.num import fft2, fft2shift

    rng = numpy.random.default_rng(2)
    data = (rng.normal(size=(2, 8, 8)) + 1j * rng.normal(size=(2, 8, 8))).astype(numpy.complex64)
    sampling = {'shape': [8, 8], 'extent': [8.0, 8.0], 'sampling': [1.0, 1.0]}

    cache = Cache()
    cache.update_raw({'probe': _wire_probe(data, sampling)})

    out = decode_obj(probes_recip_view(cache, {}))

    assert out.shape == (2, 8, 8)
    numpy.testing.assert_allclose(out, fft2shift(fft2(data)), rtol=1e-5, atol=1e-6)


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


def test_progress_probes_are_raw_passthrough():
    cache = Cache()
    positions = encode_obj(numpy.zeros((4, 5, 2), dtype=numpy.float32))
    cache.update_raw({
        'progress': {'total_loss': {'iters': [1], 'values': [0.5]}},
        'probe': {'sampling': {}, 'data': 'x'},
        'scan': {'data': positions, 'initial': 'i', 'tilt': 'ti', 'meta': {}},
    })

    assert VIEWS['progress'].compute(cache, {}) == {'total_loss': {'iters': [1], 'values': [0.5]}}
    # the bulk array itself, not the wrapper -- and the same object, not a copy
    assert VIEWS['probes'].compute(cache, {}) is cache.raw['probe']['data']
    # `ScanState.data` only, in whatever (..., 2) shape the worker sent; the client flattens
    assert VIEWS['positions'].compute(cache, {}) is positions
    assert VIEWS['positions'].deps == frozenset({'scan'})


def test_positions_absent_field_is_none():
    assert VIEWS['positions'].compute(Cache(), {}) is None


def test_probe_sum_matches_numpy_reference():
    from phaser.utils.num import abs2, fft2, fft2shift

    rng = numpy.random.default_rng(3)
    data = (rng.normal(size=(3, 8, 8)) + 1j * rng.normal(size=(3, 8, 8))).astype(numpy.complex64)
    sampling = {'shape': [8, 8], 'extent': [8.0, 8.0], 'sampling': [1.0, 1.0]}

    cache = Cache()
    cache.update_raw({'probe': _wire_probe(data, sampling)})

    real = decode_obj(VIEWS['probe_sum'].compute(cache, {}))
    recip = decode_obj(VIEWS['probe_sum_recip'].compute(cache, {}))

    assert real.shape == recip.shape == (8, 8)
    numpy.testing.assert_allclose(real, numpy.sum(abs2(data), axis=0), rtol=1e-5, atol=1e-6)
    numpy.testing.assert_allclose(recip, numpy.sum(abs2(fft2shift(fft2(data))), axis=0), rtol=1e-5, atol=1e-6)
    # `fft2` is unitary, so the incoherent total is conserved between the two
    numpy.testing.assert_allclose(numpy.sum(real), numpy.sum(recip), rtol=1e-4)


def test_views_deps_are_disjoint_from_unrelated_fields():
    # sanity check on the registry: 'obj_phase_sum'/'obj' only fire on 'object' changes,
    # not on unrelated fields like 'iter' or 'progress' (laziness/dirty-set correctness).
    assert VIEWS['obj_phase_sum'].deps == frozenset({'object'})
    assert VIEWS['obj'].deps == frozenset({'object'})
    assert VIEWS['obj_meta'].deps == frozenset({'object'})
    assert VIEWS['state'].deps == frozenset({'state'})
    assert VIEWS['progress'].deps == frozenset({'progress'})
    assert VIEWS['probes'].deps == frozenset({'probe'})
    assert VIEWS['probes_recip'].deps == frozenset({'probe'})
    assert VIEWS['probe_meta'].deps == frozenset({'probe', 'wavelength'})
    # a meta topic and its bulk topic share a dep, so one `publish_dirty` recomputes both
    # and they reach the client in a single `update` batch -- see `Broker.publish_dirty`.
    # `probe_meta` also reads `wavelength`, which the worker sends alongside the probe.
    assert VIEWS['obj_meta'].deps == VIEWS['obj'].deps
    assert VIEWS['probes'].deps <= VIEWS['probe_meta'].deps
