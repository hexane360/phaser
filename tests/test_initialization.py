# type: ignore

import logging
import re
import typing as t

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

import pane
from phaser.execute import (
    _normalize_scan_shape,
    initialize_reconstruction,
    load_raw_data,
)
from phaser.hooks import DropNanProps, RasterScanProps, RawData
from phaser.hooks.preprocessing import drop_nan_patterns
from phaser.hooks.scan import raster_scan
from phaser.plan import ReconsPlan
from phaser.state import PartialReconsState, Patterns, ProbeState, ScanState
from phaser.utils.misc import freeze
from phaser.utils.num import Sampling

from .utils import make_recons_state


def load_empty(args, props) -> RawData:
    scan_shape = props['scan_shape']
    det_shape = props['det_shape']

    return {
        'patterns': numpy.zeros((*scan_shape, *det_shape), dtype=numpy.float32),
        'mask': numpy.ones(det_shape, dtype=numpy.float32),
        'sampling': Sampling(det_shape, sampling=(1.0, 1.0)),
        'wavelength': 1.0,
        'scan_hook': None,
        'probe_hook': None,
        'seed': None,
    }


def load_no_probe(args, props) -> RawData:
    return {
        'patterns': numpy.zeros((32, 32, 64, 64), dtype=numpy.float32),
        'mask': numpy.ones((64, 64), dtype=numpy.float32),
        'sampling': Sampling((64, 64), sampling=(1.0, 1.0)),
        'wavelength': 1.0,
        'scan_hook': {
            'type': 'raster',
            'shape': (32, 32),
            'step_size': (0.6, 0.6),
        },
        'probe_hook': None,
        'seed': None,
    }


def test_load_raw_data_missing():
    plan = ReconsPlan.from_data({
        'name': 'test',
        'raw_data': 'tests.test_initialization:load_no_probe',
        'engines': [],
    })
    xp = numpy

    with pytest.raises(ValueError, match=re.escape('`probe` must be specified by raw data, previous state, or manually in `init.probe`')):
        load_raw_data(plan, xp)


def test_load_raw_data_override():
    plan = {
        'name': 'test',
        'raw_data': 'tests.test_initialization:load_no_probe',
        'engines': [],
        'init': {
            'probe': {
                'type': 'focused',
                'conv_angle': 20.0,
                'defocus': 200.0,
            },
            'scan': {
                'type': 'raster',
                'step_size': (1.0, 1.0),
            }
        }
    }
    xp = numpy

    raw_data = load_raw_data(ReconsPlan.from_data(plan), xp)

    assert pane.into_data(raw_data['probe_hook']) == {  # type: ignore
        'type': 'focused',
        'conv_angle': 20.0,
        'defocus': 200.0,
        'aberrations': (),
    }

    assert pane.into_data(raw_data['scan_hook']) == {  # type: ignore
        'type': 'raster',
        'rotation': None,
        'shape': (32, 32),
        'affine': None,

        # overridden by init.scan
        'step_size': (1.0, 1.0),
    }

    plan['init']['scan'] = 'custom.package:raster2'

    raw_data = load_raw_data(ReconsPlan.from_data(plan), xp)
    # instead of merging different hooks, the new one takes precedence
    assert pane.into_data(raw_data['scan_hook']) == {'type': 'custom.package:raster2'}


def test_load_raw_data_prev_state(caplog):
    plan = {
        'name': 'test',
        'raw_data': 'tests.test_initialization:load_no_probe',
        'engines': [],
    }

    probe_state = ProbeState(Sampling((64, 64), sampling=(1.0, 1.0)), numpy.zeros((64, 64), dtype=numpy.complex64))
    scan_state = ScanState(numpy.zeros((32, 32, 2)), numpy.zeros((32, 32, 2)))

    xp = numpy
    with caplog.at_level(logging.WARNING):
        recons = initialize_reconstruction(ReconsPlan.from_data(plan), xp=xp, init_state=PartialReconsState(
            wavelength=2.0, probe=probe_state,
        ))

    assert "Wavelength of reconstruction (1.00e+00) doesn't match wavelength of previous state (2.00e+00)" in caplog.text
    assert "Mean pattern intensity is very low (0.0 particles)." in caplog.text
    assert numpy.all(numpy.isclose(recons.state.probe.data, probe_state.data))

    plan['init'] = {
        'scan': {}
    }

    recons = initialize_reconstruction(ReconsPlan.from_data(plan), xp=xp, init_state=PartialReconsState(
        wavelength=2.0, probe=probe_state, scan=scan_state
    ))

    # probe from state overrides probe from raw data
    assert numpy.all(numpy.isclose(recons.state.probe.data, probe_state.data))
    # but scan should be modeled
    assert ~numpy.all(numpy.isclose(recons.state.scan.data, scan_state.data))

    plan['init'] = {
        'scan': {},
        'probe': {
            'type': 'focused',
            'conv_angle': 25.0,
            'defocus': 200.0,
        }
    }

    recons = initialize_reconstruction(ReconsPlan.from_data(plan), xp=xp, init_state=PartialReconsState(
        wavelength=2.0, probe=probe_state, scan=scan_state
    ))

    # both should be modeled
    assert ~numpy.all(numpy.isclose(recons.state.probe.data, probe_state.data))
    assert ~numpy.all(numpy.isclose(recons.state.scan.data, scan_state.data))


def test_load_3d_raw_data():
    scan_shape = (64, 64)
    det_shape = (128, 128)

    plan = ReconsPlan.from_data({
        'name': 'test',
        'raw_data': {
            'type': 'tests.test_initialization:load_empty',
            'scan_shape': (4096,),
            'det_shape': det_shape,
        },
        'init': {
            'scan': {
                'type': 'raster',
                'shape': scan_shape,
                'step_size': (1.0, 1.0),
            },
            'probe': {
                'type': 'focused',
                'conv_angle': 20.0,
                'defocus': 300.0,
            }
        },
        'engines': [],
    })
    recons = initialize_reconstruction(plan)

    assert recons.state.scan.data.shape == (*scan_shape, 2)
    assert recons.patterns.patterns.shape == (*scan_shape, *det_shape)

def _plan_with_scan(**init: object) -> ReconsPlan:
    return ReconsPlan.from_data({
        'name': 'test',
        'raw_data': {
            'type': 'tests.test_initialization:load_empty',
            'scan_shape': (8, 8),
            'det_shape': (32, 32),
        },
        'engines': [],
        'init': init,
    })


def test_initialize_scan_initial_independent():
    plan = _plan_with_scan(
        scan={'type': 'raster', 'shape': (8, 8), 'step_size': (1., 1.)},
        probe={'type': 'focused', 'conv_angle': 20.0, 'defocus': 100.0},
    )
    scan = initialize_reconstruction(plan, xp=numpy).state.scan

    assert not numpy.shares_memory(scan.data, scan.initial)

    initial = scan.initial.copy()
    scan.data += 1.
    assert numpy.array_equal(scan.initial, initial)


def test_initialize_reuses_scan_state():
    plan = _plan_with_scan()

    probe = ProbeState(
        Sampling((32, 32), sampling=(1.0, 1.0)),
        numpy.zeros((1, 32, 32), dtype=numpy.complex64),
    )
    # a previous state, as read back from disk
    prev = ScanState(
        numpy.arange(128.).reshape(8, 8, 2),
        numpy.arange(128.).reshape(8, 8, 2) + 100.,
        numpy.arange(128.).reshape(8, 8, 2) * 0.01,
    )

    scan = initialize_reconstruction(plan, xp=numpy, init_state=PartialReconsState(
        wavelength=1.0, probe=probe, scan=prev,
    )).state.scan

    # `initial` is carried over, not reset to `data`
    assert_allclose(scan.data, prev.data, rtol=1e-6)
    assert_allclose(scan.initial, prev.initial, rtol=1e-6)
    assert_allclose(scan.tilt, prev.tilt, rtol=1e-6)

    assert scan.data.dtype == scan.initial.dtype == scan.tilt.dtype == numpy.float32

    # the caller's state is left untouched
    assert prev.data.dtype == numpy.float64
    assert not numpy.shares_memory(scan.data, prev.data)
    assert not numpy.shares_memory(scan.initial, prev.initial)


@pytest.mark.parametrize('flat_scan', (True, False))
def test_normalize_scan_shape_keeps_initial(flat_scan: bool):
    flat = numpy.arange(128.).reshape(64, 2)
    shape = (64, 2) if flat_scan else (8, 8, 2)
    scan = ScanState(flat.reshape(shape), (flat + 100.).reshape(shape), (flat * 0.01).reshape(shape))

    patterns_shape = (8, 8, 4, 4) if flat_scan else (64, 4, 4)
    patterns = Patterns(
        numpy.zeros(patterns_shape, dtype=numpy.float32),
        numpy.ones((4, 4), dtype=numpy.float32),
    )

    (patterns, state) = _normalize_scan_shape(patterns, make_recons_state(scan))

    assert patterns.patterns.shape == (8, 8, 4, 4)
    assert state.scan.data.shape == (8, 8, 2)
    assert state.scan.initial.shape == (8, 8, 2)
    assert state.scan.tilt.shape == (8, 8, 2)

    assert numpy.array_equal(state.scan.initial, (flat + 100.).reshape(8, 8, 2))


def test_drop_nan_patterns_filters_initial():
    patterns = numpy.zeros((4, 4, 2, 2), dtype=numpy.float32)
    patterns[0, 0] = numpy.nan
    patterns[2, 3] = numpy.nan

    flat = numpy.arange(32.).reshape(16, 2)
    scan = ScanState(flat.reshape(4, 4, 2), flat.reshape(4, 4, 2) + 100., flat.reshape(4, 4, 2) * 0.01)

    (data, state) = drop_nan_patterns({
        'data': Patterns(patterns, numpy.ones((2, 2), dtype=numpy.float32)),
        'state': make_recons_state(scan),
        'seed': None, 'dtype': numpy.float32, 'xp': numpy,
    }, DropNanProps(threshold=0.5))

    kept = numpy.ones(16, dtype=numpy.bool_)
    kept[[0, 11]] = False

    assert data.patterns.shape == (14, 2, 2)
    assert numpy.array_equal(state.scan.data, flat[kept])
    assert numpy.array_equal(state.scan.initial, flat[kept] + 100.)
    assert numpy.array_equal(state.scan.tilt, flat[kept] * 0.01)


def _make_raster(shape: tuple[int, int], step: float = 1.) -> ScanState:
    return raster_scan(
        {'seed': None, 'dtype': numpy.float64, 'xp': numpy},
        RasterScanProps(shape=shape, step_size=(step, step)),
    )


def _grid_indices(shape: tuple[int, int]) -> NDArray[numpy.int64]:
    """Row and column index of every point of a `shape` grid, as a (..., 2) array."""
    return numpy.stack(numpy.indices(shape), axis=-1)


def _raster_grid(meta: t.Mapping[str, t.Any]) -> NDArray[numpy.int64]:
    """`raster_rows` and `raster_cols` stacked into a (..., 2) index array."""
    return numpy.stack((
        numpy.array(meta['raster_rows']), numpy.array(meta['raster_cols']),
    ), axis=-1)


def _raster_positions(grid: NDArray[numpy.int64], shape: tuple[int, int], step: float) -> NDArray[numpy.float64]:
    """Positions `make_raster_scan` assigns to the given grid indices."""
    return (grid - numpy.array(shape) / 2.) * step


def test_raster_scan_meta():
    # non-square, so a transposed grid can't pass
    shape = (3, 5)
    scan = _make_raster(shape)

    assert scan.meta['type'] == 'raster'

    grid = _raster_grid(scan.meta)
    assert grid.shape == (*shape, 2)
    assert_array_equal(grid, _grid_indices(shape))

    # each index labels the grid position of the matching scan point
    assert_allclose(scan.data, _raster_positions(grid, shape, 1.))

    # `meta` is a static pytree field, so it must be hashable
    hash(scan.meta)


@pytest.mark.parametrize('flat_meta', (True, False))
def test_normalize_scan_shape_keeps_raster_meta(flat_meta: bool):
    shape = (4, 8)
    scan = _make_raster(shape)

    if flat_meta:
        # a state resumed after `drop_nan_patterns`, which flattens scan and metadata alike
        flat = {k: numpy.array(v).ravel() if k.startswith('raster_') else v for (k, v) in scan.meta.items()}
        scan = ScanState(scan.data.reshape(-1, 2), scan.initial.reshape(-1, 2), meta=freeze(flat))

    patterns = Patterns(
        numpy.zeros((*shape, 4, 4), dtype=numpy.float32),
        numpy.ones((4, 4), dtype=numpy.float32),
    )

    (_, state) = _normalize_scan_shape(patterns, make_recons_state(scan))

    grid = _raster_grid(state.scan.meta)
    assert grid.shape == (*shape, 2)
    assert_array_equal(grid, _grid_indices(shape))
    assert_allclose(state.scan.data, _raster_positions(grid, shape, 1.))

    hash(state.scan.meta)


def test_drop_nan_patterns_keeps_raster_meta():
    shape = (4, 4)
    scan = _make_raster(shape)

    patterns = numpy.zeros((*shape, 2, 2), dtype=numpy.float32)
    patterns[0, 0] = numpy.nan
    patterns[2, 3] = numpy.nan

    (_, state) = drop_nan_patterns({
        'data': Patterns(patterns, numpy.ones((2, 2), dtype=numpy.float32)),
        'state': make_recons_state(scan),
        'seed': None, 'dtype': numpy.float32, 'xp': numpy,
    }, DropNanProps(threshold=0.5))

    kept = numpy.ones(16, dtype=numpy.bool_)
    kept[[0, 11]] = False

    # dropped positions take their indices with them, the rest keep theirs
    grid = _raster_grid(state.scan.meta)
    assert grid.shape == (14, 2)
    assert_array_equal(grid, _grid_indices(shape).reshape(-1, 2)[kept])
    assert_allclose(state.scan.data, _raster_positions(grid, shape, 1.))

    hash(state.scan.meta)
