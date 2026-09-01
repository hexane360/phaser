import numpy
import pytest
from numpy.testing import assert_array_equal

from phaser.execute import execute_engine, initialize_reconstruction
from phaser.hooks import RawData
from phaser.plan import ReconsPlan
from phaser.state import ScanState
from phaser.utils.num import Sampling, get_backend_module

from .utils import make_recons_state

SCAN_SHAPE = (8, 8)
DET_SHAPE = (32, 32)


def load_random(args, props) -> RawData:
    rng = numpy.random.default_rng(0x9E3D)

    return {
        'patterns': rng.random((*SCAN_SHAPE, *DET_SHAPE), dtype=numpy.float32) + 1e-3,
        'mask': numpy.ones(DET_SHAPE, dtype=numpy.float32),
        'sampling': Sampling(DET_SHAPE, sampling=(1.0, 1.0)),
        'wavelength': 0.0251,
        'scan_hook': {'type': 'raster', 'shape': SCAN_SHAPE, 'step_size': (0.6, 0.6)},
        'probe_hook': {'type': 'focused', 'conv_angle': 25.0, 'defocus': 0.0},
        'seed': None,
    }


def test_position_update_preserves_initial():
    plan = ReconsPlan.from_data({
        'name': 'test',
        'backend': 'numpy',
        'raw_data': 'tests.test_engines:load_random',
        'engines': [{
            'type': 'conventional',
            'probe_modes': 1,
            'niter': 2,
            'grouping': 16,
            'noise_model': {'type': 'amplitude'},
            'solver': {'type': 'lsqml'},
            'position_solver': {'type': 'momentum', 'step_size': 1e-2},
            'update_positions': True,
            'iter_constraints': [],
            'group_constraints': [],
        }],
    })

    recons = initialize_reconstruction(plan)
    initial = recons.state.scan.initial.copy()

    for engine in plan.engines:
        recons = execute_engine(recons, engine)

    scan = recons.state.scan
    assert not numpy.allclose(scan.data, initial)
    assert_array_equal(scan.initial, initial)


@pytest.mark.parametrize('solver', ['lsqml', 'epie'])
def test_conventional_mtf(solver: str):
    def make_plan(mtf) -> ReconsPlan:
        return ReconsPlan.from_data({
            'name': 'test',
            'backend': 'numpy',
            'raw_data': 'tests.test_engines:load_random',
            'engines': [{
                'type': 'conventional',
                'probe_modes': 1,
                'niter': 2,
                'grouping': 16,
                'noise_model': {'type': 'amplitude'},
                'solver': {'type': solver},
                'iter_constraints': [],
                'group_constraints': [],
                'mtf': mtf,
            }],
        })

    def run(mtf):
        plan = make_plan(mtf)
        recons = initialize_reconstruction(plan)
        for engine in plan.engines:
            recons = execute_engine(recons, engine)
        return recons.state

    state_no_mtf = run(None)
    assert numpy.all(numpy.isfinite(numpy.asarray(state_no_mtf.progress['detector_loss'].values)))

    state_mtf = run({'filter': {'type': 'gaussian', 'sigma': 1.0}, 'domain': 'recip'})
    assert numpy.all(numpy.isfinite(numpy.asarray(state_mtf.progress['detector_loss'].values)))

    assert not numpy.allclose(state_no_mtf.object.data, state_mtf.object.data)


@pytest.mark.jax
def test_gradient_mtf():
    try:
        get_backend_module('jax')
    except ValueError as e:
        pytest.skip(str(e))

    def make_plan(mtf) -> ReconsPlan:
        return ReconsPlan.from_data({
            'name': 'test',
            'backend': 'jax',
            'raw_data': 'tests.test_engines:load_random',
            'engines': [{
                'type': 'gradient',
                'probe_modes': 1,
                'niter': 2,
                'grouping': 16,
                'noise_model': {'type': 'amplitude'},
                'solvers': {('object', 'probe'): {'type': 'sgd', 'learning_rate': 1e-2}},
                'regularizers': [],
                'iter_constraints': [],
                'group_constraints': [],
                'mtf': mtf,
            }],
        })

    def run(mtf):
        plan = make_plan(mtf)
        recons = initialize_reconstruction(plan)
        for engine in plan.engines:
            recons = execute_engine(recons, engine)
        return recons.state

    state_no_mtf = run(None)
    assert numpy.all(numpy.isfinite(numpy.asarray(state_no_mtf.progress['detector_loss'].values)))

    state_mtf = run({'filter': {'type': 'gaussian', 'sigma': 1.0}, 'domain': 'recip'})
    assert numpy.all(numpy.isfinite(numpy.asarray(state_mtf.progress['detector_loss'].values)))

    assert not numpy.allclose(state_no_mtf.object.data, state_mtf.object.data)

    # bare FilterHook shorthand (no `domain` override) should also work
    state_mtf_bare = run({'type': 'gaussian', 'sigma': 1.0})
    assert numpy.all(numpy.isfinite(numpy.asarray(state_mtf_bare.progress['detector_loss'].values)))


@pytest.mark.jax
def test_gradient_group_indexing():
    try:
        get_backend_module('jax')
    except ValueError as e:
        pytest.skip(str(e))

    from phaser.engines.gradient.run import extract_vars, insert_vars

    flat = numpy.arange(32.).reshape(16, 2)
    scan = ScanState(flat, flat + 100., flat * 0.01)
    group = numpy.array([[1, 3, 5, 7]])

    (vars, stripped) = extract_vars(make_recons_state(scan), {'positions'}, group)
    assert_array_equal(vars['positions'], flat[[1, 3, 5, 7]])

    grouped = insert_vars(vars, stripped, group)

    # `initial` and `tilt` are indexed alongside `data`
    assert_array_equal(grouped.scan.data, flat[[1, 3, 5, 7]])
    assert_array_equal(grouped.scan.initial, flat[[1, 3, 5, 7]] + 100.)
    assert_array_equal(grouped.scan.tilt, flat[[1, 3, 5, 7]] * 0.01)
