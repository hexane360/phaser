import logging
import typing as t
from functools import partial

import numpy
from numpy.typing import NDArray
from typing_extensions import Self

from phaser.hooks import EngineArgs
from phaser.hooks.regularization import CostRegularizer, GroupConstraint
from phaser.hooks.solver import GradientSolver, NoiseModel
from phaser.observer import Observer
from phaser.plan import GradientEnginePlan
from phaser.state import ProgressState, ReconsState
from phaser.types import ReconsVar, process_flag
from phaser.utils import tree
from phaser.utils.image import PreparedOTF, PreparedPSF
from phaser.utils.num import (
    Float,
    abs2,
    assert_dtype,
    at,
    block_until_ready,
    cast_array_module,
    fft2,
    get_array_module,
    ifft2,
    ifft2shift,
    jit,
    to_complex_dtype,
    to_numpy,
    to_real_dtype,
)
from phaser.utils.optics import fourier_shift_filter

from ..common.simulation import (
    GroupManager,
    make_propagators,
    prepare_mtf,
    slice_forwards,
    stream_patterns,
    tilt_propagators,
)
from .synthetic import SYNTHETIC_GRADS, SYNTHETIC_VARS, SyntheticGrad, source_var

logger = logging.getLogger(__name__)

# variables whose gradients are accumulated across groups, then solved once per iteration
_PER_ITER_REAL_VARS: t.FrozenSet[ReconsVar] = frozenset({'positions', 'tilt'})
# keys to paths which should be sliced with the group
_PER_ITER_PATHS: t.FrozenSet[str] = frozenset({'initial'}) | _PER_ITER_REAL_VARS
# variables a per-iteration solver can handle
_PER_ITER_VARS: t.FrozenSet[ReconsVar] = _PER_ITER_REAL_VARS | SYNTHETIC_VARS


def process_solvers(
    plan: GradientEnginePlan
) -> t.Tuple[t.FrozenSet[ReconsVar], t.Sequence[GradientSolver[t.Any]], t.Sequence[GradientSolver[t.Any]]]:
    # process solvers, and split into per-group and per-iter solvers
    solvers = plan.solvers

    seen: t.Set[ReconsVar] = set()
    duplicate: t.Set[ReconsVar] = set()

    group_solvers: t.List[GradientSolver[t.Any]] = []
    iter_solvers: t.List[GradientSolver[t.Any]] = []

    for (vars, solver) in solvers.items():
        if len(vars) == 0:
            continue

        duplicate |= vars & seen
        seen |= vars

        if vars <= _PER_ITER_VARS:
            iter_solvers.append(solver({'plan': plan, 'params': vars}))
            continue

        if len(vars & _PER_ITER_VARS):
            # TODO: is it easier to just split the solver here?
            raise ValueError(f"The same solver can't handle both per-iteration "
                             f"({', '.join(map(repr, vars & _PER_ITER_VARS))}) and per-group "
                             f"({', '.join(map(repr, vars - _PER_ITER_VARS))}) variables")

        group_solvers.append(solver({'plan': plan, 'params': vars}))

    if len(duplicate):
        raise ValueError(f"Duplicate solvers for variable(s) {', '.join(map(repr, duplicate))}.")

    return (
        frozenset[ReconsVar](seen), tuple(group_solvers), tuple(iter_solvers)
    )


_PATH_MAP: t.Dict[t.Tuple[str, ...], str] = {
    ('object', 'data'): 'object',
    ('probe', 'data'): 'probe',
    ('scan', 'data'): 'positions',
    ('scan', 'tilt'): 'tilt',
    ('scan', 'initial'): 'initial',  # not a solver variable, but need to apply group indexing
}


def _normalize_path(path: t.Tuple[tree.GetAttrKey, ...]) -> t.Tuple[str, ...]:
    return tuple(p.name for p in path)


def extract_vars(state: ReconsState, vars: t.AbstractSet[ReconsVar], group: t.Optional[NDArray[numpy.integer]] = None) -> t.Tuple[t.Dict[ReconsVar, t.Any], ReconsState]:
    d = {}

    def f(path: t.Tuple[tree.GetAttrKey, ...], val: t.Any):
        if (var := _PATH_MAP.get(_normalize_path(path))) and var in vars:
            d[var] = val[tuple(group)] if var in _PER_ITER_REAL_VARS and group is not None else val
            return None
        return val

    state = tree.map_with_path(f, state, is_leaf=lambda x: x is None)
    return (d, state)


def extract_params(state: ReconsState, vars: t.AbstractSet[ReconsVar]) -> t.Dict[ReconsVar, t.Any]:
    """Parameter arrays for `vars`, resolving synthetic variables to where they're synthesized from."""
    params = extract_vars(state, frozenset[ReconsVar](map(source_var, vars)))[0]
    return {var: params[source_var(var)] for var in vars}


def insert_vars(vars: t.Dict[ReconsVar, t.Any], state: ReconsState, group: t.Optional[NDArray[numpy.integer]] = None) -> ReconsState:
    def f(path: t.Tuple[tree.GetAttrKey, ...], val: t.Any):
        if (var := _PATH_MAP.get(_normalize_path(path))):
            if var in vars:
                return vars[var]
            if var in _PER_ITER_PATHS and val is not None and group is not None:
                return val[tuple(group)]
        return val

    return tree.map_with_path(f, state, is_leaf=lambda x: x is None)


def apply_update(state: ReconsState, update: t.Dict[ReconsVar, numpy.ndarray]) -> ReconsState:
    for (var, val) in update.items():
        match source_var(var):
            case 'probe':
                state.probe.data += val
            case 'object':
                state.object.data += val
            case 'tilt':
                state.scan.tilt += val
            case 'positions':
                state.scan.data += val

    return state


def center_pos_updates(update: t.Dict[ReconsVar, t.Any]) -> t.Dict[ReconsVar, t.Any]:
    """Updates with the mean of each position update removed."""
    def f(val: t.Any) -> t.Any:
        xp = get_array_module(val)
        return val - xp.mean(val, tuple(range(val.ndim - 1)))

    return {
        var: f(val) if source_var(var) == 'positions' else val
        for (var, val) in update.items()
    }


def filter_vars(d: t.Dict[ReconsVar, t.Any], vars: t.AbstractSet[ReconsVar]) -> t.Dict[ReconsVar, t.Any]:
    return {k: v for (k, v) in d.items() if k in vars}


def project_grads(
    synthetics: t.Dict[ReconsVar, t.Tuple[SyntheticGrad[t.Any], t.Any]],
    state: ReconsState,
    d: t.Dict[ReconsVar, t.Any],
) -> t.Dict[ReconsVar, t.Any]:
    """`d` with grads (or updates) of synthetic variables projected"""
    def f(var: ReconsVar, val: t.Any) -> t.Any:
        if (synthetic := synthetics.get(var)) is None:
            return val
        return synthetic[0].project(state, synthetic[1], val)

    return {var: f(var, val) for (var, val) in d.items()}


def synthesize_grads(
    synthetics: t.Dict[ReconsVar, t.Tuple[SyntheticGrad[t.Any], t.Any]],
    state: ReconsState,
    grads: t.Dict[ReconsVar, t.Any],
) -> t.Dict[ReconsVar, t.Any]:
    """`grads`, plus grads for synthetic vars"""
    return project_grads(synthetics, state, {
        **grads,
        **{var: grads[source_var(var)] for var in synthetics if source_var(var) in grads},
    })


_UPDATE_RMS_KEYS: t.Dict[ReconsVar, str] = {
    'positions': 'pos_update_rms',
    'tilt': 'tilt_update_rms',
}

def update_rms_key(var: ReconsVar) -> t.Optional[str]:
    """Progress key holding the RMS update to `var`, or `None` if `var` isn't tracked."""
    if source_var(var) not in {'positions', 'tilt'}:
        return None
    return _UPDATE_RMS_KEYS.get(var, f'{var}_update_rms')


@tree.tree_dataclass
class SolverStates:
    noise_model_state: t.Any
    group_solver_states: t.List[t.Any]
    regularizer_states: t.List[t.Any]
    group_constraint_states: t.List[t.Any]

    @classmethod
    def init_state(
        cls, sim: ReconsState, xp: t.Any,
        noise_model: NoiseModel,
        group_solvers: t.Iterable[GradientSolver[t.Any]],
        regularizers: t.Iterable[CostRegularizer[t.Any]],
        group_constraints: t.Iterable[GroupConstraint[t.Any]],
    ) -> Self:
        noise_model_state = noise_model.init_state(sim)
        group_solver_states = [solver.init_state(sim) for solver in group_solvers]
        regularizer_states = [reg.init_state(sim) for reg in regularizers]
        group_constraint_states = [reg.init_state(sim) for reg in group_constraints]

        return cls(
            noise_model_state, group_solver_states, regularizer_states, group_constraint_states
        )


def run_engine(args: EngineArgs, props: GradientEnginePlan) -> ReconsState:
    #jax.config.update('jax_traceback_filtering', 'off')
    xp = cast_array_module(args['xp'])
    dtype = args['dtype']
    cdtype = to_complex_dtype(dtype)
    observer: Observer = args.get('observer', Observer())
    state = args['state']
    seed = args['seed']
    # default to 10 slices
    jit_unroll_slices = 10 if props.jit_unroll_slices is None else props.jit_unroll_slices

    noise_model = props.noise_model(None)

    (all_vars, group_solvers, iter_solvers) = process_solvers(props)
    # real variables we need gradients for
    grad_vars = frozenset[ReconsVar](map(source_var, all_vars))

    regularizers = tuple(reg(None) for reg in props.regularizers)
    group_constraints = tuple(reg(None) for reg in props.group_constraints)
    iter_constraints = tuple(reg(None) for reg in props.iter_constraints)

    flags = {
        'probe': process_flag(props.update_probe),
        'object': process_flag(props.update_object),
        'positions': process_flag(props.update_positions),
        'tilt': process_flag(props.update_tilt),
    }
    # shuffle_groups defaults to True for sparse groups, False for compact groups
    shuffle_groups = process_flag(props.shuffle_groups or not props.compact)
    groups = GroupManager(state.scan.data, props.grouping, props.compact, seed)

    observer.init_engine(
        state, recons_name=args['recons_name'],
        plan=props, noise_model=noise_model.name(),
    )
    start_i = int(state.iter.total_iter)

    # check patterns dtype
    assert_dtype(args['data'].patterns, dtype)
    assert_dtype(args['data'].pattern_mask, dtype)
    # load pattern mask
    pattern_mask = xp.asarray(args['data'].pattern_mask)

    # and load/stream patterns
    if props.buffer_n_groups is None:
        logging.info("Loading raw data to GPU ('buffer_n_groups' is disabled)...")
        patterns = xp.asarray(args['data'].patterns)
    else:
        logging.info(f"Streaming raw data to GPU (buffering {props.buffer_n_groups} groups)")
        patterns = args['data'].patterns

    def iter_patterns(groups: t.Iterable[NDArray[numpy.int_]]) -> t.Iterable[t.Tuple[NDArray[numpy.int_], NDArray[numpy.floating]]]:
        if props.buffer_n_groups is None:
            return ((group, patterns[tuple(xp.asarray(group))]) for group in groups)
        return stream_patterns(
            groups, patterns, xp=xp, buf_n=props.buffer_n_groups
        )

    propagators = make_propagators(state, props.bwlim_frac)
    mtf = None if props.mtf is None else prepare_mtf(props.mtf, state, dtype, xp)

    # runs rescaling
    rescale_factors = []
    for (group_i, (group, group_patterns)) in enumerate(iter_patterns(groups.iter(state.scan.data))):
        group_rescale_factors = dry_run(
            state, group, propagators, group_patterns,
            xp=xp, dtype=dtype,
        )
        rescale_factors.append(group_rescale_factors)

    rescale_factors = xp.concatenate(rescale_factors, axis=0)
    rescale_factor = xp.mean(rescale_factors)

    logger.info("Pre-calculated intensities")
    logger.info(f"Rescaling initial probe intensity by {float(rescale_factor):.2e}")
    state.probe.data *= xp.sqrt(rescale_factor)
    probe_int = xp.sum(abs2(state.probe.data))

    observer.start_engine(state)

    solver_states = SolverStates.init_state(state, xp, noise_model, group_solvers, regularizers, group_constraints)
    iter_solver_states = [solver.init_state(state) for solver in iter_solvers]
    iter_constraint_states = [reg.init_state(state) for reg in iter_constraints]
    synthetic_states: t.Dict[ReconsVar, t.Tuple[SyntheticGrad[t.Any], t.Any]] = {
        var: (synthetic := SYNTHETIC_GRADS[var](), synthetic.init_state(state))
        for var in all_vars & SYNTHETIC_VARS
    }

    loss_keys = (
        'detector_loss', 'total_loss', *(reg.name() for reg in regularizers),
    )
    other_keys = t.cast(tuple[ReconsVar], tuple(filter(lambda v: v is not None, (
        update_rms_key(var) for var in sorted(all_vars & _PER_ITER_VARS)
    ))))

    # populate missing keys in progress dictionary
    for k in (*loss_keys, *other_keys):
        if k not in state.progress:
            state.progress[k] = ProgressState()

    # progress gets clobbered by the jits, so we keep track of it manually
    progress = state.progress

    for i in range(1, props.niter+1):
        state.iter.engine_iter = i
        state.iter.total_iter = start_i + i

        # mask vars we're updating this iteration
        iter_vars = frozenset[ReconsVar](
            var for var in grad_vars
            if flags[var]({'state': state, 'niter': props.niter})
        )
        # gradients for per-iteration solvers
        iter_grads = tree.zeros_like(extract_vars(state, iter_vars & _PER_ITER_REAL_VARS)[0])
        # whether to shuffle groups this iteration
        iter_shuffle_groups = shuffle_groups({'state': state, 'niter': props.niter})

        # accumulated losses across groups
        losses_gpu = {k: t.cast(numpy.floating, xp.array(0.0)) for k in loss_keys}

        # update schedules for this iteration
        # this needs to be done outside the JIT context, which makes this kinda hacky
        solver_states.group_solver_states = [
            solver.update_for_iter(state, solver_state, props.niter)
            for (solver, solver_state) in zip(group_solvers, solver_states.group_solver_states)
        ]
        iter_solver_states = [
            solver.update_for_iter(state, solver_state, props.niter)
            for (solver, solver_state) in zip(iter_solvers, iter_solver_states)
        ]

        for (group_i, (group, group_patterns)) in enumerate(iter_patterns(groups.iter(state.scan.data, i, iter_shuffle_groups))):
            # prevent the loop running ahead of the GPU stream
            block_until_ready(losses_gpu['total_loss'])

            (state, losses_gpu, iter_grads, solver_states) = run_group(
                state, group=group, vars=iter_vars,
                noise_model=noise_model,
                group_solvers=group_solvers,
                group_constraints=group_constraints,
                regularizers=regularizers,
                losses=losses_gpu,
                iter_grads=iter_grads,
                solver_states=solver_states,
                props=propagators,
                mtf=mtf,
                group_patterns=group_patterns, #load_group(group),
                pattern_mask=pattern_mask,
                probe_int=probe_int,
                xp=xp, dtype=dtype,
                jit_unroll_slices=jit_unroll_slices,
            )
            if props.check_every_group and not numpy.isfinite(float(losses_gpu['total_loss'])):
                raise ValueError(f"NaN or inf encountered, group {group_i}")
            observer.update_group(state, props.send_every_group)

        if not numpy.isfinite(float(losses_gpu['total_loss'])):
            raise ValueError(f"NaN or inf encountered, iteration {i}")

        # report losses normalized by # of probe positions
        # this also moves losses to CPU
        losses: t.Dict[str, float] = tree.map(lambda v: float(v) / groups.n_pos, losses_gpu)
        for (k, v) in losses.items():
            progress[k].iters.append(i + start_i)
            progress[k].values.append(v)

        # synthesize the gradients projected from the per-iteration gradients
        grads = synthesize_grads(synthetic_states, state, iter_grads)

        # update per-iteration solvers
        for (sol_i, solver) in enumerate(iter_solvers):
            solver_grads = filter_vars(grads, solver.params)
            if len(solver_grads) == 0:
                continue
            (update, iter_solver_states[sol_i]) = solver.update(
                state, iter_solver_states[sol_i], solver_grads, losses['total_loss']
            )
            # solvers may take a synthetic update out of its subspace
            update = project_grads(synthetic_states, state, update)
            # make sure position updates are centered
            update = center_pos_updates(update)

            for (var, val) in update.items():
                key = update_rms_key(var)
                if not key:
                    continue

                update_rms = float(xp.mean(xp.linalg.norm(val, axis=-1)))
                progress[key].iters.append(i + start_i)
                progress[key].values.append(update_rms)

                if source_var(var) == 'tilt':
                    # signed per-axis mean, [y, x]
                    [y_mean, x_mean] = map(float, to_numpy(xp.mean(val, tuple(range(val.ndim - 1)))))
                    logger.info(f"{var} update: mean [{y_mean:5.3f}, {x_mean:5.3f}] mrad, {update_rms:5.3f} mrad RMS")
                else:  # source_var(var) == 'positions'
                    logger.info(f"{var} update: {update_rms:5.3f} A RMS")

            state = apply_update(state, update)

        for (reg_i, reg) in enumerate(iter_constraints):
            (state, iter_constraint_states[reg_i]) = reg.apply_iter(
                state, iter_constraint_states[reg_i]
            )

        assert_dtype(state.object.data, cdtype)
        assert_dtype(state.probe.data, cdtype)

        if 'positions' in iter_vars:
            # check positions are at least overlapping object
            state.object.sampling.check_scan(state.scan.data, state.probe.sampling.extent / 2.)
            assert_dtype(state.scan.data, dtype)

        state.progress = progress
        observer.update_iteration(state, i, props.niter, losses)

    observer.finish_engine(state)
    return state


@partial(
    jit,
    static_argnames=('vars', 'xp', 'dtype', 'noise_model', 'group_solvers', 'group_constraints', 'regularizers', 'jit_unroll_slices'),
    donate_argnames=('state', 'iter_grads', 'solver_states'),
)
def run_group(
    state: ReconsState,
    group: NDArray[numpy.integer],
    vars: t.AbstractSet[ReconsVar], *,
    noise_model: NoiseModel[t.Any],
    group_solvers: t.Sequence[GradientSolver[t.Any]],
    group_constraints: t.Sequence[GroupConstraint[t.Any]],
    regularizers: t.Sequence[CostRegularizer[t.Any]],
    losses: t.Dict[str, numpy.floating],
    iter_grads: t.Dict[ReconsVar, t.Any],
    solver_states: SolverStates,
    props: t.Optional[NDArray[numpy.complexfloating]],
    mtf: t.Optional[t.Union[PreparedOTF, PreparedPSF[numpy.floating]]],
    group_patterns: NDArray[numpy.floating],
    pattern_mask: NDArray[numpy.floating],
    probe_int: t.Union[float, numpy.floating],
    xp: t.Any,
    dtype: t.Type[numpy.floating],
    jit_unroll_slices: t.Union[int, bool],
) -> t.Tuple[ReconsState, t.Dict[str, numpy.floating], t.Dict[ReconsVar, t.Any], SolverStates]:
    xp = cast_array_module(xp)

    (grad, (solver_states, group_losses)) = tree.grad(run_model, has_aux=True, xp=xp, sign=-1)(
        *extract_vars(state, vars, group),
        group=group, props=props, mtf=mtf, group_patterns=group_patterns, pattern_mask=pattern_mask,
        noise_model=noise_model, regularizers=regularizers, solver_states=solver_states,
        xp=xp, dtype=dtype, jit_unroll_slices=jit_unroll_slices
    )
    for k in grad:
        # scale gradients appropriately
        # per-pattern variables are normalized by the grouping `group.shape[-1]`
        # Additionally, all gradients except the probe should be normalized by probe intensity
        grad[k] /= xp.array(
            (1.0 if k in _PER_ITER_REAL_VARS else group.shape[-1]) * (1.0 if k == 'probe' else probe_int),
            dtype=dtype
        )

    # update iter grads at group
    iter_grads = tree.map(lambda v1, v2: at(v1, tuple(group)).set(v2), iter_grads, filter_vars(grad, vars & _PER_ITER_REAL_VARS))

    for (sol_i, solver) in enumerate(group_solvers):
        solver_grads = filter_vars(grad, solver.params)
        if len(solver_grads) == 0:
            continue
        (update, solver_states.group_solver_states[sol_i]) = solver.update(
            state, solver_states.group_solver_states[sol_i], solver_grads, group_losses['total_loss']
        )
        state = apply_update(state, update)

    for (reg_i, reg) in enumerate(group_constraints):
        (state, solver_states.group_constraint_states[reg_i]) = reg.apply_group(
            group, state, solver_states.group_constraint_states[reg_i]
        )

    losses = tree.map(xp.add, losses, group_losses)
    return (state, losses, iter_grads, solver_states)


@partial(
    jit,
    static_argnames=('xp', 'dtype', 'noise_model', 'regularizers', 'jit_unroll_slices'),
    donate_argnames=('solver_states',),
)
def run_model(
    vars: t.Dict[ReconsVar, t.Any],
    sim: ReconsState,
    group: NDArray[numpy.integer],
    props: t.Optional[NDArray[numpy.complexfloating]], # base propagator, shape (n_slices-1, ny, nx)
    mtf: t.Optional[t.Union[PreparedOTF, PreparedPSF[numpy.floating]]],
    group_patterns: NDArray[numpy.floating],
    pattern_mask: NDArray[numpy.floating],
    noise_model: NoiseModel[t.Any],
    regularizers: t.Sequence[CostRegularizer[t.Any]],
    solver_states: SolverStates,
    xp: t.Any,
    dtype: t.Type[numpy.floating],
    jit_unroll_slices: t.Union[int, bool],
) -> t.Tuple[Float, t.Tuple[SolverStates, t.Dict[str, Float]]]:
    # apply vars to simulation
    sim = insert_vars(vars, sim, group)
    group_scan = sim.scan.data
    group_tilts = sim.scan.tilt

    (ky, kx) = sim.probe.sampling.recip_grid(dtype=dtype, xp=xp)
    xp = get_array_module(sim.probe.data)
    dtype = to_real_dtype(sim.probe.data.dtype)

    # preshift probe and object
    probes = ifft2shift(sim.probe.data)
    group_obj = ifft2shift(sim.object.sampling.get_view_at_pos(sim.object.data, group_scan, probes.shape[-2:]))
    group_subpx_filters = fourier_shift_filter(ky, kx, sim.object.sampling.get_subpx_shifts(group_scan, probes.shape[-2:]))
    # (group, mode, y, x)
    probes = ifft2(fft2(probes, shift=False) * group_subpx_filters[:, None], shift=False)

    def sim_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], psi):
        if prop is not None:
            return ifft2(fft2(psi * group_obj[:, slice_i, None], shift=False) * prop[:, None], shift=False)
        return psi * group_obj[:, slice_i, None]

    t_props = tilt_propagators(ky, kx, sim, props, group_tilts)
    model_wave = fft2(slice_forwards(t_props, probes, sim_slice, jit_unroll_slices=jit_unroll_slices), shift=False)

    model_intensity = xp.sum(abs2(model_wave), axis=1)
    if mtf is not None:
        model_intensity = mtf(model_intensity)
    (loss, solver_states.noise_model_state) = noise_model.calc_loss(
        model_wave, model_intensity, group_patterns, pattern_mask, solver_states.noise_model_state
    )

    losses: t.Dict[str, Float] = {'detector_loss': loss}

    for (reg_i, reg) in enumerate(regularizers):
        (reg_loss, solver_states.regularizer_states[reg_i]) = reg.calc_loss_group(
            group, sim, solver_states.regularizer_states[reg_i]
        )
        losses[reg.name()] = reg_loss
        # NOT `loss += reg_loss`: on torch, `+=` on a tensor is an in-place `add_`,
        # which would also mutate `losses['detector_loss']` (line above) since it's
        # the same tensor object, silently turning the per-term breakdown into a
        # running total instead of the detector-only loss.
        loss = loss + reg_loss

    losses['total_loss'] = loss

    return (loss, (solver_states, losses))


# TODO: DRY
@partial(
    jit,
    static_argnames=('xp', 'dtype'),
)
def dry_run(
    sim: ReconsState,
    group: NDArray[numpy.integer],
    props: t.Optional[NDArray[numpy.complexfloating]],
    group_patterns: NDArray[numpy.floating],
    xp: t.Any,
    dtype: t.Type[numpy.floating],
) -> NDArray[numpy.floating]:
    (ky, kx) = sim.probe.sampling.recip_grid(dtype=dtype, xp=xp)
    group_scan = sim.scan.data[tuple(group)]
    group_tilt = sim.scan.tilt[tuple(group)] if sim.scan.tilt is not None else None

    probes = ifft2shift(sim.probe.data)
    group_obj = ifft2shift(sim.object.sampling.get_view_at_pos(sim.object.data, group_scan, probes.shape[-2:]))
    group_subpx_filters = fourier_shift_filter(ky, kx, sim.object.sampling.get_subpx_shifts(group_scan, probes.shape[-2:]))
    probes = ifft2(fft2(probes, shift=False) * group_subpx_filters[:, None], shift=False)

    def sim_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], psi):
        if prop is not None:
            return ifft2(fft2(psi * group_obj[:, slice_i, None], shift=False) * prop[:, None], shift=False)
        return psi * group_obj[:, slice_i, None]

    t_props = tilt_propagators(ky, kx, sim, props, group_tilt)
    model_wave = fft2(slice_forwards(t_props, probes, sim_slice), shift=False)
    model_intensity = xp.sum(abs2(model_wave), axis=(1, -2, -1))
    exp_intensity = xp.sum(group_patterns, axis=(-2, -1))

    return exp_intensity / model_intensity
