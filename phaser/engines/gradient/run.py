import logging
from functools import partial
import typing as t

import numpy
from numpy.typing import NDArray
from typing_extensions import Self

from phaser.hooks.solver import NoiseModel
from phaser.utils.num import (
    assert_dtype, get_array_module, cast_array_module, jit, to_numpy,
    fft2, ifft2, abs2, check_finite, at, Float, to_complex_dtype, to_real_dtype
)
import phaser.utils.tree as tree
from phaser.utils.optics import fourier_shift_filter
from phaser.observer import Observer
from phaser.state import ProgressState, ReconsState
from phaser.hooks import EngineArgs
from phaser.hooks.solver import GradientSolver
from phaser.hooks.regularization import CostRegularizer, GroupConstraint
from phaser.plan import GradientEnginePlan
from phaser.types import process_flag, ReconsVar
from ..common.simulation import GroupManager, make_propagators, tilt_propagators, slice_forwards, stream_patterns


logger = logging.getLogger(__name__)
_PER_ITER_VARS: t.FrozenSet[ReconsVar] = frozenset({'positions', 'tilt'})


def process_solvers(
    plan: GradientEnginePlan
) -> t.Tuple[t.FrozenSet[ReconsVar], t.Sequence[GradientSolver[t.Any]], t.Sequence[GradientSolver[t.Any]]]:
    # process solvers, and split into per-group and per-iter solvers
    solvers = plan.solvers

    seen = set()
    duplicate = set()

    group_solvers = []
    iter_solvers = []

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
        frozenset(seen), tuple(group_solvers), tuple(iter_solvers)
    )


_PATH_MAP: t.Dict[t.Tuple[str, ...], ReconsVar] = {
    ('object', 'data'): 'object',
    ('probe', 'data'): 'probe',
    ('scan',): 'positions',
    ('tilt',): 'tilt'
}

def _normalize_path(path: t.Tuple[tree.GetAttrKey, ...]) -> t.Tuple[str, ...]:
    return tuple(p.name for p in path)

def extract_vars(state: ReconsState, vars: t.AbstractSet[ReconsVar], group: t.Optional[NDArray[numpy.integer]] = None) -> t.Tuple[t.Dict[ReconsVar, t.Any], ReconsState]:
    d = {}

    def f(path: t.Tuple[tree.GetAttrKey, ...], val: t.Any):
        if (var := _PATH_MAP.get(_normalize_path(path))) and var in vars:
            if var in _PER_ITER_VARS and group is not None:
                d[var] = val[tuple(group)]
            else:
                d[var] = val
            return None
        return val

    state = tree.map_with_path(f, state, is_leaf=lambda x: x is None)
    return (d, state)

def insert_vars(vars: t.Dict[ReconsVar, t.Any], state: ReconsState, group: t.Optional[NDArray[numpy.integer]] = None) -> ReconsState:
    def f(path: t.Tuple[tree.GetAttrKey, ...], val: t.Any):
        if (var := _PATH_MAP.get(_normalize_path(path))):
            if var in vars:
                return vars[var]
            if var in _PER_ITER_VARS and val is not None and group is not None:
                return val[tuple(group)]
        return val

    return tree.map_with_path(f, state, is_leaf=lambda x: x is None)


def apply_update(state: ReconsState, update: t.Dict[ReconsVar, numpy.ndarray]) -> ReconsState:
    if 'probe' in update:
        state.probe.data += update['probe']
    if 'object' in update:
        state.object.data += update['object']
    if 'tilt' in update:
        state.tilt += update['tilt']
    if 'positions' in update:
        # subtract mean position update
        xp = get_array_module(update['positions'])
        update['positions'] -= xp.mean(update['positions'], tuple(range(update['positions'].ndim - 1)))

        state.scan += update['positions']

    return state


def filter_vars(d: t.Dict[ReconsVar, t.Any], vars: t.AbstractSet[ReconsVar]) -> t.Dict[ReconsVar, t.Any]:
    return {k: v for (k, v) in d.items() if k in vars}


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
    patterns = args['data'].patterns
    pattern_mask = xp.array(args['data'].pattern_mask)
    assert_dtype(patterns, dtype)
    assert_dtype(pattern_mask, dtype)

    noise_model = props.noise_model(None)

    (all_vars, group_solvers, iter_solvers) = process_solvers(props)

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
    groups = GroupManager(state.scan, props.grouping, props.compact, seed)

    observer.init_engine(
        state, recons_name=args['recons_name'],
        plan=props, noise_model=noise_model.name(),
    )
    start_i = int(state.iter.total_iter)

    propagators = make_propagators(state, props.bwlim_frac)

    # runs rescaling
    rescale_factors = []
    for (group_i, (group, group_patterns)) in enumerate(stream_patterns(groups.iter(state.scan),
                                                                        patterns, xp=xp, buf_n=props.buffer_n_groups)):
        group_rescale_factors = dry_run(state, group, propagators, group_patterns, xp=xp, dtype=dtype)
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

    loss_keys = (
        'detector_loss', 'total_loss', *(reg.name() for reg in regularizers),
    )
    other_keys = (
        *(('pos_update_rms',) if 'positions' in all_vars else ()),
        *(('tilt_update_rms',) if 'tilt' in all_vars else ()),
    )

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
        iter_vars = all_vars & t.cast(t.Set[ReconsVar],
            set(k for (k, flag) in flags.items() if flag({'state': state, 'niter': props.niter}))
        )
        # gradients for per-iteration solvers
        iter_grads = tree.zeros_like(extract_vars(state, iter_vars & _PER_ITER_VARS)[0])
        # whether to shuffle groups this iteration
        iter_shuffle_groups = shuffle_groups({'state': state, 'niter': props.niter})

        # accumulated losses across groups
        losses: t.Dict[str, float] = {k: 0.0 for k in loss_keys}

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

        for (group_i, (group, group_patterns)) in enumerate(stream_patterns(groups.iter(state.scan, i, iter_shuffle_groups),
                                                                            patterns, xp=xp, buf_n=props.buffer_n_groups)):
            (state, group_losses, iter_grads, solver_states) = run_group(
                state, group=group, vars=iter_vars,
                noise_model=noise_model,
                group_solvers=group_solvers,
                group_constraints=group_constraints,
                regularizers=regularizers,
                iter_grads=iter_grads,
                solver_states=solver_states,
                props=propagators,
                group_patterns=group_patterns, #load_group(group),
                pattern_mask=pattern_mask,
                probe_int=probe_int,
                xp=xp, dtype=dtype
            )

            losses = tree.map(xp.add, losses, group_losses)

            check_finite(state.object.data, state.probe.data, context=f"object or probe, group {group_i}")
            observer.update_group(state, props.send_every_group)

        # report losses normalized by # of probe positions
        losses = tree.map(lambda v: float(v / groups.n_pos), losses)
        for (k, v) in losses.items():
            progress[k].iters.append(i + start_i)
            progress[k].values.append(v)

        # update per-iteration solvers
        for (sol_i, solver) in enumerate(iter_solvers):
            solver_grads = filter_vars(iter_grads, solver.params)
            if len(solver_grads) == 0:
                continue
            (update, iter_solver_states[sol_i]) = solver.update(
                state, iter_solver_states[sol_i], filter_vars(iter_grads, solver.params), losses['total_loss']
            )
            state = apply_update(state, update)

            if 'positions' in update:
                pos_update_rms = float(xp.mean(xp.linalg.norm(update['positions'], axis=-1)))
                progress['pos_update_rms'].iters.append(i + start_i)
                progress['pos_update_rms'].values.append(pos_update_rms)
                logger.info(f"Position update: mean {pos_update_rms}")  # RMS position update

            if 'tilt' in update:
                mean_tilt_update = to_numpy(xp.mean(xp.abs(update['tilt']), tuple(range(update['tilt'].ndim - 1))))
                tilt_update_rms = float(xp.mean(xp.linalg.norm(update['tilt'], axis=-1)))
                progress['tilt_update_rms'].iters.append(i + start_i)
                progress['tilt_update_rms'].values.append(tilt_update_rms)
                logger.info(f"Tilt update: mean {mean_tilt_update} mrad")  # average tilt update, [y, x]


        for (reg_i, reg) in enumerate(iter_constraints):
            (state, iter_constraint_states[reg_i]) = reg.apply_iter(
                state, iter_constraint_states[reg_i]
            )

        assert_dtype(state.object.data, cdtype)
        assert_dtype(state.probe.data, cdtype)

        if 'positions' in iter_vars:
            # check positions are at least overlapping object
            state.object.sampling.check_scan(state.scan, state.probe.sampling.extent / 2.)
            assert_dtype(state.scan, dtype)

        state.progress = progress
        observer.update_iteration(state, i, props.niter, losses)

    observer.finish_engine(state)
    return state


@partial(
    jit,
    static_argnames=('vars', 'xp', 'dtype', 'noise_model', 'group_solvers', 'group_constraints', 'regularizers'),
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
    iter_grads: t.Dict[ReconsVar, t.Any],
    solver_states: SolverStates,
    props: t.Optional[NDArray[numpy.complexfloating]],
    group_patterns: NDArray[numpy.floating],
    pattern_mask: NDArray[numpy.floating],
    probe_int: t.Union[float, numpy.floating],
    xp: t.Any,
    dtype: t.Type[numpy.floating],
) -> t.Tuple[ReconsState, t.Dict[str, Float], t.Dict[ReconsVar, t.Any], SolverStates]:
    xp = cast_array_module(xp)

    (grad, (solver_states, losses)) = tree.grad(run_model, has_aux=True, xp=xp, sign=-1)(
        *extract_vars(state, vars, group),
        group=group, props=props, group_patterns=group_patterns, pattern_mask=pattern_mask,
        noise_model=noise_model, regularizers=regularizers, solver_states=solver_states,
        xp=xp, dtype=dtype
    )
    for k in grad.keys():
        # scale gradients appropriately
        # per-pattern variables are normalized by the grouping `group.shape[-1]`
        # Additionally, all gradients except the probe should be normalized by probe intensity
        grad[k] /= xp.array(
            (1.0 if k in _PER_ITER_VARS else group.shape[-1]) * (1.0 if k == 'probe' else probe_int),
            dtype=dtype
        )

    # update iter grads at group
    iter_grads = tree.map(lambda v1, v2: at(v1, tuple(group)).set(v2), iter_grads, filter_vars(grad, vars & _PER_ITER_VARS))

    for (sol_i, solver) in enumerate(group_solvers):
        solver_grads = filter_vars(grad, solver.params)
        if len(solver_grads) == 0:
            continue
        (update, solver_states.group_solver_states[sol_i]) = solver.update(
            state, solver_states.group_solver_states[sol_i], solver_grads, losses['total_loss']
        )
        state = apply_update(state, update)

    for (reg_i, reg) in enumerate(group_constraints):
        (state, solver_states.group_constraint_states[reg_i]) = reg.apply_group(
            group, state, solver_states.group_constraint_states[reg_i]
        )

    return (state, losses, iter_grads, solver_states)


@partial(
    jit,
    static_argnames=('xp', 'dtype', 'noise_model', 'regularizers'),
    donate_argnames=('solver_states',),
)
def run_model(
    vars: t.Dict[ReconsVar, t.Any],
    sim: ReconsState,
    group: NDArray[numpy.integer],
    props: t.Optional[NDArray[numpy.complexfloating]], # base propagator, shape (n_slices-1, ny, nx)
    group_patterns: NDArray[numpy.floating],
    pattern_mask: NDArray[numpy.floating],
    noise_model: NoiseModel[t.Any],
    regularizers: t.Sequence[CostRegularizer[t.Any]],
    solver_states: SolverStates,
    xp: t.Any,
    dtype: t.Type[numpy.floating],
) -> t.Tuple[Float, t.Tuple[SolverStates, t.Dict[str, Float]]]:
    # apply vars to simulation
    sim = insert_vars(vars, sim, group)
    group_scan = sim.scan
    group_tilts = sim.tilt

    (ky, kx) = sim.probe.sampling.recip_grid(dtype=dtype, xp=xp)
    xp = get_array_module(sim.probe.data)
    dtype = to_real_dtype(sim.probe.data.dtype)
    #complex_dtype = to_complex_dtype(dtype)

    probes = sim.probe.data
    group_obj = sim.object.sampling.get_view_at_pos(sim.object.data, group_scan, probes.shape[-2:])
    group_subpx_filters = fourier_shift_filter(ky, kx, sim.object.sampling.get_subpx_shifts(group_scan, probes.shape[-2:]))[:, None, ...]
    probes = ifft2(fft2(probes) * group_subpx_filters)

    def sim_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], psi):
        # psi: (batch, n_probe, Ny, Nx)
        if prop is not None:
            return ifft2(fft2(psi * group_obj[:, slice_i, None]) * prop[:, None])
        return psi * group_obj[:, slice_i, None]

    t_props = tilt_propagators(ky, kx, sim, props, group_tilts)
    model_wave = fft2(slice_forwards(t_props, probes, sim_slice))
    model_intensity = xp.sum(abs2(model_wave), axis=1)

    (loss, solver_states.noise_model_state) = noise_model.calc_loss(
        model_wave, model_intensity, group_patterns, pattern_mask, solver_states.noise_model_state
    )

    losses: t.Dict[str, Float] = {'detector_loss': loss}

    for (reg_i, reg) in enumerate(regularizers):
        (reg_loss, solver_states.regularizer_states[reg_i]) = reg.calc_loss_group(
            group, sim, solver_states.regularizer_states[reg_i]
        )
        losses[reg.name()] = reg_loss
        loss += reg_loss

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

    probes = sim.probe.data
    group_obj = sim.object.sampling.get_view_at_pos(sim.object.data, sim.scan[tuple(group)], probes.shape[-2:])
    group_subpx_filters = fourier_shift_filter(ky, kx, sim.object.sampling.get_subpx_shifts(sim.scan[tuple(group)], probes.shape[-2:]))[:, None, ...]
    probes = ifft2(fft2(probes) * group_subpx_filters)

    def sim_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], psi):
        if prop is not None:
            return ifft2(fft2(psi * group_obj[:, slice_i, None]) * prop[:, None])
        return psi * group_obj[:, slice_i, None]

    t_props = tilt_propagators(ky, kx, sim, props, sim.tilt[tuple(group)] if sim.tilt is not None else None)
    model_wave = fft2(slice_forwards(t_props, probes, sim_slice))
    model_intensity = xp.sum(abs2(model_wave), axis=(1, -2, -1))
    exp_intensity = xp.sum(group_patterns, axis=(-2, -1))

    return exp_intensity / model_intensity
