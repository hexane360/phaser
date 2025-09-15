import logging

import numpy

from phaser.utils.misc import mask_fraction_of_groups
from phaser.utils.num import assert_dtype, cast_array_module, to_numpy, to_complex_dtype
from phaser.observer import Observer
from phaser.hooks import EngineArgs
from phaser.plan import ConventionalEnginePlan
from phaser.state import ReconsState
from phaser.types import process_flag
from ..common.simulation import SimulationState, make_propagators, GroupManager


def run_engine(args: EngineArgs, props: ConventionalEnginePlan) -> ReconsState:
    logger = logging.getLogger(__name__)

    xp = cast_array_module(args['xp'])
    dtype = args['dtype']
    cdtype = to_complex_dtype(dtype)
    observer: Observer = args.get('observer', Observer())
    seed = args['seed']

    noise_model = props.noise_model(None)
    group_constraints = tuple(reg(None) for reg in props.group_constraints)
    iter_constraints = tuple(reg(None) for reg in props.iter_constraints)

    update_probe = process_flag(props.update_probe)
    update_object = process_flag(props.update_object)
    update_positions = process_flag(props.update_positions)
    calc_error = process_flag(props.calc_error)
    # shuffle_groups defaults to True for sparse groups, False for compact groups
    shuffle_groups = process_flag(props.shuffle_groups or not props.compact)

    sim = SimulationState(
        state=args['state'], noise_model=noise_model,
        group_constraints=group_constraints, iter_constraints=iter_constraints,
        xp=xp, dtype=dtype
    )
    patterns = args['data'].patterns
    pattern_mask = xp.asarray(args['data'].pattern_mask)

    assert_dtype(patterns, dtype)
    assert_dtype(pattern_mask, dtype)
    assert_dtype(sim.state.object.data, cdtype)
    assert_dtype(sim.state.probe.data, cdtype)

    solver = props.solver(props)
    sim = solver.init(sim)
    groups = GroupManager(sim.state.scan, props.grouping, props.compact, seed=seed)

    calc_error_mask = mask_fraction_of_groups(len(groups), props.calc_error_fraction)

    position_solver = None if props.position_solver is None else props.position_solver(None)
    position_solver_state = None if position_solver is None else position_solver.init_state(sim.state)

    observer.init_engine(
        sim.state, recons_name=args['recons_name'],
        plan=props, noise_model=noise_model.name(),
    )
    start_i = sim.state.iter.total_iter

    propagators = make_propagators(sim.state, props.bwlim_frac)

    # runs rescaling
    sim = solver.presolve(
        sim, groups.iter(sim.state.scan),
        patterns=patterns, pattern_mask=pattern_mask,
        propagators=propagators
    )

    observer.start_engine(sim.state)

    for i in range(1, props.niter+1):
        sim.state.iter.engine_iter = i
        sim.state.iter.total_iter = start_i + i

        iter_update_positions = update_positions({'state': sim.state, 'niter': props.niter})
        iter_shuffle_groups = shuffle_groups({'state': sim.state, 'niter': props.niter})

        sim, pos_update, group_errors = solver.run_iteration(
            sim, groups.iter(sim.state.scan, i, iter_shuffle_groups),
            patterns=patterns, pattern_mask=pattern_mask, propagators=propagators,
            update_object=update_object({'state': sim.state, 'niter': props.niter}),
            update_probe=update_probe({'state': sim.state, 'niter': props.niter}),
            update_positions=iter_update_positions,
            calc_error=calc_error({'state': sim.state, 'niter': props.niter}),
            calc_error_mask=calc_error_mask,
            observer=observer,
        )
        assert_dtype(sim.state.object.data, cdtype)
        assert_dtype(sim.state.probe.data, cdtype)

        sim = sim.apply_iter_constraints()

        if iter_update_positions:
            if not position_solver:
                raise ValueError("Updating positions with no PositionSolver specified")

            # subtract mean position update
            pos_update -= xp.mean(pos_update, tuple(range(pos_update.ndim - 1)))
            pos_update, position_solver_state = position_solver.perform_update(sim.state.scan, pos_update, position_solver_state)
            # subtract mean again (this can change with momentum)
            pos_update -= xp.mean(pos_update, tuple(range(pos_update.ndim - 1)))
            update_mag = xp.linalg.norm(pos_update, axis=-1, keepdims=True)
            logger.info(f"Position update: mean {xp.mean(update_mag)}")
            sim.state.scan += pos_update
            assert_dtype(sim.state.scan, dtype)

            # check positions are at least overlapping object
            sim.state.object.sampling.check_scan(sim.state.scan, sim.state.probe.sampling.extent / 2.)

        error = None
        if group_errors is not None and len(group_errors):
            error = float(to_numpy(xp.nanmean(xp.concatenate(group_errors))))

            # TODO don't do this
            sim.state.progress.iters = numpy.concatenate([sim.state.progress.iters, [i + start_i]])
            sim.state.progress.detector_errors = numpy.concatenate([sim.state.progress.detector_errors, [error]])

        observer.update_iteration(sim.state, i, props.niter, error)

    observer.finish_engine(sim.state)
    return sim.state