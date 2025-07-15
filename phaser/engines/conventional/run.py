import logging
from pathlib import Path
import typing as t

import numpy

from phaser.utils.misc import mask_fraction_of_groups
from phaser.utils.num import cast_array_module, to_numpy, to_complex_dtype
from phaser.utils.io import OutputDir
from phaser.execute import Observer
from phaser.hooks import EngineArgs
from phaser.hooks.observer import early_stop
from phaser.plan import ConventionalEnginePlan
from phaser.state import ReconsState
from phaser.types import process_flag, flag_any_true
from ..common.output import output_images, output_state
from ..common.simulation import SimulationState, make_propagators, GroupManager


def run_engine(args: EngineArgs, props: ConventionalEnginePlan) -> ReconsState:
    logger = logging.getLogger(__name__)

    xp = cast_array_module(args['xp'])
    dtype = args['dtype']
    observer: Observer = early_stop(props.observer)
    recons_name = args['recons_name']
    engine_i = args['engine_i']
    seed = args['seed']

    logger.info(f"Starting engine #{args['engine_i'] + 1}...")

    noise_model = props.noise_model(None)
    group_constraints = tuple(reg(None) for reg in props.group_constraints)
    iter_constraints = tuple(reg(None) for reg in props.iter_constraints)

    update_probe = process_flag(props.update_probe)
    update_object = process_flag(props.update_object)
    update_positions = process_flag(props.update_positions)
    calc_error = process_flag(props.calc_error)
    save = process_flag(props.save)
    save_images = process_flag(props.save_images)
    # shuffle_groups defaults to True for sparse groups, False for compact groups
    shuffle_groups = process_flag(props.shuffle_groups or not props.compact)

    sim = SimulationState(
        state=args['state'], noise_model=noise_model,
        group_constraints=group_constraints, iter_constraints=iter_constraints,
        xp=xp, dtype=dtype
    )
    patterns = args['data'].patterns
    pattern_mask = xp.array(args['data'].pattern_mask)

    assert patterns.dtype == sim.dtype
    assert pattern_mask.dtype == sim.dtype
    assert sim.state.object.data.dtype == to_complex_dtype(sim.dtype)
    assert sim.state.probe.data.dtype == to_complex_dtype(sim.dtype)

    solver = props.solver(props)
    sim = solver.init(sim)
    groups = GroupManager(sim.state.scan, props.grouping, props.compact, seed=seed)

    any_output = flag_any_true(save, props.niter) or flag_any_true(save_images, props.niter)

    with OutputDir(
        props.save_options.out_dir, any_output,
        engine_i=engine_i, name=recons_name,
        group=groups.grouping, niter=props.niter,
        solver=solver.name(),
        noise_model=noise_model.name(),
    ) as out_dir:
        calc_error_mask = mask_fraction_of_groups(len(groups), props.calc_error_fraction)

        position_solver = None if props.position_solver is None else props.position_solver(None)
        position_solver_state = None if position_solver is None else position_solver.init_state(sim.state)

        propagators = make_propagators(sim.state, props.bwlim_frac)

        start_i = sim.state.iter.total_iter
        observer.init_solver(sim.state, engine_i)

        # runs rescaling
        sim = solver.presolve(
            sim, groups.iter(sim.state.scan),
            patterns=patterns, pattern_mask=pattern_mask,
            propagators=propagators
        )

        observer.start_solver()

        for i in range(1, props.niter+1):
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
            assert sim.state.object.data.dtype == to_complex_dtype(sim.dtype)
            assert sim.state.probe.data.dtype == to_complex_dtype(sim.dtype)

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
                assert sim.state.scan.dtype == sim.dtype

                # check positions are at least overlapping object
                sim.state.object.sampling.check_scan(sim.state.scan, sim.state.probe.sampling.extent / 2.)

            error = None
            if group_errors is not None:
                error = float(to_numpy(xp.nanmean(xp.concatenate(group_errors))))

                # TODO don't do this
                sim.state.progress.iters = numpy.concatenate([sim.state.progress.iters, [i + start_i]])
                sim.state.progress.detector_errors = numpy.concatenate([sim.state.progress.detector_errors, [error]])

            if observer.update_iteration(sim.state, i, props.niter, error):
                break

            if save({'state': sim.state, 'niter': props.niter}):
                output_state(sim.state, out_dir, props.save_options)

            if save_images({'state': sim.state, 'niter': props.niter}):
                output_images(sim.state, out_dir, props.save_options)

    output_state(sim.state, out_dir, props.save_options)
    output_images(sim.state, out_dir, props.save_options)
    observer.finish_solver()
    return sim.state