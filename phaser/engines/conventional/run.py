import logging

from phaser.utils.misc import mask_fraction_of_groups
from phaser.utils.num import assert_dtype, cast_array_module, check_finite, to_numpy, to_complex_dtype
from phaser.observer import Observer
from phaser.hooks import EngineArgs
from phaser.plan import ConventionalEnginePlan
from phaser.state import ReconsState, ProgressState
from phaser.types import process_flag, flag_any_true
from ..common.simulation import SimulationState, make_propagators, prepare_mtf, GroupManager


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
    assert_dtype(args['data'].patterns, dtype)
    assert_dtype(args['data'].pattern_mask, dtype)
    assert_dtype(sim.state.object.data, cdtype)
    assert_dtype(sim.state.probe.data, cdtype)

    pattern_mask = xp.asarray(args['data'].pattern_mask)
    # load/stream patterns
    if props.buffer_n_groups is None:
        logging.info("Loading raw data to GPU ('buffer_n_groups' is disabled)...")
        patterns = xp.asarray(args['data'].patterns)
    else:
        logging.info(f"Streaming raw data to GPU (buffering {props.buffer_n_groups} groups)")
        patterns = args['data'].patterns

    solver = props.solver(props)
    sim = solver.init(sim)
    groups = GroupManager(sim.state.scan.data, props.grouping, props.compact, seed=seed)

    calc_error_mask = mask_fraction_of_groups(len(groups), props.calc_error_fraction)

    position_solver = None if props.position_solver is None else props.position_solver(None)
    position_solver_state = None if position_solver is None else position_solver.init_state(sim.state)
    if position_solver is not None and flag_any_true(update_positions, props.niter):
        other_keys = ('pos_update_rms',)
    else:
        other_keys = ()

    for k in ('detector_loss', 'total_loss', *other_keys):
        if k not in sim.state.progress:
            sim.state.progress[k] = ProgressState()

    # save progress, it will get clobbered by JIT kernels
    progress = sim.state.progress

    observer.init_engine(
        sim.state, recons_name=args['recons_name'],
        plan=props, noise_model=noise_model.name(),
    )
    start_i = int(sim.state.iter.total_iter)

    propagators = make_propagators(sim.state, props.bwlim_frac)
    mtf = None if props.mtf is None else prepare_mtf(props.mtf, sim.state, dtype, xp)

    # runs rescaling
    sim = solver.presolve(
        sim, groups.iter(sim.state.scan.data),
        patterns=patterns, pattern_mask=pattern_mask,
        propagators=propagators
    )

    sim.state.progress = progress
    observer.start_engine(sim.state)

    for i in range(1, props.niter+1):
        sim.state.iter.engine_iter = i
        sim.state.iter.total_iter = start_i + i

        iter_update_positions = update_positions({'state': sim.state, 'niter': props.niter})
        iter_shuffle_groups = shuffle_groups({'state': sim.state, 'niter': props.niter})

        sim, pos_update, group_errors = solver.run_iteration(
            sim, groups.iter(sim.state.scan.data, i, iter_shuffle_groups),
            patterns=patterns, pattern_mask=pattern_mask, propagators=propagators, mtf=mtf,
            update_object=update_object({'state': sim.state, 'niter': props.niter}),
            update_probe=update_probe({'state': sim.state, 'niter': props.niter}),
            update_positions=iter_update_positions,
            calc_error=calc_error({'state': sim.state, 'niter': props.niter}),
            calc_error_mask=calc_error_mask,
            observer=observer,
        )
        assert_dtype(sim.state.object.data, cdtype)
        assert_dtype(sim.state.probe.data, cdtype)
        check_finite(sim.state.object.data, sim.state.probe.data, context=f"NaN or inf encountered, iteration {i}")

        sim = sim.apply_iter_constraints()

        if iter_update_positions:
            if not position_solver:
                raise ValueError("Updating positions with no PositionSolver specified")

            # subtract mean position update
            pos_update -= xp.mean(pos_update, tuple(range(pos_update.ndim - 1)))
            pos_update, position_solver_state = position_solver.perform_update(sim.state.scan.data, pos_update, position_solver_state)
            # subtract mean again (this can change with momentum)
            pos_update -= xp.mean(pos_update, tuple(range(pos_update.ndim - 1)))
            pos_update_rms = float(xp.mean(xp.linalg.norm(pos_update, axis=-1, keepdims=True)))
            logger.info(f"Position update: mean {pos_update_rms}")
            sim.state.scan.data += pos_update
            assert_dtype(sim.state.scan.data, dtype)

            # check positions are at least overlapping object
            sim.state.object.sampling.check_scan(sim.state.scan.data, sim.state.probe.sampling.extent / 2.)

            progress['pos_update_rms'].iters.append(i + start_i)
            progress['pos_update_rms'].values.append(pos_update_rms)

        error = None
        if group_errors is not None and len(group_errors):
            error = float(to_numpy(xp.nanmean(xp.concatenate(group_errors))))

            for k in ('detector_loss', 'total_loss'):
                progress[k].iters.append(i + start_i)
                progress[k].values.append(error)

        sim.state.progress = progress
        observer.update_iteration(sim.state, i, props.niter, {'total_loss': error} if error is not None else {})

    observer.finish_engine(sim.state)
    return sim.state