import dataclasses
import itertools
import logging
import math
import sys
import typing as t

from frozendict import frozendict
import numpy
import pane

from phaser.types import EarlyTermination
from phaser.utils.num import Device, cast_array_module, get_array_module, get_backend_devices, get_backend_module, set_default_device, to_device, xp_is_jax, Sampling, to_complex_dtype, xp_is_torch
from phaser.utils.object import ObjectSampling
from phaser.utils.misc import freeze, unwrap
from .hooks import EngineHook, Hook, ObjectHook, RawData
from .plan import GradientEnginePlan, ReconsPlan, EnginePlan, ScanHook, ProbeHook, TiltHook
from .state import Patterns, ReconsState, PartialReconsState, IterState, PreparedRecons, ScanState
from .observer import Observer, LoggingObserver, PatienceObserver, SaveObserver, ObserverSet
from .version import version_info


def execute_plan(
    plan: ReconsPlan, *, xp: t.Any = None, seed: t.Any = None,
    name: t.Optional[str] = None,
    init_state: t.Union[ReconsState, PartialReconsState, None] = None,
    observers: t.Union[Observer, t.Iterable[Observer], None] = None,
    override_observers: t.Union[Observer, t.Iterable[Observer], None] = None,
):
    logging.basicConfig(level=logging.INFO)
    logging.info(str(version_info()))

    recons = initialize_reconstruction(
        plan, xp=xp, seed=seed, name=name, init_state=init_state,
        observers=observers, override_observers=override_observers
    )
    recons.state.iter.n_total_iters = sum(
        t.cast(EnginePlan, engine.props).niter
        for engine in plan.engines
    )

    try:
        try:
            for engine in plan.engines:
                recons = execute_engine(recons, engine)
        except EarlyTermination as e:
            recons.state = e.state
        recons.observer.finish_recons(recons.state)
        logging.info("Reconstruction finished!")
    finally:
        # pass any excpetion info to observers
        recons.observer.close(sys.exc_info()[1])


def execute_engine(
    recons: PreparedRecons,
    engine: EngineHook,
) -> PreparedRecons:
    xp = get_array_module(recons.state.object.data, recons.state.probe.data)
    dtype = recons.patterns.patterns.dtype.type
    plan = t.cast(EnginePlan, engine.props)

    engine_i = recons.state.iter.engine_num

    if plan.early_termination:
        engine_observer = ObserverSet((recons.observer, PatienceObserver(
            plan.early_termination, plan.early_termination_smoothing
        )))
    else:
        engine_observer = recons.observer

    logging.info(f"Preparing for engine #{engine_i + 1}...")
    recons.patterns, recons.state = prepare_for_engine(recons.patterns, recons.state, xp, plan)
    recons.state.iter = IterState(
        engine_num=engine_i + 1,
        engine_iter=0,
        n_engine_iters=plan.niter,
        total_iter=recons.state.iter.total_iter,
        n_total_iters=recons.state.iter.n_total_iters,
    )
    try:
        recons.state = engine({
            'data': recons.patterns,
            'state': recons.state,
            'dtype': dtype,
            'xp': xp,
            'recons_name': recons.name,
            'observer': engine_observer,
            'seed': None,
        })
    except EarlyTermination as e:
        recons.state = e.state
        engine_observer.finish_engine(recons.state)

        if not e.continue_next_engine:
            engine_observer.finish_recons(recons.state)
            raise

    return recons


def _normalize_observers(
    observers: t.Union[Observer, t.Iterable[Observer], None],
    override_observers: t.Union[Observer, t.Iterable[Observer], None],
) -> ObserverSet:
    if override_observers is not None:
        if observers is not None:
            raise TypeError("Cannot specify both 'observers' and 'override_observers")
        if isinstance(override_observers, Observer):
            obs = (override_observers,)
        elif isinstance(override_observers, t.Iterable):
            obs = tuple(override_observers)
        else:
            raise TypeError(f"'override_observers' expected an Observer or list of Observers, instead got type {type(override_observers)}")

        return ObserverSet(obs)

    obs = [
        SaveObserver(),
        LoggingObserver(),
    ]

    if isinstance(observers, Observer):
        obs.append(observers)
    elif isinstance(observers, t.Iterable):
        obs.extend(observers)
    elif observers is not None:
        raise TypeError(f"'observers' expected an Observer or list of Observers, instead got type {type(observers)}")

    return ObserverSet(obs)


def _normalize_scan_shape(
    patterns: Patterns, state: ReconsState
) -> t.Tuple[Patterns, ReconsState]:
    """
    Normalizes 'patterns' and 'state' to share a common scan shape.

    Requires that there are an equal number of patterns and scan positions.
    Reshapes 'state.scan' (scan and tilt) and 'patterns' to match shape,
    choosing the highest dimensional shape of the two.
    """
    scan = state.scan

    patterns_shape = patterns.patterns.shape[:-2]
    scan_shape = scan.data.shape[:-1]

    n_patterns = math.prod(patterns_shape)
    n_scan = math.prod(scan_shape)
    if n_scan != n_patterns:
        raise ValueError(f"# of scan positions {n_scan} doesn't match # of patterns {n_patterns}")

    # choose the highest dimensional shape
    new_shape = scan_shape if len(scan_shape) > len(patterns_shape) else patterns_shape

    patterns.patterns = patterns.patterns.reshape((*new_shape, *patterns.patterns.shape[-2:]))
    scan.data = scan.data.reshape((*new_shape, 2))
    scan.initial = scan.initial.reshape((*new_shape, 2))

    if scan.tilt is not None:
        n_tilt = math.prod(scan.tilt.shape[:-1])
        if n_tilt != n_patterns:
            raise ValueError(f"# of tilt positions {n_scan} doesn't match # of patterns {n_patterns}")

        scan.tilt = scan.tilt.reshape((*new_shape, 2))

    # also normalize raster_rows and raster_cols metadata
    scan.meta = frozendict(
        scan.meta,
        **({'raster_rows': freeze(numpy.array(scan.meta['raster_rows']).reshape(new_shape))} if 'raster_rows' in scan.meta else {}),
        **({'raster_cols': freeze(numpy.array(scan.meta['raster_cols']).reshape(new_shape))} if 'raster_cols' in scan.meta else {}),
    )

    return patterns, state


def load_raw_data(
    plan: ReconsPlan, xp: t.Any, seed: t.Any = None,
    init_state: t.Union[ReconsState, PartialReconsState, None] = None
) -> RawData:
    dtype: type = numpy.float32 if plan.dtype == 'float32' else numpy.float64

    raw_data = plan.raw_data(None)

    wavelength = plan.wavelength or raw_data.get('wavelength', None)
    if wavelength is None:
        raise ValueError("`wavelength` must be specified by raw_data or manually")

    if init_state is None:
        init_state = PartialReconsState()

    if init_state.wavelength is not None and not numpy.isclose(init_state.wavelength, wavelength):
        logging.warning(f"Wavelength of reconstruction ({wavelength:.2e}) doesn't match wavelength " \
                        f"of previous state ({init_state.wavelength:.2e})")

    raw_data['scan_hook'] = pane.into_data(merge(  # type: ignore
        pane.from_data(t.cast(dict, raw_data.get('scan_hook', None)), ScanHook) if raw_data.get('scan_hook', None) is not None else None,
        None if plan.init.scan in (None, {}) else plan.init.scan
    ))
    raw_data['tilt_hook'] = pane.into_data(merge(  # type: ignore
        pane.from_data(t.cast(dict, raw_data.get('tilt_hook', None)), TiltHook) if raw_data.get('tilt_hook', None) is not None else None,
        None if plan.init.tilt in (None, {}) else plan.init.tilt
    ))
    raw_data['probe_hook'] = pane.into_data(merge(  # type: ignore
        pane.from_data(t.cast(dict, raw_data.get('probe_hook', None)), ProbeHook) if raw_data.get('probe_hook', None) is not None else None,
        None if plan.init.probe in (None, {}) else plan.init.probe
    ))
    #print(f"scan_hook: {raw_data['scan_hook']}")
    #print(f"probe_hook: {raw_data['probe_hook']}")

    if raw_data['scan_hook'] is None and init_state.scan is None:
        raise ValueError("`scan` must be specified by raw data, previous state, or manually in `init.scan`")
    if raw_data['probe_hook'] is None and init_state.probe is None:
        raise ValueError("`probe` must be specified by raw data, previous state, or manually in `init.probe`")
    if raw_data['scan_hook'] == {}:
        raise ValueError("Manual `init.scan` specified to override initial state, but scan was not provided by the raw data")
    if raw_data['tilt_hook'] == {}:
        raise ValueError("Manual `init.tilt` specified to override initial state, but tilt was not provided by the raw data")
    if raw_data['probe_hook'] == {}:
        raise ValueError("Manual `init.probe` specified to override initial state, but probe was not provided by the raw data")

    raw_data['wavelength'] = wavelength
    raw_data['seed'] = seed

    # normalize pattern intensity
    #raw_data['patterns'] /= numpy.mean(numpy.sum(raw_data['patterns'], axis=(-1, -2)))
    # ensure raw data is of the correct type
    if raw_data['patterns'].dtype != dtype:
        raw_data['patterns'] = raw_data['patterns'].astype(dtype)

    # process post_load hooks:
    for p in plan.post_load:
        raw_data = p(raw_data)

    # materialize memmap
    if isinstance(raw_data['patterns'], numpy.memmap):
        raw_data['patterns'] = raw_data['patterns'].copy()

    return raw_data


def initialize_reconstruction(
    plan: ReconsPlan, *, xp: t.Any = None, device: t.Optional[Device] = None,
    seed: t.Any = None, name: t.Optional[str] = None,
    init_state: t.Union[ReconsState, PartialReconsState, None] = None,
    observers: t.Union[Observer, t.Iterable[Observer], None] = None,
    override_observers: t.Union[Observer, t.Iterable[Observer], None] = None,
) -> PreparedRecons:
    logging.basicConfig(level=logging.INFO)

    if xp is not None:
        xp = cast_array_module(xp)
        # TODO: nicer output here
        logging.info(f"Using manually-specified backend {xp}")
        devices = get_backend_devices(xp)
        logging.info(f"Available devices: {list(devices)}")
        manual = device is not None
        device = to_device(device, xp) if device is not None else devices[0]
        logging.info(f"Using {'manually-specified ' if manual else ''}device {device}")
    else:
        xp = get_backend_module(plan.backend)
        logging.info(f"Using {'plan-specified' if plan.backend is not None else 'default'} backend {xp}")
        devices = get_backend_devices(xp)
        logging.info(f"Available devices: {list(devices)}")

        device = to_device(plan.device, xp) if plan.device is not None else devices[0]
        logging.info(f"Using {'plan-specified ' if plan.device is not None else ''}device {device}")

    set_default_device(device, xp)

    observer = _normalize_observers(observers, override_observers)

    logging.info("Executing plan...")
    observer.init_recons(plan)

    dtype: t.Type[numpy.floating] = numpy.float32 if plan.dtype == 'float32' else numpy.float64
    cdtype: t.Type[numpy.complexfloating] = to_complex_dtype(dtype)

    logging.info(f"dtype: {dtype} array backend: {xp.__name__}")
    if xp_is_jax(xp):
        import jax
        logging.info(f"jax backend: {jax.default_backend()} devices: {jax.devices()}")

    if init_state is None:
        if plan.init.state is not None:
            path = plan.init.state.expanduser()
            logging.info(f"Loading inital state from '{path}'...")
            init_state = PartialReconsState.read_hdf5(path)
        else:
            init_state = PartialReconsState()

    raw_data = load_raw_data(plan, xp, seed, init_state=init_state)

    data = Patterns(raw_data['patterns'], raw_data['mask'])
    sampling = raw_data['sampling']
    wavelength = unwrap(raw_data.get('wavelength', None))
    probe_hook = raw_data.get('probe_hook', None)
    scan_hook = raw_data.get('scan_hook', None)
    tilt_hook = raw_data.get('tilt_hook', None)

    del raw_data

    if init_state.probe is not None and plan.init.probe is None:
        logging.info("Re-using probe from initial state...")
        probe = init_state.probe
        probe.data = probe.data.astype(cdtype)

        if probe.sampling != sampling:
            logging.info("Resampling patterns to probe from initial state...")
            data.patterns = sampling.resample_recip(data.patterns, probe.sampling)
            data.pattern_mask = sampling.resample_recip(data.pattern_mask, probe.sampling)
            sampling = probe.sampling

    else:
        logging.info("Initializing probe...")
        probe = pane.from_data(probe_hook, ProbeHook)(  # type: ignore
            {'sampling': sampling, 'wavelength': wavelength, 'dtype': dtype, 'seed': seed, 'xp': xp}
        )
    if probe.data.ndim == 2:
        probe.data = probe.data.reshape((1, *probe.data.shape))

    if init_state.scan is not None and plan.init.scan is None:
        logging.info("Re-using scan from initial state...")
        scan = init_state.scan.copy()
        scan.data = scan.data.astype(dtype)
        scan.initial = scan.initial.astype(dtype)
    else:
        logging.info("Initializing scan...")
        scan = pane.from_data(scan_hook, ScanHook)(  # type: ignore
            {'dtype': dtype, 'seed': seed, 'xp': xp}
        )

    if init_state.scan is not None and init_state.scan.tilt is not None and plan.init.tilt is None:
        logging.info("Re-using tilt from initial state...")
        scan.tilt = init_state.scan.tilt.astype(dtype)
    elif tilt_hook is not None:
        logging.info("Initializing tilt...")
        scan.tilt = pane.from_data(tilt_hook, TiltHook)(  # type: ignore
            {'dtype': dtype, 'xp': xp, 'shape': scan.data.shape[:-1]}
        )

    obj_pad_px: float = plan.engines[0].obj_pad_px if len(plan.engines) > 0 else 5.0  # type: ignore
    obj_sampling = ObjectSampling.from_scan(
        scan.data, sampling.sampling, sampling.extent / 2. + obj_pad_px * sampling.sampling
    )

    if init_state.object is not None and plan.init.object is None:
        logging.info("Re-using object from initial state...")
        obj = init_state.object
        obj.data = obj.data.astype(cdtype)
    else:
        logging.info("Initializing object...")

        obj = (plan.init.object or pane.from_data('random', ObjectHook))({
            'sampling': obj_sampling, 'slices': plan.slices, 'wavelength': wavelength,
            'dtype': dtype, 'seed': seed, 'xp': xp
        })
    if obj.data.ndim == 2:
        obj.data = obj.data.reshape((1, *obj.data.shape))
        obj.thicknesses = numpy.array([], dtype=dtype)

    state = ReconsState(
        iter=IterState(0, 0, 0),
        probe=probe,
        object=obj,
        scan=scan,
        wavelength=wavelength
    )
    state = state.to_xp(xp)  # TODO: figure out why this isn't already the case
    data, state = _normalize_scan_shape(data, state)

    # process post_init hooks
    for p in plan.post_init:
        (data, state) = p({
            'data': data, 'state': state,
            'dtype': dtype, 'seed': seed, 'xp': xp
        })

    # perform some checks on preprocessed data

    avg_pattern_intensity = float(numpy.nanmean(numpy.nansum(data.patterns, axis=(-1, -2))))

    if avg_pattern_intensity < 5.0:
        logging.warning(
            f"Mean pattern intensity is very low ({avg_pattern_intensity} particles). "
            "Ensure that it is being scaled correctly to units of electrons/photons. "
            "For simulated data, use the 'scale' or 'poisson' preprocessing"
        )

    observer.start_recons(state)
    return PreparedRecons(
        data, state,
        name or plan.name,
        observer
    )


def prepare_for_engine(patterns: Patterns, state: ReconsState, xp: t.Any, engine: EnginePlan) -> t.Tuple[Patterns, ReconsState]:
    # TODO: more graceful
    if isinstance(engine, GradientEnginePlan) and not (xp_is_jax(xp) or xp_is_torch(xp)):
        raise ValueError("The gradient descent engine requires the 'jax' or 'torch' backend.")

    state = state.to_xp(xp)

    if engine.sim_shape is not None and engine.sim_shape != state.probe.data.shape[-2:]:
        if engine.resize_method == 'pad_crop':
            new_sampling = Sampling(engine.sim_shape, extent=tuple(state.probe.sampling.extent))
        else:
            new_sampling = Sampling(engine.sim_shape, sampling=tuple(state.probe.sampling.sampling))

        logging.info(f"Resampling probe and patterns to shape {new_sampling.shape}...")
        state.probe.data = state.probe.sampling.resample(state.probe.data, new_sampling)
        # also resample patterns
        patterns.patterns = state.probe.sampling.resample_recip(patterns.patterns, new_sampling)
        # and pattern mask
        patterns.pattern_mask = state.probe.sampling.resample_recip(patterns.pattern_mask, new_sampling)

        state.probe.sampling = new_sampling

    obj_sampling = state.object.sampling

    if not numpy.allclose(state.probe.sampling.sampling, state.object.sampling.sampling):
        # resample object -> probe
        logging.info(f"Resampling object to pixel size {list(map(float, state.probe.sampling.sampling))}...")
        obj_sampling = obj_sampling.with_sampling(state.probe.sampling.sampling)

    obj_sampling_pad = obj_sampling.expand_to_scan(
        state.scan.data, state.probe.sampling.extent / 2. + engine.obj_pad_px * state.probe.sampling.sampling
    )

    if obj_sampling_pad != obj_sampling:
        logging.info(f"Padding object to shape {obj_sampling_pad.shape}")
        obj_sampling = obj_sampling_pad

    if obj_sampling != state.object.sampling:
        state.object.data = state.object.sampling.resample(state.object.data, obj_sampling)
        state.object.sampling = obj_sampling

    current_probe_modes = state.probe.data.shape[0]
    if engine.probe_modes != current_probe_modes:
        # fix probe modes
        if engine.probe_modes < current_probe_modes:
            # TODO: redistribute intensity here
            state.probe.data = state.probe.data[:engine.probe_modes]
        else:
            from phaser.utils.optics import make_hermetian_modes
            if current_probe_modes != 1:
                logging.info("Summing probe modes (in real-space) before recreating with different # of modes")

            base_mode = xp.sum(state.probe.data, axis=0)
            state.probe.data = make_hermetian_modes(base_mode, engine.probe_modes, base_mode_power=engine.base_mode_power)

    if engine.slices is not None and (len(engine.slices.thicknesses) != len(state.object.thicknesses)
                                      or not numpy.allclose(engine.slices.thicknesses, state.object.thicknesses)):
        from phaser.utils.object import resample_slices
        logging.info(f"Reslicing object from {max(1, len(state.object.thicknesses))} to {max(1, len(engine.slices.thicknesses))} slice(s)...")
        state.object.data = resample_slices(state.object.data, state.object.thicknesses, engine.slices.thicknesses)
        state.object.thicknesses = xp.array(engine.slices.thicknesses, dtype=state.object.thicknesses.dtype)

    if isinstance(engine, GradientEnginePlan):
        solver_vars = set(itertools.chain.from_iterable(engine.solvers.keys()))
        if 'tilt' in solver_vars and state.scan.tilt is None:
            logging.info("Creating new, zeroed tilt map...")
            state.scan.tilt = xp.zeros_like(state.scan.data)

    return patterns, state


#_MISSING = object()


def merge(left: t.Any, right: t.Any) -> t.Any:
    def _as_dict(val) -> t.Optional[dict]:
        if isinstance(val, dict):
            return val
        if dataclasses.is_dataclass(val):
            return dataclasses.asdict(val)  # type: ignore
        if isinstance(val, pane.PaneBase):
            return val.dict(set_only=True)
        return None

    if left is None or right is None:
        return left if right is None else right

    if isinstance(left, Hook) and isinstance(right, Hook):
        if left.ref != right.ref:
            return right
        d = merge(left.props or {}, right.props or {})
        d['type'] = right.type if right.type is not None else right.ref
        return pane.convert(d, right.__class__)

    if (left_d := _as_dict(left)) is not None and (right_d := _as_dict(right)) is not None:
        keys = set(left_d.keys()) | set(right_d.keys())
        return {k: merge(left_d.get(k, None), right_d.get(k, None)) for k in keys}

    return left if right is None else right


__all__ = [
    'execute_plan', 'execute_engine',
    'initialize_reconstruction',
    'prepare_for_engine'
]