from functools import partial
import logging
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import cast_array_module, at, abs2, fft2, ifft2, jit, check_finite, to_complex_dtype, to_numpy
from phaser.hooks.solver import ConventionalSolver
from phaser.types import process_schedule
from phaser.plan import ConventionalEnginePlan, LSQMLSolverPlan, EPIESolverPlan
from phaser.execute import Observer
from phaser.engines.common.simulation import (
    stream_patterns, SimulationState, cutout_group, tilt_propagators, slice_forwards, slice_backwards
)


class LSQMLSolver(ConventionalSolver):
    def __init__(self, plan: ConventionalEnginePlan, props: LSQMLSolverPlan):
        self.plan: LSQMLSolverPlan = props
        self.engine_plan: ConventionalEnginePlan = plan

    @classmethod
    def name(cls) -> str:
        return "LSQML"

    def init(self, sim: SimulationState) -> SimulationState:
        self.logger = logging.getLogger(__name__)
        xp = sim.xp

        self.obj_mag: NDArray[numpy.floating] = xp.zeros(sim.state.probe.data.shape[-2:], dtype=sim.dtype)
        self.probe_mag: NDArray[numpy.floating] = xp.zeros_like(sim.state.object.data, dtype=sim.dtype)

        return sim

    def presolve(
        self,
        sim: SimulationState,
        groups: t.Iterator[NDArray[numpy.int_]], *,
        patterns: NDArray[numpy.floating],
        pattern_mask: NDArray[numpy.floating],
        propagators: t.Optional[NDArray[numpy.complexfloating]],
    ) -> SimulationState:
        rescale_factors = []

        # precompute obj_mag, probe_mag, and rescale probe intensity
        for (group, group_patterns) in stream_patterns(groups, patterns, xp=sim.xp, buf_n=self.engine_plan.buffer_n_groups):
            (self.obj_mag, self.probe_mag, group_rescale_factors) = lsqml_dry_run(
                sim, group, group_patterns, props=propagators, pattern_mask=pattern_mask,
                obj_mag=self.obj_mag, probe_mag=self.probe_mag
            )

            rescale_factors.append(to_numpy(group_rescale_factors))

        rescale_factors = numpy.concatenate(rescale_factors, axis=0)
        rescale_factor = numpy.mean(rescale_factors)

        self.logger.info("Pre-calculated intensities")
        self.logger.info(f"Rescaling initial probe intensity by {rescale_factor:.2e}")
        sim.state.probe.data *= numpy.sqrt(rescale_factor)
        self.probe_mag *= rescale_factor

        return sim

    def run_iteration(
        self,
        sim: SimulationState,
        groups: t.Iterator[NDArray[numpy.int_]], *,
        patterns: NDArray[numpy.floating],
        pattern_mask: NDArray[numpy.floating],
        propagators: t.Optional[NDArray[numpy.complexfloating]],
        update_object: bool,
        update_probe: bool,
        update_positions: bool,
        calc_error: bool,
        calc_error_mask: NDArray[numpy.bool_],
        observer: 'Observer',
    ) -> t.Tuple[SimulationState, NDArray[numpy.floating], t.List[NDArray[numpy.floating]]]:
        xp = sim.xp

        beta_object = process_schedule(self.plan.beta_object)({'state': sim.state, 'niter': self.engine_plan.niter})
        beta_probe = process_schedule(self.plan.beta_probe)({'state': sim.state, 'niter': self.engine_plan.niter})
        illum_reg_object = process_schedule(self.plan.illum_reg_object)({'state': sim.state, 'niter': self.engine_plan.niter})
        illum_reg_probe = process_schedule(self.plan.illum_reg_probe)({'state': sim.state, 'niter': self.engine_plan.niter})
        gamma = process_schedule(self.plan.gamma)({'state': sim.state, 'niter': self.engine_plan.niter})

        new_obj_mag = xp.zeros_like(self.obj_mag)
        new_probe_mag = xp.zeros_like(self.probe_mag)
        pos_update = xp.zeros_like(sim.state.scan, dtype=sim.dtype)
        iter_errors = []

        for (group_i, (group, group_patterns)) in enumerate(stream_patterns(groups, patterns, xp=xp,
                                                                            buf_n=self.engine_plan.buffer_n_groups)):
            group_calc_error = calc_error and calc_error_mask[group_i]

            (sim, new_obj_mag, new_probe_mag, errors, group_pos_update) = lsqml_run(
                sim, group, group_patterns, pattern_mask=pattern_mask, props=propagators,
                obj_mag=self.obj_mag, probe_mag=self.probe_mag,
                new_obj_mag=new_obj_mag, new_probe_mag=new_probe_mag,
                beta_object=beta_object, beta_probe=beta_probe,
                update_object=update_object,
                update_probe=update_probe,
                update_position=update_positions,
                calc_error=group_calc_error,
                illum_reg_object=illum_reg_object,
                illum_reg_probe=illum_reg_probe,
                gamma=gamma,
            )
            check_finite(sim.state.object.data, sim.state.probe.data, context=f"object or probe, group {group_i}")
            #assert sim.state.object.data.dtype == to_complex_dtype(sim.dtype)
            #assert sim.state.probe.data.dtype == to_complex_dtype(sim.dtype)

            sim = sim.apply_group_constraints(group)

            if update_positions:
                assert group_pos_update is not None
                pos_update = at(pos_update, tuple(group)).set(group_pos_update)

            observer.update_group(sim.state, self.engine_plan.send_every_group)

            if group_calc_error:
                assert errors is not None
                iter_errors.append(errors)

        self.obj_mag = new_obj_mag
        self.probe_mag = new_probe_mag

        return (sim, pos_update, iter_errors)


@partial(jit, donate_argnames=('obj_mag', 'probe_mag'))
def lsqml_dry_run(
    sim: SimulationState,
    group: NDArray[numpy.integer],
    group_patterns: NDArray[numpy.floating], *,
    pattern_mask: NDArray[numpy.floating],
    props: t.Optional[NDArray[numpy.complexfloating]],
    obj_mag: NDArray[numpy.floating],
    probe_mag: NDArray[numpy.floating]
) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating], NDArray[numpy.floating]]:
    xp = cast_array_module(sim.xp)
    (psi, group_obj, group_scan) = cutout_group(sim.ky, sim.kx, sim.state, group)

    obj_mag += xp.sum(abs2(xp.prod(group_obj, axis=1)), axis=0)
    obj_grid = sim.state.object.sampling

    def run_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], state):
        (probe_mag, psi) = state
        probe_mag = at(probe_mag, slice_i).set(
            obj_grid.add_view_at_pos(probe_mag[slice_i], group_scan, xp.sum(abs2(psi), axis=1))
        )

        if prop is not None:
            psi = ifft2(fft2(psi * group_obj[:, slice_i, None]) * prop[:, None])

        return (probe_mag, psi)

    props = tilt_propagators(sim.ky, sim.kx, sim.state, props,
                             sim.state.tilt[tuple(group)] if sim.state.tilt is not None else None)
    (probe_mag, psi) = slice_forwards(props, (probe_mag, psi), run_slice)

    # modeled and experimental intensity
    # summed over incoherent modes and over the pattern
    model_intensity = xp.sum(abs2(fft2(psi)), axis=(1, -2, -1))
    exp_intensity = xp.sum(group_patterns * pattern_mask, axis=(-2, -1))

    return (obj_mag, probe_mag, exp_intensity / model_intensity)

# TODO: pass LSQMLSolverPlan in here for parameters

@partial(
    jit,
    donate_argnames=('sim', 'new_obj_mag', 'new_probe_mag'),
    static_argnames=('update_object', 'update_probe', 'update_position', 'calc_error'),
)
def lsqml_run(
    sim: SimulationState,
    group: NDArray[numpy.integer],
    group_patterns: NDArray[numpy.floating], *,
    pattern_mask: NDArray[numpy.floating],
    props: t.Optional[NDArray[numpy.complexfloating]],
    obj_mag: NDArray[numpy.floating],
    probe_mag: NDArray[numpy.floating],
    new_obj_mag: NDArray[numpy.floating],
    new_probe_mag: NDArray[numpy.floating],
    beta_object: float = 0.9,
    beta_probe: float = 0.9,
    update_object: bool = True,
    update_probe: bool = True,
    update_position: bool = True,
    calc_error: bool = True,
    illum_reg_object: float,
    illum_reg_probe: float,
    gamma: float,
) -> t.Tuple[SimulationState, NDArray[numpy.floating], NDArray[numpy.floating], t.Optional[NDArray[numpy.floating]], t.Optional[NDArray[numpy.floating]]]:
    xp = cast_array_module(sim.xp)
    obj_grid = sim.state.object.sampling
    n_slices = sim.state.object.data.shape[0]

    eps = 1e-16

    (probes, group_obj, group_scan, subpx_filters) = cutout_group(sim.ky, sim.kx, sim.state, group, return_filters=True)
    psi = xp.zeros((n_slices, *probes.shape), dtype=probes.dtype)
    psi = at(psi, 0).set(probes)

    group_probe_mag = xp.zeros_like(probe_mag)
    #group_obj_mag = xp.sum(abs2(group_obj[:, 0]), axis=0)
    group_obj_mag = xp.sum(abs2(xp.prod(group_obj, axis=1)), axis=0)

    def sim_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], state):
        (group_probe_mag, psi) = state

        group_probe_mag = at(group_probe_mag, slice_i).set(
            obj_grid.add_view_at_pos(group_probe_mag[slice_i], group_scan, xp.sum(abs2(psi[slice_i]), axis=1))
        )

        if prop is not None:
            psi = at(psi, slice_i + 1).set(
                ifft2(fft2(psi[slice_i] * group_obj[:, slice_i, None]) * prop[:, None])
            )

        return (group_probe_mag, psi)

    props = tilt_propagators(sim.ky, sim.kx, sim.state, props,
                             sim.state.tilt[tuple(group)] if sim.state.tilt is not None else None)
    (group_probe_mag, psi) = slice_forwards(props, (group_probe_mag, psi), sim_slice)

    new_obj_mag += group_obj_mag
    new_probe_mag += group_probe_mag

    model_wave = fft2(psi[-1] * group_obj[:, -1, None])
    # sum over incoherent modes
    model_intensity = xp.sum(abs2(model_wave), axis=1, keepdims=True)
    # experimental data
    # group_patterns = xp.array(sim.patterns[tuple(group)])[:, None]

    errors = xp.sqrt(xp.nansum((model_intensity - group_patterns[:, None])**2, axis=(1, -1, -2))) if calc_error else None

    (chi, sim.noise_model_state) = sim.noise_model.calc_wave_update(model_wave, model_intensity, group_patterns[:, None], pattern_mask, sim.noise_model_state)
    chi = ifft2(chi)

    def update_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], state):
        (sim, chi) = state

        delta_P = chi * xp.conj(group_obj[:, slice_i, None])

        if update_object:
            delta_O = chi * xp.conj(psi[slice_i])
            alpha_O = xp.sum(xp.sum(xp.real(chi * xp.conj(delta_O * psi[slice_i])), axis=(-1, -2), keepdims=True), axis=1) / (xp.sum(abs2(delta_O * psi[slice_i])) + gamma)

            # average object update
            delta_O_avg = xp.zeros_like(sim.state.object.data[0])
            delta_O_avg = obj_grid.add_view_at_pos(delta_O_avg, group_scan, xp.sum(delta_O, axis=1))
            delta_O_avg /= (group_probe_mag[slice_i] + illum_reg_object)

            obj_update = beta_object * xp.sum(alpha_O * delta_O_avg * group_probe_mag[slice_i], axis=0) / (group_probe_mag[slice_i] + eps)
            sim.state.object.data = at(sim.state.object.data, slice_i).add(obj_update)

        if prop is not None:
            chi = ifft2(fft2(delta_P) * prop.conj()[:, None])
        elif update_probe:
            delta_P_avg = ifft2(xp.sum(fft2(delta_P) * subpx_filters.conj(), axis=0))
            delta_P_avg /= (group_obj_mag + illum_reg_probe)

            # update step per probe mode
            alpha_P = xp.sum(xp.real(chi * xp.conj(delta_P * group_obj[:, slice_i, None])), axis=(-1, -2), keepdims=True) / (xp.sum(abs2(delta_P * group_obj[:, slice_i, None])) + gamma)

            probe_update = beta_probe * xp.sum(alpha_P * delta_P_avg * group_obj_mag, axis=0) / (group_obj_mag + eps)
            sim.state.probe.data += probe_update

        return (sim, chi)

    (sim, chi) = slice_backwards(props, (sim, chi), update_slice)

    if update_position:
        def calc_pos_step(probes_fft: NDArray[numpy.complexfloating], kx: NDArray[numpy.floating]) -> NDArray[numpy.floating]:
            delta_P_x = ifft2(probes_fft * -2.j*numpy.pi * kx)

            prod = delta_P_x * group_obj[:, 0, None]
            alpha = xp.sum(xp.real(chi * xp.conj(prod)), axis=(1, -1, -2)) / xp.sum(abs2(prod))
            return alpha

        # update directions
        probes_fft = fft2(probes)
        probes_fft /= xp.sum(abs2(probes), axis=(1, -1, -2), keepdims=True)
        pos_update = xp.stack(tuple(calc_pos_step(probes_fft, k) for k in (sim.ky, sim.kx)), axis=-1)
    else:
        pos_update = None

    return (sim, new_obj_mag, new_probe_mag, errors, pos_update)


class EPIESolver(ConventionalSolver):
    def __init__(self, engine_plan: ConventionalEnginePlan, props: EPIESolverPlan):
        self.engine_plan: ConventionalEnginePlan = engine_plan
        self.plan: EPIESolverPlan = props

    @classmethod
    def name(cls) -> str:
        return "ePIE"

    def init(self, sim: SimulationState) -> SimulationState:
        self.logger = logging.getLogger(__name__)
        return sim

    def presolve(
        self,
        sim: SimulationState,
        groups: t.Iterator[NDArray[numpy.int_]], *,
        patterns: NDArray[numpy.floating],
        pattern_mask: NDArray[numpy.floating],
        propagators: t.Optional[NDArray[numpy.complexfloating]],
    ) -> SimulationState:
        rescale_factors = []
        for (group, group_patterns) in stream_patterns(groups, patterns, xp=sim.xp,
                                                       buf_n=self.engine_plan.buffer_n_groups):
            group_rescale_factors = epie_dry_run(
                sim, group, group_patterns, pattern_mask=pattern_mask, props=propagators
            )
            rescale_factors.append(to_numpy(group_rescale_factors))

        rescale_factors = numpy.concatenate(rescale_factors, axis=0)
        rescale_factor = numpy.mean(rescale_factors)

        self.logger.info("Pre-calculated intensities")
        self.logger.info(f"Rescaling initial probe intensity by {rescale_factor:.2e}")
        sim.state.probe.data *= numpy.sqrt(rescale_factor)

        return sim

    def run_iteration(
        self,
        sim: SimulationState,
        groups: t.Iterator[NDArray[numpy.int_]], *,
        patterns: NDArray[numpy.floating],
        pattern_mask: NDArray[numpy.floating],
        propagators: t.Optional[NDArray[numpy.complexfloating]],
        update_object: bool,
        update_probe: bool,
        update_positions: bool,
        calc_error: bool,
        calc_error_mask: NDArray[numpy.bool_],
        observer: 'Observer',
    ) -> t.Tuple[SimulationState, NDArray[numpy.floating], t.List[NDArray[numpy.floating]]]:
        xp = sim.xp

        # TODO: ePIE position update
        pos_update = xp.zeros_like(sim.state.scan)
        iter_errors = []

        beta_object = process_schedule(self.plan.beta_object)({'state': sim.state, 'niter': self.engine_plan.niter})
        beta_probe = process_schedule(self.plan.beta_probe)({'state': sim.state, 'niter': self.engine_plan.niter})

        for (group_i, (group, group_patterns)) in enumerate(stream_patterns(groups, patterns, xp=xp,
                                                                            buf_n=self.engine_plan.buffer_n_groups)):
            group_calc_error = calc_error and calc_error_mask[group_i]

            (sim, errors, group_pos_update) = epie_run(
                sim, group, group_patterns,
                pattern_mask=pattern_mask,
                props=propagators,
                beta_object=beta_object,
                beta_probe=beta_probe,
                update_object=update_object,
                update_probe=update_probe,
            )
            check_finite(sim.state.object.data, sim.state.probe.data, context=f"object or probe, group {group_i}")
            #assert sim.state.object.data.dtype == to_complex_dtype(sim.dtype)
            #assert sim.state.probe.data.dtype == to_complex_dtype(sim.dtype)

            sim = sim.apply_group_constraints(group)

            if update_positions:
                assert group_pos_update is not None
                pos_update = at(pos_update, tuple(group)).set(group_pos_update)

            observer.update_group(sim.state, self.engine_plan.send_every_group)

            if group_calc_error:
                assert errors is not None
                iter_errors.append(errors)

        return (sim, pos_update, iter_errors)


@partial(jit)
def epie_dry_run(
    sim: SimulationState,
    group: NDArray[numpy.integer],
    group_patterns: NDArray[numpy.floating], *,
    pattern_mask: NDArray[numpy.floating],
    props: t.Optional[NDArray[numpy.complexfloating]],
) -> NDArray[numpy.floating]:
    xp = cast_array_module(sim.xp)
    (psi, group_obj, group_scan) = cutout_group(sim.ky, sim.kx, sim.state, group)

    def run_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], psi):
        if prop is not None:
            psi = ifft2(fft2(psi * group_obj[:, slice_i, None]) * prop[:, None])

        return psi

    props = tilt_propagators(sim.ky, sim.kx, sim.state, props,
                             sim.state.tilt[tuple(group)] if sim.state.tilt is not None else None)
    psi = slice_forwards(props, psi, run_slice)

    # modeled and experimental intensity
    # summed over incoherent modes and over the pattern
    model_intensity = xp.sum(abs2(fft2(psi)), axis=(1, -2, -1))
    exp_intensity = xp.sum(group_patterns * pattern_mask, axis=(-2, -1))

    return exp_intensity / model_intensity


@partial(jit, donate_argnames=('sim',), static_argnames=('update_object', 'update_probe', 'update_position', 'calc_error'))
def epie_run(
    sim: SimulationState,
    group: NDArray[numpy.integer],
    group_patterns: NDArray[numpy.floating], *,
    pattern_mask: NDArray[numpy.floating],
    props: t.Optional[NDArray[numpy.complexfloating]],
    beta_object: float = 0.9,
    beta_probe: float = 0.9,
    update_object: bool = True,
    update_probe: bool = True,
    update_position: bool = True,
    calc_error: bool = True,
) -> t.Tuple[SimulationState, t.Optional[NDArray[numpy.floating]], t.Optional[NDArray[numpy.floating]]]:
    xp = cast_array_module(sim.xp)
    obj_grid = sim.state.object.sampling
    n_slices = sim.state.object.data.shape[0]

    (probes, group_obj, group_scan, subpx_filters) = cutout_group(sim.ky, sim.kx, sim.state, group, return_filters=True)
    psi = xp.zeros((n_slices, *probes.shape), dtype=probes.dtype)
    psi = at(psi, 0).set(probes)

    def sim_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], psi):
        if prop is not None:
            psi = at(psi, slice_i + 1).set(
                ifft2(fft2(psi[slice_i] * group_obj[:, slice_i, None]) * prop[:, None])
            )

        return psi

    props = tilt_propagators(sim.ky, sim.kx, sim.state, props,
                             sim.state.tilt[tuple(group)] if sim.state.tilt is not None else None)
    psi = slice_forwards(props, psi, sim_slice)

    model_wave = fft2(psi[-1] * group_obj[:, -1, None])
    # sum over incoherent modes
    model_intensity = xp.sum(abs2(model_wave), axis=1, keepdims=True)

    errors = xp.sqrt(xp.nansum((model_intensity - group_patterns[:, None])**2, axis=(1, -1, -2))) if calc_error else None
    (chi, sim.noise_model_state) = sim.noise_model.calc_wave_update(
        model_wave, model_intensity, group_patterns[:, None], pattern_mask, sim.noise_model_state
    )
    chi = ifft2(chi)

    def update_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], state):
        (sim, chi) = state

        probe_update = group_obj[:, slice_i, None].conj() * chi / xp.max(
            abs2(group_obj[:, slice_i, None]),
            axis=(-1, -2), keepdims=True
        )

        if update_object:
            # average incoherent modes
            group_obj_update = beta_object/n_slices * xp.sum(psi[slice_i].conj() * chi, axis=1) / xp.max(
                xp.sum(abs2(psi[slice_i]), axis=1),
                axis=(-1, -2), keepdims=True
            )

            sim.state.object.data = at(sim.state.object.data, slice_i).set(
                obj_grid.add_view_at_pos(sim.state.object.data[slice_i], group_scan, group_obj_update)
            )

        if prop is not None:
            chi = ifft2(fft2(probe_update) * prop.conj()[:, None])
        elif update_probe:
            # average probe updates in group
            probe_update = ifft2(xp.mean(fft2(probe_update) * subpx_filters.conj(), axis=0))
            sim.state.probe.data += beta_probe * probe_update

        return (sim, chi)

    (sim, chi) = slice_backwards(props, (sim, chi), update_slice)

    if update_position:
        def calc_pos_step(probes_fft: NDArray[numpy.complexfloating], kx: NDArray[numpy.floating]) -> NDArray[numpy.floating]:
            delta_P_x = ifft2(probes_fft * -2.j*numpy.pi * kx)

            prod = delta_P_x * group_obj[:, 0, None]
            alpha = xp.sum(xp.real(chi * xp.conj(prod)), axis=(1, -1, -2)) / xp.sum(abs2(prod))
            return alpha

        # update directions
        probes_fft = fft2(probes)
        probes_fft /= xp.sum(abs2(probes), axis=(1, -1, -2), keepdims=True)
        pos_update = xp.stack(tuple(calc_pos_step(probes_fft, k) for k in (sim.ky, sim.kx)), axis=-1)
    else:
        pos_update = None

    return (sim, errors, pos_update)
