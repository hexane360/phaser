import collections
import logging
import typing as t

import numpy
from numpy.typing import NDArray, DTypeLike
from typing_extensions import Self

from phaser.utils.num import (
    get_array_module, to_real_dtype, to_complex_dtype,
    fft2, ifft2, is_jax, to_numpy, block_until_ready, ufunc_outer
)
from phaser.utils.misc import FloatKey, jax_dataclass, create_compact_groupings, create_sparse_groupings, shuffled
from phaser.utils.optics import fresnel_propagator, fourier_shift_filter
from phaser.state import ReconsState
from phaser.hooks.solver import NoiseModel
from phaser.hooks.regularization import GroupConstraint, IterConstraint, StateT

logger = logging.getLogger(__name__)


class GroupManager:
    def __init__(
        self,
        scan: NDArray[numpy.floating],
        grouping: t.Optional[int] = None,
        compact: bool = False,
        seed: t.Any = None,
    ):
        self.grouping = grouping or 64
        self.compact = compact
        self.seed = seed
        self.groups: t.Optional[t.List[NDArray[numpy.int64]]] = None
        self.n_groups: int = int(numpy.ceil(numpy.prod(scan.shape[:-1]) / self.grouping))

    def _make(self, scan: NDArray[numpy.floating], i: int = 0) -> t.List[NDArray[numpy.int64]]:
        if self.compact:
            return create_compact_groupings(scan, self.grouping, seed=self.seed, i=i)
        else:
            return create_sparse_groupings(scan, self.grouping, seed=self.seed, i=i)

    def iter(
        self, scan: NDArray[numpy.floating],
        i: int = 0, shuffle_groups: bool = False,
    ) -> t.Iterator[NDArray[numpy.int64]]:
        if shuffle_groups or self.groups is None:
            self.groups = self._make(scan, i)
            return iter(self.groups)
        # shuffle order of groups (though not the groups themselves)
        return shuffled(self.groups, seed=self.seed, i=i)

    def __len__(self) -> int:
        return self.n_groups


def stream_patterns(
    groups: t.Iterable[NDArray[numpy.int64]], patterns: NDArray[numpy.floating],
    xp: t.Any, buf_n: int = 1
) -> t.Iterator[t.Tuple[NDArray[numpy.int64], NDArray[numpy.floating]]]:
    if buf_n == 0:
        for group in groups:
            group_patterns = xp.asarray(patterns[tuple(group)])
            yield group, block_until_ready(group_patterns)
        return

    buf = collections.deque()
    it = iter(groups)

    for group in it:
        buf.append((group, xp.asarray(patterns[tuple(group)])))
        if len(buf) >= buf_n:
            break

    while len(buf) > 0:
        (group, group_patterns) = buf.popleft()
        yield group, block_until_ready(group_patterns)

        # attempt to feed queue
        try:
            group = next(it)
            buf.append((group, xp.asarray(patterns[tuple(group)])))
        except StopIteration:
            continue


@jax_dataclass(init=False, static_fields=('xp', 'dtype', 'noise_model', 'group_constraints', 'iter_constraints'), drop_fields=('ky', 'kx'))
class SimulationState:
    state: ReconsState

    ky: NDArray[numpy.floating]
    kx: NDArray[numpy.floating]

    noise_model: NoiseModel
    group_constraints: t.Tuple[GroupConstraint[t.Any], ...]
    iter_constraints: t.Tuple[IterConstraint[t.Any], ...]

    noise_model_state: t.Any
    group_constraint_states: t.Tuple[t.Any, ...]
    iter_constraint_states: t.Tuple[t.Any, ...]

    xp: t.Any
    dtype: DTypeLike
    start_iter: int

    def __init__(
        self, *,
        state: ReconsState,
        noise_model: NoiseModel[t.Any],
        group_constraints: t.Tuple[GroupConstraint[t.Any], ...],
        iter_constraints: t.Tuple[IterConstraint[t.Any], ...],
        xp: t.Any,
        dtype: DTypeLike,
        noise_model_state: t.Optional[t.Any] = None,
        group_constraint_states: t.Optional[t.Tuple[t.Any, ...]] = None,
        iter_constraint_states: t.Optional[t.Tuple[t.Any, ...]] = None,
        start_iter: t.Optional[int] = None,
    ):
        self.xp = xp
        self.dtype = dtype
        self.state = state

        self.noise_model = noise_model
        self.group_constraints = group_constraints
        self.iter_constraints = iter_constraints

        self.noise_model_state = noise_model_state or noise_model.init_state(self.state)
        self.group_constraint_states = group_constraint_states if group_constraint_states is not None else tuple(
            reg.init_state(self.state) for reg in group_constraints
        )
        self.iter_constraint_states = iter_constraint_states if iter_constraint_states is not None else tuple(
            reg.init_state(self.state) for reg in iter_constraints
        )

        self.start_iter = start_iter if start_iter is not None else self.state.iter.total_iter
        (self.ky, self.kx) = state.probe.sampling.recip_grid(dtype=dtype, xp=xp)

    def apply_group_constraints(self, group: NDArray[numpy.integer]) -> Self:
        def apply_reg(reg: GroupConstraint[t.Any], state: t.Any):
            (self.state, state) = reg.apply_group(group, self.state, state)
            return state

        self.group_constraint_states = tuple(
            apply_reg(reg, state) for (reg, state) in zip(self.group_constraints, self.group_constraint_states)
        )

        return self

    def apply_iter_constraints(self) -> Self:
        def apply_reg(reg: IterConstraint[t.Any], state: t.Any):
            (self.state, state) = reg.apply_iter(self.state, state)
            return state

        self.iter_constraint_states = tuple(
            apply_reg(reg, state) for (reg, state) in zip(self.iter_constraints, self.iter_constraint_states)
        )

        return self


def make_propagators(state: ReconsState, bwlim_frac: t.Optional[float] = 2/3) -> t.Optional[NDArray[numpy.complexfloating]]:
    xp = get_array_module(state.probe.data)
    dtype = to_real_dtype(state.probe.data.dtype)
    complex_dtype = to_complex_dtype(dtype)

    (ky, kx) = state.probe.sampling.recip_grid(xp=xp, dtype=dtype)

    # ignore last slice; we don't need it
    delta_zs = to_numpy(state.object.thicknesses)[:-1]
    if len(delta_zs) == 0:
        return None

    unique_zs = set(map(FloatKey, delta_zs))

    if bwlim_frac is not None:
        bwlim = numpy.min(state.probe.sampling.k_max) * bwlim_frac
        k2 = ky**2 + kx**2
        bwlim_mask = k2 <= bwlim**2
        logger.info(f"Bandwidth limit: {bwlim * state.wavelength * 1e3:6.2f} mrad")
    else:
        bwlim_mask = xp.ones(ky.shape, dtype=numpy.bool_)

    props = {
        z: fresnel_propagator(ky, kx, state.wavelength, z).astype(complex_dtype) * bwlim_mask
        for z in unique_zs
    }

    return xp.stack(
        [props[FloatKey(z)] for z in delta_zs],
        axis = 0
    )


def tilt_propagators(
    state: ReconsState, 
    base_props: t.Optional[NDArray[numpy.complexfloating]],  # shape: (Nz-1, Ny, Nx)
    group_tilts: NDArray[numpy.floating],                    # shape: (batch, 2), in mrad
    kx: NDArray[numpy.floating],                             # shape: (Ny, Nx)
    ky: NDArray[numpy.floating],                             # shape: (Ny, Nx)
    delta_zs: NDArray[numpy.floating],                       # shape: (Nz-1)
) -> t.Optional[NDArray[numpy.complexfloating]]:
    """
    Applies tilt and slice-dependent propagation phase shifts to base_props.
    -------
    NDArray[complex] or None
        Tilted propagators of shape (batch, Nz-1, Ny, Nx), or None if no slices.
    """
    if base_props is None:
        return None

    xp = get_array_module(state.probe.data)
    dtype = to_real_dtype(state.probe.data.dtype)
    complex_dtype = to_complex_dtype(dtype)

    ## Convert mrad to rad via tan(mrad / 1000)
    # tiltx = xp.tan(group_tilts[:, 0] * 1e-3)  # shape (batch,)
    # tilty = xp.tan(group_tilts[:, 1] * 1e-3)  # shape (batch,)
    # tiltx_kx = ufunc_outer(xp.multiply, tiltx, kx)  # (batch, Ny, Nx)
    # tilty_ky = ufunc_outer(xp.multiply, tilty, ky)  # (batch, Ny, Nx)
    # tilt_sum = 2j * xp.pi * tiltx_kx + tilty_ky # (batch, Ny, Nx)
    # phase = ufunc_outer(xp.multiply, delta_zs, tilt_sum).transpose(1, 0, 2, 3) #(batch, Nz-1, Ny, Nx)
    # tilt_ramps = xp.exp(phase)

    tilt_ramps = xp.exp(
        ufunc_outer(
            xp.multiply,
            delta_zs,
            2j * xp.pi * (
                ufunc_outer(xp.multiply, xp.tan(group_tilts[:, 0] * 1e-3), kx) +
                ufunc_outer(xp.multiply, xp.tan(group_tilts[:, 1] * 1e-3), ky)
                )
            ).transpose(1, 0, 2, 3))  # (batch, Nz-1, Ny, Nx)
    
    return (base_props[None, :, :, :] * tilt_ramps).astype(complex_dtype) 


@t.overload
def cutout_group(
    ky: NDArray[numpy.floating], kx: NDArray[numpy.floating],
    state: ReconsState, group: NDArray[numpy.integer],
    return_filters: t.Literal[False] = False
) -> t.Tuple[NDArray[numpy.complexfloating], NDArray[numpy.complexfloating], NDArray[numpy.floating]]:
    ...

@t.overload
def cutout_group(
    ky: NDArray[numpy.floating], kx: NDArray[numpy.floating],
    state: ReconsState, group: NDArray[numpy.integer],
    return_filters: t.Literal[True]
) -> t.Tuple[NDArray[numpy.complexfloating], NDArray[numpy.complexfloating], NDArray[numpy.floating], NDArray[numpy.complexfloating]]:
    ...

def cutout_group(
    ky: NDArray[numpy.floating], kx: NDArray[numpy.floating],
    state: ReconsState, group: NDArray[numpy.integer],
    return_filters: bool = False
):
    """Returns (probe, obj) in the cutout region"""
    probes = state.probe.data

    group_scan = state.scan[tuple(group)]
    group_obj = state.object.sampling.get_view_at_pos(state.object.data, group_scan, probes.shape[-2:])
    # group probes in real space
    # shape (len(group), 1, Ny, Nx)
    group_subpx_filters = fourier_shift_filter(ky, kx, state.object.sampling.get_subpx_shifts(group_scan, probes.shape[-2:]))[:, None, ...]
    # shape (len(group), probe modes, Ny, Nx)
    shifted_probes = ifft2(fft2(probes) * group_subpx_filters)

    if return_filters:
        return (shifted_probes, group_obj, group_scan, group_subpx_filters)

    return (shifted_probes, group_obj, group_scan)


def slice_forwards(
    props: t.Optional[NDArray[numpy.complexfloating]],
    state: StateT,
    f: t.Callable[[int, t.Optional[NDArray[numpy.complexfloating]], StateT], StateT]
) -> StateT:
    if props is None:
        return f(0, None, state)

    n_slices = props.shape[1] + 1  # props shape: (batch, Nz-1, Ny, Nx)

    if is_jax(props):
        import jax
        def step_fn(carry, slice_i):
            # props[:, slice_i] has shape (batch, Ny, Nx)
            new_state = f(slice_i, props[:, slice_i], carry)
            return new_state, None

        state, _ = jax.lax.scan(step_fn, state, jax.numpy.arange(n_slices - 1))
        return f(n_slices - 1, None, state)

    # fallback numpy mode
    for slice_i in range(n_slices - 1):
        state = f(slice_i, props[:, slice_i], state)
    return f(n_slices - 1, None, state)


def slice_backwards(
    props: t.Optional[NDArray[numpy.complexfloating]],
    state: StateT,
    f: t.Callable[[int, t.Optional[NDArray[numpy.complexfloating]], StateT], StateT]
) -> StateT:
    if props is None:
        return f(0, None, state)

    n_slices = len(props) + 1

    if is_jax(props):
        import jax
        state = jax.lax.fori_loop(1, n_slices, lambda i, state: f(n_slices - i, props[n_slices - i - 1], state), state, unroll=False)
        return f(0, None, state)

    for slice_i in range(n_slices - 1, 0, -1):
        state = f(slice_i, props[slice_i - 1], state)

    return f(0, None, state)