"""Simulation state, scan grouping, and multislice helpers."""

import collections
import logging
import typing as t

import numpy
from numpy.typing import NDArray
from typing_extensions import Self

from phaser.hooks.filter import FilterArgs, FilterHook
from phaser.hooks.regularization import GroupConstraint, IterConstraint, StateT
from phaser.hooks.solver import NoiseModel
from phaser.plan import MtfPlan
from phaser.state import ReconsState
from phaser.utils.image import (
    PreparedOTF,
    PreparedPSF,
    prepare_convolve2d,
    prepare_convolve2d_recip,
)
from phaser.utils.misc import (
    FloatKey,
    create_compact_groupings,
    create_sparse_groupings,
    shuffled,
)
from phaser.utils.num import (
    Sampling,
    block_until_ready,
    fft2,
    get_array_module,
    ifft2,
    is_jax,
    to_complex_dtype,
    to_numpy,
    to_real_dtype,
    ufunc_outer,
)
from phaser.utils.optics import fourier_shift_filter, fresnel_propagator
from phaser.utils.tree import tree_dataclass

logger = logging.getLogger(__name__)


class GroupManager:
    """Manager for creating and iterating over groups of probe positions."""

    def __init__(
        self,
        scan: NDArray[numpy.floating],
        grouping: t.Optional[int] = None,
        compact: bool = False,
        seed: t.Any = None,
    ):
        """
        Args:
            scan: Probe positions `(..., 2)`.
            grouping: Group size. Defaults to 64.
            compact: Form spatially compact groups instead of sparse ones.
            seed: Seed for group formation and shuffling.
        """
        self.grouping = grouping or 64
        self.compact = compact
        self.seed = seed
        self.groups: t.Optional[t.List[NDArray[numpy.int64]]] = None
        self.n_pos = numpy.prod(scan.shape[:-1])
        self.n_groups: int = int(numpy.ceil(self.n_pos / self.grouping))

    def _make(self, scan: NDArray[numpy.floating], i: int = 0) -> t.List[NDArray[numpy.int64]]:
        if self.compact:
            return create_compact_groupings(scan, self.grouping, seed=self.seed, i=i)
        else:
            return create_sparse_groupings(scan, self.grouping, seed=self.seed, i=i)

    def iter(
        self, scan: NDArray[numpy.floating],
        i: int = 0, shuffle_groups: bool = False,
    ) -> t.Iterator[NDArray[numpy.int64]]:
        """Iterate over groups of scan indices.

        Groups are re-formed if `shuffle_groups` is set or none exist yet. Otherwise
        the existing groups are visited in a new order.
        """
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
    """Yield `(group, patterns)`, transferring patterns to `xp` and prefetching `buf_n` groups ahead.

    Each yielded array is ready on the device.
    """
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
        # attempt to feed queue
        try:
            group = next(it)
            buf.append((group, xp.asarray(patterns[tuple(group)])))
        except StopIteration:
            pass

        # and yield from front of queue
        (group, group_patterns) = buf.popleft()
        yield group, block_until_ready(group_patterns)


@tree_dataclass(init=False, static_fields=('xp', 'dtype', 'noise_model', 'group_constraints', 'iter_constraints'), drop_fields=('ky', 'kx'))
class SimulationState:
    """Reconstruction state with the noise model, constraints, and their states."""

    state: ReconsState
    """Reconstruction state."""

    ky: NDArray[numpy.floating]
    """Probe reciprocal-space y coordinates [1/angstrom]."""
    kx: NDArray[numpy.floating]
    """Probe reciprocal-space x coordinates [1/angstrom]."""

    noise_model: NoiseModel
    """Noise model."""
    group_constraints: t.Tuple[GroupConstraint[t.Any], ...]
    """Constraints applied after each group."""
    iter_constraints: t.Tuple[IterConstraint[t.Any], ...]
    """Constraints applied after each iteration."""

    noise_model_state: t.Any
    """Noise model state."""
    group_constraint_states: t.Tuple[t.Any, ...]
    """States of `group_constraints`."""
    iter_constraint_states: t.Tuple[t.Any, ...]
    """States of `iter_constraints`."""

    xp: t.Any
    """Array module."""
    dtype: t.Type[numpy.floating]
    """Real floating-point dtype."""
    start_iter: int
    """Total iteration count at the start of the run."""

    def __init__(
        self, *,
        state: ReconsState,
        noise_model: NoiseModel[t.Any],
        group_constraints: t.Tuple[GroupConstraint[t.Any], ...],
        iter_constraints: t.Tuple[IterConstraint[t.Any], ...],
        xp: t.Any,
        dtype: t.Type[numpy.floating],
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
        """Apply `group_constraints` for `group`, updating `state` and the constraint states in place."""
        def apply_reg(reg: GroupConstraint[t.Any], state: t.Any):
            (self.state, state) = reg.apply_group(group, self.state, state)
            return state

        self.group_constraint_states = tuple(
            apply_reg(reg, state) for (reg, state) in zip(self.group_constraints, self.group_constraint_states)
        )

        return self

    def apply_iter_constraints(self) -> Self:
        """Apply `iter_constraints`, updating `state` and the constraint states in place."""
        def apply_reg(reg: IterConstraint[t.Any], state: t.Any):
            (self.state, state) = reg.apply_iter(self.state, state)
            return state

        self.iter_constraint_states = tuple(
            apply_reg(reg, state) for (reg, state) in zip(self.iter_constraints, self.iter_constraint_states)
        )

        return self


def make_propagators(state: ReconsState, bwlim_frac: t.Optional[float] = 2/3) -> t.Optional[NDArray[numpy.complexfloating]]:
    """Return Fresnel propagators between adjacent object slices, shape `(n_slices - 1, Ny, Nx)`.

    Args:
        bwlim_frac: Bandwidth limit as a fraction of the minimum probe `k_max`. `None` disables the limit.

    Returns:
        `None` if the object has a single slice.
    """
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


def prepare_mtf(
    mtf: t.Union[MtfPlan, FilterHook],
    state: ReconsState,
    dtype: t.Type[numpy.floating],
    xp: t.Any,
) -> t.Union[PreparedOTF[numpy.inexact], PreparedPSF[numpy.floating]]:
    """Prepare a detector MTF for convolution with simulated intensities of shape `state.probe.data.shape[-2:]`."""
    mtf_args: FilterArgs = {'wavelength': float(state.wavelength)}
    if isinstance(mtf, MtfPlan):
        filt = mtf.filter(mtf_args)
        domain = mtf.domain
    else:
        filt = mtf(mtf_args)
        domain = 'recip'

    samp = Sampling(state.probe.data.shape[-2:], sampling=(1., 1.))
    prepare = prepare_convolve2d if domain == 'real' else prepare_convolve2d_recip
    return prepare(filt, samp, xp=xp, dtype=dtype)


def tilt_propagators(
    ky: NDArray[numpy.floating], kx: NDArray[numpy.floating],
    state: ReconsState, 
    props: t.Optional[NDArray[numpy.complexfloating]],  # shape: (Nz-1, Ny, Nx)
    tilts: t.Optional[NDArray[numpy.floating]]          # shape: (..., 2), in mrad
) -> t.Optional[NDArray[numpy.complexfloating]]:
    """Return `props` with phase ramps for beam `tilts` [mrad] applied.

    Args:
        props: Propagators `(n_slices - 1, Ny, Nx)`, as from `make_propagators`.
        tilts: Tilts `(..., 2)` in `(y, x)` order [mrad]. `None` for untilted.

    Returns:
        Propagators `(n_slices - 1, ..., Ny, Nx)`, or `None` if `props` is `None`.
        Without `tilts`, a length-1 axis is inserted after the slice axis.
    """
    if props is None:
        return None
    if tilts is None:
        return props[:, None, ...]

    xp = get_array_module(state.probe.data)
    dtype = to_real_dtype(state.probe.data.dtype)
    complex_dtype = to_complex_dtype(dtype)
    delta_zs = state.object.thicknesses[:-1]

    tilt_ramps = xp.exp(  # (n_layers-1, batch, Ny, Nx)
        2.j * xp.pi * ufunc_outer(xp.multiply, delta_zs, (
            ufunc_outer(xp.multiply, xp.tan(tilts[..., 0] * 1e-3), ky) +
            ufunc_outer(xp.multiply, xp.tan(tilts[..., 1] * 1e-3), kx)
        ))
    )

    return props[(slice(None), *(None,)*(tilts.ndim - 1), Ellipsis)] * tilt_ramps.astype(complex_dtype)


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
    """Return `(probes, obj, scan)`: sub-pixel shifted probes, object cutouts, and scan positions for `group`.

    Probes have shape `(len(group), modes, Ny, Nx)`. If `return_filters`, the sub-pixel shift
    filters are appended.
    """
    probes = state.probe.data

    group_scan = state.scan.data[tuple(group)]
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
    f: t.Callable[[int, t.Optional[NDArray[numpy.complexfloating]], StateT], StateT], *,
    jit_unroll_slices: t.Union[int, bool] = False,
) -> StateT:
    """Fold `f` forwards over slices.

    `f` receives the slice index, the propagator to the next slice (`None` for the last
    slice), and the state, and returns the updated state. `jit_unroll_slices` is passed
    to `jax.lax.scan` as `unroll`.
    """
    if props is None:
        return f(0, None, state)

    n_slices = len(props) + 1  # props shape: (Nz-1, batch, Ny, Nx)

    if is_jax(props):
        import jax
        def step_fn(carry, slice_i):
            new_state = f(slice_i, props[slice_i], carry)
            return new_state, None

        state, _ = jax.lax.scan(step_fn, state, jax.numpy.arange(n_slices - 1), unroll=jit_unroll_slices)
        return f(n_slices - 1, None, state)

    # fallback numpy mode
    for slice_i in range(n_slices - 1):
        state = f(slice_i, props[slice_i], state)
    return f(n_slices - 1, None, state)


def slice_backwards(
    props: t.Optional[NDArray[numpy.complexfloating]],
    state: StateT,
    f: t.Callable[[int, t.Optional[NDArray[numpy.complexfloating]], StateT], StateT],
    jit_unroll_slices: t.Union[int, bool] = False,
) -> StateT:
    """Fold `f` backwards over slices.

    `f` receives the slice index, the propagator from the previous slice (`None` for the
    first slice), and the state, and returns the updated state. `jit_unroll_slices` is
    passed to `jax.lax.fori_loop` as `unroll`.
    """
    if props is None:
        return f(0, None, state)

    n_slices = len(props) + 1

    if is_jax(props):
        import jax
        state = jax.lax.fori_loop(
            1, n_slices, lambda i, state: f(n_slices - i, props[n_slices - i - 1], state), state,
            unroll=jit_unroll_slices
        )
        return f(0, None, state)

    for slice_i in range(n_slices - 1, 0, -1):
        state = f(slice_i, props[slice_i - 1], state)

    return f(0, None, state)
