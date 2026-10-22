"""Simulation state, scan grouping, and multislice helpers."""

import abc
import collections
import concurrent.futures
import logging
import threading
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
    host_transfer_fn,
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


@tree_dataclass
class GroupPatterns:
    """Lazy view of the patterns of a group, within patterns already on the device.

    Passed to (JIT) kernels in place of the group's patterns, so that the gather happens
    inside the kernel: an eager gather costs several ms of host time per group on JAX.
    Kernels call `resolve_patterns` to get the patterns.
    """

    patterns: NDArray[numpy.floating]
    """Patterns of all positions, `(*scan_shape, Ny, Nx)`."""
    group: NDArray[numpy.int64]
    """Group indices into the scan dimensions, `(scan ndim, group size)`."""

    def get(self) -> NDArray[numpy.floating]:
        """Return the patterns of the group, `(group size, Ny, Nx)`."""
        xp = get_array_module(self.patterns)
        return self.patterns[tuple(xp.asarray(self.group))]


PatternsLike: t.TypeAlias = t.Union[NDArray[numpy.floating], GroupPatterns]
"""The patterns of a group, as yielded by a `PatternSource`."""


def resolve_patterns(patterns: PatternsLike) -> NDArray[numpy.floating]:
    """Return the patterns of a group, gathering them if `patterns` is a `GroupPatterns` view."""
    if isinstance(patterns, GroupPatterns):
        return patterns.get()
    return patterns


class PatternSource(abc.ABC):
    """Iterable of `(group, group_patterns)`, which limits how far the caller runs ahead of the device.

    `group_patterns` may be a `GroupPatterns` view, so kernels must call `resolve_patterns` on it.

    After dispatching work on a group, call `release_after` with an output of that work. What
    a released group allows depends on the implementation (see `PatternStream` and
    `ResidentPatterns`). The output must stay valid until it's ready, so it must not be
    donated to later work.

    Use as a context manager (or call `close`) to stop any background work promptly when
    iteration is abandoned.
    """

    def __init__(self):
        self._release: t.Any = None
        self._it: t.Optional[t.Generator[t.Tuple[NDArray[numpy.int64], PatternsLike], None, None]] = None

    def release_after(self, arr: t.Any):
        """Release the most recently yielded group once `arr` is ready."""
        self._release = arr

    @abc.abstractmethod
    def _iter(self) -> t.Generator[t.Tuple[NDArray[numpy.int64], PatternsLike], None, None]:
        ...

    def __iter__(self) -> t.Iterator[t.Tuple[NDArray[numpy.int64], PatternsLike]]:
        self.close()
        self._it = self._iter()
        return self._it

    def close(self):
        """Stop iteration, and any background work."""
        if self._it is not None:
            self._it.close()
            self._it = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: t.Any) -> None:
        self.close()


class PatternStream(PatternSource):
    """`PatternSource` streaming patterns from the host to the device.

    At most `buf_n` groups are kept on the device at once, counted from the start of their
    transfer until they are released. Patterns are gathered and transferred in background
    threads, so the next groups are transferred while earlier groups are processed.

    A released group allows the transfer of a later group to start, once the output passed
    to `release_after` is ready. Groups which aren't released explicitly are released once
    their own transfer completes.

    With `buf_n == 0`, each group is transferred synchronously, when it is requested.
    """

    def __init__(
        self, groups: t.Iterable[NDArray[numpy.int64]], patterns: NDArray[numpy.floating],
        xp: t.Any, buf_n: int = 1,
    ):
        super().__init__()
        if buf_n < 0:
            raise ValueError(f"'buf_n' must be non-negative, got {buf_n}")

        self._groups = groups
        self._patterns = patterns
        self._transfer = host_transfer_fn(xp)
        self._buf_n = buf_n

    def _load(
        self, group: NDArray[numpy.int64], release: t.Any, closed: t.Optional[threading.Event] = None
    ) -> t.Optional[NDArray[numpy.floating]]:
        # wait for a previous group to be released
        if release is not None:
            block_until_ready(release)
        # don't start a transfer if iteration was abandoned while waiting
        if closed is not None and closed.is_set():
            return None
        return self._transfer(self._patterns[tuple(group)])

    def _iter(self) -> t.Generator[t.Tuple[NDArray[numpy.int64], PatternsLike], None, None]:
        if self._buf_n == 0:
            for group in self._groups:
                yield group, block_until_ready(t.cast(NDArray[numpy.floating], self._load(group, None)))
            return

        groups = iter(self._groups)
        # (group, future of group_patterns)
        pending: t.Deque[t.Tuple[NDArray[numpy.int64], concurrent.futures.Future]] = collections.deque()

        # numpy releases the GIL while gathering, as do the device transfers
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self._buf_n, thread_name_prefix='pattern_stream')
        closed = threading.Event()

        def submit(release: t.Any):
            if (group := next(groups, None)) is not None:
                pending.append((group, pool.submit(self._load, group, release, closed)))

        try:
            for _ in range(self._buf_n):
                submit(None)

            while len(pending) > 0:
                (group, fut) = pending.popleft()
                group_patterns = t.cast(NDArray[numpy.floating], fut.result())
                self._release = group_patterns
                yield group, group_patterns
                # releasing this group allows another to be loaded
                submit(self._release)
        finally:
            closed.set()
            pool.shutdown(wait=False, cancel_futures=True)


class ResidentPatterns(PatternSource):
    """`PatternSource` for patterns already on the device, yielding `GroupPatterns` views.

    Before yielding a group, waits until the group `max_in_flight` groups earlier is
    released, i.e. until the output passed to `release_after` is ready. Groups which aren't
    released explicitly don't limit iteration.

    The default `max_in_flight=2` keeps a group queued behind the one running, so the
    device doesn't idle while the next group is dispatched.
    """

    def __init__(
        self, groups: t.Iterable[NDArray[numpy.int64]], patterns: NDArray[numpy.floating],
        max_in_flight: int = 2,
    ):
        super().__init__()
        if max_in_flight < 1:
            raise ValueError(f"'max_in_flight' must be positive, got {max_in_flight}")

        self._groups = groups
        self._patterns = patterns
        self._max_in_flight = max_in_flight

    def _iter(self) -> t.Generator[t.Tuple[NDArray[numpy.int64], PatternsLike], None, None]:
        # outputs of groups which haven't been released yet
        in_flight: t.Deque[t.Any] = collections.deque()

        for group in self._groups:
            while len(in_flight) >= self._max_in_flight:
                block_until_ready(in_flight.popleft())

            self._release = None
            yield group, GroupPatterns(self._patterns, group)
            if self._release is not None:
                in_flight.append(self._release)


def iter_patterns(
    groups: t.Iterable[NDArray[numpy.int64]], patterns: NDArray[numpy.floating],
    xp: t.Any, buffer_n_groups: t.Optional[int],
) -> PatternSource:
    """Return a `PatternSource` of `(group, patterns)` for `groups`.

    With `buffer_n_groups=None`, `patterns` must already be on the device. Otherwise, they're
    streamed from the host, keeping at most `buffer_n_groups` groups on the device.
    """
    if buffer_n_groups is None:
        return ResidentPatterns(groups, patterns)
    return PatternStream(groups, patterns, xp, buffer_n_groups)


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

    Probes have shape `(len(group), modes, Ny, Nx)`, and object cutouts have shape
    `(slices, len(group), Ny, Nx)`. If `return_filters`, the sub-pixel shift
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
    obj: NDArray[numpy.complexfloating],
    state: StateT,
    f: t.Callable[[int, t.Optional[NDArray[numpy.complexfloating]], NDArray[numpy.complexfloating], StateT], StateT], *,
    jit_unroll_slices: t.Union[int, bool] = False,
) -> StateT:
    """Fold `f` forwards over slices.

    `obj` holds the per-slice object (e.g. object cutouts), with slices along the first axis.
    `f` receives the slice index, the propagator to the next slice (`None` for the last
    slice), that slice of `obj`, and the state, and returns the updated state.
    `jit_unroll_slices` is passed to `jax.lax.scan` as `unroll`.

    On JAX, `obj` is passed to `f` as a scan input rather than indexed with the (traced)
    slice index, so gradients with respect to `obj` are stacked per slice rather than
    accumulated with dynamic-update-slices.
    """
    if props is None:
        return f(0, None, obj[0], state)

    n_slices = len(props) + 1  # props shape: (Nz-1, batch, Ny, Nx)

    if is_jax(props):
        import jax
        def step_fn(carry, xs):
            (slice_i, prop, obj_slice) = xs
            return f(slice_i, prop, obj_slice, carry), None

        state, _ = jax.lax.scan(
            step_fn, state, (jax.numpy.arange(n_slices - 1), props, obj[:-1]), unroll=jit_unroll_slices
        )
        return f(n_slices - 1, None, obj[-1], state)

    # fallback numpy mode
    for slice_i in range(n_slices - 1):
        state = f(slice_i, props[slice_i], obj[slice_i], state)
    return f(n_slices - 1, None, obj[-1], state)


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
