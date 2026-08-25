import functools
import math
import typing as t
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from types import ModuleType

import numpy
from frozendict import frozendict
from numpy.random import PCG64, BitGenerator, Generator, SeedSequence
from numpy.typing import NDArray

T = t.TypeVar('T')


def _proc_seed(seed: object, entropy: object = None) -> SeedSequence:
    """
    Process a random seed, along with additional entropy to use the same
    seed for multiple applications.
    """
    if seed is None:
        return SeedSequence()
    if isinstance(seed, SeedSequence):
        seed = seed.entropy

    # hash our seed and our extra entropy

    import json
    from hashlib import sha256

    state = sha256()
    state.update(json.dumps(seed).encode('utf-8'))
    if entropy is not None:
        state.update(json.dumps(entropy).encode('utf-8'))
    return SeedSequence(numpy.frombuffer(state.digest(), dtype=numpy.uint32))


def create_rng(seed: object = None, entropy: object = None) -> Generator:
    """
    Create a numpy `PCG64` `Generator` using the initial seed (if specified),
    and some additional entropy.

    If `seed` is an existing `Generator` or `BitGenerator`, it's returned.
    Otherwise, `seed` is used along with `entropy` to construct a high-quality
    initial seed.

    The seed and entropy can be anything JSON-writable.

    If no seed is specified, numpy's default methods are used to construct a high-quality
    seed. With a fixed seed specified, this function is designed to provide deterministic
    behavior across different platforms and for long periods of time.
    """
    if isinstance(seed, Generator):
        return seed
    elif isinstance(seed, BitGenerator):
        return Generator(seed)
    seq = _proc_seed(seed, entropy)
    return Generator(PCG64(seq))


def create_rng_group(n: int, seed: object = None, entropy: object = None) -> tuple[Generator, ...]:
    """
    Create a group of `n` distinct `PCG64` `BitGenerator`s using the initial seed (if specified),
    and some additional entropy.

    If `seed` is an existing `Generator` or `BitGenerator`, its underlying seed
    sequence is used to construct the group. Otherwise, `seed` is used along with
    `entropy` to construct a high-quality initial seed.

    The seed and entropy can be anything JSON-writable.

    If no seed is specified, numpy's default methods are used to construct a high-quality
    seed. With a fixed seed specified, this function is designed to provide deterministic
    behavior across different platforms and for long periods of time.
    """
    if isinstance(seed, Generator):
        seq = seed.bit_generator.seed_seq
    elif isinstance(seed, BitGenerator):
        seq = seed.seed_seq
    else:
        seq = _proc_seed(seed, entropy)

    return tuple(map(Generator, map(PCG64, t.cast(SeedSequence, seq).spawn(n))))


def shuffled(vals: t.Sequence[T], seed: t.Any = None, i: int = 0) -> t.Iterator[T]:
    """
    Return an iterator which gives `vals` in a random order.
    """
    idxs = numpy.arange(len(vals))
    rng = create_rng(seed, f"shuffle_{i}")
    rng.shuffle(idxs)

    for idx in idxs:
        yield vals[int(idx)]


def create_sparse_groupings(shape: int | t.Iterable[int] | NDArray[numpy.floating], max_grouping: int = 8,
                            seed: t.Any = None, i: int = 0) -> list[NDArray[numpy.int64]]:
    """
    Randomly partition the indices of `shape` into groups of maximum size `max_grouping`.

    Returns a list of `ceil(n / max_grouping)` groups, of sizes `ceil(n/k)` and `floor(n/k)`.
    Each group can be used to index an array `arr` of shape `shape`: `arr[tuple(group)]`.

    `seed` and `i` are used to randomize the group assignment.
    """
    if isinstance(shape, int):
        shape = (shape,)
    if not isinstance(shape, (tuple, list)):
        # assume `shape` is a list of positions
        shape = shape.shape[:-1]  # type: ignore

    idxs = numpy.indices(shape)  # type: ignore
    idxs = idxs.reshape(idxs.shape[0], -1).T

    rng = create_rng(seed, f'groupings_{i}' if i != 0 else 'groupings')
    rng.shuffle(idxs)
    return numpy.array_split(idxs.T, numpy.ceil(idxs.shape[0] / max_grouping).astype(numpy.int64), axis=-1)


def create_compact_groupings(positions: NDArray[numpy.floating], max_grouping: int = 8,
                             seed: t.Any = None, i: int = 0) -> list[NDArray[numpy.int64]]:
    """
    Partition the indices of `positions` into groups of maximum size `max_grouping`,
    such that each group is spatially compact.

    Returns a list of `ceil(n / max_grouping)` groups, of sizes `ceil(n/k)` and `floor(n/k)`.
    Each group can be used to index an array `arr` of shape `shape`: `arr[tuple(group)]`

    The current algorithm is deterministic, so `seed` and `i` are not used.
    """
    # this is slop

    positions = numpy.asarray(positions, dtype=numpy.float64)
    assert positions.ndim >= 2
    assert max_grouping >= 1

    idxs = numpy.indices(positions.shape[:-1])
    idxs = idxs.reshape(idxs.shape[0], -1)
    positions = positions.reshape(-1, positions.shape[-1])
    n_points = positions.shape[0]
    if n_points == 0:
        return []
    n_groups = -(-n_points // max_grouping)  # ceil
    if n_groups == 1:
        return [idxs]
    if n_groups == n_points:
        return numpy.split(idxs, n_points, -1)
 
    positions = positions - positions.mean(axis=0)  # conditions the sum-of-squares identities
    group_bounds, point_index = _compact_bisect(positions, n_groups)
 
    group_of_point = numpy.empty(n_points, dtype=numpy.int64)
    group_of_point[point_index] = numpy.repeat(
        numpy.arange(n_groups, dtype=numpy.int64), numpy.diff(group_bounds)
    )
    group_of_point = _compact_polish_groups(
        positions, group_of_point, n_groups, float(_sq_norm(positions).sum())
    )
 
    by_group = numpy.argsort(group_of_point, kind="stable")
    group_ends = numpy.cumsum(numpy.bincount(group_of_point, minlength=n_groups))[:-1]
    return [idxs[:, numpy.sort(group)] for group in numpy.split(by_group, group_ends)]


def _prefix_sum(values: numpy.ndarray) -> numpy.ndarray:
    """Prefix sum along axis 0 of `values`, such that values[..., 0] = 0"""
    return numpy.concatenate((
        numpy.zeros((1,) + values.shape[1:], values.dtype),
        numpy.cumsum(values, axis=0)
    ), axis=0)
 
 
def _sq_norm(vectors: numpy.ndarray) -> numpy.ndarray:
    """Squared norm of `vectors` along last axis"""
    return numpy.sum(vectors**2, axis=-1)
 
 
def _compact_bisect(points: numpy.ndarray, n_groups: int):
    """Recursively halve into balanced groups, one whole tree level per pass.
 
    Returns ``(group_bounds, point_index)``: group ``g`` owns the point indices
    ``point_index[group_bounds[g]:group_bounds[g + 1]]``.
    """
    n_points = points.shape[0]
    point_index = numpy.arange(n_points, dtype=numpy.int64)
    # coords is kept in sync with point_index: coords[p] == points[point_index[p]]
    coords = points.copy()
 
    # A segment is one node of the tree: a contiguous slice of point_index that
    # still owes some number of groups. Segments tile [0, n_points) in order.
    seg_bounds = numpy.array([0, n_points], dtype=numpy.int64)
    groups_owed = numpy.array([n_groups], dtype=numpy.int64)
 
    while (groups_owed > 1).any():
        seg_start, seg_stop = seg_bounds[:-1], seg_bounds[1:]
        seg_size = seg_stop - seg_start
        splitting = groups_owed > 1
        seg_of_point = numpy.repeat(numpy.arange(seg_start.size, dtype=numpy.int64), seg_size)
 
        # Exact size split, derived from (seg_size, groups_owed) alone. A node
        # owing k groups from n points owes `n mod k` of size `ceil(n/k)` and
        # the rest of `floor(n/k)`; sending `min(n mod k, k//2)` of the big ones
        # left keeps that true for both children, all the way to the leaves.
        left_groups = groups_owed // 2
        left_size = numpy.where(
            splitting,
            left_groups * (seg_size // groups_owed)
            + numpy.minimum(seg_size % groups_owed, left_groups),
            0,  # finished nodes are not cut
        )
        seg_split = seg_start + left_size
 
        # Cut each segment across its widest coordinate axis, at the rank that
        # hands the left child exactly the point count it owes.
        sum_x = _prefix_sum(coords)
        sum_x2 = _prefix_sum(coords * coords)
        seg_mean = (sum_x[seg_stop] - sum_x[seg_start]) / seg_size[:, None]
        seg_var = (sum_x2[seg_stop] - sum_x2[seg_start]) / seg_size[:, None]
        seg_var -= seg_mean * seg_mean
        cut_axis = numpy.argmax(seg_var, axis=1)
 
        sort_key = coords[numpy.arange(n_points), cut_axis[seg_of_point]]
        # lexsort is stable, so tied coordinates keep their existing order
        rank_order = numpy.lexsort((sort_key, seg_of_point))
        coords, point_index = coords[rank_order], point_index[rank_order]
 
        # Emit children: splitting nodes produce two, finished nodes pass through.
        child_slot = numpy.concatenate(([0], numpy.cumsum(numpy.where(splitting, 2, 1))))
        child_start = numpy.empty(child_slot[-1], dtype=numpy.int64)
        child_groups = numpy.empty(child_slot[-1], dtype=numpy.int64)
        child_start[child_slot[:-1]] = seg_start
        child_groups[child_slot[:-1]] = numpy.where(splitting, left_groups, groups_owed)
        split_seg = numpy.flatnonzero(splitting)
        child_start[child_slot[split_seg] + 1] = seg_split[split_seg]
        child_groups[child_slot[split_seg] + 1] = (groups_owed - left_groups)[split_seg]
 
        seg_bounds = numpy.concatenate((child_start, [n_points]))
        groups_owed = child_groups
 
    return seg_bounds, point_index
 
 
def _compact_neighboring_groups(points, centroids, group_of_point):
    """Flat list of ``(point, other group)`` candidates worth examining.
 
    In d dimensions, d + 1 cells meet at a generic vertex, so the d nearest
    centroids other than a point's own cover the groups it could plausibly move
    to. Looking at more changes nothing.
    """
    from scipy.spatial import KDTree

    n_points, n_dims = points.shape
    n_groups = centroids.shape[0]
    nearest = KDTree(centroids).query(
        points, k=min(n_dims + 1, n_groups - 1), workers=-1
    )[1]
    nearest = numpy.asarray(nearest, dtype=numpy.int64).reshape(n_points, -1)

    point = numpy.repeat(numpy.arange(n_points, dtype=numpy.int64), nearest.shape[1])
    alt_group = nearest.reshape(-1)
    keep = alt_group != group_of_point[point]
    return point[keep], alt_group[keep]


def _compact_find_pairs(bucket, affinity, payload, in_lo_group, n_buckets):
    """Pair swap candidates within each bucket by how misplaced they are.
 
    ``affinity`` is ``||z - c_lo||^2 - ||z - c_hi||^2``; positive means the point
    would rather be in the ``hi`` group. Only points wanting to leave are worth
    considering, and the r-th most misplaced on one side pairs with the r-th
    most misplaced on the other.
 
    Returns the two sides' ``payload`` values plus the bucket of each pair.
    """
    empty = (numpy.empty(0, numpy.int64),) * 3
    leaving_lo = in_lo_group & (affinity > 0.0)
    leaving_hi = ~in_lo_group & (affinity < 0.0)
    if not leaving_lo.any() or not leaving_hi.any():
        return empty
 
    # most misplaced first, grouped by bucket
    order_lo = numpy.lexsort((-affinity[leaving_lo], bucket[leaving_lo]))
    order_hi = numpy.lexsort((affinity[leaving_hi], bucket[leaving_hi]))
    bucket_lo = bucket[leaving_lo][order_lo]
    bucket_hi = bucket[leaving_hi][order_hi]
 
    count_lo = numpy.bincount(bucket_lo, minlength=n_buckets)
    count_hi = numpy.bincount(bucket_hi, minlength=n_buckets)
    start_lo = numpy.concatenate(([0], numpy.cumsum(count_lo)[:-1]))
    start_hi = numpy.concatenate(([0], numpy.cumsum(count_hi)[:-1]))
    rank_lo = numpy.arange(bucket_lo.size) - start_lo[bucket_lo]
    rank_hi = numpy.arange(bucket_hi.size) - start_hi[bucket_hi]
 
    n_pairs = numpy.minimum(count_lo, count_hi)
    keep_lo = rank_lo < n_pairs[bucket_lo]
    keep_hi = rank_hi < n_pairs[bucket_hi]
    # both sides are sorted by bucket and truncated to the same per-bucket
    # count, so element-wise pairing lines up
    return (
        payload[leaving_lo][order_lo][keep_lo],
        payload[leaving_hi][order_hi][keep_hi],
        bucket_lo[keep_lo],
    )
 
 
def _compact_match_pairs(first_side, second_side, n_slots):
    """Greedy disjoint matching, vectorized.
 
    Keeps a pair only if it is the highest-priority pair for both of its
    endpoints. Input must already be in priority order.
    """
    n_pairs = first_side.size
    endpoint = numpy.concatenate((first_side, second_side))
    pair_of_endpoint = numpy.tile(numpy.arange(n_pairs), 2)
 
    by_endpoint = numpy.lexsort((pair_of_endpoint, endpoint))
    endpoint = endpoint[by_endpoint]
    pair_of_endpoint = pair_of_endpoint[by_endpoint]
 
    is_first = numpy.empty(endpoint.size, dtype=bool)
    is_first[0] = True
    numpy.not_equal(endpoint[1:], endpoint[:-1], out=is_first[1:])
 
    claimed_by = numpy.full(n_slots, -1, dtype=numpy.int64)
    claimed_by[endpoint[is_first]] = pair_of_endpoint[is_first]
    pair_id = numpy.arange(n_pairs)
    return (claimed_by[first_side] == pair_id) & (claimed_by[second_side] == pair_id)
 
 
def _compact_polish_groups(points, group_of_point, n_groups, total_sq_norm):
    """Swap points between neighbouring groups until nothing improves.
 
    For a group of size g, trading x for y changes its SSE by exactly
    ``||y-c||^2 - ||x-c||^2 - ||x-y||^2 / g``, the last term being the centroid
    shift. Summed over the two groups a swap is worth
    ``affinity(y) - affinity(x) - ||x-y||^2 (1/g_a + 1/g_b)``.
 
    Gains are exact for one swap but only first order for a batch, so each round
    is scored against the previous one and the last round is rolled back if it
    did not help. Total SSE therefore falls strictly every round, which bounds
    the loop without needing an iteration limit.
    """
    n_points, n_dims = points.shape
    best_sse = numpy.inf
    best_assignment = group_of_point.copy()
    eps = 1e-12
 
    while True:
        group_size = numpy.bincount(group_of_point, minlength=n_groups).astype(numpy.float64)
        group_sum = numpy.empty((n_groups, n_dims), dtype=numpy.float64)
        for dim in range(n_dims):  # recomputed exactly each round, so no drift
            group_sum[:, dim] = numpy.bincount(
                group_of_point, weights=points[:, dim], minlength=n_groups
            )
 
        sse = total_sq_norm - float((_sq_norm(group_sum) / group_size).sum())
        if not sse < best_sse * (1.0 - eps):
            return best_assignment
        best_sse = sse
        best_assignment = group_of_point.copy()
        centroids = group_sum / group_size[:, None]
 
        point, alt_group = _compact_neighboring_groups(points, centroids, group_of_point)
        own_group = group_of_point[point]
 
        # Bucket candidates by the unordered group pair they straddle.
        pair_code, pair_of_candidate = numpy.unique(
            numpy.minimum(own_group, alt_group) * numpy.int64(n_groups)
            + numpy.maximum(own_group, alt_group),
            return_inverse=True,
        )
        pair_lo = pair_code // n_groups
        pair_hi = pair_code % n_groups
        affinity = _sq_norm(points[point] - centroids[pair_lo[pair_of_candidate]])
        affinity -= _sq_norm(points[point] - centroids[pair_hi[pair_of_candidate]])
 
        candidate_row = numpy.arange(point.size, dtype=numpy.int64)
        row_lo, row_hi, pair_of_swap = _compact_find_pairs(
            pair_of_candidate,
            affinity,
            candidate_row,
            own_group == pair_lo[pair_of_candidate],
            pair_code.size,
        )
        if row_lo.size == 0:
            return group_of_point
 
        point_lo, point_hi = point[row_lo], point[row_hi]
        coupling = 1.0 / group_size[pair_lo] + 1.0 / group_size[pair_hi]
        gain = affinity[row_hi] - affinity[row_lo]
        gain -= _sq_norm(points[point_lo] - points[point_hi]) * coupling[pair_of_swap]
 
        improving = (gain < -eps) & (point_lo != point_hi)
        point_lo, point_hi, gain = point_lo[improving], point_hi[improving], gain[improving]
        if point_lo.size == 0:
            return group_of_point
 
        # best first, point indices breaking ties; a point swaps at most once
        by_gain = numpy.lexsort((point_hi, point_lo, gain))
        point_lo, point_hi = point_lo[by_gain], point_hi[by_gain]
        disjoint = _compact_match_pairs(point_lo, point_hi, n_points)
        point_lo, point_hi = point_lo[disjoint], point_hi[disjoint]
        group_of_point[point_lo], group_of_point[point_hi] = (
            group_of_point[point_hi].copy(),
            group_of_point[point_lo].copy(),
        )


def mask_fraction_of_groups(n_groups: int, fraction: float) -> NDArray[numpy.bool_]:
    n_required = max(1, math.ceil(n_groups * fraction))
    if n_required >= n_groups:
        return numpy.ones(n_groups, dtype=numpy.bool_)

    every = n_groups // n_required  # guaranteed > 1
    mask = numpy.zeros(n_groups, dtype=numpy.bool_)
    mask[::every] = 1

    return mask


class FloatKey(float):
    def __hash__(self):
        return float.__hash__(round(self, 5))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, float) and \
            round(self, 5) == round(other, 5)


def unwrap(val: T | None) -> T:
    assert val is not None
    return val


def freeze(obj: object) -> t.Any:
    """Attempt to freeze an object, making it immutable."""
    return _freeze(obj, set())


def _freeze(obj: object, stack: set[int]) -> t.Any:
    if id(obj) in stack:
        raise ValueError(f"Cannot freeze self-referential object {type(obj).__name__}")

    stack.add(id(obj))
    try:
        # handle numpy types
        if isinstance(obj, numpy.generic):
            return _freeze(obj.item(), stack)
        if isinstance(obj, numpy.ndarray):
            return _freeze(obj.tolist(), stack)

        # mappings
        if isinstance(obj, Mapping):
            return frozendict((_freeze(k, stack), _freeze(v, stack)) for (k, v) in obj.items())

        # sets
        if isinstance(obj, AbstractSet):
            return frozenset(_freeze(v, stack) for v in obj)

        # sequences
        if isinstance(obj, Sequence):
            # immutable sequence types
            if isinstance(obj, (str, bytes, range)):
                return obj
            # byte arrays
            if isinstance(obj, (bytearray, memoryview)):
                return bytes(obj)
            return tuple(_freeze(v, stack) for v in obj)

        try:
            hash(obj)
            return obj
        except TypeError as e:
            raise TypeError(f"Don't know how to freeze type '{type(obj)}'") from e
    finally:
        stack.discard(id(obj))


class _MockModule:
    def __init__(self, module: ModuleType, rewrites: dict[str, t.Callable], wrap: t.Callable):
        self._inner: ModuleType = module
        self._rewrites: dict[str, t.Callable] = rewrites
        self._wrap: t.Callable = wrap

        self.__name__ = module.__name__
        """
        self.__spec__ = module.__spec__
        self.__package__ = module.__package__
        self.__loader__ = module.__loader__
        self.__path__ = module.__path__
        self.__doc__ = module.__doc__
        self.__annotations__ = module.__annotations__
        if hasattr(module, '__file__') and hasattr(module, '__cached__'):
            self.__file__ = module.__file__
            self.__cached__ = module.__cached__
        """

        self.__setattr__ = lambda name, val: setattr(self._inner, name, val)

    def __getattr__(self, name: t.Any) -> t.Any:
        fullpath = f"{self.__name__}.{name}"
        if (rewrite := self._rewrites.get(fullpath, None)):
            if (val := getattr(self._inner, name, None)) is not None:
                return functools.update_wrapper(rewrite, val)
            return rewrite

        val = getattr(self._inner, name)

        if isinstance(val, ModuleType):
            return _MockModule(val, self._rewrites, self._wrap)

        if hasattr(val, '__call__') and not isinstance(val, type):  # noqa: B004
            def inner(*args, **kwargs):
                return self._wrap(val, *args, **kwargs)

            return inner
            return functools.update_wrapper(inner, val)

        return val


__all__ = [
    'create_rng', 'create_rng_group',
    'create_sparse_groupings', 'create_compact_groupings',
    'mask_fraction_of_groups', 'FloatKey',
    'unwrap', 'freeze',
]
