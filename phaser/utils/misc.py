import functools
import math
from types import ModuleType
import typing as t

import numpy
from numpy.typing import NDArray
from numpy.random import SeedSequence, PCG64, BitGenerator, Generator


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

    from hashlib import sha256
    import json

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


def create_rng_group(n: int, seed: object = None, entropy: object = None) -> t.Tuple[Generator, ...]:
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


def create_sparse_groupings(shape: t.Union[int, t.Iterable[int], NDArray[numpy.floating]], max_grouping: int = 8,
                            seed: t.Any = None, i: int = 0) -> list[NDArray[numpy.int64]]:
    """
    Randomly partition the indices of `shape` into groups of maximum size `grouping`.

    Returns a list of groups. Each group can be used to index an array `arr` of shape `shape`:
    `arr[tuple(group)]`
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
    Partition the indices of `positions` into groups of maximum size `max_grouping`, such that each group is spatially compact.

    Returns a list of groups. Each group can be used to index `positions`: `positions[tuple(group)]`
    """
    from scipy.spatial import KDTree
    from scipy.cluster.vq import kmeans2

    pos_flat = positions.reshape(-1, 2)
    idxs = numpy.indices(positions.shape[:-1])
    idxs = idxs.reshape(idxs.shape[0], -1)

    n_pos = pos_flat.shape[0]
    n_groups = numpy.ceil(n_pos / max_grouping).astype(numpy.int64)
    if n_groups == 1:
        return [idxs]

    # actual grouping
    grouping = numpy.ceil(n_pos / n_groups).astype(numpy.int64)
    #print(f"n_pos: {n_pos} n_groups: {n_groups} grouping: {grouping}")

    # fist, we seed the group centroids using k-means clustering
    rng = create_rng(seed, f'groupings_{i}' if i != 0 else 'groupings')
    centroids, labels = kmeans2(pos_flat, n_groups, minit='++', missing='raise', rng=rng)

    # then, we use a greedy algorithm to assign points to groups

    # current group sizes
    group_sizes = numpy.zeros(n_groups)
    # position assignments
    assignments = numpy.full(n_pos, -1)
    # currently unassigned indices
    unassigned = numpy.arange(n_pos)

    tree = KDTree(centroids)

    # we run two phases:
    #   first, we assign positions into groups of exactly size k - 1
    #   second, we assign the remaining points (less than n - k) into groups
    for (max_group_size, n_assigned) in ((grouping - 1, (grouping - 1) * n_groups), (grouping, n_pos)):
        n_neighbors = min(4, n_groups)

        while True:
            # compute distances to the unassigned points
            # TODO this should probably exclude ineligible groups
            distances, indices = t.cast(t.Tuple[NDArray[numpy.floating], NDArray[numpy.integer]], tree.query(pos_flat[unassigned], n_neighbors))
            # start by assigning the closest points
            scan_idxs = numpy.argsort(distances[..., 0])
            indices = indices[scan_idxs]
            scan_idxs = unassigned[scan_idxs]

            # loop through unassigned probe positions, assigning to the nearest possible group
            for i, group_idxs in zip(scan_idxs, indices):
                for group_idx in group_idxs:
                    if group_sizes[group_idx] < max_group_size:
                        assignments[i] = group_idx
                        group_sizes[group_idx] += 1
                        break

            # check if we've assigned enough points
            unassigned = numpy.where(assignments == -1)[0]
            if n_pos - len(unassigned) >= n_assigned:
                break

            # we should always assign enough points after we've checked every group
            assert n_neighbors < n_groups
            n_neighbors = min(n_neighbors * 2, n_groups)

    # check that our assignments are correct
    group_sizes = numpy.bincount(assignments)
    assert numpy.min(group_sizes) in (grouping - 1, grouping)
    assert numpy.max(group_sizes) == grouping

    return [
        idxs[..., assignments == i]
        for i in range(n_groups)
    ]


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

    def __eq__(self, other: t.Any) -> bool:
        return isinstance(other, float) and \
            round(self, 5) == round(other, 5)


def unwrap(val: t.Optional[T]) -> T:
    assert val is not None
    return val


class _MockModule:
    def __init__(self, module: ModuleType, rewrites: t.Dict[str, t.Callable], wrap: t.Callable):
        self._inner: ModuleType = module
        self._rewrites: t.Dict[str, t.Callable] = rewrites
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

        if hasattr(val, '__call__') and not isinstance(val, type):
            def inner(*args, **kwargs):
                return self._wrap(val, *args, **kwargs)

            return inner
            return functools.update_wrapper(inner, val)

        return val


__all__ = [
    'create_rng', 'create_rng_group',
    'create_sparse_groupings', 'create_compact_groupings',
    'mask_fraction_of_groups', 'FloatKey',
    'unwrap',
]
