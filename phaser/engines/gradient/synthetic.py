import typing as t

import numpy
from numpy.typing import NDArray

from phaser.hooks.solver import HasState, StateT
from phaser.state import ReconsState
from phaser.types import ReconsVar
from phaser.utils.num import get_array_module


class SyntheticGrad(HasState[StateT], t.Protocol[StateT]):
    """
    Gradient constructed by projection onto a constrained subspace.

    Applied to the gradient going into a solver, and to the update coming out.
    """

    source: t.ClassVar[ReconsVar]
    """Variable this gradient is synthesized from."""

    def init_state(self, sim: ReconsState) -> StateT:
        """Precompute the data `project` needs."""
        ...

    def project(
        self, sim: ReconsState, state: StateT, arr: NDArray[numpy.floating]
    ) -> NDArray[numpy.floating]:
        """Project a gradient or update."""
        ...


class AffineSyntheticGrad(SyntheticGrad[None]):
    """
    Constrains position updates to a linear transform of the current positions,
    `u = pos @ M.T` for a 2x2 matrix `M`. `pos` is mean-centered, so `M` carries
    no translation.
    """

    source: t.ClassVar[ReconsVar] = 'positions'
    # regularization, in machine epsilons
    eps: t.Final[float] = 100.0

    def init_state(self, sim: ReconsState) -> None:
        return None

    def project(
        self, sim: ReconsState, state: None, arr: NDArray[numpy.floating]
    ) -> NDArray[numpy.floating]:
        xp = get_array_module(arr)
        pos = xp.reshape(sim.scan.data, (-1, 2))
        pos = pos - xp.mean(pos, axis=0)
        gram = pos.T @ pos
        eps = xp.finfo(gram.dtype).eps
        # small tikhonov regularization, for when `gram` is ill-conditioned
        lam = self.eps * eps * (gram[0, 0] + gram[1, 1] + eps) / 2.
        sol = xp.linalg.solve(gram + lam * xp.eye(2, dtype=gram.dtype), pos.T @ xp.reshape(arr, (-1, 2)))
        return xp.reshape(pos @ sol, arr.shape)


class LineState(t.NamedTuple):
    raster_rows: NDArray[numpy.integer]
    """Contiguous line index of each (flattened) position, shape `(n,)`."""
    counts: NDArray[numpy.integer]
    """Number of positions in each row, shape `(n_rows,)`."""


class LineSyntheticGrad(SyntheticGrad[LineState]):
    """
    Constrains position updates to be constant along each fast-scan line, as
    grouped by the `raster_rows` scan metadata. `project` is the per-line mean.
    """

    source: t.ClassVar[ReconsVar] = 'positions'

    def init_state(self, sim: ReconsState) -> LineState:
        xp = get_array_module(sim.scan.data)

        if sim.scan.meta.get('type', 'raster') != 'raster':
            raise ValueError("'positions_line' gradient requires a raster scan")
        if (raster_rows := sim.scan.meta.get('raster_rows')) is None:
            raise ValueError("'positions_line' gradient requires raster scan metadata")

        # return indices of the unique array
        # if raster_rows is zero-indexed with no gaps this is equivalent to just using `raster_rows` directly
        # in other cases this makes sure bincount works well
        raster_rows = numpy.unique(numpy.asarray(raster_rows, dtype=numpy.int32).ravel(), return_inverse=True)[1]
        counts = numpy.bincount(raster_rows)

        return LineState(xp.asarray(raster_rows), xp.asarray(counts))

    def project(
        self, sim: ReconsState, state: LineState, arr: NDArray[numpy.floating]
    ) -> NDArray[numpy.floating]:
        xp = get_array_module(arr)

        row_grads = xp.stack(tuple(
            (xp.bincount(state.raster_rows, weights=arr[..., i].ravel()) / state.counts).astype(arr.dtype)
            for i in range(arr.shape[-1])
        ), axis=-1)
        return row_grads[state.raster_rows].reshape(arr.shape)


SYNTHETIC_GRADS: t.Mapping[ReconsVar, t.Type[SyntheticGrad[t.Any]]] = {
    'positions_affine': AffineSyntheticGrad,
    'positions_line': LineSyntheticGrad,
}

SYNTHETIC_VARS: t.FrozenSet[ReconsVar] = frozenset[ReconsVar](SYNTHETIC_GRADS)

SYNTHETIC_SOURCES: t.Mapping[ReconsVar, ReconsVar] = {
    var: cls.source for (var, cls) in SYNTHETIC_GRADS.items()
}


def source_var(var: ReconsVar) -> ReconsVar:
    """Get the source variable for a (possibly synthetic) variable `var`."""
    return SYNTHETIC_SOURCES.get(var, var)


__all__ = [
    'SYNTHETIC_GRADS', 'SYNTHETIC_SOURCES', 'SYNTHETIC_VARS',
    'AffineSyntheticGrad', 'LineSyntheticGrad', 'SyntheticGrad',
    'source_var',
]
