import typing as t

import numpy
from numpy.typing import NDArray

from phaser.hooks.solver import (
    AdaptiveMomentumPositionSolverProps,
    MomentumPositionSolverProps,
    PositionSolver,
    SteepestDescentPositionSolverProps,
)
from phaser.state import ReconsState
from phaser.utils.num import get_array_module


class AdaptiveMomentumState(t.NamedTuple):
    velocity: NDArray[numpy.floating]
    """Accumulated velocity, shape (n_pos, 2)"""
    history: NDArray[numpy.floating]
    """Ring buffer of recent updates, shape (memory + 1, n_pos, 2). Index 0 is most recent."""
    n_seen: int
    """Number of updates recorded so far, to know when the buffer is usable."""


class SteepestDescentPositionSolver(PositionSolver[None]):
    def __init__(self, args: None, props: SteepestDescentPositionSolverProps):
        self.step_size = props.step_size
        self.max_step_size = props.max_step_size

    def init_state(self, sim: ReconsState) -> None:
        return None

    def perform_update(
        self,
        positions: NDArray[numpy.floating],
        gradients: NDArray[numpy.floating],
        state: None
    ) -> t.Tuple[NDArray[numpy.floating], None]:
        xp = get_array_module(positions, gradients)
        update = self.step_size * gradients

        if self.max_step_size is not None:
            update_mag = xp.linalg.norm(update, axis=-1, keepdims=True)
            update *= xp.minimum(update_mag, self.max_step_size) / update_mag

        return (update, state)


class MomentumPositionSolver(PositionSolver[NDArray[numpy.floating]]):
    def __init__(self, args: None, props: MomentumPositionSolverProps):
        self.step_size = props.step_size
        self.max_step_size = props.max_step_size
        self.momentum = props.momentum

    def init_state(self, sim: ReconsState) -> NDArray[numpy.floating]:
        xp = get_array_module(sim.scan.data)
        return xp.zeros_like(sim.scan.data)

    def perform_update(
        self,
        positions: NDArray[numpy.floating],
        gradients: NDArray[numpy.floating],
        state: NDArray[numpy.floating]
    ) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
        xp = get_array_module(positions, gradients, state)

        update = self.step_size * gradients + self.momentum * state

        if self.max_step_size is not None:
            update_mag = xp.linalg.norm(update, axis=-1, keepdims=True)
            update *= xp.minimum(update_mag, self.max_step_size) / update_mag

        # state is just previous update step
        return (update, update)


class AdaptiveMomentumPositionSolver(PositionSolver[AdaptiveMomentumState]):
    """
    Momentum whose friction is estimated from the decorrelation rate of recent updates.

    Slowly-decorrelating updates indicate consistent drift and earn a long memory;
    anticorrelation indicates oscillation and disables momentum outright.
    """
    def __init__(self, args: None, props: AdaptiveMomentumPositionSolverProps):
        self.step_size = props.step_size
        self.max_step_size = props.max_step_size
        self.memory = int(props.memory)
        self.gain = props.gain
        self.friction_scale = props.friction_scale
        self.oscillation_friction = props.oscillation_friction
        self.momentum_max_update = (
            props.max_step_size if props.momentum_max_update is None else props.momentum_max_update
        )
        self.per_position = props.per_position

        if self.memory < 1:
            raise ValueError(f"'memory' must be at least 1, got {self.memory}")

    def init_state(self, sim: ReconsState) -> AdaptiveMomentumState:
        xp = get_array_module(sim.scan.data)
        return AdaptiveMomentumState(
            velocity=xp.zeros_like(sim.scan.data),
            history=xp.zeros((self.memory + 1, *sim.scan.data.shape), dtype=sim.scan.data.dtype),
            n_seen=0,
        )

    def _friction(self, history: NDArray[numpy.floating], xp: t.Any) -> t.Any:
        """
        Estimate friction from the decay of correlation between the newest update and each
        of the `memory` preceding ones.

        Correlations are taken over the scan-position axis (matching fold_slice) unless
        `per_position` is set, in which case they are taken over the (y, x) components of
        each position independently.
        """
        newest = history[0]
        axes = (-1,) if self.per_position else (0, 1)

        def corr(a, b):
            am = a - xp.mean(a, axis=axes, keepdims=True)
            bm = b - xp.mean(b, axis=axes, keepdims=True)
            num = xp.sum(am * bm, axis=axes)
            den = xp.sqrt(xp.sum(am * am, axis=axes) * xp.sum(bm * bm, axis=axes))
            return num / xp.maximum(den, 1e-30)

        # correlation at lags 1..memory
        corrs = xp.stack([corr(newest, history[lag]) for lag in range(1, self.memory + 1)], axis=0)

        # Fit log(corr) linearly against lag, with the lag-0 correlation pinned to 1 so
        # log(1) = 0. Least-squares slope of log(c) on lag over lags 0..memory.
        lags = xp.arange(self.memory + 1, dtype=corrs.dtype)
        shape = (self.memory + 1,) + (1,) * (corrs.ndim - 1)
        lags_b = lags.reshape(shape)
        logs = xp.concatenate([xp.zeros((1, *corrs.shape[1:]), dtype=corrs.dtype),
                               xp.log(xp.maximum(corrs, 1e-30))], axis=0)
        lag_mean = xp.mean(lags)
        slope = (xp.sum((lags_b - lag_mean) * logs, axis=0)
                 / xp.maximum(xp.sum((lags - lag_mean) ** 2), 1e-30))

        # any anticorrelation at any lag means oscillation: kill momentum
        oscillating = xp.any(corrs <= 0.0, axis=0)
        friction = self.friction_scale * xp.maximum(-slope, 0.0)
        return (
            xp.where(oscillating, self.oscillation_friction, friction),
            xp.where(oscillating, 0.0, self.gain),
        )

    def perform_update(
        self,
        positions: NDArray[numpy.floating],
        gradients: NDArray[numpy.floating],
        state: AdaptiveMomentumState
    ) -> t.Tuple[NDArray[numpy.floating], AdaptiveMomentumState]:
        xp = get_array_module(positions, gradients, state.velocity)

        update = self.step_size * gradients
        history = xp.concatenate([update[None], state.history[:-1]], axis=0)

        if state.n_seen < self.memory:
            # not enough history yet; behave as plain steepest descent while it fills
            velocity = state.velocity
        else:
            (friction, gain) = self._friction(history, xp)
            if not self.per_position:
                velocity = state.velocity * (1.0 - friction) + update
                accelerated = update + gain * velocity
            else:
                velocity = state.velocity * (1.0 - friction[..., None]) + update
                accelerated = update + gain[..., None] * velocity

            if self.momentum_max_update is not None:
                # only accelerate positions that are not already moving quickly. fold_slice
                # gates on the group maximum; gating per position is the natural analogue
                # of the per-position clamp below.
                mag = xp.linalg.norm(update, axis=-1, keepdims=True)
                accelerated = xp.where(mag < self.momentum_max_update, accelerated, update)

            update = accelerated

        if self.max_step_size is not None:
            update_mag = xp.linalg.norm(update, axis=-1, keepdims=True)
            update = update * (xp.minimum(update_mag, self.max_step_size)
                               / xp.maximum(update_mag, 1e-30))

        return (update, AdaptiveMomentumState(
            velocity=velocity, history=history, n_seen=state.n_seen + 1,
        ))