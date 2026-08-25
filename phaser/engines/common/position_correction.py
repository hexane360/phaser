import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import get_array_module
from phaser.hooks.solver import (
    PositionSolver, SteepestDescentPositionSolverProps, MomentumPositionSolverProps
)
from phaser.state import ReconsState


class SteepestDescentPositionSolver(PositionSolver[None]):
    def __init__(self, args: None, props: SteepestDescentPositionSolverProps):
        self.step_size = props.step_size
        self.max_step_size = props.step_size

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