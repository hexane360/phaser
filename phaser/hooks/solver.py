"""Noise model and solver hooks."""

import abc
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.types import Dataclass, ReconsVar
from . import Hook

if t.TYPE_CHECKING:
    from phaser.utils.num import Float
    from phaser.engines.common.simulation import SimulationState
    from phaser.execute import Observer
    from phaser.plan import ConventionalEnginePlan, GradientEnginePlan  # noqa: F401
    from phaser.state import ReconsState
    from phaser.utils.image import PreparedOTF, PreparedPSF


StateT = t.TypeVar('StateT')


class HasState(t.Protocol[StateT]):  # type: ignore
    """Protocol for reconstruction objects with mutable state."""

    def init_state(self, sim: 'ReconsState') -> StateT:
        """Return the initial state for the reconstruction `sim`."""
        ...


class NoiseModel(HasState[StateT], t.Protocol[StateT]):
    """Noise model giving the loss and wave update of a reconstruction."""

    @classmethod
    def name(cls) -> str:
        """Return the name of the noise model."""
        ...

    def calc_loss(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: StateT,
    ) -> t.Tuple['Float', StateT]:
        """
        Return the calculated loss, summed across the detector and averaged across the scan.

        May be called in a JAX jit context, so must have no side effects.
        """
        ...

    def calc_wave_update(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating],
        state: StateT,
        mtf: t.Optional[t.Union['PreparedOTF', 'PreparedPSF']] = None,
    ) -> t.Tuple[NDArray[numpy.complexfloating], StateT]:
        """
        Return the calculated wave update `chi` in reciprocal space.

        If `mtf` is given, `model_intensity` has already been blurred by it; the
        intensity-domain residual must be backprojected through `mtf.adjoint()`
        before being combined with the (unblurred) `model_wave`.

        May be called in a JAX jit context, so must have no side effects.
        """
        ...


class NoiseModelHook(Hook[None, NoiseModel]):
    """Hook that creates a [`NoiseModel`][phaser.hooks.solver.NoiseModel]."""

    known = {}


class PositionSolver(HasState[StateT], t.Protocol[StateT]):
    """Solver that computes probe position updates from gradients."""

    def perform_update(
        self,
        positions: NDArray[numpy.floating],
        gradients: NDArray[numpy.floating],
        state: StateT
    ) -> t.Tuple[NDArray[numpy.floating], StateT]:
        """Return the position updates `(update, state)`."""
        ...


class SteepestDescentPositionSolverProps(Dataclass):
    """Properties for the steepest descent position solver."""

    step_size: float = 1e-2
    """Fraction of the optimal step to take."""
    max_step_size: t.Optional[float] = None
    """Maximum step size [angstrom]."""


class MomentumPositionSolverProps(Dataclass):
    """Properties for the momentum position solver."""

    step_size: float = 1e-2
    """Fraction of the optimal step to take."""
    max_step_size: t.Optional[float] = None
    """Maximum step size [angstrom]."""
    momentum: float = 0.9
    """Momentum decay rate."""


class PositionSolverHook(Hook[None, PositionSolver]):
    """Hook that creates a [`PositionSolver`][phaser.hooks.solver.PositionSolver]."""

    known = {
        'steepest_descent': ('phaser.engines.common.position_correction:SteepestDescentPositionSolver', SteepestDescentPositionSolverProps),
        'momentum': ('phaser.engines.common.position_correction:MomentumPositionSolver', MomentumPositionSolverProps),
    }


class ConventionalSolver(abc.ABC):
    """Solver for conventional (non-gradient) engines."""

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """Return the name of the solver."""
        ...

    @abc.abstractmethod
    def init(self, sim: 'SimulationState') -> 'SimulationState':
        """Return the simulation state initialized for this solver."""
        ...

    @abc.abstractmethod
    def presolve(
        self,
        sim: 'SimulationState',
        groups: t.Iterator[NDArray[numpy.int_]], *,
        patterns: NDArray[numpy.floating],
        pattern_mask: NDArray[numpy.floating],
        propagators: t.Optional[NDArray[numpy.complexfloating]],
    ) -> 'SimulationState':
        """Return the simulation state after any setup run before the first iteration."""
        ...

    @abc.abstractmethod
    def run_iteration(
        self,
        sim: 'SimulationState',
        groups: t.Iterator[NDArray[numpy.int_]], *,
        patterns: NDArray[numpy.floating],
        pattern_mask: NDArray[numpy.floating],
        propagators: t.Optional[NDArray[numpy.complexfloating]],
        mtf: t.Optional[t.Union['PreparedOTF', 'PreparedPSF']],
        update_object: bool,
        update_probe: bool,
        update_positions: bool,
        calc_error: bool,
        calc_error_mask: NDArray[numpy.bool_],
        observer: 'Observer',
    ) -> t.Tuple['SimulationState', NDArray[numpy.floating], t.List[NDArray[numpy.floating]]]:
        """
        Run an iteration of the reconstruction. Return a tuple `(sim, pos_update, iter_errors)`.
        """
        ...


class ConventionalSolverHook(Hook['ConventionalEnginePlan', ConventionalSolver]):
    """Hook that creates a [`ConventionalSolver`][phaser.hooks.solver.ConventionalSolver]."""

    known = {}


class GradientSolver(HasState[StateT], t.Protocol[StateT]):
    """Solver that turns gradients into parameter updates for gradient engines."""

    name: str
    """Name of the solver."""
    params: t.FrozenSet[ReconsVar]
    """Parameters the solver updates."""

    def init_state(self, sim: 'ReconsState') -> StateT:
        ...

    def update_for_iter(self, sim: 'ReconsState', state: StateT, niter: int) -> StateT:
        """Return the state updated at the start of an iteration, given the total iteration count `niter`."""
        ...

    def update(
        self, sim: 'ReconsState', state: StateT, grad: t.Dict[ReconsVar, numpy.ndarray], loss: float,
    ) -> t.Tuple[t.Dict[ReconsVar, numpy.ndarray], StateT]:
        """Return the parameter updates for the gradients `grad` and the new state."""
        ...


class GradientSolverArgs(t.TypedDict):
    """Arguments passed to a [`GradientSolverHook`][phaser.hooks.solver.GradientSolverHook]."""

    plan: t.Optional['GradientEnginePlan']
    """Plan of the engine using the solver."""
    params: t.Iterable[ReconsVar]
    """Parameters to be optimized."""


class GradientSolverHook(Hook['GradientSolverArgs', GradientSolver]):
    """Hook that creates a [`GradientSolver`][phaser.hooks.solver.GradientSolver]."""

    known = {}