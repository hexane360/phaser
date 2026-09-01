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
    def init_state(self, sim: 'ReconsState') -> StateT:
        ...


class NoiseModel(HasState[StateT], t.Protocol[StateT]):
    @classmethod
    def name(cls) -> str:
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
    known = {}


class PositionSolver(HasState[StateT], t.Protocol[StateT]):
    def perform_update(
        self,
        positions: NDArray[numpy.floating],
        gradients: NDArray[numpy.floating],
        state: StateT
    ) -> t.Tuple[NDArray[numpy.floating], StateT]:
        """
        Return the calculated position updates
        """
        ...


class SteepestDescentPositionSolverProps(Dataclass):
    # fraction of optimal step to take
    step_size: float = 1e-2
    # maximum step size (in angstroms)
    max_step_size: t.Optional[float] = None


class MomentumPositionSolverProps(Dataclass):
    # fraction of optimal step to take
    step_size: float = 1e-2
    # maximum step size (in angstroms)
    max_step_size: t.Optional[float] = None
    # momentum decay rate
    momentum: float = 0.9


class PositionSolverHook(Hook[None, PositionSolver]):
    known = {
        'steepest_descent': ('phaser.engines.common.position_correction:SteepestDescentPositionSolver', SteepestDescentPositionSolverProps),
        'momentum': ('phaser.engines.common.position_correction:MomentumPositionSolver', MomentumPositionSolverProps),
    }


class ConventionalSolver(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        ...

    @abc.abstractmethod
    def init(self, sim: 'SimulationState') -> 'SimulationState':
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
    known = {}


class GradientSolver(HasState[StateT], t.Protocol[StateT]):
    name: str
    params: t.FrozenSet[ReconsVar]

    def init_state(self, sim: 'ReconsState') -> StateT:
        ...

    def update_for_iter(self, sim: 'ReconsState', state: StateT, niter: int) -> StateT:
        ...

    def update(
        self, sim: 'ReconsState', state: StateT, grad: t.Dict[ReconsVar, numpy.ndarray], loss: float,
    ) -> t.Tuple[t.Dict[ReconsVar, numpy.ndarray], StateT]:
        ...


class GradientSolverArgs(t.TypedDict):
    plan: t.Optional['GradientEnginePlan']
    params: t.Iterable[ReconsVar]


class GradientSolverHook(Hook['GradientSolverArgs', GradientSolver]):
    known = {}