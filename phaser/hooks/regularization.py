"""Constraint and cost regularizer hooks."""

import typing as t
import numpy
from numpy.typing import NDArray

from phaser.types import Dataclass
from . import Hook

from .solver import HasState, StateT

if t.TYPE_CHECKING:
    from phaser.utils.num import Float
    from phaser.state import ReconsState


@t.runtime_checkable
class GroupConstraint(HasState[StateT], t.Protocol[StateT]):
    """Constraint applied after each group update."""

    def apply_group(self, group: NDArray[numpy.integer], sim: 'ReconsState', state: StateT) -> t.Tuple['ReconsState', StateT]:
        """Return the state after applying the constraint to the updated scan indices `group`."""
        ...


@t.runtime_checkable
class IterConstraint(HasState[StateT], t.Protocol[StateT]):
    """Constraint applied after each iteration."""

    def apply_iter(self, sim: 'ReconsState', state: StateT) -> t.Tuple['ReconsState', StateT]:
        """Return the state after applying the constraint."""
        ...


@t.runtime_checkable
class CostRegularizer(HasState[StateT], t.Protocol[StateT]):
    """Regularization term added to the cost function."""

    def name(self) -> str:
        """Return the name of the regularizer."""
        ...

    def calc_loss_group(self, group: NDArray[numpy.integer], sim: 'ReconsState', state: StateT) -> t.Tuple['Float', StateT]:
        """Return the regularization cost for group `group`."""
        ...


class ClampObjectAmplitudeProps(Dataclass):
    """Properties for clamping the object amplitude."""

    amplitude: t.Union[float, t.List[t.Optional[float]]] = 1.1
    """Maximum amplitude, or `[min, max]`. `None` leaves a bound open."""


class LimitProbeSupportProps(Dataclass):
    """Properties for limiting the probe's reciprocal-space support."""

    max_angle: float
    """Maximum scattering angle [mrad]."""


class RegularizeLayersProps(Dataclass):
    """Properties for smoothing the object along the slice axis."""

    sigma: float = 50.0
    """Standard deviation of the Gaussian filter [angstrom]."""
    weight: float = 0.9
    """Weight of the filtered object in the update; the rest is the unfiltered object."""


class UnstructuredGaussianProps(Dataclass):
    """Properties for Gaussian smoothing of a state attribute."""

    attr_path: str
    """Dotted path of the attribute in the reconstruction state."""
    sigma: float
    """Standard deviation of the Gaussian filter [angstrom]."""

    weight: float = 0.9
    """Weight of the filtered values in the update; the rest is the unfiltered values."""


class TiltGaussianProps(UnstructuredGaussianProps):
    """Properties for Gaussian smoothing of the tilt."""

    attr_path: t.Literal['scan.tilt'] = 'scan.tilt'


class OPRGaussianProps(UnstructuredGaussianProps):
    """Properties for Gaussian smoothing of probe mode intensity data."""

    attr_path: t.Literal['opr.data'] = 'opr.data'


class ObjLowPassProps(Dataclass):
    """Properties for low-pass filtering the object."""

    max_freq: float = 0.4
    """Cutoff frequency [1/pixel]; the Nyquist frequency is `0.5`."""


class GaussianProps(Dataclass):
    """Properties for Gaussian smoothing of the object."""

    sigma: float
    """Standard deviation of the Gaussian filter [angstrom]."""
    weight: float = 0.9
    """Weight of the filtered object in the update; the rest is the unfiltered object."""


class NonNegObjectPhaseProps(Dataclass):
    """Properties for constraining the object phase to be non-negative."""

    weight: float = 1.0
    r"""Weight $w$ in `[0, 1]` of the phase clamp $\phi \to w \max(\phi, 0) + (1 - w) \phi$."""


class IterConstraintHook(Hook[None, IterConstraint]):
    """Hook that creates an [`IterConstraint`][phaser.hooks.regularization.IterConstraint]."""

    known = {
        'clamp_object_amplitude': ('phaser.engines.common.regularizers:ClampObjectAmplitude', ClampObjectAmplitudeProps),
        'limit_probe_support': ('phaser.engines.common.regularizers:LimitProbeSupport', LimitProbeSupportProps),
        'layers': ('phaser.engines.common.regularizers:RegularizeLayers', RegularizeLayersProps),
        'obj_low_pass': ('phaser.engines.common.regularizers:ObjLowPass', ObjLowPassProps),
        'obj_gaussian': ('phaser.engines.common.regularizers:ObjGaussian', GaussianProps),
        'opr_gaussian': ('phaser.engines.common.regularizers:UnstructuredGaussian', OPRGaussianProps),
        'tilt_gaussian': ('phaser.engines.common.regularizers:UnstructuredGaussian', TiltGaussianProps),
        'remove_phase_ramp': ('phaser.engines.common.regularizers:RemovePhaseRamp', t.Dict[str, t.Any]),
        'nonneg_object_phase': ('phaser.engines.common.regularizers:NonNegObjectPhase', NonNegObjectPhaseProps),

    }


class GroupConstraintHook(Hook[None, GroupConstraint]):
    """Hook that creates a [`GroupConstraint`][phaser.hooks.regularization.GroupConstraint]."""

    known = {
        'clamp_object_amplitude': ('phaser.engines.common.regularizers:ClampObjectAmplitude', ClampObjectAmplitudeProps),
        'limit_probe_support': ('phaser.engines.common.regularizers:LimitProbeSupport', LimitProbeSupportProps),
        'obj_low_pass': ('phaser.engines.common.regularizers:ObjLowPass', ObjLowPassProps),
        'obj_gaussian': ('phaser.engines.common.regularizers:ObjGaussian', GaussianProps),
        'remove_phase_ramp': ('phaser.engines.common.regularizers:RemovePhaseRamp', t.Dict[str, t.Any]),
        'nonneg_object_phase': ('phaser.engines.common.regularizers:NonNegObjectPhase', NonNegObjectPhaseProps),

    }


class CostRegularizerProps(Dataclass):
    """Properties for a cost regularizer."""

    cost: float
    """Regularization strength."""


class TVRegularizerProps(Dataclass):
    """Properties for a total variation regularizer."""

    cost: float
    """Regularization strength."""
    eps: float = 1.0e-8
    """Smoothing term that keeps the gradient finite."""


class CostRegularizerHook(Hook[None, CostRegularizer]):
    """Hook that creates a [`CostRegularizer`][phaser.hooks.regularization.CostRegularizer]."""

    known = {
        'obj_l1': ('phaser.engines.common.regularizers:ObjL1', CostRegularizerProps),
        'obj_l2': ('phaser.engines.common.regularizers:ObjL2', CostRegularizerProps),
        'obj_phase_l1': ('phaser.engines.common.regularizers:ObjPhaseL1', CostRegularizerProps),
        'obj_recip_l1': ('phaser.engines.common.regularizers:ObjRecipL1', CostRegularizerProps),
        'obj_tv': ('phaser.engines.common.regularizers:ObjTotalVariation', TVRegularizerProps),
        'obj_tikh': ('phaser.engines.common.regularizers:ObjTikhonov', CostRegularizerProps),
        'obj_tikhonov': ('phaser.engines.common.regularizers:ObjTikhonov', CostRegularizerProps),
        'layers_tv': ('phaser.engines.common.regularizers:LayersTotalVariation', CostRegularizerProps),
        'layers_tikh': ('phaser.engines.common.regularizers:LayersTikhonov', CostRegularizerProps),
        'layers_tikhonov': ('phaser.engines.common.regularizers:LayersTikhonov', CostRegularizerProps),
        'probe_phase_tikh': ('phaser.engines.common.regularizers:ProbePhaseTikhonov', CostRegularizerProps),
        'probe_phase_tikhonov': ('phaser.engines.common.regularizers:ProbePhaseTikhonov', CostRegularizerProps),
        'probe_recip_tv': ('phaser.engines.common.regularizers:ProbeRecipTotalVariation', TVRegularizerProps),
        'probe_recip_tikh': ('phaser.engines.common.regularizers:ProbeRecipTikhonov', CostRegularizerProps),
        'probe_recip_tikhonov': ('phaser.engines.common.regularizers:ProbeRecipTikhonov', CostRegularizerProps),
    }
