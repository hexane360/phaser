import typing as t

import numpy
from numpy.typing import NDArray

from phaser.types import Dataclass
from phaser.utils.num import Float
from . import Hook

from .solver import HasState, StateT

if t.TYPE_CHECKING:
    from phaser.state import ReconsState


@t.runtime_checkable
class GroupConstraint(HasState[StateT], t.Protocol[StateT]):
    def apply_group(self, group: NDArray[numpy.integer], sim: 'ReconsState', state: StateT) -> t.Tuple['ReconsState', StateT]:
        ...


@t.runtime_checkable
class IterConstraint(HasState[StateT], t.Protocol[StateT]):
    def apply_iter(self, sim: 'ReconsState', state: StateT) -> t.Tuple['ReconsState', StateT]:
        ...


@t.runtime_checkable
class CostRegularizer(HasState[StateT], t.Protocol[StateT]):
    def calc_loss_group(self, group: NDArray[numpy.integer], sim: 'ReconsState', state: StateT) -> t.Tuple[Float, StateT]:
        ...


class ClampObjectAmplitudeProps(Dataclass):
    amplitude: float = 1.1


class LimitProbeSupportProps(Dataclass):
    max_angle: float


class RegularizeLayersProps(Dataclass):
    weight: float = 0.9  # weight of regularization to apply
    sigma: float = 50.0  # standard deviation of gaussian filter (angstrom)


class RegularizeTiltProps(Dataclass):
    weight: float = 0.9  # weight of regularization to apply
    sigma: float = 50.0  # standard deviation of gaussian filter (angstrom)


class ObjLowPassProps(Dataclass):
    max_freq: float = 0.4  # 1/px (nyquist = 0.5)


class GaussianProps(Dataclass):
    sigma: float  # standard deviation of filter (angstrom)
    weight: float = 0.9


class IterConstraintHook(Hook[None, IterConstraint]):
    known = {
        'clamp_object_amplitude': ('phaser.engines.common.regularizers:ClampObjectAmplitude', ClampObjectAmplitudeProps),
        'limit_probe_support': ('phaser.engines.common.regularizers:LimitProbeSupport', LimitProbeSupportProps),
        'layers': ('phaser.engines.common.regularizers:RegularizeLayers', RegularizeLayersProps),
        'tilt': ('phaser.engines.common.regularizers:RegularizeTilt', RegularizeTiltProps),
        'obj_low_pass': ('phaser.engines.common.regularizers:ObjLowPass', ObjLowPassProps),
        'obj_gaussian': ('phaser.engines.common.regularizers:ObjGaussian', GaussianProps),
        'remove_phase_ramp': ('phaser.engines.common.regularizers:RemovePhaseRamp', t.Dict[str, t.Any]),
    }


class GroupConstraintHook(Hook[None, GroupConstraint]):
    known = {
        'clamp_object_amplitude': ('phaser.engines.common.regularizers:ClampObjectAmplitude', ClampObjectAmplitudeProps),
        'limit_probe_support': ('phaser.engines.common.regularizers:LimitProbeSupport', LimitProbeSupportProps),
        'obj_low_pass': ('phaser.engines.common.regularizers:ObjLowPass', ObjLowPassProps),
        'obj_gaussian': ('phaser.engines.common.regularizers:ObjGaussian', GaussianProps),
        'remove_phase_ramp': ('phaser.engines.common.regularizers:RemovePhaseRamp', t.Dict[str, t.Any]),
    }


class CostRegularizerProps(Dataclass):
    cost: float


class TVRegularizerProps(Dataclass):
    cost: float
    eps: float = 1.0e-8


class CostRegularizerHook(Hook[None, CostRegularizer]):
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