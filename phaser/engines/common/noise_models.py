"""Detector noise models."""

import typing as t

import numpy
from numpy.typing import NDArray

from phaser.hooks.solver import NoiseModel
from phaser.plan import AmplitudeNoisePlan, AnscombeNoisePlan, PoissonNoisePlan
from phaser.utils.num import get_array_module, Float, to_numpy
from phaser.state import ReconsState

if t.TYPE_CHECKING:
    from phaser.utils.image import PreparedOTF, PreparedPSF


class AmplitudeNoiseModel(NoiseModel[None]):
    r"""Gaussian noise on diffraction amplitudes.

    Loss is $\frac{2}{1 + \sigma^2} \sum M \left(\sqrt{I_{exp} + o} - \sqrt{I_{model} + o} - \epsilon\right)^2$,
    with offset $o$, Gaussian variance $\sigma^2$, and mask $M$. Negative $I_{exp}$ is clipped to zero.
    """

    @classmethod
    def name(cls) -> str:
        return "amplitude"

    def __init__(self, args: None, props: AmplitudeNoisePlan):
        self.offset: float = props.offset
        self.gaussian_variance: float = props.gaussian_variance

        self.var: float = 1 + self.gaussian_variance

        self.eps: float = props.eps

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(model_wave, model_intensity, exp_patterns, mask)
        patterns = xp.maximum(exp_patterns, 0.0)

        return (t.cast(numpy.floating,
            2. * xp.sum(mask * (
                xp.sqrt(patterns + self.offset) - xp.sqrt(model_intensity + self.offset) - self.eps
            )**2) / self.var).astype(exp_patterns.dtype),
            state
        )

    def calc_wave_update(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating],
        state: None,
        mtf: t.Optional[t.Union['PreparedOTF', 'PreparedPSF']] = None,
    ) -> t.Tuple[NDArray[numpy.complexfloating], None]:
        xp = get_array_module(model_wave, model_intensity, exp_patterns, mask)
        patterns = xp.maximum(exp_patterns, 0.0)

        update = xp.sqrt(patterns + self.offset) / (xp.sqrt(model_intensity + self.offset) + self.eps) - 1.0
        update *= mask # / self.var
        #print(f"min update: {xp.min(update).get()}")
        #print(f"max update: {xp.max(update).get()}")

        if mtf is not None:
            update = mtf.adjoint()(update)

        return (update * model_wave, state)


class AnscombeNoiseModel(AmplitudeNoiseModel):
    """Amplitude noise model with the Anscombe offset $o = 3/8$ as default."""

    @classmethod
    def name(cls) -> str:
        return "anscombe"

    def __init__(self, args: None, props: AnscombeNoisePlan):
        super().__init__(args, props)


class PoissonNoiseModel(NoiseModel[None]):
    r"""Poisson noise on diffraction intensities.

    Loss is $\sum M \left(I_{model} + \epsilon + I_{exp} (\ln(I_{exp} + \epsilon) - \ln(I_{model} + \epsilon) - 1)\right)$.
    """

    @classmethod
    def name(cls) -> str:
        return "poisson"

    def __init__(self, args: None, props: PoissonNoisePlan):
        self.eps: float = props.eps

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(model_wave, model_intensity, exp_patterns, mask)
        patterns = xp.maximum(exp_patterns, 0.0)

        loss = xp.sum(mask * (
            model_intensity + self.eps + patterns * (
                xp.log(patterns + self.eps) - xp.log(model_intensity + self.eps) - 1.0
            )
        )).astype(exp_patterns.dtype)
        return (loss, state)

    def calc_wave_update(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating],
        state: None,
        mtf: t.Optional[t.Union['PreparedOTF', 'PreparedPSF']] = None,
    ) -> t.Tuple[NDArray[numpy.complexfloating], None]:
        """Raise `NotImplementedError`."""
        raise NotImplementedError()