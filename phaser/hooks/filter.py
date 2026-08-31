import typing as t

from ..types import Dataclass
from .hook import Hook
from ..utils.image import Filter, GaussianFilter, SquarePixelFilter


class GaussianFilterProps(Dataclass):
    sigma: float | tuple[float, float]
    """Standard deviation of the blur, in detector pixels."""
    psf_sigma: float = 3.


class SquarePixelFilterProps(Dataclass):
    psf_radius: int = 10


def make_gaussian_filter(_args: None, props: GaussianFilterProps) -> Filter:
    return GaussianFilter(sigma=props.sigma, psf_sigma=props.psf_sigma)


def make_ideal_sq_filter(_args: None, props: SquarePixelFilterProps) -> Filter:
    return SquarePixelFilter(psf_radius=props.psf_radius)


class FilterHook(Hook[None, Filter]):
    known: t.ClassVar = {
        'gaussian': ('phaser.hooks.filter:make_gaussian_filter', GaussianFilterProps),
        'ideal_sq': ('phaser.hooks.filter:make_ideal_sq_filter', SquarePixelFilterProps),
    }
