import typing as t

from typing_extensions import NotRequired

from ..types import Dataclass
from ..utils.image import Filter, GaussianFilter, SquarePixelFilter
from ..utils.physics import Electron
from .hook import Hook


class FilterArgs(t.TypedDict):
    wavelength: NotRequired[float | None]


class GaussianFilterProps(Dataclass):
    sigma: float | tuple[float, float]
    """Standard deviation of the blur, in detector pixels."""
    psf_sigma: float = 3.


class SquarePixelFilterProps(Dataclass):
    psf_radius: int = 10


class EmpadFilterProps(Dataclass):
    kv: t.Optional[float] = None
    square: bool = True


def make_gaussian_filter(_args: FilterArgs, props: GaussianFilterProps) -> Filter:
    return GaussianFilter(sigma=props.sigma, psf_sigma=props.psf_sigma)


def make_ideal_sq_filter(_args: FilterArgs, props: SquarePixelFilterProps) -> Filter:
    return SquarePixelFilter(psf_radius=props.psf_radius)


def make_empad_filter(args: FilterArgs, props: EmpadFilterProps) -> Filter:
    if props.kv is not None:
        kv = props.kv
    elif (wavelength := args.get('wavelength', None)) is not None:
        kv = Electron.from_wavelength(wavelength).energy * 1e-3
    else:
        raise ValueError("'kv' must be specified by raw data or passed to filter")

    kv_key = str(round(kv, 0))

    # intrinsic detector blur (before square pixel response)
    match kv_key:
        # extracted from Tate 2016, fit to PSF
        case "200.0": filt = GaussianFilter(0.458)

        # extracted from Philipp 2022; fit to MTF and separated from square pixel response
        case "80.0": filt = GaussianFilter(0.1)
        case "120.0": filt = GaussianFilter(0.165)
        case "300.0": filt = (
            0.802*GaussianFilter(0.708) + 0.198*GaussianFilter(0.154)
        )
        case _: raise ValueError(f"Unsupported 'kv' {kv}. We currently support '80', '120', '200', and '300' keV")

    # additionally convolve with square pixel response
    return filt*SquarePixelFilter() if props.square else filt


class FilterHook(Hook[FilterArgs, Filter]):
    known: t.ClassVar = {
        'gaussian': ('phaser.hooks.filter:make_gaussian_filter', GaussianFilterProps),
        'ideal_sq': ('phaser.hooks.filter:make_ideal_sq_filter', SquarePixelFilterProps),
        'empad': ('phaser.hooks.filter:make_empad_filter', EmpadFilterProps),
    }
