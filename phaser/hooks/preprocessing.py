import logging
import math
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.types import cast_length
from phaser.utils.num import get_array_module, cast_array_module, to_numpy, fft2, ifft2, at, Sampling
from phaser.utils.misc import create_rng, create_sparse_groupings
from phaser.utils.optics import fourier_shift_filter
from phaser.utils.image import affine_transform
from phaser.state import Patterns, ReconsState
from . import RawData, PostInitArgs, PoissonProps, ScaleProps, DropNanProps, CropDataProps

logger = logging.getLogger(__name__)


def crop_data(raw_data: RawData, props: CropDataProps) -> RawData:
    if raw_data['patterns'].ndim != 4:
        raise ValueError(f"'crop_data' expects a 4D array of patterns, got shape {raw_data['patterns'].shape} instead")

    (y_i, y_f, x_i, x_f) = props.crop
    logging.info(f"Cropping raw data to {0 if y_i is None else y_i}:{raw_data['patterns'].shape[0] if y_f is None else y_f},"
                 f" {0 if x_i is None else x_i}:{raw_data['patterns'].shape[1] if x_f is None else x_f}")
    raw_data['patterns'] = raw_data['patterns'][slice(y_i, y_f), slice(x_i, x_f)]

    if (scan_hook := raw_data.get('scan_hook', None)) is not None:
        if scan_hook['type'] == 'raster':
            raw_data['scan_hook'] = {
                **scan_hook,
                'shape': raw_data['patterns'].shape[:2],
            }

    return raw_data


def scale_patterns(raw_data: RawData, props: ScaleProps) -> RawData:
    raw_data['patterns'] *= props.scale
    return raw_data


def add_poisson_noise(raw_data: RawData, props: PoissonProps) -> RawData:
    xp = get_array_module(raw_data['patterns'])
    dtype = raw_data['patterns'].dtype

    if props.scale is not None:
        logger.info(f"Adding poisson noise to raw patterns, after scaling by {props.scale:.2e}")
        raw_data['patterns'] *= props.scale
    else:
        logger.info("Adding poisson noise to raw patterns")

    rng = create_rng(raw_data.get('seed', None), 'poisson_noise')

    # TODO do this in batches?
    patterns = rng.poisson(to_numpy(raw_data['patterns'])).astype(dtype)

    if props.gaussian is not None:
        patterns += rng.normal(scale=props.gaussian, size=patterns.shape)

    logger.info(f"Mean pattern intensity: {numpy.nanmean(numpy.nansum(patterns, axis=(-1, -2)))}")

    raw_data['patterns'] = xp.array(patterns)
    return raw_data


def drop_nan_patterns(args: PostInitArgs, props: DropNanProps) -> t.Tuple[Patterns, ReconsState]:
    xp = get_array_module(args['data'].patterns)

    # flatten scan and patterns
    scan = args['state'].scan.reshape(-1, 2)
    patterns = args['data'].patterns.reshape(-1, *args['data'].patterns.shape[-2:])

    fraction_nan = xp.sum(xp.isnan(args['data'].patterns), axis=(-1, -2)) / xp.prod(patterns.shape[-2:])

    mask = fraction_nan > props.threshold

    if (n := int(xp.sum(mask))):
        logger.info(f"Dropping {n}/{patterns.shape[0]} patterns which are at least {props.threshold:.1%} NaN values")
        patterns = patterns[mask]

        if scan.shape[0] == mask.size:
            # apply mask to scan as well
            scan = scan[mask]
        elif scan.shape[0] != patterns.shape[0]:
            raise ValueError(f"# of scan positions {scan.shape[0]} doesn't match # of patterns"
                             f" before ({mask.size}) or after ({patterns.shape[0]}) filtering")
        # otherwise, we assume the mask has already been applied to the scan

    args['state'].scan = scan
    args['data'].patterns = patterns

    return (args['data'], args['state'])


def diffraction_align(args: PostInitArgs, props: t.Any = None) -> t.Tuple[Patterns, ReconsState]:
    patterns, state = args['data'], args['state']

    xp = cast_array_module(args['xp'])
    grouping = 128
    groups = create_sparse_groupings(patterns.patterns.shape[:-2], grouping)

    sum_pattern = xp.zeros(patterns.patterns.shape[-2:], dtype=patterns.patterns.dtype)

    for group in groups:
        pats = xp.array(patterns.patterns[tuple(group)]) * xp.array(patterns.pattern_mask)
        sum_pattern += t.cast(NDArray[numpy.floating], xp.nansum(pats, axis=tuple(range(pats.ndim - 2))))

    mean_pattern = sum_pattern / math.prod(patterns.patterns.shape[:-2])

    ky, kx = Sampling(
        cast_length(mean_pattern.shape, 2), extent=(1.0, 1.0)
    ).recip_grid(dtype=patterns.patterns.dtype, xp=xp)

    shift = xp.array([
        xp.nansum(ky * mean_pattern), xp.nansum(kx * mean_pattern)
    ]) / xp.nansum(mean_pattern)

    logging.info(f"Shifting diffraction patterns by ({shift[1]}, {shift[0]}) px")

    def bilinear_shift(arr: NDArray[numpy.floating]) -> NDArray[numpy.floating]:
        return to_numpy(xp.fft.ifftshift(affine_transform(
            xp.fft.fftshift(xp.array(arr), axes=(-2, -1)), [1., 1.], shift,
            output_shape=arr.shape[-2:], order=1
        ), axes=(-2, -1)))

    for group in groups:
        patterns.patterns[tuple(group)] = bilinear_shift(patterns.patterns[tuple(group)])

    # fftshift mask as well
    patterns.pattern_mask = bilinear_shift(patterns.pattern_mask)

    return (patterns, state)
