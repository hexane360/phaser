import logging
import math
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.state import Patterns, ReconsState, ScanState
from phaser.types import cast_length
from phaser.utils.image import (
    affine_transform,
    prepare_convolve2d,
    prepare_convolve2d_recip,
)
from phaser.utils.misc import create_rng, create_sparse_groupings, freeze
from phaser.utils.num import Sampling, cast_array_module, get_array_module, to_numpy

from . import (
    ApplyMtfProps,
    BinProps,
    CropDataProps,
    DropNanProps,
    OffsetProps,
    PoissonProps,
    PostInitArgs,
    RawData,
    ScaleProps,
)

logger = logging.getLogger(__name__)


def crop_data(raw_data: RawData, props: CropDataProps) -> RawData:
    if raw_data['patterns'].ndim != 4:
        raise ValueError(f"'crop_data' expects a 4D array of patterns, got shape {raw_data['patterns'].shape} instead")

    (y_i, y_f, x_i, x_f) = props.crop
    logging.info(f"Cropping raw data to {0 if y_i is None else y_i}:{raw_data['patterns'].shape[0] if y_f is None else y_f},"
                 f" {0 if x_i is None else x_i}:{raw_data['patterns'].shape[1] if x_f is None else x_f}")
    raw_data['patterns'] = raw_data['patterns'][slice(y_i, y_f), slice(x_i, x_f)]

    if (scan_hook := raw_data.get('scan_hook', None)) is not None and scan_hook['type'] == 'raster':
        raw_data['scan_hook'] = {
            **scan_hook,
            'shape': raw_data['patterns'].shape[:2],
        }

    return raw_data


def scale_patterns(raw_data: RawData, props: ScaleProps) -> RawData:
    raw_data['patterns'] *= props.scale
    return raw_data

def offset_patterns(raw_data: RawData, props: OffsetProps) -> RawData:
    raw_data['patterns'] -= props.offset
    return raw_data

def bin_patterns(raw_data: RawData, props: BinProps) -> RawData:
    #xp = get_array_module(raw_data['patterns'])
    bin_factor = props.bin
    patterns = raw_data['patterns']
    Ny, Nx = patterns.shape[-2:]
    patterns = patterns.reshape(*patterns.shape[:-2],
                       Ny // bin_factor, bin_factor,
                       Nx // bin_factor, bin_factor).sum(axis=(-1, -3))

    print(patterns.shape)  # (120, 45, 128, 128)
    
    raw_data['patterns'] = patterns
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

    raw_data['patterns'] = xp.asarray(patterns)
    return raw_data


def apply_mtf(raw_data: RawData, props: ApplyMtfProps) -> RawData:
    xp = get_array_module(raw_data['patterns'])
    filt = props.mtf(raw_data)

    # TODO: nice representation for `filt`, print out here
    logger.info("Applying detector MTF to raw patterns")

    patterns = raw_data['patterns']
    # sigma/psf_radius are in detector pixels, so use a unit sampling
    samp = Sampling(patterns.shape[-2:], sampling=(1., 1.))

    prepare = prepare_convolve2d if props.domain == 'real' else prepare_convolve2d_recip
    prepared = prepare(filt, samp, xp=xp, dtype=patterns.dtype)

    grouping = 128
    for group in create_sparse_groupings(patterns.shape[:-2], grouping):
        patterns[tuple(group)] = prepared(xp.asarray(patterns[tuple(group)]))

    raw_data['patterns'] = patterns
    return raw_data


def drop_nan_patterns(args: PostInitArgs, props: DropNanProps) -> t.Tuple[Patterns, ReconsState]:
    xp = get_array_module(args['data'].patterns)

    scan = args['state'].scan

    # flatten scan, tilt, and patterns
    scan_arr = scan.data.reshape(-1, 2)
    initial_arr = scan.initial.reshape(-1, 2)
    scan_meta = dict(scan.meta)
    if 'raster_rows' in scan_meta:
        scan_meta['raster_rows'] = numpy.array(scan_meta['raster_rows']).ravel()
    if 'raster_cols' in scan_meta:
        scan_meta['raster_cols'] = numpy.array(scan_meta['raster_cols']).ravel()

    tilt_arr = None if scan.tilt is None else scan.tilt.reshape(-1, 2)
    patterns = args['data'].patterns.reshape(-1, *args['data'].patterns.shape[-2:])

    fraction_nan = xp.sum(xp.isnan(patterns), axis=(-1, -2)) / xp.prod(patterns.shape[-2:])

    mask = fraction_nan > props.threshold

    if (n := int(xp.sum(mask))):
        logger.info(f"Dropping {n}/{patterns.shape[0]} patterns which are at least {props.threshold:.1%} NaN values")
        patterns = patterns[~mask]

        if scan_arr.shape[0] == xp.size(mask):
            # apply mask to scan as well
            scan_arr = scan_arr[~mask]
            initial_arr = initial_arr[~mask]
            if 'raster_rows' in scan_meta:
                scan_meta['raster_rows'] = scan_meta['raster_rows'][~to_numpy(mask)]
            if 'raster_cols' in scan_meta:
                scan_meta['raster_cols'] = scan_meta['raster_cols'][~to_numpy(mask)]
        elif scan_arr.shape[0] != patterns.shape[0]:
            raise ValueError(f"# of scan positions {scan_arr.shape[0]} doesn't match # of patterns"
                             f" before ({mask.size}) or after ({patterns.shape[0]}) filtering")
        # otherwise, we assume the mask has already been applied to the scan (and metadata)

        # tilt can come from an alternate source, so we need to check it separately
        if tilt_arr is not None:
            if tilt_arr.shape[0] == mask.size:
                tilt_arr = tilt_arr[~mask]
            elif tilt_arr.shape[0] != patterns.shape[0]:
                raise ValueError(f"# of tilt positions {tilt_arr.shape[0]} doesn't match # of patterns"
                                f" before ({mask.size}) or after ({patterns.shape[0]}) filtering")

    args['state'].scan = ScanState(
        scan_arr, initial_arr, tilt_arr, freeze(scan_meta)
    )
    args['data'].patterns = patterns

    return (args['data'], args['state'])


def diffraction_align(args: PostInitArgs, props: t.Any = None) -> t.Tuple[Patterns, ReconsState]:
    patterns, state = args['data'], args['state']

    xp = cast_array_module(args['xp'])
    grouping = 128
    groups = create_sparse_groupings(patterns.patterns.shape[:-2], grouping)

    sum_pattern = xp.zeros(patterns.patterns.shape[-2:], dtype=patterns.patterns.dtype)

    for group in groups:
        pats = xp.asarray(patterns.patterns[tuple(group)]) * xp.asarray(patterns.pattern_mask)
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
