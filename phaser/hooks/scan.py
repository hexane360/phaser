"""Scan initialization hooks."""


import logging
from pathlib import Path

import numpy
from numpy.typing import NDArray
from frozendict import frozendict

from phaser.state import ScanState
from phaser.utils.num import cast_array_module
from phaser.utils.scan import make_raster_scan

from . import ManualScanProps, RasterScanProps, ScanHookArgs


def raster_scan(args: ScanHookArgs, props: RasterScanProps) -> ScanState:
    """Return a raster scan state with shape `(ny, nx, 2)` positions.

    Raises:
        ValueError: If `shape` or `step_size` is not set.
    """
    xp = cast_array_module(args['xp'])
    logger = logging.getLogger(__name__)

    if props.shape is None:
        raise ValueError("scan 'shape' must be specified by metadata or manually")
    if props.step_size is None:
        raise ValueError("scan 'step_size' must be specified by metadata or manually")
    step_size = numpy.broadcast_to(props.step_size, (2,))
    rot = props.rotation or 0.0

    if props.affine is not None:
        affine = xp.asarray(props.affine, dtype=args['dtype'])
    else:
        affine = None

    logger.info(f"Making raster scan, shape {props.shape},"
                f" step size [{step_size[0]:.2f}, {step_size[1]:.2f}],"
                f" rotation {rot:.2f} deg"
                f" affine transformation {affine.ravel() if affine is not None else 'None'}")
    
    scan = make_raster_scan(
        props.shape, step_size, rot, affine,
        dtype=args['dtype'], xp=xp,
    )
    ii, jj = numpy.indices(props.shape, dtype=numpy.int64)
    assert ii.shape == jj.shape == scan.shape[:-1]

    return ScanState(
        scan, xp.asarray(scan, copy=True), tilt=None, meta=frozendict(
            type='raster',
            raster_rows=tuple(map(tuple, ii.tolist())),
            raster_cols=tuple(map(tuple, jj.tolist())),
        )
    )


_LENGTH_UNITS = {'m': 1e10, 'nm': 10., 'A': 1., 'angstrom': 1.}


def load_scan_positions(path) -> NDArray[numpy.float64]:
    """Load a scan positions file as an `(n_frames, 2)` array of `(y, x)` positions [angstrom], one per pattern in the raw data.

    The file is an `.npz` with 1D arrays `x` and `y`, one entry per pattern in acquisition order (NaN for a pattern not
    taken at a scan position), and optionally `units` (`'m'`, the default, `'nm'` or `'A'`). Other entries are ignored.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Scan positions file not found: {path}")

    with numpy.load(path, allow_pickle=False) as f:
        if (missing := {'x', 'y'} - set(f.files)):
            raise ValueError(f"Scan positions file {path} is missing array(s) {sorted(missing)}")
        x, y = numpy.asarray(f['x'], dtype=numpy.float64), numpy.asarray(f['y'], dtype=numpy.float64)
        units = str(f['units']) if 'units' in f.files else 'm'

    if x.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError(f"Scan positions file {path}: 'x' and 'y' must be 1D arrays of the same, nonzero length"
                         f" (got {x.shape} and {y.shape})")
    if (scale := _LENGTH_UNITS.get(units)) is None:
        raise ValueError(f"Scan positions file {path} has unknown units '{units}' (expected one of {sorted(_LENGTH_UNITS)})")

    return numpy.stack([y, x], axis=-1) * scale


def manual_scan(args: ScanHookArgs, props: ManualScanProps) -> ScanState:
    """Return a scan state with the `(n_frames, 2)` positions of a scan positions file, one per pattern in the raw data.

    Patterns with no position are NaN; `drop_nans` drops them.
    """
    xp = cast_array_module(args['xp'])
    logger = logging.getLogger(__name__)

    pts = load_scan_positions(props.path)
    n_missing = int(numpy.sum(numpy.isnan(pts).any(axis=-1)))
    logger.info(f"Loaded {pts.shape[0] - n_missing} scan positions from '{props.path}'"
                f" ({n_missing} frames with no position)")

    if props.affine is not None:
        pts = pts @ numpy.asarray(props.affine, dtype=numpy.float64).T
    if props.rotation:
        theta = props.rotation * numpy.pi / 180.
        pts = pts @ numpy.array([[numpy.cos(theta), -numpy.sin(theta)], [numpy.sin(theta), numpy.cos(theta)]]).T

    scan = xp.asarray(pts, dtype=args['dtype'])
    return ScanState(scan, xp.asarray(scan, copy=True), tilt=None, meta=frozendict(type='manual'))
