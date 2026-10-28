"""Load Gatan data."""

import logging
import warnings
from pathlib import Path

import numpy

from phaser.io.gatan import GatanMetadata, load_4d
from phaser.types import cast_length
from phaser.utils.num import Sampling
from phaser.utils.physics import Electron

from .. import LoadGatanProps, RawData


def load_gatan(args: None, props: LoadGatanProps) -> RawData:
    """Load Gatan `.dm4` data, with the scan and probe hooks taken from its metadata.

    Raises:
        ValueError: If the voltage or `diff_step` is given by neither `props` nor the metadata,
            or the data is not found.
    """
    logger = logging.getLogger(__name__)

    path = Path(props.path).expanduser()

    # file = dm.file_reader(path, lazy=True)

    # if path.suffix.lower() == '.json':  # load as metadata
    #     pass # stub for now, not implemented

    # else: # grab from dm4 file
    metadata = GatanMetadata.from_dm4(path)
    
    assert metadata.path is not None

    # path = metadata.path / metadata.gatan_filename

    voltage = props.kv * 1e3 if props.kv is not None else metadata.voltage
    diff_step = props.diff_step or metadata.diff_step
    scan_shape = metadata.scan_shape

    print(f"Scan shape: {scan_shape}, Step size: {metadata.scan_step}")
    needs_scale = not metadata.is_simulated()

    probe_hook = {
        'type': 'focused',
        'conv_angle': metadata.conv_angle,
        'defocus': metadata.defocus * 1e10 if metadata.defocus is not None else None,
    }
    # TODO: handle explicit scan_positions here
    scan_hook = {
        'type': 'raster',
        # [x, y] -> [y, x]
        'shape': tuple(reversed(metadata.scan_shape)),
        'step_size': tuple(s*1e10 for s in reversed(metadata.scan_step)),  # m to A
        'rotation': metadata.scan_rotation or 0.0,
        'affine': metadata.scan_correction[::-1, ::-1] if metadata.scan_correction is not None else None,
    }

    if voltage is None:
        raise ValueError("'kv'/'voltage' must be specified by metadata or passed to 'raw_data'")
    if diff_step is None:
        raise ValueError("'diff_step' must be specified by metadata or passed to 'raw_data'")

    wavelength = Electron(voltage).wavelength

    if not path.exists():
        raise ValueError(f"Couldn't find gatan data at path {path}")

    patterns = numpy.fft.ifftshift(load_4d(path, scan_shape, memmap=False), axes=(-1, -2)).astype(numpy.float32)

    if needs_scale:
        if metadata.e_scaling is None:
            warnings.warn("ADU not supplied for experimental dataset. This is not recommended.")
        else:
            bg_offset = metadata.background_offset or 0.
            logger.info(f"Offsetting patterns by {bg_offset:.3e} and scaling by {metadata.e_scaling:.5e}")
            patterns -= bg_offset
            patterns *= metadata.e_scaling

    # patterns = numpy.transpose(patterns, (1, 0, 2, 3))

    a = float(wavelength / (diff_step * 1e-3)) # recip. pixel size -> 1 / real space extent

    sampling = Sampling(cast_length(patterns.shape[-2:], 2), extent=(a, a))

    mask = numpy.zeros_like(patterns, shape=patterns.shape[-2:]).astype(numpy.float32)

    mask[2:-2, 2:-2] = 1.

    return {
        'patterns': patterns,
        'mask': numpy.fft.ifftshift(mask, axes=(-1, -2)).astype(numpy.float32),
        'sampling': sampling,
        'wavelength': wavelength,
        'probe_hook': probe_hook,
        'scan_hook': scan_hook,
        'seed': None,
    }