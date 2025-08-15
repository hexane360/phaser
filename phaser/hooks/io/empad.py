
from pathlib import Path
import warnings
import logging
import typing as t

import numpy

from phaser.utils.num import Sampling
from phaser.utils.physics import Electron
from phaser.io.empad import load_4d, EmpadMetadata
from phaser.types import cast_length
from .. import LoadEmpadProps, RawData


def load_empad(args: None, props: LoadEmpadProps) -> RawData:
    logger = logging.getLogger(__name__)

    path = Path(props.path).expanduser()

    if path.suffix.lower() == '.json':  # load as metadata
        meta = EmpadMetadata.from_json(path)
        assert meta.path is not None
        path = meta.path / meta.raw_filename

        if meta.empad_version is not None and meta.empad_version > 1:
            raise ValueError("EMPAD v2 import not currently supported (open a Github issue!)")

        voltage = props.kv * 1e3 if props.kv is not None else meta.voltage
        diff_step = props.diff_step or meta.diff_step
        # [x, y] -> [y, x]
        scan_shape = t.cast(t.Tuple[int, int], tuple(reversed(meta.scan_shape)))
        adu = props.adu or meta.adu
        needs_scale = not meta.is_simulated()
        det_flips = props.det_flips or meta.det_flips

        probe_hook = {
            'type': 'focused',
            'conv_angle': meta.conv_angle,
            'defocus': meta.defocus * 1e10 if meta.defocus is not None else None,
        }
        # TODO: handle explicit scan_positions here
        scan_hook = {
            'type': 'raster',
            # [x, y] -> [y, x]
            'shape': scan_shape,
            'step_size': tuple(s*1e10 for s in reversed(meta.scan_step)),  # m to A
            'affine': meta.scan_correction[::-1, ::-1] if meta.scan_correction is not None else None,
            'rotation': meta.det_rotation - meta.scan_rotation,
        }

        #TODO: add tilt to metafile
        tilt_hook= None
    else:
        voltage = props.kv * 1e3 if props.kv is not None else None
        diff_step = props.diff_step
        scan_shape = None
        probe_hook = scan_hook = tilt_hook = None
        adu = None
        needs_scale = False
        det_flips = props.det_flips

    if voltage is None:
        raise ValueError("'kv'/'voltage' must be specified by metadata or passed to 'raw_data'")
    if diff_step is None:
        raise ValueError("'diff_step' must be specified by metadata or passed to 'raw_data'")

    wavelength = Electron(voltage).wavelength

    if not path.exists():
        raise ValueError(f"Couldn't find raw data at path {path}")

    logging.info(f"Loading with detector flips: {list(map(int, det_flips or (True, False, False)))} [y, x, transpose]")
    patterns = numpy.fft.ifftshift(load_4d(path, scan_shape, memmap=True, flips=det_flips), axes=(-1, -2))

    if needs_scale:
        if adu is None:
            warnings.warn("ADU not supplied for experimental dataset. This is not recommended.")
        else:
            logger.info(f"Scaling patterns by ADU ({adu:.1f})")
            patterns /= adu

    a = wavelength / (diff_step * 1e-3)  # recip. pixel size -> 1 / real space extent
    sampling = Sampling(cast_length(patterns.shape[-2:], 2), extent=(a, a))

    mask = numpy.zeros_like(patterns, shape=patterns.shape[-2:])
    mask[2:-2, 2:-2] = 1.

    return {
        'patterns': patterns,
        'mask': numpy.fft.ifftshift(mask, axes=(-1, -2)),
        'sampling': sampling,
        'wavelength': wavelength,
        'probe_hook': probe_hook,
        'scan_hook': scan_hook,
        'tilt_hook': tilt_hook,
        'seed': None,
    }