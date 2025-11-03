from pathlib import Path
import logging
import typing as t

import numpy

from phaser.utils.num import Sampling
from phaser.utils.physics import Electron
from phaser.io.nion import load_4d, NionMetadata
from phaser.types import cast_length
from .. import LoadNionProps, RawData
import json

import zipfile as zf


def load_nion(args: None, props: LoadNionProps) -> RawData:
    logger = logging.getLogger(__name__)

    path = Path(props.path).expanduser()

    with zf.ZipFile(path, "r") as data_file:
        data_file = zf.ZipFile(path, mode="r")
        json_metadata = data_file.read(
            "metadata.json"
        )  # Get the metadata from the file
        json_metadata = json.loads(json_metadata.decode("utf8").replace("'", '"'))

        nion_metadata = NionMetadata.from_data(json_metadata)

    scan_meta = nion_metadata.metadata.scan
    instr_meta = nion_metadata.metadata.instrument

    voltage = instr_meta.high_tension
    scan_shape = scan_meta.scan_size
    scan_shape = tuple(map(int, scan_shape))
    spatial_calibration = nion_metadata.spatial_calibrations[0]
    camera_processing = nion_metadata.properties.camera_processing_parameters.processing

    spatial_units = spatial_calibration.units

    match spatial_units:
        case "nm":
            scale_factor = 1e-9
        case _:
            scale_factor = 1

    scan_step = spatial_calibration.scale * scale_factor
    diff_step = props.diff_step

    logger.info(f"Scan shape: {scan_shape}, Step size: {scan_step}")

    scan_hook = {
        "type": "raster",
        # [x, y] -> [y, x]
        "shape": tuple(reversed(scan_shape)),
        "step_size": scan_step * 1e10, 
        "rotation": (props.detector_rotation_offset or 0.0) + (scan_meta.rotation_deg or 0.0),  # may be the other way around
        # 'affine': metadata.scan_correction[::-1, ::-1] if metadata.scan_correction is not None else None,
    }

    if voltage is None:
        raise ValueError(
            "'kv'/'voltage' must be specified by metadata or passed to 'raw_data'"
        )
    if diff_step is None:
        raise ValueError(
            "'diff_step' must be specified by metadata or passed to 'raw_data'"
        )

    wavelength = Electron(voltage).wavelength

    if not path.exists():
        raise ValueError(f"Couldn't find nion data at path {path}")

    flips: t.List[bool] = [False, False, False]

    for process_step in camera_processing:
        match process_step:
            case "flip_l_r":
                flips[1] = True

    logger.info(f"Loading with flips: {flips}")

    patterns = load_4d(path, cast_length(scan_shape, 2), flips=cast_length(flips, 3), memmap=False)
    patterns = numpy.fft.ifftshift(patterns, axes=(-1, -2)).astype(numpy.float32)

    # if needs_scale:
    #     if metadata.e_scaling is None:
    #         warnings.warn("ADU not supplied for experimental dataset. This is not recommended.")
    #     else:
    #         logger.info(f"Offsetting patterns by {metadata.background_offset:.3e} and scaling by {metadata.e_scaling:.5e}")
    #         patterns -= metadata.background_offset
    #         patterns *= metadata.e_scaling

    # patterns = numpy.transpose(patterns, (1, 0, 2, 3))

    a = float(
        wavelength / (diff_step * 1e-3)
    )  # recip. pixel size -> 1 / real space extent

    sampling = Sampling(cast_length(patterns.shape[-2:], 2), extent=(a, a))

    mask = numpy.zeros_like(patterns, shape=patterns.shape[-2:]).astype(numpy.float32)

    mask[2:-2, 2:-2] = 1.0

    return {
        "patterns": patterns,
        "mask": numpy.fft.ifftshift(mask, axes=(-1, -2)).astype(numpy.float32),
        "sampling": sampling,
        "wavelength": wavelength,
        # 'probe_hook': probe_hook,
        "scan_hook": scan_hook,
        "seed": None,
    }
