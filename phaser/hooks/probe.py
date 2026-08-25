import logging

from phaser.utils.optics import make_focused_probe

from ..state import PixelatedProbeState
from . import FocusedProbeProps, ProbeHookArgs


def focused_probe(args: ProbeHookArgs, props: FocusedProbeProps) -> PixelatedProbeState:
    logger = logging.getLogger(__name__)

    if props.conv_angle is None:
        raise ValueError("Probe 'conv_angle' must be specified by metadata or manually")
    if props.defocus is None:
        raise ValueError("Probe 'defocus' must be specified by metadata or manually")

    logger.info(f"Making probe, conv_angle {props.conv_angle} mrad, defocus {props.defocus} A")
    if len(props.aberrations):
        s = '\n'.join(f"  {ab!r}" for ab in props.aberrations)
        logger.info(f"Aberrations:\n{s}")

    sampling = args['sampling']
    ky, kx = sampling.recip_grid(dtype=args['dtype'], xp=args['xp'])
    probe = make_focused_probe(
        ky, kx, args['wavelength'],
        props.conv_angle, defocus=props.defocus, aberrations=props.aberrations
    )
    return PixelatedProbeState(sampling, probe)