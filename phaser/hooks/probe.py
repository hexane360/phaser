import logging

from phaser.utils.optics import make_focused_probe, make_parameterized_probe
from ..state import ProbeState, ParameterizedProbeState
from . import ProbeHookArgs, FocusedProbeProps, ParameterizedProbeProps
from ..utils.optics import ABERRATION_SPECS

def focused_probe(args: ProbeHookArgs, props: FocusedProbeProps) -> ProbeState:
    logger = logging.getLogger(__name__)

    if props.conv_angle is None:
        raise ValueError("Probe 'conv_angle' must be specified by metadata or manually")
    if props.defocus is None:
        raise ValueError("Probe 'defocus' must be specified by metadata or manually")

    logger.info(f"Making probe, conv_angle {props.conv_angle} mrad, defocus {props.defocus} A")

    sampling = args['sampling']
    ky, kx = sampling.recip_grid(dtype=args['dtype'], xp=args['xp'])
    probe = make_focused_probe(
        ky, kx, args['wavelength'],
        props.conv_angle, defocus=props.defocus
    )
    return ProbeState(sampling, probe)


def parameterized_probe(args: ProbeHookArgs, props: ParameterizedProbeProps) -> ParameterizedProbeState:
    logger = logging.getLogger(__name__)

    if props.conv_angle is None:
        raise ValueError("Probe 'conv_angle' must be specified by metadata or manually")
    if props.defocus is None:
        raise ValueError("Probe 'defocus' must be specified by metadata or manually")

    logger.info(f"Making parameterized probe with conv_angle {props.conv_angle} mrad defocus {props.defocus} A")

    for name, _, is_complex in ABERRATION_SPECS[1:]:
        val = props.aberration_dict.get(name, 0.0) if props.aberration_dict else 0.0
        if is_complex:
            if isinstance(val, (list, tuple)) and len(val) == 2:
                val_str = f"{val[0]:.4g} + {val[1]:.4g}j"
            elif isinstance(val, (int, float)):
                val_str = f"{val:.4g} + {val:.4g}j"
            else:
                val_str = f"Invalid ({val})"
        else:
            if isinstance(val, (int, float)):
                val_str = f"{val:.4g}"
            else:
                val_str = f"Invalid ({val})"
        logger.info(f"Aberration {name}: {val_str}")

    params = props.to_aberration_array()
    sampling = args['sampling']

    return ParameterizedProbeState(sampling, conv_angle=props.conv_angle,wavelength=args['wavelength'], params=params)