
import numpy

from phaser.utils.num import cast_array_module, to_complex_dtype
from phaser.utils.object import random_phase_object

from ..state import PixelatedObjectState
from . import ObjectHookArgs, RandomObjectProps


def random_object(args: ObjectHookArgs, props: RandomObjectProps) -> PixelatedObjectState:
    sampling = args['sampling']

    if args['slices'] is not None:
        thicknesses = numpy.array(args['slices'].thicknesses, dtype=args['dtype'])
        shape = (len(thicknesses), *sampling.shape)
    else:
        thicknesses = numpy.array([], dtype=args['dtype'])
        shape = sampling.shape

    obj = random_phase_object(
        shape, props.sigma,
        dtype=to_complex_dtype(args['dtype']),
        xp=cast_array_module(args['xp'])
    )
    return PixelatedObjectState(sampling, obj, thicknesses)