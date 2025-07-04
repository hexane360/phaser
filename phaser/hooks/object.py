
import numpy

from phaser.utils.num import cast_array_module, to_complex_dtype
from phaser.utils.object import random_phase_object, parameterized_object
from ..state import ObjectState, ParameterizedObjectState
from . import ObjectHookArgs, RandomObjectProps, ParameterizedObjectProps


def random_object(args: ObjectHookArgs, props: RandomObjectProps) -> ObjectState:
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
    return ObjectState(sampling, obj, thicknesses)


def random_parameterized_object(args: ObjectHookArgs, props: ParameterizedObjectProps) -> ParameterizedObjectState:
    if args['slices'] is None:
        raise NotImplementedError("random_parameterized_object requires slices.")

    sampling = args['sampling']
    thicknesses = numpy.array(args['slices'].thicknesses, dtype=args['dtype'])
    nz = len(thicknesses)
    ny, nx = sampling.shape
    shape = (nz, ny, nx)
    layer_offset = numpy.zeros(nz, dtype=args['dtype'])

    # Convert spacing and offset from physical units (meters) to pixel units
    dy_phys, dx_phys = props.spacing
    y0_phys, x0_phys = props.offset
    dy_pix = dy_phys / sampling.sampling[0]
    dx_pix = dx_phys / sampling.sampling[1]
    y0_pix = y0_phys / sampling.sampling[0]
    x0_pix = x0_phys / sampling.sampling[1]

    # Compute number of atoms in each direction
    n_atoms_y = int(ny // dy_pix)
    n_atoms_x = int(nx // dx_pix)
    ys = y0_pix + numpy.arange(n_atoms_y) * dy_pix
    xs = x0_pix + numpy.arange(n_atoms_x) * dx_pix

    xs_grid, ys_grid = numpy.meshgrid(xs, ys, indexing='xy')
    atom_grid = numpy.stack([xs_grid, ys_grid], axis=-1).reshape(-1, 2)  # (n_atoms, 2)
    atom_defaults = numpy.array([props.H1, props.B1, props.H2, props.B2], dtype=args['dtype'])
    atom_params_slice = numpy.hstack([atom_grid, numpy.broadcast_to(atom_defaults, (atom_grid.shape[0], 4))])  # (n_atoms, 6)
    atom_params = numpy.stack([atom_params_slice] * nz, axis=0)  # (nz, n_atoms, 6)
    params = numpy.concatenate([atom_params.ravel(), layer_offset.ravel()])
    return ParameterizedObjectState(sampling, thicknesses, args['wavelength'], params, atom_params.shape[1])

