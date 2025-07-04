"""
Object & object cutout utilities
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import repeat
import warnings
import typing as t

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray
from typing_extensions import Self

from .num import get_array_module, cast_array_module, to_real_dtype, as_numpy, at, to_complex_dtype
from .num import as_array, is_cupy, is_jax, NumT, ComplexT, DTypeT
from .misc import create_rng, jax_dataclass


if t.TYPE_CHECKING:
    from phaser.utils.image import _BoundaryMode


@t.overload
def random_phase_object(shape: t.Iterable[int], sigma: float = 1e-6, *, seed: t.Optional[object] = None,
                        dtype: t.Optional[ComplexT] = None, xp: t.Any = None) -> NDArray[ComplexT]:
    ...

@t.overload
def random_phase_object(shape: t.Iterable[int], sigma: float = 1e-6, *, seed: t.Optional[object] = None,
                        dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> NDArray[numpy.complexfloating]:
    ...

def random_phase_object(shape: t.Iterable[int], sigma: float = 1e-6, *, seed: t.Optional[object] = None,
                        dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> NDArray[numpy.complexfloating]:
    """
    Construct a random phase object of shape `shape`.

    # Parameters

      - `shape`: Shape of random phase object to generate
      - `sigma`: Standard deviation of phase variation to create
      - `seed`: Random seed or existing random number generator to use. See `create_rng` for more details.
      - `dtype`: Output datatype of object. Must be complex. Defaults to `numpy.float64`
      - `xp`: Array module to create object on.
    """
    xp2 = numpy if xp is None else cast_array_module(xp)

    rng = create_rng(seed, 'random_phase_object')

    real_dtype = to_real_dtype(dtype) if dtype is not None else numpy.float64
    obj_angle = xp2.array(rng.normal(0., sigma, tuple(shape)), dtype=real_dtype)
    return xp2.cos(obj_angle) + xp2.sin(obj_angle) * 1.j

@t.overload
def parameterized_object(
    shape: t.Iterable[int],
    atom_params: NDArray[numpy.floating],
    layer_offset: NDArray[numpy.floating],
    dtype: t.Optional[ComplexT] = None,
    xp: t.Any = None,
    sigma_e: float = 1.0,
) ->  NDArray[ComplexT]:
    ...

@t.overload
def parameterized_object(
    shape: t.Iterable[int],
    atom_params: NDArray[numpy.floating],
    layer_offset: NDArray[numpy.floating],
    dtype: t.Optional[DTypeLike] = None,
    xp: t.Any = None,
    sigma_e: float = 1.0,
) -> NDArray[numpy.complexfloating]:
    ...

def parameterized_object(
    shape: t.Iterable[int],
    atom_params: NDArray[numpy.floating],
    layer_offset: NDArray[numpy.floating],
    dtype: t.Optional[DTypeLike] = None,
    xp: t.Any = None,
    sigma_e: float = 1.0,
) -> NDArray[numpy.complexfloating]:
    """
    Create a parameterized 3D object transmission function from a (nz, n_atoms, 6) array of atom parameters.

    - shape: (nz, ny, nx)
    - atom_params: (nz, n_atoms, 6)
    - layer_offset: (nz,) or scalar

    extent: (height, width) in same units as x/y. If None, defaults to (ny, nx).
    sigma_e: electron interaction constant.
    """
    xp2 = numpy if xp is None else cast_array_module(xp)
    dtype = to_real_dtype(dtype) if dtype is not None else numpy.float64
    cdtype = to_complex_dtype(dtype) if dtype is not None else numpy.complex64
    shape = tuple(shape)
    assert len(shape) == 3, "Only 3D objects for multislice simulation is supported: shape should be (nz, ny, nx)"
    nz, ny, nx = shape

    y = xp2.arange(ny, dtype=dtype)
    x = xp2.arange(nx, dtype=dtype)
    grid_x, grid_y = xp2.meshgrid(x, y)

    obj = xp2.empty((nz, ny, nx), dtype=cdtype)
    for iz in range(nz):
        atoms = atom_params[iz]
        x0 = atoms[:, 0][:, None, None]
        y0 = atoms[:, 1][:, None, None]
        H1 = atoms[:, 2][:, None, None]/1e10
        B1 = atoms[:, 3][:, None, None]
        H2 = atoms[:, 4][:, None, None]
        B2 = atoms[:, 5][:, None, None]

        dx = grid_x[None, :, :] - x0
        dy = grid_y[None, :, :] - y0
        r2 = dx**2 + dy**2

        V_l = xp2.sum(H1 * xp2.exp(-r2 / B1), axis=0)
        A_l = -xp2.sum(H2 * xp2.exp(-r2 / B2), axis=0)
        A_l += layer_offset[iz]
        if is_jax(obj):
            obj = obj.at[iz].set(A_l * xp2.exp(1j * sigma_e * V_l))
        else:
            obj[iz] = A_l * xp2.exp(1j * sigma_e * V_l)
    return obj

def resample_slices(
    obj: NDArray[NumT], old_thicknesses: ArrayLike, new_thicknesses: ArrayLike, *,
    align: t.Literal['middle', 'top', 'bottom'] = 'middle',
    pad_mode: t.Literal['edge', 'vacuum', 'average'] = 'vacuum',
    rescale_projected: bool = False,
) -> NDArray[NumT]:
    """
    Resample an object in Z, such that average potential is roughly conserved.

    `align` determines how the old object is aligned relative to the new object. The
    default, 'center' keeps the center of the object fixed, padding or cropping both
    object surfaces equally. 'top' and 'bottom' align the top and bottom surfaces,
    respectively.

    `pad_mode` determines how the edges of the sample are handled. The default,
    'vacuum', pads with vacuum slices. 'edge' duplicates the edge slice, and
    'average' duplicates with the average object potential.

    In the case of a multi <-> single slice conversion, the position or thickness
    of the single slice is not taken into account. Instead, projected potential is
    conserved such as to create a uniform potential distribution.

    Returns an object of shape `(max(1, len(new_thicknesses)), *obj.shape[1:])`.
    """
    xp = get_array_module(obj)

    while obj.ndim < 3:
        obj = obj[None, ...]

    old_thicknesses = as_numpy(old_thicknesses)
    new_thicknesses = as_numpy(new_thicknesses)

    if len(new_thicknesses) < 2:
        if obj.shape[0] == 1:
            # single -> single slice, don't resample
            return obj

        # multi -> single
        return xp.prod(obj, axis=0, keepdims=True)  # type: ignore

    if obj.shape[0] == 1:
        # single -> multi, keep projected potential constant
        # TODO more options in this case?
        new_total_thick = numpy.sum(new_thicknesses)

        slice_frac = xp.array((new_thicknesses / new_total_thick)[(slice(None), *repeat(None, obj.ndim - 1))])
        return xp.exp((xp.log(obj) * slice_frac).astype(obj.dtype))

    if obj.shape[0] != len(old_thicknesses):
        raise ValueError(f"Expected an object with {len(old_thicknesses)} slices, instead got object shape {obj.shape}")

    # center of slices
    old_zs = xp.cumsum(old_thicknesses) - old_thicknesses / 2.
    new_zs = xp.cumsum(new_thicknesses) - new_thicknesses / 2.

    # shift new_zs for proper alignment
    if align == 'bottom':
        new_zs += numpy.sum(old_thicknesses) - numpy.sum(new_thicknesses)
    elif align == 'middle':
        new_zs += (numpy.sum(old_thicknesses) - numpy.sum(new_thicknesses)) / 2.
    elif align != 'top':
        raise ValueError(f"Unknown alignment type '{align}'. Expected 'top', 'middle', or 'bottom'.")

    # log to make linear, normalize from projected potential to potential
    obj = (xp.log(obj) / old_thicknesses[:, None, None]).astype(obj.dtype)

    if pad_mode == 'edge':
        before_slice = obj[0]
        after_slice = obj[-1]
    elif pad_mode == 'average':
        before_slice = after_slice = xp.nanmean(obj, axis=0)  # type: ignore
    elif pad_mode == 'vacuum':
        before_slice = after_slice = xp.zeros_like(obj[0])
    else:
        raise ValueError(f"Unknown padding mode '{pad_mode}'. Expected 'edge', 'vacuum', or 'average'.")

    # pad old object with outer slices
    old_zs = numpy.concatenate(([old_zs[0] - old_thicknesses[0]], old_zs, [old_zs[-1] + old_thicknesses[-1]]))
    obj = xp.concatenate((before_slice[None, ...], obj, after_slice[None, ...]), axis=0)

    new_obj = _interp1d(obj, old_zs, new_zs)

    # convert back to projected potential
    new_obj *= new_thicknesses[:, None, None]

    def _calc_projected(o) -> numpy.floating:
        # TODO the object probably needs to be unwrapped before this
        proj_pot = xp.imag(o)
        slice_proj_pot = (
            xp.nanquantile(proj_pot, 0.99, axis=tuple(range(1, o.ndim))) -
            xp.nanquantile(proj_pot, 0.01, axis=tuple(range(1, o.ndim)))
        )
        return xp.abs(xp.sum(slice_proj_pot))

    if rescale_projected:
        warnings.warn("The `rescale_projected` feature is experimental. Use at your own risk.")
        # TODO: we may not be able to trust the raw potentials w/out phase ramp removal here
        scale = _calc_projected(obj) / _calc_projected(new_obj)
        new_obj *= xp.array(scale)[:, None, None]

    # undo log
    return xp.exp(new_obj.astype(obj.dtype))


def _interp1d(arr: NDArray[NumT], old_zs: NDArray[numpy.floating], new_zs: NDArray[numpy.floating]) -> NDArray[NumT]:
    """
    Interpolates along the first dimension of `arr`. Boundary values are replaced with the edge value.
    """
    xp = get_array_module(arr)
    real_dtype = to_real_dtype(arr.dtype)
    slice_is = numpy.searchsorted(old_zs, new_zs) - 1

    new_arr = xp.empty((len(new_zs), *arr.shape[1:]), dtype=arr.dtype)

    delta_zs = numpy.diff(old_zs)

    # TODO specialize this for jax?
    for i, new_z in enumerate(new_zs):
        if new_z <= old_zs[0]: # before
            slice = arr[0]
        elif old_zs[-1] <= new_z: # after
            slice = arr[-1]
        else:
            slice_i = slice_is[i]
            # linearly interpolate
            t = xp.array(float((new_z - old_zs[slice_i]) / delta_zs[slice_i]), dtype=real_dtype)
            slice = ((1-t)*arr[slice_i] + t*arr[slice_i + 1]).astype(arr.dtype)

        new_arr = at(new_arr, i).set(slice)

    return new_arr


@jax_dataclass(frozen=True, init=False)
class ObjectSampling:
    shape: NDArray[numpy.int_]
    """Sampling shape `(n_y, n_x)`"""
    sampling: NDArray[numpy.float64]
    """Sample spacing `(s_y, s_x)`"""
    corner: NDArray[numpy.float64]
    """Corner of sampling `(y_min, x_min)`"""

    region_min: t.Optional[NDArray[numpy.float64]]
    region_max: t.Optional[NDArray[numpy.float64]]

    @property
    def min(self) -> NDArray[numpy.float64]:
        """Minimum object pixel position (y, x). Alias for `corner`."""
        return self.corner

    @property
    def max(self) -> NDArray[numpy.float64]:
        """Maximum pixel position (y, x)."""
        return (self.corner + (self.shape - 1) * self.sampling).astype(numpy.float64)

    @property
    def extent(self) -> NDArray[numpy.float64]:
        return (self.shape * self.sampling).astype(numpy.float64)

    def __init__(self, shape: t.Tuple[int, int], sampling: ArrayLike, corner: t.Optional[ArrayLike] = None,
                 region_min: t.Optional[ArrayLike] = None, region_max: t.Optional[ArrayLike] = None):
        object.__setattr__(self, 'shape', numpy.broadcast_to(as_numpy(shape).astype(numpy.int_), (2,)))
        object.__setattr__(self, 'sampling', numpy.broadcast_to(as_numpy(sampling).astype(numpy.float64), (2,)))
        object.__setattr__(self, 'region_min', numpy.broadcast_to(as_numpy(region_min).astype(numpy.float64), (2,)) if region_min is not None else None)
        object.__setattr__(self, 'region_max', numpy.broadcast_to(as_numpy(region_max).astype(numpy.float64), (2,)) if region_max is not None else None)

        if corner is None:
            corner = -self.extent / 2. + self.sampling/2. #* (self.shape % 2)
        else:
            corner = numpy.broadcast_to(as_numpy(corner).astype(numpy.float64), (2,))

        object.__setattr__(self, 'corner', corner)

    def __eq__(self, other: t.Any) -> bool:
        if type(self) is not type(other):
            return False
        xp = get_array_module(self.sampling, other.sampling)
        return (
            xp.array_equal(self.shape, other.shape) and
            xp.array_equal(self.sampling, other.sampling) and
            xp.array_equal(self.corner, other.corner)
        )

    @staticmethod
    def _scan_extent(scan_positions: NDArray[numpy.floating]) -> t.Tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
        xp = get_array_module(scan_positions)
        scan_min = numpy.array(tuple(float(xp.nanmin(scan_positions[..., i])) for i in range(2)))
        scan_max = numpy.array(tuple(float(xp.nanmax(scan_positions[..., i])) for i in range(2)))
        return (scan_min, scan_max)

    @classmethod
    def from_scan(cls: t.Type[Self], scan_positions: NDArray[numpy.floating], sampling: ArrayLike, pad: ArrayLike = 0) -> Self:
        """Create an ObjectSampling around the given scan positions, padded by at least a radius `pad` in real-space."""
        sampling = as_numpy(sampling).astype(numpy.float64)
        pad = numpy.broadcast_to(pad, (2,)).astype(numpy.float64)

        (scan_min, scan_max) = cls._scan_extent(scan_positions)
        n = numpy.ceil((2.*pad + scan_max - scan_min) / sampling).astype(numpy.int_) + 1

        return cls((n[0], n[1]), sampling, scan_min - pad, scan_min, scan_max)

    def expand_to_scan(self, scan_positions: NDArray[numpy.floating], pad: ArrayLike = 0.) -> Self:
        pad = numpy.broadcast_to(pad, (2,)).astype(numpy.float64)

        (scan_min, scan_max) = self._scan_extent(scan_positions)
        pad_min = numpy.ceil(numpy.maximum(0, self.min - scan_min + pad) / self.sampling).astype(numpy.int_)
        pad_max = numpy.ceil(numpy.maximum(0, scan_max - self.max + pad) / self.sampling).astype(numpy.int_)

        if numpy.all(pad_min == 0) and numpy.all(pad_max == 0):
            return self

        region_min = numpy.minimum(self.region_min, scan_min) if self.region_min is not None else None
        region_max = numpy.maximum(self.region_max, scan_max) if self.region_max is not None else None

        return self.__class__(
            t.cast(t.Tuple[int, int], tuple(self.shape + pad_min + pad_max)),
            self.sampling,
            self.corner - pad_min * self.sampling,
            region_min, region_max
        )

    def check_scan(self, scan_positions: NDArray[numpy.floating], pad: ArrayLike = 0.):
        xp = get_array_module(scan_positions)

        pad = numpy.broadcast_to(pad, (2,)).astype(numpy.float64)
        obj_min = self.corner - pad
        obj_max = self.corner + self.extent + pad

        outside: NDArray[numpy.bool_] = (
            (scan_positions[..., 0] < obj_min[0]) | (scan_positions[..., 0] > obj_max[0]) |
            (scan_positions[..., 1] < obj_min[1]) | (scan_positions[..., 1] > obj_max[1])
        )
        if (n_outside := int(xp.sum(outside))):
            raise ValueError(f"{n_outside}/{outside.size} probe positions completely outside object")

    def _pos_to_object_idx(self, pos: ArrayLike, cutout_shape: t.Tuple[int, ...]) -> NDArray[numpy.float64]:
        """Return starting index for the cutout closest to centered around `pos` (`(y, x)`)"""

        if not is_jax(pos):  # allow jax tracers to work right
            pos = as_numpy(pos)

        # for a given cutout, shift to the top left pixel of that cutout
        # e.g. a 2x2 cutout needs shifted by s/2
        shift = -numpy.maximum(0., (numpy.array(cutout_shape[-2:]) - 1.)) / 2.

        return ((pos - self.corner) / self.sampling + shift).astype(numpy.float64)  # type: ignore

    def slice_at_pos(self, pos: ArrayLike, cutout_shape: t.Tuple[int, ...]) -> t.Tuple[slice, slice]:
        """
        Return slices to cutout a region of shape `cutout_shape` around the object position `pos`.

        # Parameters

         - `pos`: Object position to return a cutout around `(y, x)`
         - `cutout_shape`: Shape of cutout to return.

        Returns slices which can be used to index into an object. E.g. `obj[slice_at_pos(pos, (32, 32))]`
        will return an array of shape `(32, 32)`.
        """

        idxs = self._pos_to_object_idx(pos, cutout_shape)
        (start_i, start_j) = map(int, numpy.round(idxs).astype(numpy.int64))
        assert start_i >= 0 and start_j >= 0
        return (
            slice(start_i, start_i + cutout_shape[-2]),
            slice(start_j, start_j + cutout_shape[-1]),
        )

    def get_subpx_shifts(self, pos: ArrayLike, cutout_shape: t.Tuple[int, ...]) -> NDArray[numpy.float64]:
        """
        Get the subpixel shifts between `pos` and the cutout region around `pos`.

        Returns the shift from the rounded position towards the actual position, in length units.
        """
        pos = self._pos_to_object_idx(as_array(pos), cutout_shape)
        return (pos - get_array_module(pos).round(pos)).astype(numpy.float64) * self.sampling

    @t.overload
    def cutout(  # pyright: ignore[reportOverlappingOverload]
        self, arr: NDArray[DTypeT], pos: ArrayLike, shape: t.Tuple[int, ...]
    ) -> ObjectCutout[DTypeT]:
        ...

    @t.overload
    def cutout(self, arr: numpy.ndarray, pos: ArrayLike, shape: t.Tuple[int, ...]) -> ObjectCutout[numpy.generic]:
        ...

    def cutout(self, arr: numpy.ndarray, pos: ArrayLike, shape: t.Tuple[int, ...]) -> ObjectCutout[t.Any]:
        xp = get_array_module(arr, pos)
        return ObjectCutout(self, xp.array(arr), xp.array(pos), shape)

    def get_view_at_pos(self, arr: NDArray[NumT], pos: ArrayLike, shape: t.Tuple[int, ...]) -> NDArray[NumT]:
        """
        Get cutout views of `arr` of shape `shape` around positions `pos`
        """
        return self.cutout(arr, pos, shape).get()

    def set_view_at_pos(self, arr: NDArray[NumT], pos: ArrayLike, view: numpy.ndarray) -> NDArray[NumT]:
        """
        Set cutout views of `arr` of shape `shape` around positions `pos`
        """
        cutout = self.cutout(arr, pos, view.shape).set(view)
        return cutout.obj

    def add_view_at_pos(self, arr: NDArray[NumT], pos: ArrayLike, view: numpy.ndarray) -> NDArray[NumT]:
        """
        Add to cutout views of `arr` of shape `shape` around positions `pos`
        """
        cutout = self.cutout(arr, pos, view.shape).add(view)
        return cutout.obj

    def get_region_crop(self, pad: ArrayLike = 0.) -> t.Tuple[slice, slice]:
        pad = numpy.asarray(pad)
        if self.region_min is None:
            min_i, min_j = None, None
        else:
            min_i, min_j = numpy.ceil(self._pos_to_object_idx(self.region_min - pad, (0, 0))).astype(numpy.int_)
        if self.region_max is None:
            max_i, max_j = None, None
        else:
            max_i, max_j = numpy.floor(self._pos_to_object_idx(self.region_max + pad, (0, 0))).astype(numpy.int_) + 1

        return (slice(min_i, max_i), slice(min_j, max_j))

    def get_region_mask(self, pad: ArrayLike = 0., *, xp: t.Any = None) -> NDArray[numpy.bool_]:
        xp2 = numpy if xp is None else cast_array_module(xp)
        mask = xp2.zeros(self.shape, dtype=numpy.bool_)
        mask = at(mask, self.get_region_crop(pad=pad)).set(numpy.bool_(1))  # type: ignore
        return mask

    def get_region_center(self) -> NDArray[numpy.floating]:
        region_max = self.region_max if self.region_max is not None else self.max
        region_min = self.region_min if self.region_min is not None else self.min
        return (region_max + region_min) / 2.0

    def get_region_extent(self) -> NDArray[numpy.floating]:
        region_max = self.region_max if self.region_max is not None else self.max
        region_min = self.region_min if self.region_min is not None else self.min
        return region_max - region_min

    @t.overload
    def grid(  # pyright: ignore[reportOverlappingOverload]
        self, *, dtype: t.Type[NumT], xp: t.Any = None
    ) -> t.Tuple[NDArray[NumT], NDArray[NumT]]:
        ...

    @t.overload
    def grid(self, *, dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
        ...

    def grid(self, *, dtype: t.Any = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.number], NDArray[numpy.number]]:
        """Return the sampling grid `(yy, xx)` for this object"""
        xp2 = get_array_module(self.sampling, self.corner) if xp is None else cast_array_module(xp)

        if dtype is None:
            dtype = numpy.common_type(self.sampling, self.corner)

        ys = xp2.linspace(self.min[0], self.max[0], self.shape[0], endpoint=True, dtype=dtype)
        xs = xp2.linspace(self.min[1], self.max[1], self.shape[1], endpoint=True, dtype=dtype)

        return tuple(xp2.meshgrid(ys, xs, indexing='ij'))  # type: ignore

    def mpl_extent(self, center: bool = True) -> t.Tuple[float, float, float, float]:
        """
        Return the extent of the sampling grid, for use in matplotlib.

        Extent is returned as `(left, right, bottom, top)`.
        If `center` is specified (the default), samples correspond to the center of pixels.
        Otherwise, they correspond to the corners of pixels.
        """
        # shift pixel corners to centers
        shift = self.min - self.sampling/2. * int(center)
        return (shift[1], self.extent[1] + shift[1], self.extent[0] + shift[0], shift[0])

    def resample(
        self, arr: NDArray[NumT], new_samp: 'ObjectSampling', *,
        order: int = 1, mode: '_BoundaryMode' = 'grid-constant',
        cval: t.Union[NumT, float] = 1.0,
        rotation: t.Optional[float] = None,
        affine: t.Optional[ArrayLike] = None,
    ) -> NDArray[NumT]:
        from .image import affine_transform, to_affine_matrix, translation_matrix, scale_matrix, rotation_matrix

        if arr.shape[-2:] != tuple(self.shape):
            raise ValueError("Image dimension don't match sampling dimensions")

        if rotation is not None:
            if affine is not None:
                raise ValueError("`rotation` and `affine` cannot both be specified.")
            if rotation != 0.0:
                center = self.get_region_center()
                affine = translation_matrix(center) @ rotation_matrix(rotation) @ translation_matrix(-center)

        if affine is not None:
            matrix = numpy.linalg.inv(
                scale_matrix(1/new_samp.sampling) @ translation_matrix(-new_samp.corner) @
                to_affine_matrix(affine) @
                translation_matrix(self.corner) @ scale_matrix(self.sampling)
            )

            return affine_transform(
                arr, matrix, output_shape=tuple(new_samp.shape),
                order=order, mode=mode, cval=cval
            )

        matrix = new_samp.sampling / self.sampling
        offset = (new_samp.corner - self.corner) / self.sampling

        return affine_transform(
            arr, matrix, offset, output_shape=tuple(new_samp.shape),
            order=order, mode=mode, cval=cval
        )

    def with_sampling(self, sampling: ArrayLike) -> Self:
        sampling = numpy.broadcast_to(sampling, (2,))
        new_shape = numpy.ceil(self.sampling / sampling * self.shape).astype(int)
        growth = sampling * new_shape - self.sampling * self.shape
        assert numpy.all(growth >= 0.)
        return self.__class__(
            shape=tuple(new_shape),
            sampling=sampling,
            # split half of growth to each corner
            corner=self.corner - growth / 2. + (sampling - self.sampling)/2.,
            region_min=self.region_min,
            region_max=self.region_max,
        )


@dataclass
class ObjectCutout(t.Generic[DTypeT]):
    sampling: ObjectSampling
    obj: NDArray[DTypeT]
    pos: NDArray[numpy.floating]
    cutout_shape: t.Tuple[int, ...]

    _start_idxs: NDArray[numpy.int_] = field(init=False)

    def __post_init__(self):
        self._start_idxs = numpy.round(self.sampling._pos_to_object_idx(self.pos, self.cutout_shape)).astype(numpy.int_) # type: ignore
        self._start_idxs = get_array_module(self.obj).array(self._start_idxs)

    @property
    def shape(self) -> t.Tuple[int, ...]:
        return (*self.pos.shape[:-1], *self.obj.shape[:-2], *self.cutout_shape[-2:])

    def get(self) -> NDArray[DTypeT]:
        if is_jax(self.obj):
            from ._jax_kernels import get_cutouts
            return t.cast(NDArray[DTypeT], get_cutouts(self.obj, self._start_idxs, tuple(self.cutout_shape)))

        if is_cupy(self.obj):
            try:
                from ._cuda_kernels import get_cutouts
                return get_cutouts(self.obj, self._start_idxs, self.cutout_shape, 1)  # type: ignore
            except (ImportError, NotImplementedError):
                pass

        xp = get_array_module(self.obj)
        out = xp.empty(self.shape, dtype=self.obj.dtype)
        for idx in numpy.ndindex(self.pos.shape[:-1]):
            # todo make slices outside of loop
            out[tuple(idx)] = self.obj[(Ellipsis, *self.sampling.slice_at_pos(self.pos[idx], self.cutout_shape))]

        return out

    def set(self, view: NDArray[DTypeT]) -> ObjectCutout[DTypeT]:
        if is_jax(self.obj):
            from ._jax_kernels import set_cutouts
            obj = t.cast(NDArray[DTypeT], set_cutouts(self.obj, view, self._start_idxs))
            return self._with_obj(obj)

        if is_cupy(self.obj):
            try:
                from ._cuda_kernels import set_cutouts
                set_cutouts(self.obj, view, self._start_idxs)
                return self
            except (ImportError, NotImplementedError):
                pass

        for idx in numpy.ndindex(self.pos.shape[:-1]):
            # todo make slices outside of loop
            self.obj[(Ellipsis, *self.sampling.slice_at_pos(self.pos[idx], view.shape))] = view[idx]
        return self

    def add(self, view: NDArray[DTypeT]) -> ObjectCutout[DTypeT]:
        if is_jax(self.obj):
            from ._jax_kernels import add_cutouts
            obj = t.cast(NDArray[DTypeT], add_cutouts(self.obj, view, self._start_idxs))
            return self._with_obj(obj)

        if is_cupy(self.obj):
            try:
                from ._cuda_kernels import add_cutouts
                add_cutouts(self.obj, view, self._start_idxs)
                return self
            except (ImportError, NotImplementedError):
                pass

        for idx in numpy.ndindex(self.pos.shape[:-1]):
            # todo make slices outside of loop
            self.obj[(Ellipsis, *self.sampling.slice_at_pos(self.pos[idx], view.shape))] += view[idx]
        return self

    def _with_obj(self, obj: NDArray[DTypeT]) -> ObjectCutout[DTypeT]:
        return ObjectCutout(
            self.sampling,
            obj,
            self.pos.copy(),
            self.cutout_shape
        )

    def __cupy_get_ndarray__(self) -> NDArray[DTypeT]:
        if not is_cupy(self.obj):
            raise AttributeError()
        return self.get()

    def __array__(self) -> NDArray[DTypeT]:
        return self.get()


__all__ = [
    'random_phase_object', 'resample_slices',
    'ObjectSampling', 'ObjectCutout',
]
