import contextlib
import json
import typing as t
from pathlib import Path

import h5py
import numpy
from frozendict import frozendict
from numpy.typing import NDArray

from phaser.state import (
    IterState,
    ObjectState,
    PartialReconsState,
    PixelatedObjectState,
    PixelatedProbeState,
    ProbeState,
    ProgressState,
    ReconsState,
    ScanState,
)
from phaser.utils.num import Sampling, to_numpy
from phaser.utils.object import ObjectSampling
from phaser.utils.misc import freeze

HdfLike: t.TypeAlias = h5py.File | str | Path
OpenMode: t.TypeAlias = t.Literal['r', 'r+', 'w', 'w-', 'x', 'a']
DTypeT = t.TypeVar('DTypeT', bound=numpy.generic)

_DTYPE_CATEGORIES: dict[type[numpy.generic], type[numpy.generic]] = {
    numpy.bool_: numpy.bool_,
    numpy.float32: numpy.floating,
    numpy.float64: numpy.floating,
    numpy.floating: numpy.floating,
    numpy.complex64: numpy.complexfloating,
    numpy.complex128: numpy.complexfloating,
    numpy.complexfloating: numpy.complexfloating,
    numpy.inexact: numpy.inexact,
    numpy.uint8: numpy.integer,
    numpy.uint16: numpy.integer,
    numpy.uint32: numpy.integer,
    numpy.uint64: numpy.integer,
    numpy.int8: numpy.integer,
    numpy.int16: numpy.integer,
    numpy.int32: numpy.integer,
    numpy.int64: numpy.integer,
    numpy.integer: numpy.integer,
    numpy.signedinteger: numpy.signedinteger,
    numpy.unsignedinteger: numpy.unsignedinteger,
}

_CATEGORY_MIN_DTYPE: dict[type[numpy.generic], type[numpy.generic]] = {
    numpy.bool_: numpy.bool_,
    numpy.inexact: numpy.float32,
    numpy.floating: numpy.float32,
    numpy.complexfloating: numpy.complex64,
    numpy.integer: numpy.uint8,
    numpy.unsignedinteger: numpy.uint8,
    numpy.signedinteger: numpy.int8,
}


class OutputDir(contextlib.AbstractContextManager[Path]):
    def __init__(self, fmt_str: str, any_output: bool, **kwargs: t.Any):
        try:
            out_dir = fmt_str.format(**kwargs)
            self.out_dir: Path = Path(out_dir).expanduser().absolute()
        except KeyError as e:
            raise ValueError(f"Invalid format string in 'out_dir' (unknown key {e})") from None
        except Exception as e:
            raise ValueError("Invalid format string in 'out_dir'") from e

        self.any_output: bool = any_output

    def __enter__(self) -> Path:
        if self.any_output:
            try:
                self.out_dir.mkdir(exist_ok=True)
            except Exception as e:
                e.add_note(f"Unable to create output dir '{self.out_dir}'")
                raise

        return self.out_dir

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: object):
        if exc_value is None and self.any_output:
            # create finished file
            (self.out_dir / 'finished').touch(mode=0o664)


def open_hdf5(file: HdfLike, mode: str = 'r', **kwargs: t.Any) -> h5py.File:
    if mode not in ('r', 'r+', 'w', 'w-', 'x', 'a'):
        raise ValueError("Invalid mode. Must be one of 'r', 'r+', 'w', 'w-', 'x', or 'a'.")

    if isinstance(file, h5py.File):
        requested_write = mode in ('r+', 'w', 'w-', 'x', 'a')
        have_write = mode in ('r+', 'w', 'w-', 'x', 'a')
        if requested_write and not have_write:
            raise ValueError(f"Requested writable file but passed read-only file '{file}'.")
        return file

    return h5py.File(file, mode=mode, **kwargs)


def hdf5_read_state(file: HdfLike) -> PartialReconsState:
    file = open_hdf5(file, 'r')

    ty = _hdf5_read_string(file, 'type')
    version_str = _hdf5_read_string(file, 'version')
    if ty != 'phaser_state':
        raise ValueError(f"While reading file '{file.filename}':\nExpected a file of type 'phaser_state', instead got type '{ty}'")

    version = _parse_version(version_str)
    # VERSION 0.2:
    # - moved 'scan' into its own group
    # - added 'meta' to probe, object, scan (backwards compatible)
    # - added 'type' to probe, object (backwards compatible)
    if version > (0, 2):
        raise ValueError(f"While reading file '{file.filename}':\nUnsupported file version '{version_str}'. Maximum supported version is '0.2'.")
    read_scan_as_group = version >= (0, 2)

    wavelength = _hdf5_read_scalar(file, 'wavelength', numpy.float64) if 'wavelength' in file else None

    probe = hdf5_read_probe_state(_assert_group(file['probe'])) if 'probe' in file else None
    obj = hdf5_read_object_state(_assert_group(file['object'])) if 'object' in file else None
    iter = hdf5_read_iter_state(_assert_group(file['iter'])) if 'iter' in file else IterState.empty()

    if 'scan' not in file:
        scan = None
    elif read_scan_as_group:  # new behavior
        scan = hdf5_read_scan_state(_assert_group(file['scan']))
    else:  # old behavior
        scan_arr = _hdf5_read_array(file, 'scan', numpy.float64)
        tilt_arr = _hdf5_read_array(file, 'tilt', numpy.float64, nullable=True)
        if tilt_arr is not None:
            assert tilt_arr.shape == scan_arr.shape

        # use current scan as initial
        scan = ScanState(scan_arr, initial=scan_arr.copy(), tilt=tilt_arr)

    progress = hdf5_read_progress_state(_assert_group(file['progress'])) if 'progress' in file else None

    return PartialReconsState(
        wavelength=wavelength, iter=iter, probe=probe,
        object=obj, scan=scan, progress=progress
    )


def hdf5_read_probe_state(group: h5py.Group) -> ProbeState:
    ty = _hdf5_read_string(group, 'type', nullable=True)
    if ty not in (None, 'pixelated'):
        # currently we only support pixelated probes
        raise ValueError(f"While reading file '{group.file.filename}':\nUnsupported probe type '{ty}'")

    probes = _hdf5_read_array(group, 'data', numpy.complexfloating)
    assert probes.ndim == 3

    extent = _hdf5_read_array_shape(group, 'extent', numpy.float64, (2,))
    (n_y, n_x) = probes.shape[-2:]

    meta = hdf5_read_meta(group)

    return PixelatedProbeState(
        Sampling((n_y, n_x), extent=(extent[0], extent[1])),
        data=probes, meta=meta,
    )


def hdf5_read_scan_state(group: h5py.Group) -> ScanState:
    scan = _hdf5_read_array(group, 'data', numpy.float64)

    tilt = _hdf5_read_array(group, 'tilt', numpy.float64, nullable=True)
    initial = _hdf5_read_array(group, 'initial', numpy.float64, nullable=True)

    if tilt is not None:
        assert tilt.shape == scan.shape
    if initial is not None:
        assert initial.shape == scan.shape

    meta = hdf5_read_meta(group)

    return ScanState(
        scan, initial=scan.copy() if initial is None else initial, tilt=tilt, meta=meta,
    )


def hdf5_read_object_state(group: h5py.Group) -> ObjectState:
    ty = _hdf5_read_string(group, 'type', nullable=True)
    if ty not in (None, 'pixelated'):
        # currently we only support pixelated objects
        raise ValueError(f"While reading file '{group.file.filename}':\nUnsupported object type '{ty}'")

    obj = _hdf5_read_array(group, 'data', numpy.complexfloating)
    (n_z, n_y, n_x) = obj.shape

    thicknesses = _hdf5_read_array(group, 'thicknesses', numpy.floating)
    assert thicknesses.ndim == 1
    assert thicknesses.size == n_z if n_z > 1 else thicknesses.size in (0, 1)

    sampling = _hdf5_read_array_shape(group, 'sampling', numpy.float64, (2,))
    corner = _hdf5_read_array_shape(group, 'corner', numpy.float64, (2,))

    region_min = _hdf5_read_array_shape(group, 'region_min', numpy.float64, (2,), nullable=True)
    region_max = _hdf5_read_array_shape(group, 'region_max', numpy.float64, (2,), nullable=True)

    meta = hdf5_read_meta(group)

    return PixelatedObjectState(
        ObjectSampling((n_y, n_x), sampling, corner, region_min, region_max),
        data=obj, thicknesses=thicknesses, meta=meta,
    )


def hdf5_read_iter_state(group: h5py.Group) -> IterState:
    engine_num = int(_hdf5_read_scalar(group, 'engine_num', numpy.int64))
    engine_iter = int(_hdf5_read_scalar(group, 'engine_iter', numpy.int64))
    total_iter = int(_hdf5_read_scalar(group, 'total_iter', numpy.int64))

    return IterState(
        engine_num=engine_num, engine_iter=engine_iter, total_iter=total_iter
    )


def hdf5_read_progress_state(group: h5py.Group) -> dict[str, ProgressState]:
    if 'iters' in group and 'detector_errors' in group:
        # read old-style, convert to new style
        iters = _hdf5_read_array(group, 'iters', numpy.int64)
        values = _hdf5_read_array(group, 'detector_errors', numpy.float64)
        assert iters.ndim == values.ndim == 1
        assert iters.shape == values.shape

        return {'total_loss': ProgressState(iters.tolist(), values.tolist())}

    # read new-style
    d: dict[str, ProgressState] = {}

    for (k, inner) in group.items():
        if not isinstance(inner, h5py.Group):
            continue
        iters = _hdf5_read_array(inner, 'iters', numpy.int64)
        values = _hdf5_read_array(inner, 'values', numpy.float64)
        assert iters.ndim == values.ndim == 1
        assert iters.shape == values.shape

        d[k] = ProgressState(iters.tolist(), values.tolist())

    return d


def hdf5_write_state(state: ReconsState | PartialReconsState, file: HdfLike):
    file = open_hdf5(file, 'w')  # overwrite if existing
    _hdf5_write_string(file, 'type', "phaser_state")
    _hdf5_write_string(file, 'version', "0.2")
    file.create_dataset('wavelength', (), numpy.float64, state.wavelength)

    if state.probe is not None:
        hdf5_write_probe_state(state.probe, file.create_group("probe"))
    if state.object is not None:
        hdf5_write_object_state(state.object, file.create_group("object"))
    if state.scan is not None:
        hdf5_write_scan_state(state.scan, file.create_group("scan"))
    if state.iter is not None:
        hdf5_write_iter_state(state.iter, file.create_group("iter"))
    if state.progress is not None:
        hdf5_write_progress_state(state.progress, file.create_group("progress"))


def hdf5_write_probe_state(state: ProbeState, group: h5py.Group):
    # we only support pixelated probes currently
    assert state.ty == 'pixelated'
    assert state.data.ndim == 3

    _hdf5_write_string(group, 'type', state.ty)
    dataset = group.create_dataset('data', data=to_numpy(state.data))
    dataset.dims[0].label = 'mode'
    dataset.dims[1].label = 'y'
    dataset.dims[2].label = 'x'

    group.create_dataset('sampling', data=state.sampling.sampling.astype(numpy.float64))
    group.create_dataset('extent', data=state.sampling.extent.astype(numpy.float64))

    hdf5_write_meta(group, state.meta)


def hdf5_write_object_state(state: ObjectState, group: h5py.Group):
    # we only support pixelated objects currently
    assert state.ty == 'pixelated'
    assert state.data.ndim == 3
    assert state.thicknesses.ndim == 1

    n_z = state.data.shape[0]
    thick = to_numpy(state.thicknesses)
    assert thick.ndim == 1
    assert thick.size == n_z if n_z > 1 else thick.size in (0, 1)

    _hdf5_write_string(group, 'type', state.ty)
    group.create_dataset('thicknesses', data=thick)
    zs = group.create_dataset('zs', data=to_numpy(state.zs()))
    zs.make_scale("z")

    dataset = group.create_dataset('data', data=to_numpy(state.data))
    dataset.dims[0].label = 'z'
    dataset.dims[0].attach_scale(zs)
    dataset.dims[1].label = 'y'
    dataset.dims[2].label = 'x'

    group.create_dataset('sampling', data=state.sampling.sampling.astype(numpy.float64))
    group.create_dataset('extent', data=state.sampling.extent.astype(numpy.float64))
    group.create_dataset('corner', data=state.sampling.corner.astype(numpy.float64))

    _hdf5_write_nullable_dataset(group, 'region_min', state.sampling.region_min, numpy.float64)
    _hdf5_write_nullable_dataset(group, 'region_max', state.sampling.region_max, numpy.float64)

    hdf5_write_meta(group, state.meta)


def hdf5_write_scan_state(state: ScanState, group: h5py.Group):
    assert state.data.ndim >= 2
    assert state.initial.shape == state.data.shape
    if state.tilt is not None:
        assert state.tilt.shape == state.data.shape

    dataset = group.create_dataset('data', data=to_numpy(state.data).astype(numpy.float64))
    dataset.dims[dataset.ndim - 1].label = 'yx'

    dataset = group.create_dataset('initial', data=to_numpy(state.initial).astype(numpy.float64))
    dataset.dims[dataset.ndim - 1].label = 'yx'

    if state.tilt is not None:
        dataset = group.create_dataset('tilt', data=to_numpy(state.tilt).astype(numpy.float64))
        dataset.dims[dataset.ndim - 1].label = 'yx'

    hdf5_write_meta(group, state.meta)


def hdf5_write_iter_state(state: IterState, group: h5py.Group):
    group.create_dataset("engine_num", (), numpy.uint64, data=state.engine_num)
    group.create_dataset("engine_iter", (), numpy.uint64, data=state.engine_iter)
    group.create_dataset("total_iter", (), numpy.uint64, data=state.total_iter)


def hdf5_write_progress_state(state: dict[str, ProgressState], group: h5py.Group):
    for (k, v) in state.items():
        subgroup = group.require_group(k)

        iters = subgroup.create_dataset("iters", data=numpy.array(v.iters, dtype=numpy.int64))
        iters.make_scale("total_iter")
        dataset = subgroup.create_dataset("values", data=numpy.array(v.values, dtype=numpy.float64))
        dataset.dims[0].label = 'total_iter'
        dataset.dims[0].attach_scale(iters)


def hdf5_read_meta(group: h5py.Group, path: str = 'meta') -> frozendict[str, t.Any]:
    if path not in group:
        return frozendict()

    dataset = group[path]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"While reading '{group.file.filename}:\n"
                        f"Expected a dataset at path '{group.name}/{path}', instead found {type(dataset)}")

    if h5py.check_string_dtype(dataset.dtype) is None:
        raise TypeError(f"While reading '{group.file.filename}:\n"
                        f"Expected a string dataset at path '{group.name}/{path}', instead found {dataset.dtype}") from None

    s = dataset[()]

    if isinstance(s, numpy.ndarray):
        assert s.shape == ()

    try:
        s = bytes(s).decode('utf-8')
    except (TypeError, UnicodeDecodeError) as e:
        e.add_note(f"While reading '{group.file.filename}', invalid string dataset '{group.name}/{path}'")
        raise

    try:
        d = freeze(json.loads(s))
        assert isinstance(d, frozendict)  # config toplevel should be dict
        return d
    except (json.JSONDecodeError, AssertionError) as e:
        e.add_note(f"While reading '{group.file.filename}', invalid JSON dataset '{group.name}/{path}'")
        raise


def hdf5_write_meta(group: h5py.Group, meta: frozendict[str, t.Any], path: str = 'meta'):
    if len(meta):
        s = json.dumps(meta, indent='', ensure_ascii=False).encode('utf-8')
        group.create_dataset(path, data=s, dtype=h5py.string_dtype('utf-8'))


def _parse_version(version: str) -> tuple[int, ...]:
    try:
        return tuple(map(int, version.split(".")))
    except ValueError:
        raise ValueError(f"Unable to parse version '{version}'") from None


def _assert_group(group: h5py.Group | h5py.Dataset | h5py.Datatype) -> h5py.Group:
    if isinstance(group, h5py.Group):
        return group
    raise ValueError(f"While reading '{group.file.filename}':\n"
                     f"Expected a group at path '{group.name}', instead found {type(group)}.")


@t.overload
def _hdf5_read_dataset(group: h5py.Group, path: str, dtype: type[DTypeT], nullable: t.Literal[False] = ...) -> DTypeT | NDArray[DTypeT]: ...
@t.overload
def _hdf5_read_dataset(group: h5py.Group, path: str, dtype: type[DTypeT], nullable: bool = ...) -> DTypeT | NDArray[DTypeT] | None: ...

def _hdf5_read_dataset(group: h5py.Group, path: str, dtype: type[DTypeT], nullable: bool = False) -> DTypeT | NDArray[DTypeT] | None:
    dtype_category = _DTYPE_CATEGORIES[dtype]

    if path not in group:
        if nullable:
            return None
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Path '{group.name}/{path}' not found.")

    dataset = group[path]

    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"While reading '{group.file.filename}':\n"
                        f"Expected a dataset at path '{group.name}/{path}', instead found {type(dataset)}.")

    if not numpy.issubdtype(dataset.dtype, dtype_category):
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Expected a dataset of dtype '{dtype_category}' at path '{group.name}/{path}', instead found {dataset.dtype}.")

    if dataset.shape is None:
        if nullable:
            return None
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Dataset at path '{group.name}/{path}' is empty.")

    # ensure promotion is correct. eg dtype = numpy.floating promotes with numpy.float32
    out_dtype = numpy.promote_types(dataset.dtype, _CATEGORY_MIN_DTYPE.get(dtype_category, dtype))
    return dataset[()].astype(out_dtype)


@t.overload
def _hdf5_read_array(group: h5py.Group, path: str, dtype: type[DTypeT], nullable: t.Literal[False] = ...) -> NDArray[DTypeT]: ...
@t.overload
def _hdf5_read_array(group: h5py.Group, path: str, dtype: type[DTypeT], nullable: bool = ...) -> NDArray[DTypeT] | None: ...

def _hdf5_read_array(group: h5py.Group, path: str, dtype: type[DTypeT], nullable: bool = False) -> NDArray[DTypeT] | None:
    arr = _hdf5_read_dataset(group, path, dtype, nullable=nullable)
    if arr is not None:
        return numpy.asarray(arr)


@t.overload
def _hdf5_read_array_shape(
    group: h5py.Group, path: str, dtype: type[DTypeT], shape: tuple[int, ...], nullable: t.Literal[False] = ...
) -> NDArray[DTypeT]: ...
@t.overload
def _hdf5_read_array_shape(
    group: h5py.Group, path: str, dtype: type[DTypeT], shape: tuple[int, ...], nullable: bool = ...
) -> NDArray[DTypeT] | None: ...

def _hdf5_read_array_shape(
    group: h5py.Group, path: str, dtype: type[DTypeT], shape: tuple[int, ...], nullable: bool = False
) -> NDArray[DTypeT] | None:
    arr = _hdf5_read_dataset(group, path, dtype, nullable)
    if arr is None:
        return None
    arr = numpy.asarray(arr)

    if arr.shape != shape:
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Expected a dataset of shape '{shape}' at path '{group.name}/{path}', instead got shape {arr.shape}.")
    return arr


def _hdf5_read_scalar(group: h5py.Group, path: str, dtype: type[DTypeT]) -> DTypeT:
    arr = _hdf5_read_dataset(group, path, dtype)
    if isinstance(arr, numpy.ndarray):
        raise ValueError(f"While reading '{group.file.filename}':\n"  # noqa: TRY004
                         f"Expected a scalar dataset, instead got shape {arr.shape}.")
    return arr


@t.overload
def _hdf5_read_string(group: h5py.Group, path: str, nullable: t.Literal[False] = ...) -> str: ...

@t.overload
def _hdf5_read_string(group: h5py.Group, path: str, nullable: bool = ...) -> str | None: ...

def _hdf5_read_string(group: h5py.Group, path: str, nullable: bool = False) -> str | None:
    if path not in group:
        if nullable:
            return None
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Path '{group.name}/{path}' not found.")

    dataset = group[path]

    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"While reading '{group.file.filename}':\n"
                         f"Expected a string at path '{group.name}/{path}', instead found {type(dataset)}.")

    if dataset.shape is None:
        if nullable:
            return None
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Dataset at path '{group.name}/{path}' is empty.")

    dataset = dataset[()]
    if not isinstance(dataset, bytes):
        raise TypeError(f"While reading '{group.file.filename}':\n"
                         f"Expected a scalar string at path '{group.name}/{path}', instead found {dataset} (type {type(dataset)}).")

    try:
        return dataset.decode('utf-8')
    except ValueError:
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Invalid string at path '{group.name}/{path}")


def _hdf5_write_string(group: h5py.Group, name: str, data: str):
    group.create_dataset(name, (), h5py.string_dtype('utf-8'), data)


def _hdf5_write_nullable_dataset(group: h5py.Group, name: str, data: numpy.ndarray | None, dtype: t.Any):
    if data is not None:
        group.create_dataset(name, data=to_numpy(data).astype(dtype))
    else:
        group.create_dataset(name, dtype=h5py.Empty(dtype))


def tiff_write_opts(
    sampling: Sampling | ObjectSampling,
    corner: NDArray[numpy.floating] | None = None, *,
    unit: t.Literal['angstrom'] = 'angstrom',  # other units not yet supported
    n_slices: int = 1,
    zs: t.Sequence[float] | NDArray[numpy.floating] | None = None,
) -> dict[str, t.Any]:
    if corner is None:
        corner = sampling.corner

    z_dict = {}
    if zs is not None:
        n_slices = len(zs)
        z_dict['PositionZ'] = list(map(float, zs))
        z_dict['PositionZUnit'] = [unit] * n_slices

    return {
        # 1/angstrom -> 1/cm
        'resolution': tuple(float(1e8/s) for s in reversed(sampling.sampling)),
        'resolutionunit': 'CENTIMETER',
        'metadata': {
            'OME': {
                'PhysicalSizeX': float(sampling.sampling[1]),
                'PhysicalSizeXUnit': unit,
                'PhysicalSizeY': float(sampling.sampling[0]),
                'PhysicalSizeYUnit': unit,
                'Plane': {
                    'PositionX': [float(corner[1])] * n_slices,
                    'PositionXUnit': [unit] * n_slices,
                    'PositionY': [float(corner[0])] * n_slices,
                    'PositionYUnit': [unit] * n_slices,
                    **z_dict
                }
            }
        }
    }


def tiff_write_opts_recip(
    sampling: Sampling | ObjectSampling, *,
    unit: t.Literal['1/angstrom'] = '1/angstrom',  # other units not yet supported
    n_slices: int = 1,
    zs: t.Sequence[float] | NDArray[numpy.floating] | None = None,
) -> dict[str, t.Any]:
    z_dict = {}
    if zs is not None:
        n_slices = len(zs)
        z_dict['PositionZ'] = list(map(float, zs))
        z_dict['PositionZUnit'] = [unit] * n_slices

    d = {
        'metadata': {
            'OME': {
                'PhysicalSizeX': float(1/sampling.extent[1]),
                'PhysicalSizeXUnit': unit,
                'PhysicalSizeY': float(1/sampling.extent[0]),
                'PhysicalSizeYUnit': unit,
                'Plane': {
                    # TODO get recip corner here
                    'PositionX': [0.0] * n_slices,
                    'PositionXUnit': [unit] * n_slices,
                    'PositionY': [0.0] * n_slices,
                    'PositionYUnit': [unit] * n_slices,
                    **z_dict,
                }
            }
        }
    }

    return d


__all__ = [
    'HdfLike',
    'OpenMode',
    'hdf5_read_state',
    'hdf5_write_state',
    'open_hdf5',
    'tiff_write_opts',
    'tiff_write_opts_recip',
]