import contextlib
from pathlib import Path
import typing as t
import jax

import numpy
from numpy.typing import NDArray
import h5py

from phaser.utils.num import Sampling, to_numpy, get_array_module
from phaser.utils.object import ObjectSampling
from phaser.state import ReconsState, IterState, ProbeState, ObjectState, ProgressState, PartialReconsState, ParameterizedProbeState


HdfLike: t.TypeAlias = t.Union[h5py.File, str, Path]
OpenMode: t.TypeAlias = t.Literal['r', 'r+', 'w', 'w-', 'x', 'a']
DTypeT = t.TypeVar('DTypeT', bound=numpy.generic)

_DTYPE_CATEGORIES: t.Dict[t.Type[numpy.generic], t.Type[numpy.generic]] = {
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

_CATEGORY_MIN_DTYPE: t.Dict[t.Type[numpy.generic], t.Type[numpy.generic]] = {
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

    def __exit__(self, exc_type: t.Optional[type], exc_value: t.Optional[BaseException], tb: t.Any):
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
    version = _hdf5_read_string(file, 'version')
    if ty != 'phaser_state':
        raise ValueError(f"While reading file '{file.filename}':\nExpected a file of type 'phaser_state', instead got type '{ty}'")

    if _parse_version(version) > (0, 1):
        raise ValueError(f"While reading file '{file.filename}':\nUnsupported file version '{version}'. Maximum supported version is '0.1'.")

    wavelength = _hdf5_read_scalar(file, 'wavelength', numpy.float64) if 'wavelength' in file else None

    probe = hdf5_read_probe_state(_assert_group(file['probe'])) if 'probe' in file else None
    obj = hdf5_read_object_state(_assert_group(file['object'])) if 'object' in file else None
    iter = hdf5_read_iter_state(_assert_group(file['iter'])) if 'iter' in file else IterState.empty()
    scan = numpy.asarray(_hdf5_read_dataset(file, 'scan', numpy.float64)) if 'scan' in file else None
    tilt = numpy.asarray(_hdf5_read_dataset(file, 'tilt', numpy.float64)) if 'tilt' in file else None

    if tilt is not None and scan is not None:
        assert tilt.shape == scan.shape
    progress = hdf5_read_progress_state(_assert_group(file['progress'])) if 'progress' in file else None

    return PartialReconsState(
        wavelength=wavelength, iter=iter, probe=probe,
        object=obj, scan=scan, tilt=tilt, progress=progress
    )


def hdf5_read_probe_state(group: h5py.Group) -> t.Union[ProbeState, ParameterizedProbeState]:
    extent = _hdf5_read_dataset_shape(group, 'extent', numpy.float64, (2,))
    n_y, n_x = _hdf5_read_dataset_shape(group, 'extent', numpy.int16, (2,))
    probes = _hdf5_read_dataset(group, 'data', numpy.complexfloating)
    assert probes.ndim == 3

    if 'parameters' in group:
        parameters = numpy.atleast_1d(_hdf5_read_dataset(group, 'parameters', numpy.floating))
        conv_angle = float(_hdf5_read_dataset(group, 'conv_angle', numpy.floating))
        wavelength = float(_hdf5_read_dataset(group, 'wavelength', numpy.floating))
        return ParameterizedProbeState(
            Sampling((n_y, n_x), extent=(extent[0], extent[1])),
            params=parameters, conv_angle=conv_angle, wavelength=wavelength
        )
    return ProbeState(
        Sampling((n_y, n_x), extent=(extent[0], extent[1])),
        data=probes
        )


def hdf5_read_object_state(group: h5py.Group) -> ObjectState:
    obj = numpy.asarray(_hdf5_read_dataset(group, 'data', numpy.complexfloating))
    (n_z, n_y, n_x) = obj.shape
    
    thicknesses = numpy.asarray(_hdf5_read_dataset(group, 'thicknesses', numpy.floating))
    assert thicknesses.ndim == 1
    assert thicknesses.size == n_z if n_z > 1 else thicknesses.size in (0, 1)

    sampling = _hdf5_read_dataset_shape(group, 'sampling', numpy.float64, (2,))
    corner = _hdf5_read_dataset_shape(group, 'corner', numpy.float64, (2,))

    region_min = _hdf5_read_dataset_shape(group, 'region_min', numpy.float64, (2,)) if 'region_min' in group else None
    region_max = _hdf5_read_dataset_shape(group, 'region_max', numpy.float64, (2,)) if 'region_max' in group else None

    return ObjectState(
        ObjectSampling((n_y, n_x), sampling, corner, region_min, region_max),
        data=obj, thicknesses=thicknesses
    )


def hdf5_read_iter_state(group: h5py.Group) -> IterState:
    engine_num = int(_hdf5_read_scalar(group, 'engine_num', numpy.int64))
    engine_iter = int(_hdf5_read_scalar(group, 'engine_iter', numpy.int64))
    total_iter = int(_hdf5_read_scalar(group, 'total_iter', numpy.int64))

    return IterState(
        engine_num=engine_num, engine_iter=engine_iter, total_iter=total_iter
    )


def hdf5_read_progress_state(group: h5py.Group) -> ProgressState:
    iters = numpy.asarray(_hdf5_read_dataset(group, 'iters', numpy.int64))
    errors = numpy.asarray(_hdf5_read_dataset(group, 'detector_errors', numpy.float64))
    assert iters.ndim == errors.ndim == 1
    assert iters.shape == errors.shape

    return ProgressState(
        iters=iters, detector_errors=errors,
    )


def hdf5_write_state(state: t.Union[ReconsState, PartialReconsState], file: HdfLike):
    file = open_hdf5(file, 'w')  # overwrite if existing
    file.create_dataset('type', (), h5py.string_dtype(), "phaser_state")
    file.create_dataset('version', (), h5py.string_dtype(), "0.1")
    file.create_dataset('wavelength', (), numpy.float64, state.wavelength)

    if state.probe is not None:
        hdf5_write_probe_state(state.probe, file.create_group("probe"))
    if state.object is not None:
        hdf5_write_object_state(state.object, file.create_group("object"))
    if state.scan is not None:
        file.create_dataset('scan', data=to_numpy(state.scan.astype(numpy.float64)))
    if state.tilt is not None:
        file.create_dataset('tilt', data=to_numpy(state.tilt.astype(numpy.float64)))
    if state.iter is not None:
        hdf5_write_iter_state(state.iter, file.create_group("iter"))
    if state.progress is not None:
        hdf5_write_progress_state(state.progress, file.create_group("progress"))


def hdf5_write_probe_state(state: t.Union[ProbeState, ParameterizedProbeState], group: h5py.Group):
    # Pixel-wise probe data
    #TODO: not working when turn on probe limit
    assert state.data.ndim == 3
    dataset = group.create_dataset('data', data=to_numpy(state.data))
    dataset.dims[0].label = 'mode'
    dataset.dims[1].label = 'y'
    dataset.dims[2].label = 'x'

    group.create_dataset('sampling', data=state.sampling.sampling.astype(numpy.float64))
    group.create_dataset('extent', data=state.sampling.extent.astype(numpy.float64))
    group.create_dataset('shape', data=state.sampling.shape.astype(numpy.int16))
    if isinstance(state, ParameterizedProbeState):
        ##TODO more readbale?
        group.create_dataset('parameters', data=to_numpy(state.params))
        group.create_dataset('wavelength', data=float(state.wavelength))
        group.create_dataset('conv_angle', data=float(state.conv_angle))


def hdf5_write_object_state(state: ObjectState, group: h5py.Group):
    assert state.data.ndim == 3
    assert state.thicknesses.ndim == 1
    n_z = state.data.shape[0]
    assert state.thicknesses.ndim == 1
    assert state.thicknesses.size == n_z if n_z > 1 else state.thicknesses.size in (0, 1)

    thick = to_numpy(state.thicknesses)
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


def hdf5_write_iter_state(state: IterState, group: h5py.Group):
    group.create_dataset("engine_num", (), numpy.uint64, data=state.engine_num)
    group.create_dataset("engine_iter", (), numpy.uint64, data=state.engine_iter)
    group.create_dataset("total_iter", (), numpy.uint64, data=state.total_iter)


def hdf5_write_progress_state(state: ProgressState, group: h5py.Group):
    iters = group.create_dataset("iters", data=state.iters.astype(numpy.uint64))
    iters.make_scale("total_iter")
    dataset = group.create_dataset("detector_errors", data=state.detector_errors.astype(numpy.float64))
    dataset.dims[0].label = 'total_iter'
    dataset.dims[0].attach_scale(iters)


def _parse_version(version: str) -> t.Tuple[int, ...]:
    try:
        return tuple(map(int, version.split(".")))
    except ValueError:
        raise ValueError(f"Unable to parse version '{version}'") from None


def _assert_group(group: t.Union[h5py.Group, h5py.Dataset, h5py.Datatype]) -> h5py.Group:
    if isinstance(group, h5py.Group):
        return group
    raise ValueError(f"While reading '{group.file.filename}':\n"
                     f"Expected a group at path '{group.name}', instead found {type(group)}.")


def _hdf5_read_dataset(group: h5py.Group, path: str, dtype: t.Type[DTypeT]) -> t.Union[DTypeT, NDArray[DTypeT]]:
    dtype_category = _DTYPE_CATEGORIES[dtype]

    if path not in group:
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Path '{group.name}{path}' not found.")

    dataset = group[path]

    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Expected a dataset at path '{group.name}{path}', instead found {type(dataset)}.")

    if not numpy.issubdtype(dataset.dtype, dtype_category):
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Expected a dataset of dtype '{dtype_category}' at path '{group.name}{path}', instead found {dataset.dtype}.")

    # ensure promotion is correct. eg dtype = numpy.floating promotes with numpy.float32
    out_dtype = numpy.promote_types(dataset.dtype, _CATEGORY_MIN_DTYPE.get(dtype, dtype))
    return dataset[()].astype(out_dtype)


def _hdf5_read_dataset_shape(group: h5py.Group, path: str, dtype: t.Type[DTypeT], shape: t.Tuple[int, ...]) -> NDArray[DTypeT]:
    arr = numpy.asarray(_hdf5_read_dataset(group, path, dtype))
    if arr.shape != shape:
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Expected a dataset of shape '{shape}' at path '{group.name}{path}', instead got shape {arr.shape}.")
    return arr


def _hdf5_read_scalar(group: h5py.Group, path: str, dtype: t.Type[DTypeT]) -> DTypeT:
    arr = _hdf5_read_dataset(group, path, dtype)
    if isinstance(arr, numpy.ndarray):
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Expected a scalar dataset, instead got shape {arr.shape}.")
    return arr


def _hdf5_read_string(group: h5py.Group, path: str) -> str:
    if path not in group:
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Path '{group.name}{path}' not found.")

    dataset = group[path]

    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Expected a string at path '{group.name}{path}', instead found {type(dataset)}.")

    dataset = dataset[()]
    if not isinstance(dataset, bytes):
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Expected a scalar string at path '{group.name}{path}', instead found {dataset} (type {type(dataset)}).")

    try:
        return dataset.decode('utf-8')
    except ValueError:
        raise ValueError(f"While reading '{group.file.filename}':\n"
                         f"Invalid string at path '{group.name}{path}")


def _hdf5_write_nullable_dataset(group: h5py.Group, name: str, data: t.Optional[numpy.ndarray], dtype: t.Any):
    if data is not None:
        group.create_dataset(name, data=to_numpy(data.astype(dtype)))
    else:
        group.create_dataset(name, dtype=h5py.Empty(dtype))

        numpy.array(1, dtype=numpy.floating)


def tiff_write_opts(
    sampling: t.Union[Sampling, ObjectSampling],
    corner: t.Optional[NDArray[numpy.floating]] = None, *,
    unit: t.Literal['angstrom'] = 'angstrom',  # other units not yet supported
    n_slices: int = 1,
    zs: t.Union[t.Sequence[float], NDArray[numpy.floating], None] = None,
) -> t.Dict[str, t.Any]:
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
    sampling: t.Union[Sampling, ObjectSampling], *,
    unit: t.Literal['1/angstrom'] = '1/angstrom',  # other units not yet supported
    n_slices: int = 1,
    zs: t.Union[t.Sequence[float], NDArray[numpy.floating], None] = None,
) -> t.Dict[str, t.Any]:
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
    'open_hdf5',
    'hdf5_read_state', 'hdf5_write_state',
    'tiff_write_opts', 'tiff_write_opts_recip',
    'HdfLike', 'OpenMode',
]