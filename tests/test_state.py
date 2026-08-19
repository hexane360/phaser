import re
from pathlib import Path

import h5py
import numpy
import pytest
from frozendict import frozendict
from numpy.testing import assert_allclose

from phaser.state import (
    IterState,
    ObjectState,
    PartialReconsState,
    ProbeState,
    ProgressState,
    ReconsState,
    ScanState,
)
from phaser.utils.io import hdf5_read_state
from phaser.utils.num import Sampling, get_backend_module
from phaser.utils.object import ObjectSampling

from .utils import INPUT_FILES_PATH

# scan and tilt of the checked-in v0.1 fixtures
V0_1_SCAN = numpy.stack(numpy.meshgrid(
    numpy.arange(4.) * 0.6, numpy.arange(4.) * 0.6, indexing='ij',
), axis=-1)
V0_1_TILT = numpy.linspace(-1., 1., 32).reshape(4, 4, 2)


def make_state(*, region: bool = True, meta: bool = True) -> ReconsState:
    rng = numpy.random.default_rng(0x5CA4)

    probe = ProbeState(
        Sampling((8, 8), sampling=(0.5, 0.5)),
        (rng.random((2, 8, 8)) + 1.j * rng.random((2, 8, 8))).astype(numpy.complex64),
        meta=frozendict({'source': 'test'}) if meta else frozendict(),
    )
    obj = ObjectState(
        ObjectSampling(
            (16, 16), (0.5, 0.5), (-4., -4.),
            (-2., -2.) if region else None, (2., 2.) if region else None,
        ),
        (rng.random((2, 16, 16)) + 1.j * rng.random((2, 16, 16))).astype(numpy.complex64),
        numpy.array([10., 10.]),
        meta=frozendict({'slices': 2}) if meta else frozendict(),
    )
    # data, initial and tilt are all distinct
    scan = ScanState(
        rng.random((4, 4, 2)) * 2.,
        rng.random((4, 4, 2)) * 2.,
        rng.random((4, 4, 2)) * 0.1,
        meta=frozendict({'step_size': [0.6, 0.6]}) if meta else frozendict(),
    )

    return ReconsState(
        iter=IterState(2, 5, 15), wavelength=0.0251,
        probe=probe, object=obj, scan=scan,
        progress={'total_loss': ProgressState([1, 2], [3., 2.])},
    )


@pytest.mark.parametrize('region', (True, False))
def test_state_hdf5_roundtrip(tmp_path: Path, region: bool):
    state = make_state(region=region)
    path = tmp_path / 'state.h5'
    state.write_hdf5(path)

    with h5py.File(path) as f:
        assert f['probe']['type'][()] == b'pixelated'
        assert f['object']['type'][()] == b'pixelated'

    read = ReconsState.read_hdf5(path)

    assert read.wavelength == state.wavelength
    assert read.probe.ty == read.object.ty == 'pixelated'
    assert (read.iter.engine_num, read.iter.engine_iter, read.iter.total_iter) == (2, 5, 15)
    assert read.progress['total_loss'].iters == [1, 2]
    assert read.progress['total_loss'].values == [3., 2.]

    assert read.probe.sampling == state.probe.sampling
    assert_allclose(read.probe.data, state.probe.data)

    assert read.object.sampling == state.object.sampling
    assert_allclose(read.object.data, state.object.data)
    assert_allclose(read.object.thicknesses, state.object.thicknesses)

    if region:
        assert_allclose(read.object.sampling.region_min, [-2., -2.])
        assert_allclose(read.object.sampling.region_max, [2., 2.])
    else:
        assert read.object.sampling.region_min is None
        assert read.object.sampling.region_max is None

    assert_allclose(read.scan.data, state.scan.data)
    assert_allclose(read.scan.initial, state.scan.initial)
    assert_allclose(read.scan.tilt, state.scan.tilt)

    # `initial` is stored, not synthesized from `data`
    assert not numpy.allclose(read.scan.data, read.scan.initial)
    assert not numpy.shares_memory(read.scan.data, read.scan.initial)


def test_state_hdf5_roundtrip_no_tilt(tmp_path: Path):
    state = make_state()
    state.scan.tilt = None
    path = tmp_path / 'state.h5'
    state.write_hdf5(path)

    with h5py.File(path) as f:
        assert 'tilt' not in f['scan']

    assert ReconsState.read_hdf5(path).scan.tilt is None


def test_state_hdf5_meta_roundtrip(tmp_path: Path):
    meta = frozendict({
        'str': 'value', 'int': 3, 'float': 1.5, 'bool': True, 'null': None,
        'list': [1, [2, 3]], 'dict': {'inner': [4]},
    })
    # JSON arrays are frozen to tuples, making `meta` hashable
    expected = frozendict({
        'str': 'value', 'int': 3, 'float': 1.5, 'bool': True, 'null': None,
        'list': (1, (2, 3)), 'dict': frozendict({'inner': (4,)}),
    })

    state = make_state()
    state.scan.meta = meta
    path = tmp_path / 'state.h5'
    state.write_hdf5(path)
    read = ReconsState.read_hdf5(path)

    assert read.scan.meta == expected
    hash(read.scan.meta)

    assert read.probe.meta == state.probe.meta
    assert read.object.meta == state.object.meta


def test_state_hdf5_meta_empty(tmp_path: Path):
    state = make_state(meta=False)
    path = tmp_path / 'state.h5'
    state.write_hdf5(path)

    # empty metadata leaves no dataset behind
    with h5py.File(path) as f:
        assert 'meta' not in f['scan']
        assert 'meta' not in f['probe']
        assert 'meta' not in f['object']

    assert ReconsState.read_hdf5(path).scan.meta == frozendict()


@pytest.mark.jax
def test_state_pytree_leaves_are_numeric():
    # `tree` dispatches to jax directly, so the backend must be loaded before
    # the state classes are registered as pytree nodes
    try:
        get_backend_module('jax')
    except ValueError as e:
        pytest.skip(str(e))

    from phaser.utils import tree

    # every non-static field is an array; discriminators and metadata are static
    for (path, leaf) in tree.leaves_with_path(make_state()):
        assert numpy.issubdtype(numpy.asarray(leaf).dtype, numpy.number), \
            f"non-numeric pytree leaf at '{''.join(map(str, path))}'"


def test_state_meta_survives_backend_roundtrip():
    state = make_state()
    round_tripped = state.to_xp(numpy).to_numpy()

    assert round_tripped.probe.meta == state.probe.meta
    assert round_tripped.object.meta == state.object.meta
    assert round_tripped.scan.meta == state.scan.meta

    assert state.probe.resample(state.probe.sampling).meta == state.probe.meta


@pytest.mark.parametrize(('name', 'has_tilt'), (
    ('state_v0.1.h5', True),
    ('state_v0.1_no_tilt.h5', False),
))
def test_read_state_v0_1(name: str, has_tilt: bool):
    """Fixtures written by the pre-`ScanState` writer, at commit 55d531e."""
    state = hdf5_read_state(INPUT_FILES_PATH / name)
    assert state.scan is not None

    assert_allclose(state.scan.data, V0_1_SCAN)
    # v0.1 has no `initial`, so the stored scan stands in for it
    assert_allclose(state.scan.initial, V0_1_SCAN)
    assert not numpy.shares_memory(state.scan.data, state.scan.initial)

    if has_tilt:
        assert_allclose(state.scan.tilt, V0_1_TILT)
    else:
        assert state.scan.tilt is None

    assert state.scan.meta == frozendict()
    assert state.probe is not None and state.probe.meta == frozendict()
    # v0.1 has no 'type', which reads back as pixelated
    assert state.probe.ty == 'pixelated'

    # v0.1 wrote absent regions as empty datasets
    assert state.object is not None
    assert state.object.ty == 'pixelated'
    assert state.object.sampling.region_min is None
    assert state.object.sampling.region_max is None


@pytest.mark.parametrize(('group', 'ty'), (
    ('probe', 'fake_type'),
    ('object', 'fake_type'),
))
def test_read_state_unsupported_type(tmp_path: Path, group: str, ty: str):
    path = tmp_path / 'state.h5'
    make_state().write_hdf5(path)

    with h5py.File(path, 'r+') as f:
        del f[group]['type']
        f[group].create_dataset('type', data=ty, dtype=h5py.string_dtype('utf-8'))

    with pytest.raises(ValueError, match=re.escape(f"Unsupported {group} type '{ty}'")):
        hdf5_read_state(path)


def test_read_state_unsupported_version(tmp_path: Path):
    path = tmp_path / 'state.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('type', (), h5py.string_dtype(), "phaser_state")
        f.create_dataset('version', (), h5py.string_dtype(), "0.3")

    with pytest.raises(ValueError, match=re.escape("Unsupported file version '0.3'")):
        hdf5_read_state(path)


def test_partial_state_hdf5_roundtrip(tmp_path: Path):
    state = make_state()
    path = tmp_path / 'state.h5'
    PartialReconsState(wavelength=state.wavelength, probe=state.probe).write_hdf5(path)
    read = PartialReconsState.read_hdf5(path)

    assert read.scan is None
    assert read.object is None
    assert read.progress is None
    assert read.iter == IterState.empty()

    assert read.probe is not None
    assert_allclose(read.probe.data, state.probe.data)

    with pytest.raises(ValueError, match=re.escape("ReconsState missing 'object', 'scan'")):
        read.to_complete()
