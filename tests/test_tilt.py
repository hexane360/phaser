# type: ignore

import h5py
import numpy
from numpy.testing import assert_allclose
import pane
import pytest

from phaser.hooks import TiltHook


@pytest.fixture
def tilt_arrays():
    ny, nx = (4, 5)
    tilt_x = numpy.arange(ny * nx, dtype=numpy.float32).reshape(ny, nx)
    tilt_y = -2.0 * tilt_x
    return (tilt_x, tilt_y)


@pytest.fixture
def tilts_file(tmp_path, tilt_arrays):
    """Synthetic .tilts file matching the 4DSTEM Explorer 'Sample Tilt' format."""
    path = tmp_path / "sample_tilt.tilts"
    tilt_x, tilt_y = tilt_arrays

    with h5py.File(path, 'w') as f:
        f.create_dataset('scan/tilt_x', data=tilt_x)
        f.create_dataset('scan/tilt_y', data=tilt_y)
        f.create_dataset('binned/tilt_x', data=tilt_x[::2, ::2])
        f.create_dataset('binned/tilt_y', data=tilt_y[::2, ::2])

    return path


@pytest.fixture
def npy_file(tmp_path, tilt_arrays):
    path = tmp_path / "tilt.npy"
    tilt_x, tilt_y = tilt_arrays
    numpy.save(path, numpy.stack([tilt_y, tilt_x], axis=-1))
    return path


def _args(shape):
    return {'dtype': numpy.float32, 'xp': numpy, 'shape': shape}


def _hook(path, **props):
    return pane.from_data({'type': 'custom', 'path': str(path), **props}, TiltHook)


def test_load_tilts_file(tilts_file, tilt_arrays):
    tilt_x, tilt_y = tilt_arrays
    tilt = _hook(tilts_file)(_args(tilt_x.shape))

    assert tilt.shape == (*tilt_x.shape, 2)
    assert tilt.dtype == numpy.float32
    assert_allclose(tilt[..., 0], tilt_y)
    assert_allclose(tilt[..., 1], tilt_x)


def test_load_npy(npy_file, tilt_arrays):
    tilt_x, tilt_y = tilt_arrays
    tilt = _hook(npy_file)(_args(tilt_x.shape))

    assert tilt.shape == (*tilt_x.shape, 2)
    assert tilt.dtype == numpy.float32
    assert_allclose(tilt[..., 0], tilt_y)
    assert_allclose(tilt[..., 1], tilt_x)


def test_load_npy_flat(tmp_path, tilt_arrays):
    tilt_x, tilt_y = tilt_arrays
    path = tmp_path / "tilt_flat.npy"
    numpy.save(path, numpy.stack([tilt_y.ravel(), tilt_x.ravel()], axis=-1))

    tilt = _hook(path)(_args(tilt_x.shape))

    assert tilt.shape == (*tilt_x.shape, 2)
    assert_allclose(tilt[..., 0], tilt_y)
    assert_allclose(tilt[..., 1], tilt_x)


def test_load_scale(tilts_file, tilt_arrays):
    tilt_x, tilt_y = tilt_arrays
    tilt = _hook(tilts_file, scale=-0.5)(_args(tilt_x.shape))

    assert_allclose(tilt[..., 0], -0.5 * tilt_y)
    assert_allclose(tilt[..., 1], -0.5 * tilt_x)


@pytest.mark.parametrize('fixture', ['tilts_file', 'npy_file'])
def test_load_flips(fixture, request, tilt_arrays):
    path = request.getfixturevalue(fixture)
    tilt_x, tilt_y = tilt_arrays

    tilt = _hook(path, flips=(True, False, False))(_args(tilt_x.shape))
    assert_allclose(tilt[..., 0], tilt_y[::-1])
    assert_allclose(tilt[..., 1], tilt_x[::-1])

    tilt = _hook(path, flips=(False, True, False))(_args(tilt_x.shape))
    assert_allclose(tilt[..., 0], tilt_y[:, ::-1])
    assert_allclose(tilt[..., 1], tilt_x[:, ::-1])

    tilt = _hook(path, flips=(False, False, True))(_args(tilt_x.shape[::-1]))
    assert tilt.shape == (*tilt_x.shape[::-1], 2)
    assert_allclose(tilt[..., 0], tilt_y.T)
    assert_allclose(tilt[..., 1], tilt_x.T)


def test_load_wrong_shape(tilts_file):
    with pytest.raises(ValueError, match="doesn't match scan shape"):
        _hook(tilts_file)(_args((10, 10)))


def test_load_transposed_shape(tilts_file, tilt_arrays):
    tilt_x, _ = tilt_arrays
    with pytest.raises(ValueError, match="transposed scan"):
        _hook(tilts_file)(_args(tilt_x.shape[::-1]))


def test_load_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        _hook(tmp_path / "nonexistent.tilts")(_args((4, 5)))


def test_load_unsupported_extension(tmp_path):
    path = tmp_path / "tilt.txt"
    path.write_text("not a tilt map")

    with pytest.raises(ValueError, match="Unsupported tilt file extension '.txt'"):
        _hook(path)(_args((4, 5)))


def test_load_not_tilts(tmp_path):
    path = tmp_path / "other.tilts"
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=numpy.zeros((4, 5)))

    with pytest.raises(ValueError, match="doesn't look like a .tilts file"):
        _hook(path)(_args((4, 5)))
