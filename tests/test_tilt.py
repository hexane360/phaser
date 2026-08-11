# type: ignore

import h5py
import numpy
from numpy.testing import assert_allclose
import pane
import pytest

from phaser.hooks import TiltHook


@pytest.fixture
def tilts_file(tmp_path):
    """Synthetic .tilts file matching the 4DSTEM Explorer 'Sample Tilt' format."""
    path = tmp_path / "sample_tilt.tilts"

    ny, nx = (4, 5)
    tilt_x = numpy.arange(ny * nx, dtype=numpy.float32).reshape(ny, nx)
    tilt_y = -2.0 * tilt_x

    with h5py.File(path, 'w') as f:
        f.create_dataset('scan/tilt_x', data=tilt_x)
        f.create_dataset('scan/tilt_y', data=tilt_y)
        f.create_dataset('binned/tilt_x', data=tilt_x[::2, ::2])
        f.create_dataset('binned/tilt_y', data=tilt_y[::2, ::2])

    return (path, tilt_x, tilt_y)


def _args(shape):
    return {'dtype': numpy.float32, 'xp': numpy, 'shape': shape}


def test_load_tilts_file(tilts_file):
    path, tilt_x, tilt_y = tilts_file
    hook = pane.from_data({'type': 'tilts', 'path': str(path)}, TiltHook)

    tilt = hook(_args(tilt_x.shape))

    assert tilt.shape == (*tilt_x.shape, 2)
    assert tilt.dtype == numpy.float32
    assert_allclose(tilt[..., 0], tilt_y)
    assert_allclose(tilt[..., 1], tilt_x)


def test_load_tilts_file_scale(tilts_file):
    path, tilt_x, tilt_y = tilts_file
    hook = pane.from_data({'type': 'tilts', 'path': str(path), 'scale': -0.5}, TiltHook)

    tilt = hook(_args(tilt_x.shape))

    assert_allclose(tilt[..., 0], -0.5 * tilt_y)
    assert_allclose(tilt[..., 1], -0.5 * tilt_x)


def test_load_tilts_file_flips(tilts_file):
    path, tilt_x, tilt_y = tilts_file

    hook = pane.from_data({'type': 'tilts', 'path': str(path), 'flips': (True, False, False)}, TiltHook)
    tilt = hook(_args(tilt_x.shape))
    assert_allclose(tilt[..., 0], tilt_y[::-1])
    assert_allclose(tilt[..., 1], tilt_x[::-1])

    hook = pane.from_data({'type': 'tilts', 'path': str(path), 'flips': (False, True, False)}, TiltHook)
    tilt = hook(_args(tilt_x.shape))
    assert_allclose(tilt[..., 0], tilt_y[:, ::-1])
    assert_allclose(tilt[..., 1], tilt_x[:, ::-1])


def test_load_tilts_file_transpose_flip(tilts_file):
    path, tilt_x, tilt_y = tilts_file

    hook = pane.from_data({'type': 'tilts', 'path': str(path), 'flips': (False, False, True)}, TiltHook)
    tilt = hook(_args(tilt_x.shape[::-1]))

    assert tilt.shape == (*tilt_x.shape[::-1], 2)
    assert_allclose(tilt[..., 0], tilt_y.T)
    assert_allclose(tilt[..., 1], tilt_x.T)


def test_load_tilts_file_wrong_shape(tilts_file):
    path, tilt_x, _ = tilts_file
    hook = pane.from_data({'type': 'tilts', 'path': str(path)}, TiltHook)

    with pytest.raises(ValueError, match="doesn't match scan shape"):
        hook(_args((10, 10)))


def test_load_tilts_file_transposed_shape(tilts_file):
    path, tilt_x, _ = tilts_file
    hook = pane.from_data({'type': 'tilts', 'path': str(path)}, TiltHook)

    with pytest.raises(ValueError, match="transposed scan"):
        hook(_args(tilt_x.shape[::-1]))


def test_load_tilts_file_missing(tmp_path):
    hook = pane.from_data({'type': 'tilts', 'path': str(tmp_path / "nonexistent.tilts")}, TiltHook)

    with pytest.raises(FileNotFoundError):
        hook(_args((4, 5)))


def test_load_tilts_file_not_tilts(tmp_path):
    path = tmp_path / "other.h5"
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=numpy.zeros((4, 5)))

    hook = pane.from_data({'type': 'tilts', 'path': str(path)}, TiltHook)

    with pytest.raises(ValueError, match="doesn't look like a .tilts file"):
        hook(_args((4, 5)))
