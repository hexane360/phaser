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


@pytest.mark.parametrize('fixture', ['tilts_file', 'npy_file'])
def test_load_crop(fixture, request, tilt_arrays):
    path = request.getfixturevalue(fixture)
    tilt_x, tilt_y = tilt_arrays

    tilt = _hook(path, crop=(1, 3, 2, None))(_args((2, 3)))
    assert tilt.shape == (2, 3, 2)
    assert_allclose(tilt[..., 0], tilt_y[1:3, 2:])
    assert_allclose(tilt[..., 1], tilt_x[1:3, 2:])


def test_load_crop_open_ended(tilts_file, tilt_arrays):
    """`None` runs to the edge, as in `crop_data`."""
    tilt_x, _ = tilt_arrays

    tilt = _hook(tilts_file, crop=(None, None, 1, 4))(_args((4, 3)))
    assert_allclose(tilt[..., 1], tilt_x[:, 1:4])


def test_load_crop_after_flips(tilts_file, tilt_arrays):
    """The window is in scan coordinates, so it applies to the flipped map."""
    tilt_x, _ = tilt_arrays

    tilt = _hook(tilts_file, flips=(True, False, False), crop=(0, 2, None, None))(_args((2, 5)))
    assert_allclose(tilt[..., 1], tilt_x[::-1][0:2])


def test_load_crop_scale(tilts_file, tilt_arrays):
    tilt_x, _ = tilt_arrays
    tilt = _hook(tilts_file, crop=(0, 2, 0, 2), scale=-0.5)(_args((2, 2)))
    assert_allclose(tilt[..., 1], -0.5 * tilt_x[0:2, 0:2])


def test_load_crop_flat_npy(tmp_path, tilt_arrays):
    """A flat (N, 2) map has no scan dimensions to crop along."""
    tilt_x, tilt_y = tilt_arrays
    path = tmp_path / "tilt_flat.npy"
    numpy.save(path, numpy.stack([tilt_y.ravel(), tilt_x.ravel()], axis=-1))

    with pytest.raises(ValueError, match="carries no scan dimensions"):
        _hook(path, crop=(0, 2, 0, 2))(_args((2, 2)))


def test_load_crop_empty(tilts_file):
    with pytest.raises(ValueError, match="selects nothing"):
        _hook(tilts_file, crop=(3, 1, None, None))(_args((0, 5)))


def test_load_crop_wrong_shape(tilts_file):
    with pytest.raises(ValueError, match="cropped to"):
        _hook(tilts_file, crop=(0, 2, 0, 2))(_args((4, 5)))


def test_crop_data_sets_tilt_crop():
    """`crop_data` hands its window to a custom tilt hook, and leaves an explicit one alone."""
    from phaser.hooks import PostLoadHook

    def run(tilt_hook):
        raw_data = {
            'patterns': numpy.zeros((4, 5, 2, 2)),
            'scan_hook': {'type': 'raster', 'shape': (4, 5)},
            'tilt_hook': tilt_hook,
        }
        hook = pane.from_data({'type': 'crop_data', 'crop': [1, 3, 2, None]}, PostLoadHook)
        return hook(raw_data)

    out = run({'type': 'custom', 'path': 'tilt.tilts'})
    assert out['tilt_hook']['crop'] == (1, 3, 2, None)
    assert out['scan_hook']['shape'] == (2, 3)

    # an explicit crop is someone saying they've already accounted for it
    out = run({'type': 'custom', 'path': 'tilt.tilts', 'crop': (0, 1, 0, 1)})
    assert out['tilt_hook']['crop'] == (0, 1, 0, 1)

    # a global tilt is uniform, so there is nothing to crop
    out = run({'type': 'global', 'tilt': (1.0, 2.0)})
    assert 'crop' not in out['tilt_hook']


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
