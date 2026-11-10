from pathlib import Path

import numpy
import pane
import pytest

from phaser.hooks import ScanHook
from phaser.hooks.scan import load_scan_positions


def _scan(path: Path, **props):
    hook = pane.from_data({'type': 'non_raster', 'path': str(path), **props}, ScanHook)
    return hook({'seed': None, 'dtype': numpy.float64, 'xp': numpy})


def test_positions_file(tmp_path: Path):
    path = tmp_path / 'scan_positions.npz'
    # frames 0 and 3 have no position; positions in m
    nan = numpy.nan
    numpy.savez(path, x=[nan, 1e-10, 3e-10, nan, 0.], y=[nan, 2e-10, -1e-10, nan, 0.], units='m', note='ignored')
    scan = _scan(path)

    assert scan.meta['type'] == 'non_raster'
    assert scan.data.shape == (5, 2)
    # (y, x) in angstrom
    numpy.testing.assert_allclose(scan.data[[1, 2, 4]], [[2., 1.], [-1., 3.], [0., 0.]])
    assert numpy.isnan(scan.data[[0, 3]]).all()
    numpy.testing.assert_array_equal(scan.initial, scan.data)


@pytest.mark.parametrize(('units', 'scale'), [(None, 1e10), ('m', 1e10), ('nm', 10.), ('A', 1.)])
def test_positions_units(tmp_path: Path, units: str | None, scale: float):
    path = tmp_path / 'p.npz'
    numpy.savez(path, x=[1.], y=[2.], **({} if units is None else {'units': units}))
    numpy.testing.assert_allclose(load_scan_positions(path), [[2. * scale, 1. * scale]])


def test_positions_match_raster(tmp_path: Path):
    # a raster's own positions, written as a positions file, give the raster scan back (rotation and affine included)
    from phaser.utils.scan import make_raster_scan
    raster = make_raster_scan((3, 4), (2.0, 1.5)).reshape(-1, 2)
    path = tmp_path / 'p.npz'
    numpy.savez(path, x=raster[:, 1], y=raster[:, 0], units='A')
    affine = numpy.array([[1.0, 0.1], [0.0, 0.9]])
    expected = make_raster_scan((3, 4), (2.0, 1.5), 30., affine).reshape(-1, 2)
    numpy.testing.assert_allclose(_scan(path, rotation=30., affine=affine.tolist()).data, expected, atol=1e-9)


@pytest.mark.parametrize(('arrays', 'match'), [
    ({'x': [0.]}, "missing array"),
    ({'x': [0., 1.], 'y': [0.]}, "same, nonzero length"),
    ({'x': [[0.]], 'y': [[0.]]}, "1D arrays"),
    ({'x': [], 'y': []}, "nonzero length"),
    ({'x': [0.], 'y': [0.], 'units': 'furlong'}, "unknown units"),
])
def test_positions_errors(tmp_path: Path, arrays: dict, match: str):
    path = tmp_path / 'p.npz'
    numpy.savez(path, **arrays)
    with pytest.raises(ValueError, match=match):
        load_scan_positions(path)
