import logging
import typing as t

import numpy
from numpy.typing import NDArray
from numpy.testing import assert_allclose
import pane
import pytest
from phaser.utils.num import get_backend_module
import tifffile

from phaser.utils.num import Sampling
from phaser.hooks import RawDataHook, RawData
from .utils import EXPECTED_PATH, INPUT_FILES_PATH


@pytest.fixture(scope="module")
def expected_raw_data() -> RawData:
    patterns = numpy.fft.ifftshift(
        t.cast(NDArray[numpy.floating], tifffile.imread(EXPECTED_PATH / "mit.tiff"))
    )

    return {
        'patterns': patterns,
        'mask': numpy.ones_like(patterns, dtype=numpy.float32),
        'sampling': Sampling((128, 128), extent=(62.75, 62.75)),
        'wavelength': 0.0251,
    }


def assert_raw_data_matches(actual: RawData, expected: RawData, scan_shape: t.Tuple[int, ...], rtol: float = 1e-7, atol: float = 0.0):
    expected_shape = (*scan_shape, *expected['patterns'].shape)

    assert actual['patterns'].shape == expected_shape
    assert actual['patterns'].dtype == expected['patterns'].dtype
    assert_allclose(actual['patterns'], numpy.broadcast_to(expected['patterns'], expected_shape), rtol=rtol, atol=atol, strict=True, err_msg="Patterns do not match")
    assert_allclose(actual['mask'], expected['mask'], rtol=1e-5, strict=True, err_msg="Mask does not match")
    assert_allclose(actual['sampling'].extent, expected['sampling'].extent, err_msg="Sampling does not match")
    assert actual.get('wavelength', 'fail') == expected['wavelength']  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert actual.get('scan_hook', 'fail') is None
    assert actual.get('tilt_hook', 'fail') is None
    assert actual.get('probe_hook', 'fail') is None


def test_load_manual_empad_v1(caplog, expected_raw_data: RawData):
    caplog.set_level(logging.INFO)
    get_backend_module("numpy")

    hook = pane.from_data(
        {
            "type": "manual",
            "wavelength": 0.0251,
            "path": INPUT_FILES_PATH / "empad_v1.raw",
            "det_shape": (128, 128),
            "det_flips": (True, False, False),
            "gap": 1024,  # 2 * 128 * 4 bytes
            "offset": 0,
            "diff_step": 0.4,
        },  # type: ignore
        RawDataHook,
    )
    raw_data = hook(None)

    assert caplog.record_tuples == [
        ("phaser.hooks.io.manual", logging.INFO, "Loading as raw binary..."),
        ("phaser.hooks.io.manual", logging.WARNING, "dtype not specified when loading raw file, defaulting to 'float32'..."),
        ("phaser.hooks.io.manual", logging.INFO, "Applying detector flips: [1, 0, 0] [y, x, transpose]"),
        ("phaser.hooks.io.manual", logging.WARNING, "ADU not supplied for experimental dataset. This is not recommended."),
    ]

    assert_raw_data_matches(raw_data, expected_raw_data, (4,))


def test_load_manual_h5py(caplog, expected_raw_data: RawData):
    caplog.set_level(logging.INFO)
    get_backend_module("numpy")

    hook = pane.from_data(
        {
            "type": "manual",
            "wavelength": 0.0251,
            "path": INPUT_FILES_PATH / "dp.h5",
            "diff_step": 0.4,
            "fftshifted": True,
        },  # type: ignore
        RawDataHook,
    )
    raw_data = hook(None)

    assert caplog.record_tuples == [
        ("phaser.hooks.io.manual", logging.INFO, "Loading as HDF5..."),
        ("phaser.hooks.io.manual", logging.INFO, "Found patterns at key 'dp' (inferred) in HDF5 file."),
        ("phaser.hooks.io.manual", logging.INFO, "Applying detector flips: [0, 0, 0] [y, x, transpose]"),
        ("phaser.hooks.io.manual", logging.WARNING, "ADU not supplied for experimental dataset. This is not recommended."),
    ]

    assert_raw_data_matches(raw_data, expected_raw_data, (2, 2))


def test_load_manual_h5py_customkey(caplog, expected_raw_data: RawData):
    caplog.set_level(logging.INFO)
    get_backend_module("numpy")

    hook = pane.from_data(
        {
            "type": "manual",
            "wavelength": 0.0251,
            "path": INPUT_FILES_PATH / "dp_customkey.h5",
            "key": "a/b/c/d",
            "diff_step": 0.4,
            "fftshifted": True,
        },  # type: ignore
        RawDataHook,
    )
    raw_data = hook(None)

    assert caplog.record_tuples == [
        ("phaser.hooks.io.manual", logging.INFO, "Loading as HDF5..."),
        ("phaser.hooks.io.manual", logging.INFO, "Loaded patterns from key 'a/b/c/d' in HDF5 file."),
        ("phaser.hooks.io.manual", logging.INFO, "Applying detector flips: [0, 0, 0] [y, x, transpose]"),
        ("phaser.hooks.io.manual", logging.WARNING, "ADU not supplied for experimental dataset. This is not recommended."),
    ]

    assert_raw_data_matches(raw_data, expected_raw_data, (2, 2))


def test_load_manual_tiff(caplog, expected_raw_data: RawData):
    caplog.set_level(logging.INFO)
    get_backend_module("numpy")

    hook = pane.from_data(
        {
            "type": "manual",
            "wavelength": 0.0251,
            "path": INPUT_FILES_PATH / "tiff_uint32.tiff",
            "diff_step": 0.4,
            "adu": 300.,
        },  # type: ignore
        RawDataHook,
    )
    raw_data = hook(None)

    assert caplog.record_tuples == [
        ("phaser.hooks.io.manual", logging.INFO, "Loading as TIFF..."),
        ("phaser.hooks.io.manual", logging.INFO, "Casting patterns to floating-point dtype float32"),
        ("phaser.hooks.io.manual", logging.INFO, "Applying detector flips: [0, 0, 0] [y, x, transpose]"),
        ("phaser.hooks.io.manual", logging.INFO, "Scaling patterns by ADU (300.0)"),
    ]

    assert_raw_data_matches(raw_data, expected_raw_data, (2, 2), atol=0.005)