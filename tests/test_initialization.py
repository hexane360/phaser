# type: ignore

import re
import logging
import typing as t

import numpy
import pane
import pytest

from phaser.utils.num import Sampling
from phaser.hooks import RawData
from phaser.plan import ReconsPlan
from phaser.execute import load_raw_data, initialize_reconstruction
from phaser.state import PartialReconsState, ProbeState


def load_empty(args, props) -> RawData:
    return {
        'patterns': numpy.zeros((32, 32, 64, 64), dtype=numpy.float32),
        'mask': numpy.ones((64, 64), dtype=numpy.float32),
        'sampling': Sampling((64, 64), sampling=(1.0, 1.0)),
        'wavelength': 1.0,
        'scan_hook': {
            'type': 'raster',
            'shape': (32, 32),
            'step_size': (0.6, 0.6),
        },
        'probe_hook': None,
        'seed': None,
    }


def test_load_raw_data_missing():
    plan = ReconsPlan.from_data({
        'name': 'test',
        'raw_data': 'tests.test_initialization:load_empty',
        'engines': [],
    })
    xp = numpy

    with pytest.raises(ValueError, match=re.escape('`probe` must be specified by raw data, previous state, or manually in `init.probe`')):
        load_raw_data(plan, xp)


def test_load_raw_data_override():
    plan = {
        'name': 'test',
        'raw_data': 'tests.test_initialization:load_empty',
        'engines': [],
        'init': {
            'probe': {
                'type': 'focused',
                'conv_angle': 20.0,
                'defocus': 200.0,
            },
            'scan': {
                'type': 'raster',
                'step_size': (1.0, 1.0),
            }
        }
    }
    xp = numpy

    raw_data = load_raw_data(ReconsPlan.from_data(plan), xp)

    assert pane.into_data(raw_data['probe_hook']) == {  # type: ignore
        'type': 'focused',
        'conv_angle': 20.0,
        'defocus': 200.0,
    }

    assert pane.into_data(raw_data['scan_hook']) == {  # type: ignore
        'type': 'raster',
        'rotation': None,
        'shape': (32, 32),
        'affine': None,

        # overridden by init.scan
        'step_size': (1.0, 1.0),
    }

    plan['init']['scan'] = 'custom.package:raster2'

    raw_data = load_raw_data(ReconsPlan.from_data(plan), xp)
    # instead of merging different hooks, the new one takes precedence
    assert pane.into_data(raw_data['scan_hook']) == {'type': 'custom.package:raster2'}


def test_load_raw_data_prev_state(caplog):
    plan = {
        'name': 'test',
        'raw_data': 'tests.test_initialization:load_empty',
        'engines': [],
    }

    probe_state = ProbeState(Sampling((64, 64), sampling=(1.0, 1.0)), numpy.zeros((64, 64), dtype=numpy.complex64))
    scan_state = numpy.zeros((32, 32, 2))

    xp = numpy
    with caplog.at_level(logging.WARNING):
        patterns, state = initialize_reconstruction(ReconsPlan.from_data(plan), xp, init_state=PartialReconsState(
            wavelength=2.0, probe=probe_state,
        ))

    assert "Wavelength of reconstruction (1.00e+00) doesn't match wavelength of previous state (2.00e+00)" in caplog.text
    assert "Mean pattern intensity is very low (0.0 particles)." in caplog.text
    assert state.probe is probe_state

    plan['init'] = {
        'scan': {}
    }

    patterns, state = initialize_reconstruction(ReconsPlan.from_data(plan), xp, init_state=PartialReconsState(
        wavelength=2.0, probe=probe_state, scan=scan_state
    ))

    # probe from state overrides probe from raw data
    assert state.probe is probe_state
    # but scan should be modeled
    assert state.scan is not scan_state

    plan['init'] = {
        'scan': {},
        'probe': {
            'type': 'focused',
            'conv_angle': 25.0,
            'defocus': 200.0,
        }
    }

    patterns, state = initialize_reconstruction(ReconsPlan.from_data(plan), xp, init_state=PartialReconsState(
        wavelength=2.0, probe=probe_state, scan=scan_state
    ))

    # both should be modeled
    assert state.probe is not probe_state
    assert state.scan is not scan_state