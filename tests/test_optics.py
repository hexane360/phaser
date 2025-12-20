
import numpy
import pytest

from phaser.utils.physics import Electron

from .utils import with_backends, check_array_equals_file

from phaser.utils.num import abs2, get_backend_module, BackendName, Sampling, to_numpy, fft2, ifft2
from phaser.utils.optics import (
    make_focused_probe, fresnel_propagator,
    Aberration, AberrationList, _normalize_aberrations,
    Krivanek, Cartesian, Polar, KrivanekComplex, KrivanekCartesian, KrivanekPolar,
)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@check_array_equals_file('probe_10mrad_focused_mag.tiff', decimal=5)
def test_focused_probe(backend: BackendName) -> numpy.ndarray:
    xp = get_backend_module(backend)

    sampling = Sampling((1024, 1024), extent=(25., 25.))

    probe = make_focused_probe(*sampling.recip_grid(dtype=numpy.float32, xp=xp), wavelength=0.0251, aperture=10.)
    return to_numpy(xp.abs(probe))


@with_backends('numpy', 'jax', 'cupy', 'torch')
@check_array_equals_file('probe_10mrad_20over.tiff', decimal=5)
def test_defocused_probe(backend: BackendName) -> numpy.ndarray:
    xp = get_backend_module(backend)

    sampling = Sampling((1024, 1024), extent=(25., 25.))

    probe = make_focused_probe(*sampling.recip_grid(dtype=numpy.float32, xp=xp), wavelength=0.0251, aperture=10., defocus=200.)
    return to_numpy(probe)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@check_array_equals_file('probe_25mrad_aberrated.tiff', decimal=5)
def test_aberrated_probe(backend: BackendName) -> numpy.ndarray:
    xp = get_backend_module(backend)
    sampling = Sampling((1024, 1024), extent=(39.05, 39.05))
    wavelength = Electron(200e3).wavelength

    probe = make_focused_probe(
        *sampling.recip_grid(dtype=numpy.float32, xp=xp), wavelength, aperture=25., defocus=10.,
        aberrations=[
            {'a1': -12.0+5.0j},
            KrivanekCartesian(2, 3, a=300., b=-400.),
            KrivanekComplex(3, 0, val=-500_000.0),
        ]
    )
    return to_numpy(probe)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@check_array_equals_file('fresnel_200kV_1nm_phase.tiff', decimal=5)
def test_fresnel_propagator(backend: BackendName) -> numpy.ndarray:
    xp = get_backend_module(backend)

    sampling = Sampling((1024, 1024), extent=(100., 100.))

    return to_numpy(xp.angle(
        fresnel_propagator(*sampling.recip_grid(dtype=numpy.float64, xp=xp), 0.0251, 10., tilt=(8., 5.))
    ))


@with_backends('numpy', 'jax', 'cupy', 'torch')
@check_array_equals_file('probe_10mrad_focused_mag.tiff', decimal=5)
def test_propagator_sign(backend: BackendName) -> numpy.ndarray:
    xp = get_backend_module(backend)

    sampling = Sampling((1024, 1024), extent=(25., 25.))
    (ky, kx) = sampling.recip_grid(dtype=numpy.float32, xp=xp)

    # make sure defocus sign agrees with propagator sign
    # 200 angstrom underfocused + 200 angstrom propagation = focused
    probe = make_focused_probe(ky, kx, wavelength=0.0251, aperture=10., defocus=-200.)
    prop = fresnel_propagator(ky, kx, wavelength=0.0251, delta_z=200.)

    probe = ifft2(fft2(probe) * prop)
    return to_numpy(xp.abs(probe))


def test_parse_aberrations():
    import pane
    result = pane.convert([
        {'c3': 5.0},                         # haider complex
        {'b2': {'a': 5.0, 'b': -2.0}},       # haider cartesian
        {'a1': {'mag': 5.0, 'angle': 90.0}}, # haider polar
        {'n': 4, 'm': 1, 'val': 1+1.j},      # krivanek complex
        {'n': 1, 'm': 0, 'a': 5.0},          # krivanek cartesian
        {'n': 5, 'm': 0, 'mag': 5.0},        # krivanek polar
    ], AberrationList)

    assert result == [
        {'c3': complex(5.0)},
        {'b2': Cartesian(a=5.0, b=-2.0)},
        {'a1': Polar(mag=5.0, angle=90.0)},
        KrivanekComplex(4, 1, val=1+1.j),
        KrivanekCartesian(1, 0, a=5.0, b=0.0),
        KrivanekPolar(5, 0, mag=5.0, angle=0.0),
    ]

    assert list(_normalize_aberrations(result)) == [
        KrivanekComplex.make_unchecked(3, 0, val=complex(5.0)),
        KrivanekComplex.make_unchecked(2, 1, val=15.0-6.0j),
        KrivanekComplex.make_unchecked(1, 2, val=pytest.approx(5.0j)),
        KrivanekComplex.make_unchecked(4, 1, val=1+1.j),
        KrivanekComplex.make_unchecked(1, 0, val=complex(5.0)),
        KrivanekComplex.make_unchecked(5, 0, val=complex(5.0)),
    ]
    # TODO test failures