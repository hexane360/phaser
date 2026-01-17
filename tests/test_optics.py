
import numpy
import pytest

from phaser.utils.physics import Electron

from .utils import with_backends, check_array_equals_file

from phaser.types import (
    ComplexCartesian, ComplexPolar, KrivanekComplex, KrivanekCartesian, KrivanekPolar,
    Aberration, process_aberrations
)
from phaser.utils.num import get_backend_module, BackendName, Sampling, to_numpy, fft2, ifft2
from phaser.utils.optics import make_focused_probe, fresnel_propagator


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
@check_array_equals_file('probe_30mrad_aberrated.tiff', decimal=5,
                         out_name='probe_30mrad_aberrated_{backend}.tiff')
def test_aberrated_probe(backend: BackendName) -> numpy.ndarray:
    xp = get_backend_module(backend)
    sampling = Sampling((1024, 1024), extent=(50.0, 50.0))
    wavelength = Electron(300e3).wavelength

    probe = make_focused_probe(
        *sampling.recip_grid(dtype=numpy.float32, xp=xp), wavelength, aperture=30., defocus=0.,
        aberrations=[
            {'a1': 10.0+10.0j},
            {'b2': (1e3+2e3j) / 3},
            KrivanekComplex(3, 2, val=50e3j),
        ]
    )
    return to_numpy(probe)


@with_backends('numpy', 'jax', 'cupy', 'torch')
@check_array_equals_file('probe_15mrad_spherical.tiff', decimal=5,
                         out_name='probe_15mrad_spherical_{backend}.tiff')
def test_spherical_probe(backend: BackendName) -> numpy.ndarray:
    xp = get_backend_module(backend)
    sampling = Sampling((1024, 1024), extent=(50.0, 50.0))
    wavelength = Electron(200e3).wavelength

    probe = make_focused_probe(
        *sampling.recip_grid(dtype=numpy.float32, xp=xp), wavelength, aperture=15.,
        defocus=-578.266,
        aberrations=[
            {'c3': 1.0e+7},
            {'a1': 20.0+20.0j},
            KrivanekCartesian(3, 2, re=1.5e6, im=2.0e6),
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
        {'b2': {'re': 5.0, 'im': -2.0}},     # haider cartesian
        {'a1': {'mag': 5.0, 'angle': 90.0}}, # haider polar
        {'n': 4, 'm': 1, 'val': 1+1.j},      # krivanek complex
        {'n': 1, 'm': 0, 're': 5.0},         # krivanek cartesian
        {'n': 5, 'm': 0, 'mag': 5.0},        # krivanek polar
    ], list[Aberration])

    assert result == [
        {'c3': complex(5.0)},
        {'b2': ComplexCartesian(re=5.0, im=-2.0)},
        {'a1': ComplexPolar(mag=5.0, angle=90.0)},
        KrivanekComplex(4, 1, val=1+1.j),
        KrivanekCartesian(1, 0, re=5.0, im=0.0),
        KrivanekPolar(5, 0, mag=5.0, angle=0.0),
    ]

    assert list(process_aberrations(result)) == [
        KrivanekComplex.make_unchecked(3, 0, val=complex(5.0)),
        KrivanekComplex.make_unchecked(2, 1, val=15.0-6.0j),
        KrivanekComplex.make_unchecked(1, 2, val=pytest.approx(5.0j)),
        KrivanekComplex.make_unchecked(4, 1, val=1+1.j),
        KrivanekComplex.make_unchecked(1, 0, val=complex(5.0)),
        KrivanekComplex.make_unchecked(5, 0, val=complex(5.0)),
    ]
    # TODO test failures