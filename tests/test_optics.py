
import numpy

from .utils import with_backends, check_array_equals_file

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
