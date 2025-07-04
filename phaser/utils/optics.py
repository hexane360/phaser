"""
Probe/optics utilities
"""

import logging
import typing as t

import numpy
from numpy.typing import NDArray, ArrayLike

from .num import get_array_module, ifft2, abs2, NumT, ufunc_outer, is_jax, cast_array_module
from .num import Float, Sampling, to_complex_dtype, to_real_dtype, split_array, to_numpy

ABERRATION_SPECS: t.List[t.Tuple[str, t.Tuple[int, int], bool]] = [
    ('C1', (1, 0), False),
    ('C3', (3, 0), False),
    ('C5', (5, 0), False),
    ('B2', (2, 1), True),
    ('B4', (4, 1), True),
    ('S3', (3, 2), True),
    ('D4', (4, 3), True),
    ('A1', (1, 2), True),
    ('A2', (2, 3), True),
    ('A3', (3, 4), True),
    ('A4', (4, 5), True),
    ('A5', (5, 6), True),
]

ABERRATION_KEYS = [name for name, *_ in ABERRATION_SPECS]

@t.overload
def make_focused_probe(ky: NDArray[numpy.float64], kx: NDArray[numpy.float64], wavelength: Float,
                       aperture: Float, *, defocus: Float = 0.) -> NDArray[numpy.complex128]:
    ...

@t.overload
def make_focused_probe(ky: NDArray[numpy.float32], kx: NDArray[numpy.float32], wavelength: Float,
                       aperture: Float, *, defocus: Float = 0.) -> NDArray[numpy.complex64]:
    ...

@t.overload
def make_focused_probe(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], wavelength: Float,
                       aperture: Float, *, defocus: Float = 0.) -> NDArray[numpy.complexfloating]:
    ...

def make_focused_probe(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], wavelength: Float,
                       aperture: Float, *, defocus: Float = 0.) -> NDArray[numpy.complexfloating]:
    """
    Create a focused probe from a circular aperture of semi-angle `aperture` (in mrad).

    Optionally, give it some defocus (positive corresponds to overfocus).
    """
    xp = get_array_module(ky, kx)

    thetay, thetax = ky * wavelength, kx * wavelength
    theta2 = thetay**2 + thetax**2

    phase = (defocus/(2. * wavelength)) * theta2
    probe = xp.exp(-2.j*numpy.pi * phase)

    mask = theta2 <= (aperture * 1e-3)**2
    probe *= mask

    # normalize intensity of probe
    probe /= xp.sqrt(xp.sum(abs2(probe)))

    return ifft2(probe)


@t.overload
def make_parameterized_probe(ky: NDArray[numpy.float64], kx: NDArray[numpy.float64], wavelength: Float,
                       aperture: Float, *, aberrations: NDArray[numpy.float64]) -> NDArray[numpy.complex128]:
    ...

@t.overload
def make_parameterized_probe(ky: NDArray[numpy.float32], kx: NDArray[numpy.float32], wavelength: Float,
                       aperture: Float, *, aberrations: NDArray[numpy.float32]) -> NDArray[numpy.complex64]:
    ...

@t.overload
def make_parameterized_probe(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], wavelength: Float,
                       aperture: Float, *, aberrations: NDArray[numpy.floating]) -> NDArray[numpy.complexfloating]:
    ...

def make_parameterized_probe(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating],wavelength: Float, 
                         aperture: Float, *, aberrations: NDArray[numpy.floating]) -> NDArray[numpy.complexfloating]:
    """
    Create an aberrated probe from a circular aperture and a list of aberration coefficients.

    `aberrations` is a 1D array matching the order in ABERRATION_SPECS and give in magnitude (, angle if complex)
    Each term is either:
        - 1 value if imag=False (real only)
        - 2 values (cnma, cnmb) if imag=True
    """
    xp = get_array_module(ky, kx)
    dtype = to_real_dtype(ky.dtype)
    complex_dtype = to_complex_dtype(dtype)

    thetay, thetax = ky * wavelength, kx * wavelength
    theta2 = thetay**2 + thetax**2
    phi = xp.arctan2(ky, kx)

    chi = xp.zeros_like(theta2, dtype=complex_dtype)
    i = 0
    for _, (n, m), imag in ABERRATION_SPECS:
        if imag:
            mag = aberrations[i]
            ang = aberrations[i+1]
            cnma = mag * xp.cos(ang)
            cnmb = mag * xp.sin(ang)
            i += 2
        else:
            cnma = aberrations[i]
            cnmb = 0.0
            i += 1
        z = xp.exp(phi * m * 1.j)
        power = (n + 1) / 2
        chi += theta2**power / (n + 1) * (xp.real(z) * cnma + xp.imag(z) * cnmb)

    mask = theta2 <= (aperture * 1e-3)**2
    probe_freq = xp.exp(-2.j * xp.pi * chi /wavelength) * mask

    probe_freq /= xp.sqrt(xp.sum(abs2(probe_freq)))
    return ifft2(probe_freq)


def make_hermetian_modes(
    base_probe: NDArray[NumT], n_modes: int, base_mode_power: float = 0.7, rel_powers: ArrayLike = 1.0
) -> NDArray[NumT]:
    """
    Create Hermitian-Gauss probe modes based on `base_probe`.

    Creates `n_modes` modes. The base mode will have power `base_mode_power`, while the
    other modes will split the remaining power according to `rel_powers` (which may be a list).
    """
    xp = get_array_module(base_probe)

    if not (0.0 <= base_mode_power < 1.0):
        raise ValueError(f"Invalid base_mode_power '{base_mode_power}'. Expected a value between 0 and 1")

    # TODO: ensure intensity is preserved

    rel_powers = numpy.array(rel_powers, dtype=to_real_dtype(base_probe.dtype)).ravel()
    rel_powers = numpy.pad(rel_powers, (0, n_modes - rel_powers.size - 1), mode='edge')[:n_modes - 1]
    powers = numpy.concatenate([[base_mode_power], (1.0 - base_mode_power) * rel_powers])
    logging.debug(f"Hermetian mode powers: {powers.tolist()}")

    # we make modes like the following:
    #       0 1 2 3 4
    #     +----------
    #   0 | 0 2 5   |
    #   1 | 1 4 8   |
    #   2 | 3 7     |
    #   3 | 6       |
    #   4 +----------

    # the diagonal each mode is on (indexing starting from 0):
    i = numpy.arange(n_modes)
    diag_num = numpy.ceil((numpy.sqrt(8*i + 9) - 3) / 2).astype(numpy.int_)

    # and the corresponding grid indices
    xx = (i - diag_num*(diag_num+1)/2).astype(numpy.int_)
    yy = diag_num - xx

    modes = hermetian_modes(base_probe, yy, xx)
    modes *= xp.array(numpy.sqrt(powers)[:, None, None], dtype=modes.dtype)
    return t.cast(NDArray[NumT], modes)


def hermetian_modes(base_probe: NDArray[NumT], y_orders: ArrayLike, x_orders: ArrayLike) -> NDArray[NumT]:
    """
    Split `base_probe` into Hermite-Gaussian modes. y_orders and x_orders (broadcast together)
    are the orders of modes to return.

    Return a ndarray of shape `(*y_orders.shape, *base_probe.shape)`.
    Each mode is orthogonal and normalized.
    """
    xp = get_array_module(base_probe)
    real_dtype = to_real_dtype(base_probe.dtype)
    y_orders, x_orders = numpy.broadcast_arrays(y_orders, x_orders)

    (yy, xx) = xp.indices(base_probe.shape, dtype=real_dtype)
    base_probe_mag = abs2(base_probe)

    (com_y, com_x) = (xp.sum(a * base_probe_mag) / xp.sum(base_probe_mag) for a in (yy, xx))
    yy -= com_y
    xx -= com_x
    #(var_y, var_x) = (xp.sum(a**2. * base_probe_mag) / xp.sum(base_probe_mag) for a in (yy, xx))

    modes = xp.empty((y_orders.size, *base_probe.shape), dtype=base_probe.dtype)

    for (i, (y_power, x_power)) in enumerate(zip(y_orders.flat, x_orders.flat)):
        mode = yy**y_power * xx**x_power * base_probe
        #if y_power > 0 or x_power > 0:
        #    mode = mode * xp.exp(-xx**2./(2 * var_x) - yy*2./(2 * var_y))
        #    mode /= xp.sqrt(xp.sum(abs2(mode)))

        # orthogonalize to other modes
        for prev_i in range(i):  # TODO do this in a smarter way
            mode -= modes[prev_i] * xp.sum(modes[prev_i] * xp.conj(mode))
        # renormalize
        mode /= xp.sqrt(xp.sum(abs2(mode)))

        if is_jax(modes) and not t.TYPE_CHECKING:
            modes = modes.at[i].set(mode)
        else:
            modes[i] = mode

    return modes.reshape((*y_orders.shape, *base_probe.shape))


@t.overload
def fourier_shift_filter(ky: NDArray[numpy.float64], kx: NDArray[numpy.float64], shifts: ArrayLike) -> NDArray[numpy.complex128]:
    ...

@t.overload
def fourier_shift_filter(ky: NDArray[numpy.float32], kx: NDArray[numpy.float32], shifts: ArrayLike) -> NDArray[numpy.complex64]:
    ...

@t.overload
def fourier_shift_filter(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], shifts: ArrayLike) -> NDArray[numpy.complexfloating]:
    ...

def fourier_shift_filter(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], shifts: ArrayLike) -> NDArray[numpy.complexfloating]:
    """
    Create a phase ramp / Fourier shift filter, using the reciprocal space frequencies `ky` & `kx`.

    A frequency-shift filter can be created using `fourier_shift_filter(yy, xx, -shifts)`.

    # Parameters:

     - `ky`, `kx`: Frequency grid filter is created with
     - `shifts`: Vector(s) to shift by. Should be an array of shape `(..., 2)`, with the last dimension
       representing `(y, x)` coordinates.

    Returns a ndarray of shape `(*shifts.shape[:-1], *ky.shape)`
    """
    xp = get_array_module(ky, kx)
    dtype = to_complex_dtype(ky.dtype)

    (y, x) = split_array(xp.array(shifts, dtype=ky.dtype), axis=-1)

    return xp.exp(xp.array(-2.j*numpy.pi, dtype=dtype) * (ufunc_outer(xp.multiply, x, kx) + ufunc_outer(xp.multiply, y, ky)))


@t.overload
def fresnel_propagator(ky: NDArray[numpy.float64], kx: NDArray[numpy.float64], wavelength: Float,
                       delta_z: Float, tilt: t.Tuple[float, float] = (0., 0.)) -> NDArray[numpy.complex128]:
    ...

@t.overload
def fresnel_propagator(ky: NDArray[numpy.float32], kx: NDArray[numpy.float32], wavelength: Float,
                       delta_z: Float, tilt: t.Tuple[float, float] = (0., 0.)) -> NDArray[numpy.complex64]:
    ...


@t.overload
def fresnel_propagator(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], wavelength: Float,
                       delta_z: Float, tilt: t.Tuple[float, float] = (0., 0.)) -> NDArray[numpy.complexfloating]:
    ...


def fresnel_propagator(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], wavelength: Float,
                       delta_z: Float, tilt: t.Tuple[float, float] = (0., 0.)) -> NDArray[numpy.complexfloating]:
    """
    Return a Fresnel diffraction filter in frequency space, for use in free-space propagation. Roughly taken from Kirkland [1].

    # Parameters

     - `ky`, `kx`: Frequency grid filter is created with
     - `wavelength`: Wavelength
     - `delta_z`: Distance to propagate by
     - `tilt`: `(x, y)` mistilt to apply (in mrad). 

    [1]. Kirkland, E. J. Advanced Computing in Electron Microscopy. (Springer US, Boston, MA, 2010). doi:10.1007/978-1-4419-6533-2.

    """
    xp = get_array_module(ky, kx)

    (tiltx, tilty) = numpy.tan(tilt[0]*1e-3), numpy.tan(tilt[1]*1e-3)

    k2 = ky**2 + kx**2
    return xp.exp(-1.j * numpy.pi * delta_z * (wavelength * k2 - 2.*(kx*tiltx + ky*tilty))) \
        .astype(to_complex_dtype(k2.dtype))


def estimate_probe_radius(wavelength: Float, aperture: Float, defocus: Float, *,
                          threshold: t.Union[t.Literal['geom'], Float] = 0.9, xp: t.Any = None) -> float:
    """
    Estimate the radius of a probe created by a circular aperture of
    maximum semi-angle `aperture` (in mrad) defocused by `defocus` (in length units).

    `threshold` is used to determine the intensity cutoff for size calcuation.
    'geom' may be specified to use the geometric approximation (valid for large defocuses).
    """
    xp2 = numpy if xp is None else cast_array_module(xp)

    aperture *= 1e-3  # mrad -> rad

    if threshold == 'geom':
        return float(defocus * aperture)

    rel_defocus = numpy.abs(defocus) / wavelength

    # 8*nyquist
    sampling = 1/(16 * aperture)
    min_box_size = 5. * rel_defocus * aperture
    # number of points required is size / sampling
    # we round up to nearest power of 2
    with numpy.errstate(divide='ignore'):
        n = max(1024, int(numpy.exp2(numpy.ceil(numpy.log2(min_box_size / sampling)))))
    if n > 2**14:
        raise ValueError("Can't calculate probe radii (requires too much phase space)") from None

    # TODO: this should be possible in 1D, using a Henkel transform or similar
    # also, we can cache a lot of this

    samp = Sampling((n, n), sampling=(sampling, sampling))
    ky, kx = samp.recip_grid(xp=xp2)
    yy, xx = samp.real_grid(xp=xp2)

    probe_int = abs2(make_focused_probe(ky, kx, 1.0, aperture*1e3, defocus=rel_defocus))

    rs = xp2.sqrt(yy**2 + xx**2)
    r_samp = samp.sampling[0] / 3.
    r_i = xp2.floor(rs / r_samp).astype(int)

    probe_ints = xp2.bincount(r_i.ravel(), weights=probe_int.ravel())
    cum_ints = to_numpy(xp2.cumsum(probe_ints))

    try:
        probe_n = int(numpy.argwhere(cum_ints > float(threshold))[0, 0])
        # redimensionalize
        return float(probe_n * r_samp * wavelength)
    except IndexError:
        raise ValueError("Couldn't calculate probe radii (didn't reach threshold, is probe too big for calculation box?)") from None


def calc_metrics(*,
    wavelength: t.Optional[float] = None, voltage: t.Optional[float] = None,
    conv_angle: float, defocus: float, scan_step: float, diff_step: float,
    probe_radius: t.Optional[float] = None, xp: t.Any = None,
    threshold: t.Union[t.Literal['geom'], float] = 0.9,
) -> t.Dict[str, float]:
    """
    Calculate sampling metrics for the given parameters. Units are as follows:

    voltage: volts
    wavelength, scan_step, defocus: length units (must be consistent)
    conv_angle, diff_step: mrad
    """

    if wavelength is None:
        if voltage is None:
            raise ValueError("One of 'wavelength' or 'voltage' must be specified")
        from phaser.utils.physics import Electron
        wavelength = Electron(voltage).wavelength

    if probe_radius is None:
        probe_radius = estimate_probe_radius(
            wavelength=wavelength, aperture=conv_angle,
            defocus=defocus, threshold=threshold, xp=xp
        )

    # diff_step in 1/A
    d_step = diff_step*1e-3 / wavelength

    return {
        'probe_radius': probe_radius,
        'fund_samp': 1/(scan_step * d_step),
        'probe_samp': 1/(2.*probe_radius * d_step),
        'linear_oversamp': 2.*probe_radius / scan_step,
        'areal_oversamp': numpy.pi*probe_radius**2 / scan_step**2,
        'ronchi_mag': conv_angle / (diff_step * probe_radius),

        'wavelength': wavelength,
        'conv_angle': conv_angle,
        'defocus': defocus,
        'scan_step': scan_step,
        'diff_step': diff_step,
    }


class _FittedQDA:
    def __init__(self, means: ArrayLike, rotations: t.Sequence[ArrayLike],
                 scalings: t.Sequence[ArrayLike], priors: t.Optional[ArrayLike] = None):
        self.means_ = numpy.asarray(means)
        self.n_classes = self.means_.shape[0]
        self.rotations_ = list(map(numpy.asarray, rotations))
        self.scalings_ = list(map(numpy.asarray, scalings))
        self.priors_ = numpy.asarray(priors) if priors is not None else numpy.full(self.n_classes, 1/self.n_classes)

    def predict_log_proba(self, X: ArrayLike) -> NDArray[numpy.floating]:
        X = numpy.asarray(X)
        assert X.ndim == 2

        norm2 = []
        for i in range(self.n_classes):
            X2 = (X - self.means_[i]) @ (self.rotations_[i] * (self.scalings_[i] ** -0.5))
            norm2.append(numpy.sum(X2**2, axis=-1))

        norm2 = numpy.array(norm2).T
        u = numpy.asarray([numpy.sum(numpy.log(s)) for s in self.scalings_])
        scores = -0.5 * (norm2 + u) + numpy.log(self.priors_)

        log_likelihood = scores - numpy.max(scores, axis=-1, keepdims=True)
        log_likelihood -= numpy.log(numpy.sum(numpy.exp(log_likelihood), axis=-1, keepdims=True))

        return log_likelihood

    def predict_proba(self, X: ArrayLike) -> NDArray[numpy.floating]:
        return numpy.exp(self.predict_log_proba(X))

    def predict_prob_success(self, X: ArrayLike) -> NDArray[numpy.floating]:
        return numpy.exp(self.predict_log_proba(X)[..., 1])


def predict_recons_success(ronchi_mag: ArrayLike, areal_oversamp: ArrayLike) -> NDArray[numpy.floating]:
    """
    Empirically predict the probability of reconstruction success, given the Ronchigram magnification
    and areal oversampling.

    Broadcasts `ronchi_mag` and `areal_oversamp` together, and returns an array of the same shape,
    with values indicating the estimated probability of success.

    Fitted on simulated Si data, using an intensity threshold of 90% to calculate the probe radius.
    """

    clf = _FittedQDA(
        means=[[0.49027803, 1.82918678], [0.80980859, 2.00158048]],
        rotations=[
            [[-0.2316652028331075, 0.972795576571098], [0.972795576571098, 0.2316652028331075]],
            [[-0.26985970170864165, 0.9628996528162853], [0.9628996528162853, 0.26985970170864165]]
        ],
        scalings=[[0.56440416, 0.07769331], [0.4110058 , 0.06621369]],
    )

    ronchi_mag, areal_oversamp = numpy.broadcast_arrays(ronchi_mag, areal_oversamp)

    return clf.predict_prob_success(
        numpy.log10(numpy.stack((ronchi_mag, areal_oversamp), axis=-1).reshape(-1, 2))
    ).reshape(ronchi_mag.shape)


__all__ = [
    'make_focused_probe', 'make_parameterized_probe',
    'make_hermetian_modes', 'hermetian_modes',
    'fourier_shift_filter', 'fresnel_propagator',
    'estimate_probe_radius', 'calc_metrics', 'predict_recons_success',
]