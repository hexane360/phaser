import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray

from phaser.utils.num import fft2, ifft2, abs2, Sampling, get_array_module, at, to_numpy, NumT
from phaser.utils.object import ObjectSampling
from phaser.utils.image import remove_linear_ramp, translation_matrix, rotation_matrix
from phaser.state import ObjectState, ParameterizedObjectState
from phaser.types import cast_length


def split_image(img: NDArray[NumT]) -> t.Tuple[NDArray[NumT], NDArray[NumT], NDArray[NumT], NDArray[NumT]]:
    """
    Split image into 4 subimages (upper left, upper right, lower left, lower right).
    """
    shape = img.shape[-2:]
    # crop image divisible by two
    shape = tuple(s - s%2 for s in shape)
    img = img[(Ellipsis, *(slice(0, s) for s in shape))]

    return cast_length((
        img[(Ellipsis, slice(row_start, shape[0], 2), slice(col_start, shape[1], 2))]
        for row_start in (0, 1)
        for col_start in (0, 1)
    ), 4)


def fourier_correlate(img1: NDArray[NumT], img2: NDArray[NumT]) -> NDArray[numpy.complex128]:
    assert img1.shape == img2.shape
    from skimage.filters import window

    xp = get_array_module(img1, img2)

    win = xp.asarray(window('hann', img1.shape))
    img1_fft = numpy.fft.fftshift(fft2(img1.astype(numpy.float64) * win), axes=(-2, -1))
    img2_fft = xp.conj(numpy.fft.fftshift(fft2(img2.astype(numpy.float64) * win), axes=(-2, -1)))
    fft1_mag = abs2(img1_fft)
    fft2_mag = abs2(img2_fft)

    with xp.errstate(invalid='ignore'):
        corr = img1_fft * img2_fft / xp.sqrt(fft1_mag * fft2_mag)
    return xp.nan_to_num(corr, nan=1.)


def _calc_r_binning(
    shape: t.Tuple[int, ...], r_spacing: float = 1.0,
    xp: t.Any = None
) -> NDArray[numpy.int64]:
    if xp is None or t.TYPE_CHECKING:
        xp = numpy

    c_y, c_x = tuple(int(s//2) for s in shape[-2:])
    y, x = xp.indices(shape[-2:])
    r = xp.sqrt((y - c_y)**2 + (x - c_x)**2)
    return xp.floor(r / r_spacing).astype(numpy.int64).ravel()


def contrast_transfer(
    img1: NDArray[NumT], img2: NDArray[NumT], *,
    inscribed: bool = True, r_spacing: float = 1.0
) -> t.Tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64]]:
    assert img1.shape == img2.shape
    from skimage.filters import window

    xp = get_array_module(img1, img2)

    win = xp.asarray(window(('tukey', 0.3), img1.shape), dtype=numpy.float64)
    img1_fft = xp.fft.fftshift(fft2(img1.astype(numpy.float64) * win), axes=(-2, -1))
    img2_fft = xp.conj(xp.fft.fftshift(fft2(img2.astype(numpy.float64) * win), axes=(-2, -1)))
    fft1_mag = abs2(img1_fft)
    fft2_mag = abs2(img2_fft)

    r_i = _calc_r_binning(img1.shape, r_spacing, xp=xp)
    mag1_count = xp.bincount(r_i, fft1_mag.ravel())
    mag2_count = xp.bincount(r_i, fft2_mag.ravel())

    with numpy.errstate(invalid='ignore', divide='ignore'):
        vals = to_numpy(xp.sqrt(mag1_count / mag2_count))
    vals = numpy.nan_to_num(vals, nan=1.)

    rs = numpy.linspace(0., len(vals) * r_spacing, len(vals), endpoint=False, dtype=numpy.float64)
    freq = rs / numpy.sqrt(numpy.prod(img1.shape[-2:]))

    if inscribed:
        # crop to inner radii
        n_r = int(numpy.floor(min(img1.shape[-2:]) / (2.*r_spacing)))
        return (vals[:n_r], rs[:n_r], freq[:n_r])
    return (vals, rs, freq)


def fourier_ring_correlate(
    img1: NDArray[NumT], img2: NDArray[NumT], *,
    inscribed: bool = True, r_spacing: float = 1.0
) -> t.Tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64]]:
    assert img1.shape == img2.shape
    from skimage.filters import window

    xp = get_array_module(img1, img2)

    win = xp.asarray(window(('tukey', 0.3), img1.shape), dtype=numpy.float64)
    img1_fft = xp.fft.fftshift(fft2(img1.astype(numpy.float64) * win), axes=(-2, -1))
    img2_fft = xp.conj(xp.fft.fftshift(fft2(img2.astype(numpy.float64) * win), axes=(-2, -1)))
    fft1_mag = abs2(img1_fft)
    fft2_mag = abs2(img2_fft)

    r_i = _calc_r_binning(img1.shape, r_spacing, xp=xp)
    # sum real and imaginary separately
    real_count = xp.bincount(r_i, (img1_fft * xp.conj(img2_fft)).real.ravel())
    mag1_count = xp.bincount(r_i, fft1_mag.ravel())
    mag2_count = xp.bincount(r_i, fft2_mag.ravel())

    with numpy.errstate(invalid='ignore', divide='ignore'):
        vals = to_numpy(real_count / xp.sqrt(mag1_count * mag2_count))
    vals = numpy.nan_to_num(vals, posinf=0., nan=1.)

    #vals = numpy.bincount(r_i.ravel(), corr.ravel()) / numpy.bincount(r_i.ravel())
    rs = numpy.linspace(0., len(vals) * r_spacing, len(vals), endpoint=False, dtype=numpy.float64)
    freq = rs / numpy.sqrt(numpy.prod(img1.shape[-2:]))

    if inscribed:
        # crop to inner radii
        n_r = int(numpy.floor(min(img1.shape[-2:]) / (2.*r_spacing)))
        return (vals[:n_r], rs[:n_r], freq[:n_r])
    return (vals, rs, freq)


def frc_intersect_threshold(frc: NDArray[numpy.floating], freq: NDArray[numpy.floating], threshold: float) -> t.Tuple[float, float]:
    diff = frc - threshold
    lastdiff = numpy.roll(diff, shift=1)
    # find negative zero crossing
    try:
        i = numpy.nonzero((diff[1:] < 0) & (lastdiff[1:] > 0))[0][0]
    except IndexError:
        raise ValueError("No crossing found when evaluating FRC resolution")

    m = (diff[i+1] - diff[i]) / (freq[i+1] - freq[i])
    m_f = (frc[i+1] - frc[i]) / (freq[i+1] - freq[i])
    x_d = diff[i]/m
    return (float(freq[i] - x_d), float(frc[i] - x_d * m_f))


def _cross_correlate(x: NDArray[numpy.floating], y: NDArray[numpy.floating], max_shift: float) -> t.Tuple[float, float]:
    xp = get_array_module(x, y)

    samp = Sampling(tuple(x.shape), sampling=(1.0, 1.0))  # type: ignore
    yy, xx = samp.real_grid(xp=xp)

    cross_corr = ifft2(fft2(x) * xp.conj(fft2(y))).real
    # limit maximum shift
    cross_corr = at(cross_corr, yy**2 + xx**2 > max_shift**2).set(numpy.nan)  # type: ignore

    max_i = xp.nanargmax(cross_corr)
    y = yy.ravel()[max_i]
    x = xx.ravel()[max_i]
    return (float(y), float(x))


def align_object_to_ground_truth(
    object: t.Union[ObjectState, ParameterizedObjectState],
    ground_truth: NDArray[numpy.floating],
    ground_truth_sampling: ArrayLike,
    rotation_angle: float = 0.0,
    refinement_niter: int = 0,
    order: int = 1,
) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
    """
    ground_truth: Ground truth phase (in radians, or radians/angstrom for multislice data)

    Returns a tuple (object, ground_truth)
    """
    xp = get_array_module(object.data, ground_truth)
    ground_truth = xp.asarray(ground_truth)
    object_phase = xp.angle(xp.asarray(object.data))

    # normalize multislice objects to radians/angstrom
    if len(object.thicknesses):
        object_phase /= object.thicknesses[:, None, None]

    # remove linear ramp
    object_roi = object.sampling.get_region_mask(xp=xp)
    object_phase = remove_linear_ramp(object_phase, object_roi)
    object_phase -= xp.nanquantile(object_phase[..., object_roi], 0.01, axis=-1)[:, None, None]
    # and get average
    object_mean = xp.mean(object_phase, axis=0)

    # initially center ground truth on object roi
    obj_center = object.sampling.get_region_center()
    ground_truth_corner = obj_center - numpy.array(ground_truth.shape) * ground_truth_sampling / 2.
    ground_truth_samp = ObjectSampling(
        t.cast(t.Tuple[int, int], ground_truth.shape),
        sampling=ground_truth_sampling, corner=ground_truth_corner,
        region_min=object.sampling.region_min, region_max=object.sampling.region_max,
    )
    crop = ground_truth_samp.get_region_crop(pad=-0.05 * object.sampling.get_region_extent())

    # perform initial resampling
    upsamp_obj = object.sampling.resample(object_mean, ground_truth_samp, cval=0., rotation=rotation_angle, order=order)

    # cross correlate
    max_shift = float(numpy.min(
        (ground_truth_samp.extent - object.sampling.get_region_extent()) / (3.0 * ground_truth_samp.sampling)
    ))
    if max_shift < 0:
        raise ValueError("Error: Ground truth extent smaller than object extent")
    #print(f"max_shift: {max_shift:.3f} px ({max_shift * ground_truth_samp.sampling[0]:.3f} angstrom)")

    shift = numpy.array(_cross_correlate(upsamp_obj[tuple(crop)], ground_truth[tuple(crop)], max_shift))

    # shift ground truth to match experiment
    ground_truth_samp = ObjectSampling(
        tuple(ground_truth_samp.shape),
        sampling=ground_truth_sampling, corner=ground_truth_corner + shift * ground_truth_sampling,
        region_min=object.sampling.region_min, region_max=object.sampling.region_max, 
    )
    crop = ground_truth_samp.get_region_crop()

    if refinement_niter == 0:
        # perform final upsampling
        upsamp_obj = object.sampling.resample(object_phase, ground_truth_samp, cval=0., rotation=rotation_angle, order=order)
        return upsamp_obj[(slice(None), *crop)], ground_truth[tuple(crop)]

    import scipy.optimize

    def _make_affine(mat: NDArray[numpy.floating]) -> NDArray[numpy.floating]:
        (a, b, c, d, e, f) = mat
        affine = translation_matrix(obj_center) @ numpy.array([
            [a, b, e],
            [c, d, f],
            [0., 0., 1],
        ]) @ rotation_matrix(rotation_angle) @ translation_matrix(-obj_center)
        return affine

    def align_and_correlate(mat: NDArray[numpy.floating]) -> NDArray[numpy.floating]:
        affine = _make_affine(mat)
        upsamp_obj = object.sampling.resample(object_mean, ground_truth_samp, cval=0., affine=affine, order=order)
        return to_numpy(upsamp_obj[tuple(crop)] - ground_truth[tuple(crop)]).ravel()

    max_shift_refine = max_shift * ground_truth_samp.sampling[0] / 5.

    # (a, b, c, d, e, f)
    init_mat = numpy.array([1., 0., 0., 1., 0., 0.])
    min_bound = numpy.array([0.9, -0.1, -0.1, 0.9, -max_shift_refine, -max_shift_refine])
    max_bound = numpy.array([1.1, 0.1, 0.1, 1.1, max_shift_refine, max_shift_refine])

    result = scipy.optimize.least_squares(align_and_correlate, init_mat, bounds=(min_bound, max_bound),
                                          method='dogbox', max_nfev=refinement_niter, xtol=1e-4)
    print(f"""\
Refinement result: {result.message}
    nfev: {result.nfev}
    matrix: {result.x}""")

    affine = _make_affine(result.x)
    (a, b, c, d, e, f) = result.x
    affine = translation_matrix(obj_center) @ numpy.array([
        [a, b, e],
        [c, d, f],
        [0., 0., 1],
    ]) @ rotation_matrix(rotation_angle) @ translation_matrix(-obj_center)

    # perform final upsampling
    upsamp_obj = object.sampling.resample(object_phase, ground_truth_samp, cval=0., affine=affine, order=order)
    return upsamp_obj[(slice(None), *crop)], ground_truth[tuple(crop)]