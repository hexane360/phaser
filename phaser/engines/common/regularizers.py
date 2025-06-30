from functools import partial
import logging
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import (
    get_array_module, get_scipy_module, Float,
    jit, fft2, ifft2, abs2, xp_is_jax, to_real_dtype
)
from phaser.state import ReconsState
from phaser.hooks.regularization import (
    ClampObjectAmplitudeProps, LimitProbeSupportProps,
    RegularizeLayersProps, ObjLowPassProps, GaussianProps,
    CostRegularizerProps, TVRegularizerProps, RegularizeTiltProps
)


class ClampObjectAmplitude:
    def __init__(self, args: None, props: ClampObjectAmplitudeProps):
        self.amplitude = props.amplitude

    def init_state(self, sim: ReconsState) -> None:
        return None

    def apply_group(self, group: NDArray[numpy.integer], sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        return self.apply_iter(sim, state)

    def apply_iter(self, sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        amp = to_real_dtype(sim.object.data.dtype)(self.amplitude)
        sim.object.data = clamp_amplitude(sim.object.data, amp)
        return (sim, None)


@partial(jit, donate_argnames=('obj',), cupy_fuse=True)
def clamp_amplitude(obj: NDArray[numpy.complexfloating], amplitude: t.Union[float, numpy.floating]) -> NDArray[numpy.complexfloating]:
    xp = get_array_module(obj)

    obj_amp = xp.abs(obj)
    scale = xp.minimum(obj_amp, amplitude) / obj_amp
    return obj * scale


class LimitProbeSupport:
    def __init__(self, args: None, props: LimitProbeSupportProps):
        self.max_angle = props.max_angle

    def init_state(self, sim: ReconsState) -> NDArray[numpy.bool_]:
        xp = get_array_module(sim.probe.data)
        (ky, kx) = sim.probe.sampling.recip_grid(xp=xp)
        mask = kx**2 + ky**2 <= (self.max_angle*1e-3 / sim.wavelength)**2
        return mask

    def apply_group(self, group: NDArray[numpy.integer], sim: ReconsState, state: NDArray[numpy.bool_]) -> t.Tuple[ReconsState, NDArray[numpy.bool_]]:
        return self.apply_iter(sim, state)

    def apply_iter(self, sim: ReconsState, state: NDArray[numpy.bool_]) -> t.Tuple[ReconsState, NDArray[numpy.bool_]]:
        mask = state
        #xp = get_array_module(sim.state.probe.data)
        #print(f"intensity before: {xp.sum(abs2(sim.state.probe.data))}")
        sim.probe.data = ifft2(fft2(sim.probe.data) * mask)
        #print(f"intensity after: {xp.sum(abs2(sim.state.probe.data))}")
        return (sim, mask)


class RemovePhaseRamp:
    def __init__(self, args: None, props: t.Any):
        ...

    def init_state(self, sim: ReconsState) -> NDArray[numpy.bool_]:
        xp = get_array_module(sim.object.data)
        return sim.object.sampling.get_region_mask(xp=xp)

    def apply_group(self, group: NDArray[numpy.integer], sim: ReconsState, state: NDArray[numpy.bool_]) -> t.Tuple[ReconsState, NDArray[numpy.bool_]]:
        return self.apply_iter(sim, state)

    def apply_iter(self, sim: ReconsState, state: NDArray[numpy.bool_]) -> t.Tuple[ReconsState, NDArray[numpy.bool_]]:
        from phaser.utils.image import remove_linear_ramp
        xp = get_array_module(sim.object.data)
        phase = remove_linear_ramp(xp.angle(sim.object.data), state)
        sim.object.data = t.cast(NDArray[numpy.complexfloating], xp.abs(sim.object.data) * xp.exp(1.j * phase))
        return (sim, state)


class RegularizeLayers:
    def __init__(self, args: None, props: RegularizeLayersProps):
        self.weight = props.weight
        self.sigma = props.sigma

    def init_state(self, sim: ReconsState) -> None:
        return None

    def apply_iter(self, sim: ReconsState, state: None) -> t.Tuple[ReconsState, None]:
        xp = get_array_module(sim.object.data)
        scipy = get_scipy_module(sim.object.data)
        dtype = to_real_dtype(sim.object.data)

        if len(sim.object.thicknesses) < 2:
            return (sim, None)

        # approximate layers as equally spaced
        layer_spacing = numpy.mean(sim.object.thicknesses)
        # calculate size of filter (go to ~sigma in each direction)
        r = int(numpy.ceil(2. * self.sigma / layer_spacing))
        n = 2*r + 1

        # make Gaussian filter
        zs = ((xp.arange(0, n) - (n-1)//2) * layer_spacing).astype(dtype)
        kernel = xp.exp(-(zs / self.sigma)**2 / 2.)
        kernel /= xp.sum(kernel)

        # we convolve the log of object, because the transmission
        # function is multiplicative, not additive

        if xp_is_jax(xp):
            new_obj = xp.exp(scipy.signal.convolve(
                xp.pad(xp.log(sim.object.data), ((r, r), (0, 0), (0, 0)), mode='edge'),
                kernel[:, None, None],
                mode="valid"
            ))
        else:
            new_obj = xp.exp(scipy.ndimage.convolve1d(xp.log(
                sim.object.data
            ), kernel, axis=0, mode='nearest'))

        assert new_obj.shape == sim.object.data.shape
        assert new_obj.dtype == sim.object.data.dtype
        sim.object.data = (
            self.weight * new_obj + (1 - self.weight) * sim.object.data
        )
        return (sim, None)


class ObjLowPass:
    def __init__(self, args: None, props: ObjLowPassProps):
        self.logger = logging.getLogger(__name__)
        self.max_freq = props.max_freq

    def init_state(self, sim: ReconsState) -> NDArray[numpy.bool_]:
        samp = sim.object.sampling
        xp = get_array_module(sim.object.data)

        ky = xp.fft.fftfreq(samp.shape[0], 1.0)
        kx = xp.fft.fftfreq(samp.shape[1], 1.0)
        (ky, kx) = xp.meshgrid(ky, kx, indexing='ij')
        k2 = ky**2 + kx**2

        return k2 <= self.max_freq**2

    def apply_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: NDArray[numpy.bool_]
    ) -> t.Tuple[ReconsState, NDArray[numpy.bool_]]:
        return self.apply_iter(sim, state)

    def apply_iter(
        self, sim: ReconsState, state: NDArray[numpy.bool_]
    ) -> t.Tuple[ReconsState, NDArray[numpy.bool_]]:
        # TODO: should this be done in-place?
        sim.object.data = ifft2(state * fft2(sim.object.data))
        return (sim, state)


class ObjGaussian:
    def __init__(self, args: None, props: GaussianProps):
        self.sigma = props.sigma
        self.weight = props.weight

    def init_state(self, sim: ReconsState) -> NDArray[numpy.floating]:
        samp = sim.object.sampling
        xp = get_array_module(sim.object.data)
        dtype = to_real_dtype(sim.object.data.dtype)

        ky = xp.fft.fftfreq(samp.shape[0], samp.sampling[0]).astype(dtype)
        kx = xp.fft.fftfreq(samp.shape[1], samp.sampling[1]).astype(dtype)
        (ky, kx) = xp.meshgrid(ky, kx, indexing='ij')
        k2 = ky**2 + kx**2

        filt = xp.exp(-(2 * numpy.pi**2 * self.sigma**2) * k2).astype(dtype)
        return (1. - self.weight * (1. - filt)).astype(dtype)

    def apply_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: NDArray[numpy.floating]
    ) -> t.Tuple[ReconsState, NDArray[numpy.floating]]:
        return self.apply_iter(sim, state)

    def apply_iter(
        self, sim: ReconsState, state: NDArray[numpy.floating]
    ) -> t.Tuple[ReconsState, NDArray[numpy.floating]]:
        # TODO: should this be done in-place?
        sim.object.data = ifft2(state * fft2(sim.object.data))
        return (sim, state)


class ObjL1:
    def __init__(self, args: None, props: CostRegularizerProps):
        self.cost: float = props.cost

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(sim.object.data)

        cost = xp.sum(xp.abs(sim.object.data - 1.0))
        cost_scale = (group.shape[-1] / numpy.prod(sim.scan.shape[:-1])).astype(cost.dtype)
        return (cost * cost_scale * self.cost, state)


class ObjL2:
    def __init__(self, args: None, props: CostRegularizerProps):
        self.cost: Float = props.cost

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(sim.object.data)

        cost = xp.sum(abs2(sim.object.data - 1.0))
        cost_scale = (group.shape[-1] / numpy.prod(sim.scan.shape[:-1])).astype(cost.dtype)
        return (cost * cost_scale * self.cost, state)


class ObjPhaseL1:
    def __init__(self, args: None, props: CostRegularizerProps):
        self.cost: float = props.cost

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(sim.object.data)

        cost = xp.sum(xp.abs(xp.angle(sim.object.data)))
        cost_scale = (group.shape[-1] / numpy.prod(sim.scan.shape[:-1])).astype(cost.dtype)
        return (cost * cost_scale * self.cost, state)


class ObjRecipL1:
    def __init__(self, args: None, props: CostRegularizerProps):
        self.cost: float = props.cost

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(sim.object.data)

        # l1 norm of diff. pattern amplitude
        # TODO log object before this?
        cost = xp.sum(
            xp.abs(fft2(xp.prod(sim.object.data, axis=0)))
        )
        # scale cost by fraction of the total reconstruction in the group
        cost_scale = (group.shape[-1] / numpy.prod(sim.scan.shape[:-1])).astype(cost.dtype)

        return (cost * cost_scale * self.cost, state)


class ObjTotalVariation:
    def __init__(self, args: None, props: TVRegularizerProps):
        self.cost: float = props.cost
        self.eps: float = props.eps

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(sim.object.data)

        # isotropic total variation
        g_y, g_x = img_grad(sim.object.data)
        cost = xp.sum(xp.sqrt(abs2(g_y) + abs2(g_x) + self.eps))
        # anisotropic total variation
        #cost = (
        #    xp.sum(xp.abs(xp.diff(sim.object.data, axis=-1))) +
        #    xp.sum(xp.abs(xp.diff(sim.object.data, axis=-2)))
        #)
        # scale cost by fraction of the total reconstruction in the group
        # TODO also scale by # of pixels or similar?
        cost_scale = (group.shape[-1] / numpy.prod(sim.scan.shape[:-1])).astype(cost.dtype)

        return (cost * cost_scale * self.cost, state)


class ObjTikhonov:
    def __init__(self, args: None, props: CostRegularizerProps):
        self.cost: float = props.cost

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(sim.object.data)

        cost = (
            xp.sum(abs2(xp.diff(sim.object.data, axis=-1))) +
            xp.sum(abs2(xp.diff(sim.object.data, axis=-2)))
        )
        # scale cost by fraction of the total reconstruction in the group
        cost_scale = (group.shape[-1] / numpy.prod(sim.scan.shape[:-1])).astype(cost.dtype)

        return (cost * cost_scale * self.cost, state)


class LayersTotalVariation:
    def __init__(self, args: None, props: CostRegularizerProps):
        self.cost: float = props.cost

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(sim.object.data)

        if sim.object.data.shape[0] < 2:
            return (0.0, state)

        cost = xp.sum(xp.abs(xp.diff(sim.object.data, axis=0)))
        # scale cost by fraction of the total reconstruction in the group
        cost_scale = (group.shape[-1] / numpy.prod(sim.scan.shape[:-1])).astype(cost.dtype)

        return (cost * cost_scale * self.cost, state)


class LayersTikhonov:
    def __init__(self, args: None, props: CostRegularizerProps):
        self.cost: float = props.cost

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(sim.object.data)

        if sim.object.data.shape[0] < 2:
            return (0.0, state)

        cost = xp.sum(abs2(xp.diff(sim.object.data, axis=0)))
        # scale cost by fraction of the total reconstruction in the group
        cost_scale = (group.shape[-1] / numpy.prod(sim.scan.shape[:-1])).astype(cost.dtype)

        return (cost * cost_scale * self.cost, state)


class ProbePhaseTikhonov:
    def __init__(self, args: None, props: CostRegularizerProps):
        self.cost: float = props.cost

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(sim.probe.data)

        phase = xp.angle(fft2(sim.probe.data))

        cost = (
            xp.sum(abs2(xp.diff(phase, axis=-1))) +
            xp.sum(abs2(xp.diff(phase, axis=-2)))
        )
        cost_scale = 1.0

        return (cost * cost_scale * self.cost, state)


class ProbeRecipTikhonov:
    def __init__(self, args: None, props: CostRegularizerProps):
        self.cost: float = props.cost

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(sim.probe.data)
        probe_recip = xp.fft.fftshift(fft2(sim.probe.data), axes=(-1, -2))

        cost = (
            xp.sum(abs2(xp.diff(probe_recip, axis=-1))) +
            xp.sum(abs2(xp.diff(probe_recip, axis=-2)))
        )
        cost_scale = 1.0

        return (cost * cost_scale * self.cost, state)


class ProbeRecipTotalVariation:
    def __init__(self, args: None, props: TVRegularizerProps):
        self.cost: float = props.cost
        self.eps: float = props.eps

    def init_state(self, sim: ReconsState) -> None:
        return None

    def calc_loss_group(
        self, group: NDArray[numpy.integer], sim: ReconsState, state: None
    ) -> t.Tuple[Float, None]:
        xp = get_array_module(sim.probe.data)
        probe_recip = xp.fft.fftshift(fft2(sim.probe.data), axes=(-1, -2))

        g_y, g_x = img_grad(probe_recip)
        cost = xp.sum(xp.sqrt(abs2(g_y) + abs2(g_x) + self.eps))
        cost_scale = 1.0

        return (cost * cost_scale * self.cost, state)


class RegularizeTilt:
    # https://pypi.org/project/jaxkd/0.1.1/ ??
    # Currently take about 5s per iteration for 256*256 scan positions, faster if keep 2D scan
    def __init__(self, args: None, props:RegularizeTiltProps):
        self.weight = props.weight
        self.sigma = props.sigma

    def init_state(self, sim):
        return None

    def apply_iter(self, sim, state):
        xp = get_array_module(sim.tilt)
        scan = numpy.array(sim.scan)
        tilt = numpy.array(sim.tilt)

        shape = tilt.shape[:-1]
        N = int(numpy.prod(numpy.array(shape)))

        xy = scan.reshape(-1, 2)
        tilt_flat = tilt.reshape(-1, 2)

        from scipy.spatial import cKDTree
        tree = cKDTree(xy)
        radius = 3 * self.sigma
        tilt_blurred = numpy.zeros_like(tilt_flat)
        for i in range(N):
            indices = tree.query_ball_point(xy[i], r=radius)
            if not indices:
                tilt_blurred[i] = tilt_flat[i]
                continue
            neighbor_vecs = tilt_flat[indices]
            dists = numpy.linalg.norm(xy[indices] - xy[i], axis=1)
            weights = numpy.exp(-dists**2 / (2 * self.sigma**2))
            weights /= numpy.sum(weights)
            tilt_blurred[i] = numpy.sum(weights[:, None] * neighbor_vecs, axis=0)

        tilt_blurred = xp.asarray(tilt_blurred).reshape(*shape, 2)
        sim.tilt = self.weight * tilt_blurred + (1 - self.weight) * sim.tilt
        return sim, state


def img_grad(img: numpy.ndarray) -> t.Tuple[numpy.ndarray, numpy.ndarray]:
    xp = get_array_module(img)
    return (
        xp.diff(img, axis=-2, append=img[..., -1:, :]),
        xp.diff(img, axis=-1, append=img[..., :, -1:]),
    )