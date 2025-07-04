import typing as t

import numpy
from numpy.typing import NDArray
import matplotlib
from matplotlib import pyplot
from matplotlib.colors import Colormap, Normalize, LinearSegmentedColormap
from matplotlib.transforms import Affine2DBase, Affine2D
import matplotlib.patheffects as path_effects

from .num import (
    get_array_module, abs2, fft2, ifft2,
    to_numpy, Sampling, to_real_dtype
)
from .optics import fourier_shift_filter
from .image import remove_linear_ramp, colorize_complex
from .object import ObjectSampling
from .misc import create_sparse_groupings


if t.TYPE_CHECKING:
    from ..state import ObjectState, ProbeState, ParameterizedProbeState ##TODO add ploting function of ParameterizedProbeState
    from matplotlib.axes import Axes
    from matplotlib.text import Text
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage


ColormapLike: t.TypeAlias = t.Union[str, Colormap]
NormLike: t.TypeAlias = Normalize


def plot_pacbed(raw: NDArray[numpy.floating],
    *, ax: t.Optional['Axes'] = None, log: bool = True,
    diff_step: t.Union[t.Tuple[float, float], float, None] = None
) -> 'Axes':
    from matplotlib.colors import LogNorm

    if ax is None:
        fig, ax = pyplot.subplots()
        ax = t.cast('Axes', ax)
    # else:
    #     fig = ax.get_figure()

    pacbed = numpy.nansum(raw, axis=tuple(range(raw.ndim - 2)))

    if diff_step is not None:
        if isinstance(diff_step, (int, float)):
            diff_step = (diff_step, diff_step)
        ax.set_ylabel("k_y [mrad]")
        ax.set_xlabel("k_x [mrad]")
        extent = (
            (-pacbed.shape[1]/2. - 0.5) * diff_step[1],
            (pacbed.shape[1]/2. - 0.5) * diff_step[1],
            (pacbed.shape[1]/2. - 0.5) * diff_step[0],
            (-pacbed.shape[1]/2. - 0.5) * diff_step[0],
        )
    else:
        ax.set_ylabel("k_y [px]")
        ax.set_xlabel("k_x [px]")
        extent = None

    ax.imshow(
        pacbed,
        norm=t.cast('Normalize', LogNorm() if log else None),
        extent=t.cast(t.Tuple[float, float, float, float], extent),
    )

    return ax


def plot_raw(raw: NDArray[numpy.floating],
    *, fig: t.Optional['Figure'] = None,
    mask: t.Optional[NDArray[numpy.bool_]] = None,
    interactive: bool = True, log: bool = False,
    scan_step: t.Union[t.Tuple[float, float], float, None] = None,
    diff_step: t.Union[t.Tuple[float, float], float, None] = None,
) -> 'Figure':
    from matplotlib.colors import Normalize, LogNorm

    if raw.ndim != 4:
        raise ValueError("Expected a 4D STEM dataset")

    idx = (0, 0)

    if fig is None:
        fig = pyplot.figure(constrained_layout=True)

    if mask is None:
        mask = numpy.ones(raw.shape[-2:], dtype=numpy.bool_)

    real_img = numpy.tensordot(raw, mask, axes=((-1, -2), (-1, -2)))
    recip_img = raw[tuple(idx)]
    vmin = float(numpy.nanquantile(raw, 0.001))  # used for lognorm
    vmax = float(numpy.nanquantile(raw, 0.999))

    (recip_ax, real_ax) = fig.subplots(ncols=2)

    real_ax.set_aspect(1.)
    recip_ax.set_aspect(1.)

    if scan_step is not None:
        if isinstance(scan_step, (int, float)):
            scan_step = (float(scan_step), float(scan_step))
        real_ax.set_ylabel("y [$\\mathrm{\\AA}$]")
        real_ax.set_xlabel("x [$\\mathrm{\\AA}$]")
        real_extent = (
            -0.5 * scan_step[1],
            (real_img.shape[1] - 0.5) * scan_step[1],
            (real_img.shape[0] - 0.5) * scan_step[0],
            -0.5 * scan_step[0]
        )
    else:
        real_ax.set_ylabel("y [px]")
        real_ax.set_xlabel("x [px]")
        real_extent = None

    if diff_step is not None:
        if isinstance(diff_step, (int, float)):
            diff_step = (float(diff_step), float(diff_step))
        recip_ax.set_ylabel("k_y [mrad]")
        recip_ax.set_xlabel("k_x [mrad]")
        recip_extent = (
            (-recip_img.shape[1]/2. - 0.5) * diff_step[1],
            (recip_img.shape[1]/2. - 0.5) * diff_step[1],
            (recip_img.shape[1]/2. - 0.5) * diff_step[0],
            (-recip_img.shape[1]/2. - 0.5) * diff_step[0],
        )
    else:
        recip_ax.set_ylabel("k_y [px]")
        recip_ax.set_xlabel("k_x [px]")
        recip_extent = None

    real_ax.imshow(
        real_img,
        vmin=float(numpy.nanquantile(real_img, 0.01)),
        vmax=float(numpy.nanquantile(real_img, 0.99)),
        extent=t.cast(t.Sequence[float], real_extent),
    )

    recip_imshow = recip_ax.imshow(
        recip_img,
        norm=t.cast('Normalize', LogNorm(vmin, vmax) if log else Normalize(0.0, vmax)),
        extent=t.cast(t.Sequence[float], recip_extent),
    )

    if interactive:
        _raw_interact(
            fig, real_ax, recip_ax, raw, idx, recip_imshow,
            numpy.array(scan_step if scan_step is not None else [1., 1.]),
        )

    return fig


def _raw_interact(
    fig: 'Figure', real_ax: 'Axes', recip_ax: 'Axes', raw: NDArray[numpy.floating],
    idx: t.Tuple[int, int], recip_imshow: 'AxesImage',
    scan_step: NDArray[numpy.floating],
):
    import threading

    import matplotlib.path as path
    from matplotlib.patches import PathPatch
    from matplotlib.backend_bases import KeyEvent, MouseEvent, MouseButton, Event

    class Timer(threading.Thread):
        def __init__(self, interval: float, fn: t.Callable[[], bool]):
            super().__init__()
            self.stop_event: threading.Event = threading.Event()
            self.interval: float = interval
            self.fn: t.Callable[[], bool] = fn

        def run(self):
            while not self.stop_event.wait(self.interval):
                if not self.fn():
                    break

        def stop(self):
            self.stop_event.set()

        def __del__(self):
            self.stop()
            self.join()

    (y, x) = idx

    crosshair = path.Path(numpy.array([
        [-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5],
        [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]
    ]), list(map(int, [path.Path.MOVETO, 2, 2, 2, path.Path.MOVETO, 2, 2, 2])))
    marker = PathPatch(crosshair, fc='red', fill=True, linestyle='None', transform=Affine2D().translate(x * scan_step[1], y * scan_step[0]) + real_ax.transData)
    real_ax.add_patch(marker)

    def update():
        #print(f"\rpos: ({x}, {y})   ", end='')
        recip_imshow.set_data(raw[y, x])
        marker.set_transform(Affine2D().translate(x * scan_step[1], y * scan_step[0]) + real_ax.transData)
        fig.canvas.draw_idle()

    repeat_timer: t.Optional[Timer] = None
    pressed: t.Set[str] = set()

    def handle_presses():
        nonlocal x, y

        if 'left' in pressed:
            if x > 0:
                x -= 1
        elif 'right' in pressed:
            if x < raw.shape[1] - 1:
                x += 1
        elif 'up' in pressed:
            if y > 0:
                y -= 1
        elif 'down' in pressed:
            if y < raw.shape[0] - 1:
                y += 1

        update()

    def key_pressed(event: KeyEvent):
        nonlocal repeat_timer

        if event.key not in ('left', 'right', 'up', 'down'):
            return

        pressed.add(event.key)
        handle_presses()

        if repeat_timer is not None:
            repeat_timer.stop()
            repeat_timer.join()
        repeat_timer = Timer(0.2, key_repeat)
        repeat_timer.start()

    def key_repeat() -> bool:
        if len(pressed):
            handle_presses()
            return True  # continue
        return False

    def key_released(event: KeyEvent):
        if event.key in pressed:
            pressed.remove(event.key)

    def mouse_event(event: MouseEvent):
        nonlocal x, y
        if (event.button is MouseButton.LEFT
            or event.buttons is not None and MouseButton.LEFT in event.buttons  # type: ignore
        ) and event.x is not None and event.y is not None:

            (click_x, click_y) = real_ax.transData.inverted().transform(tuple(map(int, (event.x, event.y))))
            click_x /= scan_step[1]
            click_y /= scan_step[0]
            (click_x, click_y) = map(int, map(round, (click_x, click_y)))
            if not 0 <= click_x < raw.shape[1] or not 0 <= click_y < raw.shape[0]:
                return
            x, y = click_x, click_y
            update()

    fig.canvas.mpl_connect('key_press_event', t.cast(t.Callable[[Event], t.Any], key_pressed))
    fig.canvas.mpl_connect('key_release_event', t.cast(t.Callable[[Event], t.Any], key_released))
    fig.canvas.mpl_connect('button_press_event', t.cast(t.Callable[[Event], t.Any], mouse_event))
    fig.canvas.mpl_connect('motion_notify_event', t.cast(t.Callable[[Event], t.Any], mouse_event))


@t.overload
def plot_object_phase(
    data: 'ObjectState', sampling: None = None, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    unwrap: bool = True, remove_ramp: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    ...

@t.overload
def plot_object_phase(
    data: NDArray[numpy.complexfloating], sampling: ObjectSampling, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    unwrap: bool = True, remove_ramp: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    ...

def plot_object_phase(
    data: t.Union['ObjectState', NDArray[numpy.complexfloating]],
    sampling: t.Optional[ObjectSampling] = None, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    unwrap: bool = True, remove_ramp: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    if hasattr(data, 'sampling'):
        sampling = t.cast(ObjectSampling, getattr(data, 'sampling'))
        data = t.cast(NDArray[numpy.complexfloating], getattr(data, 'data'))
    elif sampling is None:
        raise ValueError("'sampling' must be specified, or an 'ObjectState' passed")
    else:
        data = t.cast(NDArray[numpy.complexfloating], data)

    xp = get_array_module(data)
    mask = sampling.get_region_mask()
    phase = xp.mean(xp.angle(data), axis=0)

    if unwrap:
        phase = xp.unwrap(xp.unwrap(phase, axis=1), axis=0)
    if remove_ramp:
        phase = remove_linear_ramp(phase, mask)

    return _plot_object_data(
        phase, mask, sampling,
        ax=ax, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        zoom_roi=zoom_roi, **imshow_kwargs
    )


@t.overload
def plot_object_mag(
    data: 'ObjectState', sampling: None = None, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    ...

@t.overload
def plot_object_mag(
    data: NDArray[numpy.complexfloating], sampling: ObjectSampling, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    ...

def plot_object_mag(
    data: t.Union['ObjectState', NDArray[numpy.complexfloating]],
    sampling: t.Optional[ObjectSampling] = None, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    if hasattr(data, 'sampling'):
        sampling = t.cast(ObjectSampling, getattr(data, 'sampling'))
        data = t.cast(NDArray[numpy.complexfloating], getattr(data, 'data'))
    elif sampling is None:
        raise ValueError("'sampling' must be specified, or an 'ObjectState' passed")
    else:
        data = t.cast(NDArray[numpy.complexfloating], data)

    xp = get_array_module(data)
    mag = abs2(xp.prod(data, axis=0))

    return _plot_object_data(
        mag, sampling.get_region_mask(), sampling,
        ax=ax, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        zoom_roi=zoom_roi, **imshow_kwargs
    )


def _plot_object_data(
    data: NDArray[numpy.floating], mask: NDArray[numpy.bool_], sampling: ObjectSampling, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    xp = get_array_module(data)

    if ax is None:
        fig, ax = pyplot.subplots()
        ax = t.cast('Axes', ax)
    # else:
    #     fig = ax.get_figure()

    ax.set_xlabel(r"X [$\mathrm{\AA}$]")
    ax.set_ylabel(r"Y [$\mathrm{\AA}$]")

    if norm is None:
        vmin = vmin if vmin is not None else float(xp.nanmin(data[mask]))
        vmax = vmax if vmax is not None else float(xp.nanmax(data[mask]))
        norm = Normalize(vmin, vmax)

    cmap = t.cast('Colormap', pyplot.get_cmap(cmap))  # type: ignore

    img = ax.imshow(to_numpy(data), cmap=cmap, norm=norm, extent=sampling.mpl_extent(), **imshow_kwargs)

    if zoom_roi:
        min = sampling.region_min if sampling.region_min is not None else sampling.min
        max = sampling.region_max if sampling.region_max is not None else sampling.max
        ax.set_xlim(min[1], max[1])
        ax.set_ylim(max[0], min[0])

    return img


@t.overload
def plot_probes(
    data: 'ProbeState', sampling: None = None, *,
    fig: t.Optional['Figure'] = None,
):
    ...

@t.overload
def plot_probes(
    data: NDArray[numpy.complexfloating], sampling: Sampling, *,
    fig: t.Optional['Figure'] = None,
):
    ...

def plot_probes(
    data: t.Union[NDArray[numpy.complexfloating], 'ProbeState'], sampling: t.Optional[Sampling] = None, *,
    fig: t.Optional['Figure'] = None,
):
    if hasattr(data, 'sampling'):
        sampling = t.cast(Sampling, getattr(data, 'sampling'))
        data = t.cast(NDArray[numpy.complexfloating], getattr(data, 'data'))
    elif sampling is None:
       raise ValueError("'sampling' must be specified, or a 'ProbeState' passed") 
    else:
        data = t.cast(NDArray[numpy.complexfloating], data)

    if fig is None:
        fig = pyplot.figure()

    n_probes = data.shape[0]

    axs = fig.subplots(ncols=n_probes, sharex=True, sharey=True, squeeze=False,
                       gridspec_kw={'wspace': 0.0})

    if n_probes == 0:
        return

    axs[0, 0].set_xlabel(r"X [$\mathrm{\AA}$]")
    axs[0, 0].set_ylabel(r"Y [$\mathrm{\AA}$]")

    imgs = colorize_complex(data)
    extent = sampling.mpl_real_extent()

    for (ax, img) in zip(axs.flat, imgs):
        ax.imshow(img, extent=extent)


def plot_probe_overlap(
    data: 'ProbeState', scan: NDArray[numpy.floating],
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    subpx: bool = False, grouping: int = 16,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    xp = get_array_module(data.data, scan)
    dtype = to_real_dtype(data.data.dtype)

    probes = data.data
    #if normalize_probe:
    #    probes = probes / xp.sqrt(xp.sum(abs2(probes)))

    if ax is None:
        fig, ax = pyplot.subplots()
        ax = t.cast('Axes', ax)

    pad = data.sampling.extent / 2. + data.sampling.sampling
    obj_samp = ObjectSampling.from_scan(scan, data.sampling.sampling, pad)

    obj = xp.zeros(obj_samp.shape, dtype=dtype)
    ky, kx = data.sampling.recip_grid(xp=xp, dtype=dtype)

    sum_probe = xp.sum(abs2(probes), axis=0)

    for group in create_sparse_groupings(scan, grouping):
        group_scan = scan[tuple(group)]
        if subpx:
            group_subpx_filters = fourier_shift_filter(
                ky, kx, obj_samp.get_subpx_shifts(group_scan, probes.shape[-2:])
            )[:, None, ...]
            shifted_probes = ifft2(fft2(probes) * group_subpx_filters)
            obj = obj_samp.cutout(obj, group_scan, ky.shape).add(
                xp.sum(abs2(shifted_probes), axis=1)
            ).obj
        else:
            obj = obj_samp.cutout(obj, group_scan, ky.shape).add(
                xp.broadcast_to(sum_probe, (group_scan.shape[0], *sum_probe.shape))
            ).obj

    img = ax.imshow(to_numpy(obj), cmap=cmap, norm=norm, vmin=vmin or 0., vmax=vmax, **imshow_kwargs)
    return img


def plot_metrics(metrics: t.Dict[str, float]) -> 'Figure':
    fig, (ax1, ax2, ax3) = pyplot.subplots(
        ncols=3,
        gridspec_kw={
            'width_ratios': [2., 1., 2.],
            'wspace': 0.05,
        },
        constrained_layout=True
    )
    fig.set_size_inches(12, 4)
    ax1.set_box_aspect(1.)
    ax2.set_aspect(1.)
    ax3.set_box_aspect(1.)

    _plot_linear_metrics(ax1, metrics)
    _plot_probe_overlap(ax2, metrics)
    _plot_predicted_success(ax3, metrics)

    return fig


def _plot_linear_metrics(ax: 'Axes', metrics: t.Dict[str, float]):
    # mrad
    diff_scale = metrics['wavelength']*1e3/(2. * metrics['probe_radius'])
    step_scale = 2.*metrics['probe_radius']

    x = metrics['scan_step']
    y = metrics['diff_step']

    ax.set_xlabel("Scan step [$\\mathrm{\\AA/px}$]")
    ax.set_xscale('log')
    ax.set_ylabel("Diff. pixel size [mrad/px]")
    ax.set_yscale('log')

    ylim = (
        min(y/1.5, 0.1*diff_scale),
        max(y*1.5, 10.*diff_scale),
    )
    xlim = (
        min(x/1.5, 0.1*step_scale),
        max(x*1.5, 10.*step_scale),
    )
    yy = numpy.geomspace(*ylim, 201, endpoint=True)
    xx = numpy.geomspace(*xlim, 201, endpoint=True)
    (yy, xx) = numpy.meshgrid(yy, xx, indexing='ij')

    fps = 1.0 / (yy / diff_scale * xx / step_scale)

    ax.axhline(diff_scale, linestyle='dashed', color='#4363d8') # blue
    ax.axvline(step_scale, linestyle='dashdot', color='#f58231') # orange
    ax.contour(xx, yy, fps, [1.0], linestyles=['solid'], colors=['#e6194b']) # red

    ax.scatter([x], [y], marker='x', color='black', s=80)

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)


def _plot_probe_overlap(ax: 'Axes', metrics: t.Dict[str, float]):
    from matplotlib.patches import Circle, Rectangle

    probe_r = metrics['probe_radius']
    scan_r = metrics['scan_step'] / 2.0
    box_r = metrics['scan_step'] * metrics['fund_samp'] / 2.

    ax_r = scan_r + max(probe_r, box_r) * 1.1

    ax.set_xlim(-ax_r, ax_r)
    ax.set_ylim(-ax_r, ax_r)

    ax.set_axis_off()

    theta = 20.0 * numpy.pi / 180.

    pos1 = numpy.array([numpy.cos(theta), numpy.sin(theta)]) * scan_r

    _ = [ax.add_patch(Circle(
        t.cast(t.Tuple[float, float], tuple(pos)), probe_r, facecolor='green', alpha=0.6, edgecolor='black',
        linewidth=2.0,
    )) for pos in (pos1, -pos1)]

    ax.plot([pos1[0], -pos1[0]], [pos1[1], -pos1[1]], '.-k', linewidth=2.0)

    _ = [ax.add_patch(Rectangle(
        (pos[0] - box_r, pos[1] - box_r), 2*box_r, 2*box_r,
        edgecolor='black', linewidth=2.0, fill=False,
    )) for pos in (pos1, -pos1)]


def _plot_predicted_success(ax: 'Axes', metrics: t.Dict[str, float]):
    from phaser.utils.optics import predict_recons_success

    bwr_r: Colormap = matplotlib.colormaps['bwr_r']  # type: ignore

    gamma = 2.0
    lin_norm = Normalize(0.5, 1.0)
    prob_cmap = LinearSegmentedColormap.from_list(
        'prob', numpy.stack([
            bwr_r(lin_norm.inverse(numpy.abs(lin_norm(x))**gamma * numpy.sign(lin_norm(x))))
            for x in numpy.linspace(0., 1., 512, endpoint=True)
        ], axis=0)
    )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel("Ronchi. mag [$\\mathrm{px/\\AA}$]")
    ax.set_ylabel("Areal oversampling")

    ronchi_mag = metrics['ronchi_mag']
    areal_oversamp = metrics['areal_oversamp']

    ylim = (
        min(8e-1, areal_oversamp / 3),
        max(1e4, areal_oversamp * 3),
    )
    xlim = (
        min(8e-1, ronchi_mag / 1.5),
        max(1e2, ronchi_mag * 4),
    )

    yy = numpy.geomspace(ylim[1], ylim[0], 100)
    xx = numpy.geomspace(xlim[0], xlim[1], 100)
    yy, xx = numpy.meshgrid(yy, xx, indexing='ij')

    pp = predict_recons_success(xx, yy)
    ax.pcolormesh(xx, yy, pp, alpha=0.8, cmap=prob_cmap, vmin=0.0, vmax=1.0)

    prob = predict_recons_success(ronchi_mag, areal_oversamp)
    ax.scatter([ronchi_mag], [areal_oversamp], marker='x', s=80, c='black')
    ax.annotate(f"{prob:.1%}", (ronchi_mag, areal_oversamp),
                (0.5, 0.7), textcoords='offset fontsize')


class _ScalebarTransform(Affine2DBase):
    def __init__(self, ax, origin):
        self._invalid: bool
        super().__init__("ScalebarTransform")
        self.origin = origin
        self._mtx: t.Optional[numpy.ndarray] = None
        self.transAxes = ax.transAxes
        self.transLimits = ax.transLimits
        self.set_children(self.transAxes, self.transLimits)

    def _calc_matrix(self) -> numpy.ndarray:
        sx, sy = numpy.diag(self.transLimits.get_matrix())[:2]
        m = numpy.diag([sx, 1., 1.]) @ self.transAxes.get_matrix()
        m[0, 2] = (m[0, 2] + m[0, 0] * self.origin[0]) / sx
        m[1, 2] += m[1, 1] * self.origin[1]
        return m

    def get_matrix(self) -> numpy.ndarray:
        if self._invalid or self._mtx is None:
            self._mtx = self._calc_matrix()
        return self._mtx


def add_scalebar(ax: 'Axes', size: float, height: float = 0.05):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.0, 0.0), size, height, fc='white', ec='black', linewidth=2.0, transform=_ScalebarTransform(ax, (0.04, 0.06))))


def add_scalebar_top(ax: 'Axes', size: float, height: float = 0.05):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.0, 0.0), size, height, fc='white', ec='black', linewidth=2.0, transform=_ScalebarTransform(ax, (0.04, 1.0 - 0.06 - height))))


def label_inside_ax(ax: 'Axes', text: str, pos: t.Tuple[float, float] = (0.02, 0.95), color: t.Any = 'white',
                    stroke: t.Any = 'black', strokewidth: float = 3.0, **text_kwargs: t.Any) -> 'Text':
    kwargs = {
        'ha': 'left',
        'va': 'top',
        'fontsize': 16.0,
        'transform': ax.transAxes,
        'color': color,
    }
    kwargs.update(text_kwargs)

    t = ax.text(pos[0], pos[1], text, **kwargs)
    if strokewidth > 0.0:
        t.set_path_effects([
            path_effects.Stroke(linewidth=strokewidth, foreground=stroke),
            path_effects.Normal()
        ])
    return t


__all__ = [
    'plot_pacbed', 'plot_raw',
    'plot_object_phase', 'plot_object_mag',
    'plot_metrics',

    'add_scalebar', 'add_scalebar_top',
    'ColormapLike', 'NormLike',
]
