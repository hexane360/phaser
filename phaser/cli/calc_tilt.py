import math
import typing as t
from pathlib import Path

import click
import numpy
from matplotlib import pyplot
from matplotlib.backend_bases import MouseButton, MouseEvent, PickEvent
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from rich.console import Console
from rich.prompt import FloatPrompt, Prompt

# TODO: generalize to other detectors
from phaser.io.empad import EmpadMetadata, load_4d


@click.command()
@click.argument('path', type=click.Path(exists=True, dir_okay=False, file_okay=True))
def calc_tilt(path: t.Union[str, Path]):
    """
    Calculate the mistilt present in a ptychography datase.

    PATH should be the path to a JSON metadata file.
    """
    console = Console()

    path = Path(path).resolve()
    meta = EmpadMetadata.from_json(path)

    exp_path = meta.path or Path('.')
    raw_path = exp_path / (meta.raw_filename or "scan_x128_y128.raw")
    if not raw_path.exists():
        raise ValueError(f"Can't find raw data at path '{raw_path}'")

    raw = load_4d(raw_path)

    params = {
        'det': 'bf',
        'inner': 0.0,
        'outer': 2.0,
    }

    params['det'] = Prompt.ask("Detector type", choices=['bf', 'adf', 'af'], default=params['det'], console=console)
    if params['det'] in ('af', 'adf'):
        params['inner'] = float(FloatPrompt.ask(r"Inner radius \[mrad]", default=params['inner'], console=console))
        inner = params['inner']
    else:
        inner = 0.

    if params['det'] in ('af', 'bf'):
        params['outer'] = float(FloatPrompt.ask(r"Outer radius \[mrad]", default=params['outer'], console=console))
        outer = params['outer']
    else:
        outer = numpy.inf

    def get_pattern_norm(pattern) -> Normalize:
        max_int = numpy.nanquantile(pattern, 0.99)
        min_int = numpy.nanmin(pattern)
        return Normalize(
            0.4*(max_int - min_int) + min_int,
            max_int
        )

    console.print("Loading dataset...")
    kx = numpy.arange(raw.shape[-2], dtype=numpy.float32) - raw.shape[-2] / 2.
    kx *= meta.diff_step
    ky = numpy.arange(raw.shape[-1], dtype=numpy.float32) - raw.shape[-1] / 2.
    ky *= meta.diff_step
    kyy, kxx = numpy.meshgrid(kx, ky, indexing='ij')
    k2 = kyy**2 + kxx**2
    virtual_aperture = numpy.zeros(raw.shape[-2:], dtype=bool)
    virtual_aperture[(k2 >= inner**2) & (k2 <= outer**2)] = 1
    virtual_img = numpy.nansum(raw * virtual_aperture, axis=(-1, -2))
    pattern = numpy.nansum(raw, axis=(0, 1))

    scan_step = numpy.array(meta.scan_step) * 1e10  # m to A

    real_fig, real_ax = pyplot.subplots()
    real_ax.set_xlabel(r"x [$\mathrm{\AA}$]")
    real_ax.set_ylabel(r"y [$\mathrm{\AA}$]")
    real_img = real_ax.imshow(virtual_img, extent=(-0.5 * scan_step[0], (virtual_img.shape[1] + 0.5) * scan_step[0], (virtual_img.shape[0] + 0.5) * scan_step[1], -0.5 * scan_step[1]))

    recip_fig, recip_ax = pyplot.subplots()
    recip_ax.invert_xaxis()  # to match sign convention
    recip_ax.set_aspect('equal')
    recip_ax.set_xlabel(r"$\theta_x$ [mrad]")
    recip_ax.set_ylabel(r"$\theta_y$ [mrad]")
    recip_img = recip_ax.pcolormesh(kx, ky, pattern, cmap='magma', norm=get_pattern_norm(pattern))

    real_img.set_picker(True)
    real_img.set_animated(True)

    rect = None
    #bg = real_fig.canvas.copy_from_bbox(real_ax.bbox)  # type: ignore
    drag_start: t.Optional[t.Tuple[float, float]] = None

    def draw_artists():
        real_ax.draw_artist(real_img)
        if rect is not None:
            real_ax.draw_artist(rect)

        real_fig.canvas.blit(real_ax.bbox)

    def draw(event=None):
        #nonlocal bg
        #bg = real_fig.canvas.copy_from_bbox(real_ax.bbox)  # type: ignore
        #real_fig.canvas.restore_region(bg)  # type: ignore
        draw_artists()

    def on_pick(event: PickEvent):
        nonlocal rect, drag_start
        if event.mouseevent.button != MouseButton.LEFT:
            return
        if event.artist is not real_img:
            return

        drag_start = tuple(real_ax.transData.inverted().transform((event.mouseevent.x, event.mouseevent.y)))
        rect = Rectangle(drag_start, 1., 1., fc='none', ec='red', transform=real_ax.transData)

    def on_release(event: MouseEvent):
        nonlocal drag_start
        if event.button != MouseButton.LEFT:
            return
        drag_start = None

        draw()
        if rect is not None:
            sum_in_rect(rect)

    def on_move(event: MouseEvent):
        nonlocal rect
        if drag_start is None or event.inaxes is None or event.button != MouseButton.LEFT:
            return
        if event.x is None or event.y is None or rect is None:
            return
        pt: t.Tuple[float, float] = tuple(real_ax.transData.inverted().transform((event.x, event.y)))

        rect.set_width(abs(pt[0] - drag_start[0]))
        rect.set_height(abs(pt[1] - drag_start[1]))
        rect.set_xy((min(pt[0], drag_start[0]), min(pt[1], drag_start[1])))
        draw()

    def sum_in_rect(rect: Rectangle):
        nonlocal pattern

        bbox = rect.get_bbox()
        min_x, min_y = numpy.array(bbox.min) / scan_step
        max_x, max_y = numpy.array(bbox.max) / scan_step

        pattern = numpy.nansum(raw[
            math.floor(min_y):math.ceil(max_y),
            math.floor(min_x):math.ceil(max_x),
        ], axis=(0, 1))
        recip_img.set_array(pattern)
        recip_img.set_norm(get_pattern_norm(pattern))
        recip_fig.canvas.draw()

    real_fig.canvas.mpl_connect('button_release_event', on_release)  # type: ignore
    real_fig.canvas.mpl_connect('motion_notify_event', on_move)      # type: ignore
    real_fig.canvas.mpl_connect('draw_event', draw)                  # type: ignore
    real_fig.canvas.mpl_connect('pick_event', on_pick)               # type: ignore

    pyplot.show()