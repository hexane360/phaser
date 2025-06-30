from functools import partial
from pathlib import Path
import typing as t

import numpy
import tifffile

from phaser.utils.num import to_numpy, abs2, fft2, get_array_module
from phaser.utils.image import remove_linear_ramp, colorize_complex, scale_to_integral_type
from phaser.utils.io import tiff_write_opts, tiff_write_opts_recip
from phaser.state import ReconsState
from phaser.plan import SaveOptions


def output_images(state: ReconsState, out_dir: Path, options: SaveOptions):
    for ty in options.images:
        if ty not in _SAVE_FUNCS:
            raise ValueError(f"Unknown image type '{ty}'")

        ext = options.plot_ext if ty in _PLOT_FUNCS else 'tiff'
        try:
            out_name = options.img_fmt.format(
                type=ty, iter=state.iter, ext=ext
            )
            out_path = out_dir / out_name
        except KeyError as e:
            raise ValueError(f"Invalid format string in 'img_fmt' (unknown key {e})") from None
        except Exception as e:
            raise ValueError("Invalid format string in 'img_fmt'") from e

        _SAVE_FUNCS[ty](state, out_path, options)


def output_state(state: ReconsState, out_dir: Path, options: SaveOptions):
    try:
        out_name = options.hdf5_fmt.format(
            iter=state.iter
        )
    except KeyError as e:
        raise ValueError(f"Invalid format string in 'hdf5_fmt' (unknown key {e})") from None
    except Exception as e:
        raise ValueError("Invalid format string in 'hdf5_fmt'") from e

    state.write_hdf5(out_dir / out_name)


def _save_probe(state: ReconsState, out_path: Path, options: SaveOptions):
    probe = to_numpy(state.probe.data)
    write_opts = tiff_write_opts(state.probe.sampling, n_slices=probe.shape[0])

    if options.img_dtype == 'float':
        # save complex image
        with tifffile.TiffWriter(out_path, ome=True) as w:
            write_opts['metadata']['axes'] = 'CYX'
            w.write(probe, **write_opts)
        return

    img = scale_to_integral_type(
        colorize_complex(probe), options.img_dtype
    )
    with tifffile.TiffWriter(out_path, ome=True) as w:
        write_opts['metadata']['axes'] = 'CYXS'
        w.write(img, photometric='rgb', **write_opts)


def _save_probe_mag(state: ReconsState, out_path: Path, options: SaveOptions):
    probe_mag = abs2(state.probe.data)
    write_opts = tiff_write_opts(state.probe.sampling, n_slices=probe_mag.shape[0])

    if options.img_dtype != 'float':
        probe_mag = scale_to_integral_type(to_numpy(probe_mag), options.img_dtype, min_range=0.2)

    with tifffile.TiffWriter(out_path, ome=True) as w:
        write_opts['metadata']['axes'] = 'CYX'
        w.write(to_numpy(probe_mag), **write_opts)


def _save_probe_recip(state: ReconsState, out_path: Path, options: SaveOptions):
    xp = get_array_module(state.probe.data)
    probe = to_numpy(xp.fft.fftshift(fft2(state.probe.data), axes=(-1, -2)))
    write_opts = tiff_write_opts_recip(state.probe.sampling, n_slices=probe.shape[0])

    if options.img_dtype == 'float':
        # save complex image
        with tifffile.TiffWriter(out_path, ome=True) as w:
            write_opts['metadata']['axes'] = 'CYX'
            w.write(probe, **write_opts)
        return

    img = scale_to_integral_type(
        colorize_complex(probe), options.img_dtype
    )
    with tifffile.TiffWriter(out_path, ome=True) as w:
        write_opts['metadata']['axes'] = 'CYXS'
        w.write(img, photometric='rgb', **write_opts)


def _save_probe_recip_mag(state: ReconsState, out_path: Path, options: SaveOptions):
    xp = get_array_module(state.probe.data)
    probe_mag = to_numpy(abs2(xp.fft.fftshift(fft2(state.probe.data), axes=(-1, -2))))
    write_opts = tiff_write_opts_recip(state.probe.sampling, n_slices=probe_mag.shape[0])

    if options.img_dtype != 'float':
        probe_mag = scale_to_integral_type(probe_mag, options.img_dtype, min_range=0.2)

    with tifffile.TiffWriter(out_path, ome=True) as w:
        write_opts['metadata']['axes'] = 'CYX'
        w.write(probe_mag, **write_opts)


def _save_object_phase(state: ReconsState, out_path: Path, options: SaveOptions, stack: bool = False):
    crop = options.crop_roi

    xp = get_array_module(state.object.data)
    obj_phase = xp.angle(state.object.data)

    obj_sampling = state.object.sampling
    write_opts = tiff_write_opts(
        obj_sampling,
        corner=obj_sampling.region_min if crop else None,
        zs=state.object.zs() if stack else None,
    )

    if crop:
        obj_phase = obj_phase[(Ellipsis, *state.object.sampling.get_region_crop())]
        mask = xp.ones(obj_phase.shape[-2:], dtype=numpy.bool_)
    else:
        # include whole image, but only scale based on ROI
        mask = state.object.sampling.get_region_mask(xp=xp)

    if options.unwrap_phase:
        obj_phase = xp.unwrap(xp.unwrap(obj_phase, axis=-1), axis=-2)

    if not stack:
        obj_phase = xp.sum(obj_phase, axis=0)

    mask = to_numpy(mask)
    obj_phase = remove_linear_ramp(to_numpy(obj_phase), mask)

    if options.img_dtype != 'float':
        obj_phase = scale_to_integral_type(obj_phase, options.img_dtype, mask)

    with tifffile.TiffWriter(out_path, ome=True) as w:
        write_opts['metadata']['axes'] = 'ZYX' if stack else 'YX'
        w.write(obj_phase, **write_opts)


def _save_object_mag(state: ReconsState, out_path: Path, options: SaveOptions, stack: bool = False):
    crop = options.crop_roi

    obj_sampling = state.object.sampling
    write_opts = tiff_write_opts(
        obj_sampling,
        corner=obj_sampling.region_min if crop else None,
        zs=state.object.zs() if stack else None,
    )

    xp = get_array_module(state.object.data)
    obj_mag = abs2(state.object.data)
    if crop:
        obj_mag = obj_mag[(Ellipsis, *state.object.sampling.get_region_crop())]
        mask = numpy.ones(obj_mag.shape[-2:], dtype=numpy.bool_)
    else:
        # include whole image, but only scale based on ROI
        mask = state.object.sampling.get_region_mask(xp=numpy)

    if not stack:
        obj_mag = xp.prod(obj_mag, axis=0)

    obj_mag = to_numpy(obj_mag)
    if options.img_dtype != 'float':
        obj_mag = scale_to_integral_type(obj_mag, options.img_dtype, mask, min_range=0.2)

    with tifffile.TiffWriter(out_path, ome=True) as w:
        write_opts['metadata']['axes'] = 'ZYX' if stack else 'YX'
        w.write(obj_mag, **write_opts)


def _plot_scan(state: ReconsState, out_path: Path, options: SaveOptions):
    from matplotlib import pyplot
    fig, ax = pyplot.subplots(figsize=(4, 4), dpi=options.plot_dpi, constrained_layout=True)

    ax.set_aspect(1.)
    [l, r, b, t] = state.object.sampling.mpl_extent()
    ax.set_xlim(l, r)
    ax.set_ylim(b, t)

    scan = to_numpy(state.scan).reshape(-1, 2)
    i = numpy.arange(scan[..., 0].size)
    ax.scatter(scan[..., 1], scan[..., 0], c=i, s=0.2, cmap='plasma')

    fig.savefig(out_path)
    pyplot.close(fig)


def _plot_tilt(state: ReconsState, out_path: Path, options: SaveOptions):
    from matplotlib import pyplot

    fig, ax = pyplot.subplots(figsize=(4, 4), dpi=options.plot_dpi, constrained_layout=True)

    ax.set_aspect(1.)
    [l, r, b, t] = state.object.sampling.mpl_extent()
    ax.set_xlim(l, r)
    ax.set_ylim(b, t)

    scan = to_numpy(state.scan).reshape(-1, 2)
    tilt = to_numpy(state.tilt).reshape(-1, 2)
    tilt = tilt[..., 1] + tilt[..., 0]*1.j
    max_tilt = max(numpy.max(numpy.abs(tilt)), 1.0)  # at least 1 mrad
    c = colorize_complex(tilt / max_tilt, amp=True, rescale=False)
    ax.scatter(scan[..., 1], scan[..., 0], c=c, s=0.2)

    fig.draw_without_rendering()
    trans = ax.transAxes + fig.transFigure.inverted()
    legend_ax_max = trans.transform([0.95, 0.02])
    legend_ax_size = (0.1, 0.1)
    legend_ax = fig.add_axes((legend_ax_max[0] - legend_ax_size[0], legend_ax_max[1], *legend_ax_size), projection='polar')

    legend_ax.set_rmax(max_tilt)  # type: ignore
    legend_ax.set_theta_direction(-1)  # type: ignore
    legend_ax.set_axis_off()

    thetas = numpy.linspace(0., 2*numpy.pi, 70)
    rs = numpy.concatenate([[0.0], numpy.geomspace(0.1, 1.0, 30)])
    rr, tt = numpy.meshgrid(rs, thetas, indexing='ij')
    c2 = colorize_complex(rr * numpy.exp(1.j * tt), rescale=False)
    legend_ax.pcolormesh(tt, rr * max_tilt, c2)
    legend_ax.text(-numpy.pi/2., max_tilt * 1.05, f"{max_tilt:.1f} mrad", ha='center', va='bottom', size='small')

    fig.savefig(out_path)
    pyplot.close(fig)


_SAVE_FUNCS: t.Dict[str, t.Callable[[ReconsState, Path, SaveOptions], t.Any]] = {
    'probe': _save_probe,
    'probe_mag': _save_probe_mag,
    'probe_recip': _save_probe_recip,
    'probe_recip_mag': _save_probe_recip_mag,
    'object_phase_stack': partial(_save_object_phase, stack=True),
    'object_phase_sum': partial(_save_object_phase, stack=False),
    'object_mag_stack': partial(_save_object_mag, stack=True),
    'object_mag_sum': partial(_save_object_mag, stack=False),
    'scan': _plot_scan,
    'tilt': _plot_tilt,
}
# save functions with special handling of file extensions
_PLOT_FUNCS: t.Set[str] = {'scan', 'tilt'}