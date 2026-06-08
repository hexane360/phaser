from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event
import fnmatch
import typing as t

import click
import numpy
from numpy.typing import NDArray, ArrayLike
import scipy.ndimage
import json
import h5py
from matplotlib import pyplot
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.backend_bases import MouseEvent, MouseButton, PickEvent, KeyEvent
from rich.console import Console
from rich.prompt import Prompt, FloatPrompt, Confirm


# TODO: generalize to other detectors
from phaser.io.empad import EmpadMetadata, load_4d


def load_adf(path: t.Union[str, Path]) -> t.Tuple[numpy.ndarray, t.Any]:
    f = h5py.File(path)

    images = t.cast(h5py.Group, f['Data/Image'])
    if len(images) == 0:
        raise ValueError("No images found in dataset.")
    if len(images) > 1:
        raise ValueError("Multi-image files not currently supported.")
    image = t.cast(h5py.Group, next(iter(images.values())))
    raw_meta: numpy.ndarray = t.cast(h5py.Dataset, image['Metadata'])[:, 0][()]
    meta_bytes = raw_meta.tobytes()
    meta_bytes = meta_bytes[:meta_bytes.find(b"\0")]
    meta = json.loads(meta_bytes)
    data = t.cast(h5py.Dataset, image['Data'])[..., 0][()]

    return (data, meta)


def normed_to_uint8(data: numpy.ndarray) -> NDArray[numpy.uint8]:
    return numpy.floor(numpy.clip(data, 0, 1) * 255.999).astype(numpy.uint8)


def normed_to_color(data: numpy.ndarray, color: ArrayLike) -> NDArray[numpy.uint8]:
    return numpy.floor(numpy.clip(data, 0, 1)[..., None] * color).astype(numpy.uint8)


def signed_angle(v1: numpy.ndarray, v2: numpy.ndarray) -> float:
    return numpy.pi - numpy.mod(numpy.arctan2(v1[0] * v2[1] - v1[1] * v2[0], numpy.dot(v1, v2)), 2.*numpy.pi)


def load_files(paths: t.Iterable[t.Union[str, Path]], inner: float, outer: float) -> t.Iterable[t.Tuple[Path, EmpadMetadata, NDArray[numpy.float32]]]:
    queue: Queue[t.Tuple[Path, EmpadMetadata, NDArray[numpy.float32]]] = Queue(1)
    finished = Event()

    # producer thread
    # should hold one in queue, one waiting for queue, one processing
    def producer():
        try:
            for path in paths:
                # eagerly load and put on queue
                path = Path(path).resolve()
                meta = EmpadMetadata.from_json(path)

                exp_path = meta.path or Path('.')
                raw_path = (exp_path / (meta.raw_filename or "scan_x128_y128.raw")).resolve()
                if not raw_path.exists():
                    raise ValueError(f"Can't find raw data at path '{raw_path}'")
                scan_shape = t.cast(t.Tuple[int, int], meta.scan_shape[::-1]) if meta.scan_shape is not None else None
                raw = load_4d(raw_path, scan_shape=scan_shape, flips=meta.det_flips)

                kx = numpy.arange(raw.shape[-2], dtype=numpy.float32) - raw.shape[-2] / 2.
                ky = numpy.arange(raw.shape[-1], dtype=numpy.float32) - raw.shape[-1] / 2.
                kyy, kxx = numpy.meshgrid(kx, ky, indexing='ij')
                k2 = kyy**2 + kxx**2
                virtual_aperture = numpy.zeros(raw.shape[-2:], dtype=bool)
                virtual_aperture[(k2 >= inner**2) & (k2 <= outer**2)] = 1
                virtual_img = numpy.sum(raw * virtual_aperture, axis=(-1, -2))

                queue.put((path, meta, virtual_img))
            # finished all files
            finished.set()
        except BaseException:
            import traceback
            traceback.print_exc()

    thread = Thread(target=producer, name='loader', daemon=True)
    thread.start()

    # periodically check that thread is still running,
    # to prevent deadlock
    while thread.is_alive():
        try:
            val = queue.get(timeout=1.)
        except Empty:
            continue
        yield val

    thread.join()

    # drain queue once thread is finished
    while True:
        try:
            val = queue.get(False)
        except Empty:
            break
        yield val

    if not finished.is_set():
        raise ValueError("Error in file loading")

    print("Finished processing files!")


def calibrated_meta_path(meta_path: Path):
    meta_path_name = meta_path.stem
    if meta_path_name.endswith('_orig'):
        meta_path_name = meta_path_name[:-5]
    new_meta_path = meta_path.with_stem(meta_path_name + "_calib")
    return new_meta_path


@click.command()
@click.argument('path', type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option('--include', type=str, multiple=True,
              help="Glob of filenames to include. If not specified, include all '.json' files")
@click.option('--exclude', type=str, multiple=True,
              help="Glob of filenames to exclude.")
@click.option('--skip-existing/--no-skip-existing', default=True,
              help="Whether to skip datasets which have already been calibrated. Defaults to true.")
def calc_drift(path: t.Union[str, Path], include: t.Sequence[str] = (), exclude: t.Sequence[str] = (), skip_existing: bool = True):
    """
    Calculate the linear drift present in a ptychography dataset, or group of datasets.

    PATH should be the path to a JSON metadata file, or to a directory
    which will be searched for JSON metadata files.

    Datasets can be included or excluded with the `--include` and `--exclude` options.
    These can be repeated multiple times.

    Calculated drift is stored in a new metadata file with the suffix `_calib`.
    By default, datasets which already have this file are skipped. This behavior can be changed
    using the `--no-skip-existing` option.
    """

    console = Console()
    path = Path(path)

    if path.is_file():
        paths = [path]
    else:
        exclude = (*exclude, "*_calib.json", '._*')
        paths = list(path.glob('**/*.json'))
        if len(include):
            paths = list(filter(lambda path: any(fnmatch.fnmatch(path.name, pat) for pat in include), paths))
        paths = list(filter(lambda path: not any(fnmatch.fnmatch(path.name, pat) for pat in exclude), paths))

        if skip_existing:
            paths = list(filter(lambda path: not calibrated_meta_path(path).exists(), paths))

        paths.sort()

    console.print(f"{len(paths)} file(s) to process.")

    params = {
        'det': 'bf',
        'inner': 0.0,
        'outer': 2.0,
        'd1': 6.,
        'd2': 6.
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

    params['scale1'] = FloatPrompt.ask("Distance 1 scale", default=1., console=console)
    params['scale2'] = FloatPrompt.ask("Distance 2 scale", default=params['scale1'], console=console)
    #params['angle'] = FloatPrompt.ask("Signed angle from distance 1 -> distance 2 (degree, CWW is +)", default=90., console=console)
    #params['angle'] *= numpy.pi/180.

    console.print("Loading files...")
    for (meta_path, meta, virtual_img) in load_files(paths, inner, outer):
        console.print(f"Loaded file '{meta.path}'")
        if not bool(Confirm.ask("Process this file?", default=True)):
            console.print("Skipping file...")
            continue

        while True:
            correction_matrix = calc_drift_one(console, meta, virtual_img, params)
            if correction_matrix is None:
                continue

            choice = Prompt.ask("Save calibration?", choices=['y', 'n', 'abort'], default='y', console=console)
            if choice == 'abort':
                break
            if choice.lower() not in ('y', 'yes'):
                continue

            new_meta_path = calibrated_meta_path(meta_path)
            new_meta = meta.__replace__()
            new_meta.scan_correction = tuple(map(tuple, correction_matrix))  # type: ignore

            new_meta.write_json(new_meta_path, indent=4)
            console.print(f"New metadata written to '{new_meta_path}'!")
            break


def calc_drift_one(
    console: Console, meta: EmpadMetadata, virtual_img: NDArray[numpy.float32], params: t.Dict[str, t.Any]
) -> t.Optional[NDArray[numpy.number]]:
    print(meta)
    scan_step_4d = numpy.array(meta.scan_step) * 1e10  # m to angstrom
    scan_size_4d = scan_step_4d * meta.scan_shape

    console.print(f" 4D pixel size: {scan_step_4d[0]:.3f} x {scan_step_4d[1]:.3f} A", style='logging.level.info')
    console.print(f" 4D image size: {scan_size_4d[0]:.2f} x {scan_size_4d[1]:.2f} A", style='logging.level.info')

    fig, ax = pyplot.subplots(constrained_layout=True)
    canvas = fig.canvas
    ax.set_xlabel('x [A]')
    ax.set_ylabel('y [A]')

    ax.set_xlim(0., scan_size_4d[0])
    ax.set_ylim(scan_size_4d[1], 0.)

    img = ax.imshow(virtual_img, extent=(-0.5 * scan_step_4d[0], (virtual_img.shape[1] + 0.5) * scan_step_4d[0], (virtual_img.shape[0] + 0.5) * scan_step_4d[1], -0.5 * scan_step_4d[1]))
    img.set_picker(True)
    img.set_animated(True)

    path: t.Optional[PathPatch] = None
    circles: t.List[Circle] = []

    selected: t.Optional[Circle] = None
    bg = canvas.copy_from_bbox(ax.bbox)  # type: ignore

    def draw_artists():
        ax.draw_artist(img)
        if path is not None:
            ax.draw_artist(path)
        for circle in circles:
            ax.draw_artist(circle)
        canvas.blit(ax.bbox)  # type: ignore

    def draw(event=None):
        nonlocal path, bg

        vertices = numpy.array([circle.center for circle in circles])
        if len(vertices):
            if path is None:
                path = PathPatch(MplPath(vertices), fill=False, lw=3.)
                ax.add_patch(path)
            else:
                path.set_path(MplPath(vertices))

        bg = canvas.copy_from_bbox(ax.bbox)  # type: ignore
        draw_artists()

    def on_release(event: MouseEvent):
        nonlocal selected
        if not event.button == MouseButton.LEFT:
            return
        selected = None

    def on_pick(event: PickEvent):
        nonlocal selected
        nonlocal path
        if event.mouseevent.button != MouseButton.LEFT:
            return
        if event.artist is not img:
            return

        for circle in circles:
            if circle.contains_point((event.mouseevent.x, event.mouseevent.y)):  # type: ignore
                selected = circle
                return

        if len(circles) < 3:
            pos = t.cast(
                t.Tuple[float, float],
                list(ax.transData.inverted().transform((event.mouseevent.x, event.mouseevent.y)))
            )

            node = Circle(pos, radius=0.5, fc='white', ec='black', transform=ax.transData)
            node.set_animated(True)
            ax.add_patch(node)
            circles.append(node)
            selected = node
            draw()

    def on_move(event: MouseEvent):
        if selected is None or event.inaxes is None or event.button != MouseButton.LEFT:
            return
        if event.x is None or event.y is None:
            return
        new_pt = ax.transData.inverted().transform((event.x, event.y))
        selected.center = new_pt

        canvas.restore_region(bg)  # type: ignore
        draw()
        #draw_artists()

    warped_img = None
    correction_matrix: t.Optional[NDArray[numpy.number]] = None

    def on_press(event: KeyEvent):
        if event.key != 'enter':
            return
        vertices = numpy.array([circle.center for circle in circles])

        if len(vertices) != 3:
            console.print("Need 3 points to compute drift")
            return

        vecs = numpy.diff(vertices, axis=0)
        #vecs_next = numpy.roll(vecs, 1, axis=0)

        print(f"angle: {signed_angle(vecs[:, 0], vecs[:, 1]) * 180./numpy.pi}")

        while True:
            dists = []
            for (i, scale, default) in zip(range(len(vertices)-1), (params['scale1'], params['scale2']), (params['d1'], params['d2'])):
                while True:
                    try:
                        s = Prompt.ask(f"Distance {i+1} (* {scale:.3f} A)", default=default, console=console)
                        val = float(eval(s))
                        dists.append(val * scale)
                    except Exception:
                        continue
                    break
            console.print(f"Distances [A]: {dists[0]:.3f}, {dists[1]:.3f}")
            if Confirm.ask("Distances correct?", default=True):
                break

        # first, we use some logic to determine which orthogonal basis the measurements are closest to.
        horz_vec = numpy.argmax(numpy.abs(vecs[:, 0]))  # vec with maximum x component is horizontal
        horz_sign = numpy.sign(vecs[horz_vec, 0])  # whether to flip horizontal vector
        vert_sign = numpy.sign(vecs[horz_vec-1, 1])  # whether to flip vertical vector

        # make target_vecs using the determinations above
        target_vecs = numpy.diag(numpy.array(dists))
        if horz_vec != 0:
            # v1 should be vertical
            target_vecs = target_vecs[::-1, :]
        # flip target_vecs based on desired signs
        target_vecs = numpy.diag([horz_sign, vert_sign]) @ target_vecs
        console.print(f"finding transformation\n{vecs.T}\nto\n{target_vecs}")

        # we try to find A which transforms `vecs` into `target_vecs`.
        a = target_vecs @ numpy.linalg.inv(vecs.T)

        nonlocal warped_img, correction_matrix

        # because we only know distances, `a` 
        q, r = numpy.linalg.qr(a)
        r: NDArray[numpy.floating] = numpy.diag(numpy.sign(numpy.diagonal(r))) @ r  # flip to ensure diagonals are positive. These are absorbed into `q`.
        console.print(f"actual dists: {dists}")
        console.print(f"measured dists: {numpy.linalg.norm(vecs, axis=-1)}")
        console.print(f"distortion:\n{r[::-1, ::-1]}")  # correct for ptychoshelves coordinate system
        #print(f"distortion:\n{r}")  # correct for ptychoshelves coordinate system
        vecs_after = (r @ vecs.T).T
        console.print(f"dists after correction: {numpy.linalg.norm(vecs_after, axis=-1)}")
        console.print(f"vecs after correction: {vecs_after[0]}, {vecs_after[1]}")
        angle_after = signed_angle(vecs_after[0], vecs_after[1])
        console.print(f"angle after correction: {180./numpy.pi * angle_after:.2f}")
        #vecs_after_a = (a @ vecs.T).T
        #print(f"vecs after a correction: {vecs_after_a[:, 0]}, {vecs_after_a[:, 1]}")
        #angle_after_a = signed_angle(vecs_after_a[:, 0], vecs_after_a[:, 1])
        #print(f"angle after a correction: {180./numpy.pi * angle_after_a:.2f}")
        correction_matrix = r

        # warp image given corrections
        warped_shape = tuple(numpy.ceil(numpy.array(virtual_img.shape) * numpy.max(numpy.abs(numpy.diagonal(r)))).astype(int))
        affine = numpy.block([[numpy.linalg.inv(r)[::-1, ::-1], numpy.zeros((2, 1))], [numpy.zeros((1, 2)), numpy.ones((1, 1))]])
        translation = numpy.eye(3)
        translation[:2, -1] += numpy.array(virtual_img.shape[-2:]) / 2.
        translation2 = numpy.eye(3)
        translation2[:2, -1] -= numpy.array(warped_shape[-2:]) / 2.
        affine = translation @ affine @ translation2
        warped_img = scipy.ndimage.affine_transform(virtual_img, affine, output_shape=warped_shape)
        pyplot.close(fig)

    # TODO: error out for non-interactive backends
    #canvas.mpl_connect('button_press_event', on_click)
    canvas.mpl_connect('button_release_event', on_release)  # type: ignore
    canvas.mpl_connect('key_press_event', on_press)         # type: ignore
    canvas.mpl_connect('motion_notify_event', on_move)      # type: ignore
    canvas.mpl_connect('draw_event', draw)                  # type: ignore
    canvas.mpl_connect('pick_event', on_pick)               # type: ignore

    pyplot.show()

    if warped_img is None:
        return

    fig, ax = pyplot.subplots()
    ax.imshow(warped_img, vmin=float(numpy.nanmin(virtual_img)), vmax=float(numpy.nanmax(virtual_img)),
              extent=(-0.5 * scan_step_4d[0], (warped_img.shape[1] + 0.5) * scan_step_4d[0], (warped_img.shape[0] + 0.5) * scan_step_4d[1], -0.5 * scan_step_4d[1]))
    ax.set_xlabel('x [A]')
    ax.set_ylabel('y [A]')
    ax.set_title("Warped image")

    def on_press_2(event: KeyEvent):
        if event.key != 'enter':
            return
        pyplot.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_press_2)  # type: ignore

    pyplot.show()

    return correction_matrix