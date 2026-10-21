"""`phaser process_empad` command."""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import time
import typing as t

from rich.logging import RichHandler

import click
from click.exceptions import Exit
import pane
from rich.prompt import Prompt
from rich.theme import Theme
from rich.console import Console

from phaser.io.empad import EmpadMetadata, UnscannedDataset
from phaser.utils.config import Config
from phaser.cli.utils import natural_key


T = t.TypeVar('T')


class ProcessEmpadConfig(pane.PaneBase):
    """Configuration for `process_empad`."""

    diff_step: t.Dict[float, t.Tuple[float, float]] = {}
    """
    Diffraction pixel size calibration, given as a mapping kV: [camera_length, diff_step].
    'camera_length' is given in meters, and diff_step in mrad/pixel. At a given kV, the
    pixel size is interpolated linearly to get different camera lengths.
    E.g.:
    diff_step:
      200: [0.689, 0.5]  # 200 kV: 689 mm camera length, 0.5 mrad/px
    """

    adu: t.Optional[t.Tuple[float, float]] = None
    """
    ADU (single electron intensity) calibration.
    Give a tuple of `[kV, adu]`. Other kVs assume linear scaling.
    """

    det_flips: t.Optional[t.Tuple[bool, bool, bool]] = None
    """Detector flips (flip_y, flip_x, transpose). Set and forget"""

    det_rotation: float = 0.0
    """Detector rotation relative to scan axes (degrees)."""

    probe_current: float = 30.0
    """Default probe current (pA)"""

    conv_angle: float = 25.0
    """Default convergence angle (mrad). Used only if missing from metadata (e.g. EMPAD1)."""


CONFIG: Config[ProcessEmpadConfig] = Config('process_empad', ProcessEmpadConfig)


def default_console() -> Console:
    """Return a themed `Console`, and configure logging to use it."""
    console = Console(theme=Theme({
        'warning': 'bold yellow',
        'error': 'bold red',
        'info': 'blue',
        'prompt.invalid': 'bold red',
    }))
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", datefmt="[%X]",
        handlers=[RichHandler(logging.INFO, console=console, enable_link_path=False)]
    )
    return console


def prompt_ask(name, unit=None, default=None, default_str: str = "", console=None, validate=float, err="Please enter a number") -> t.Tuple[str, t.Any]:
    """Prompt for a value `validate`-able from a string until valid.

    Returns:
        `(entered_string, value)`. Entering `default_str` returns `default` as the value.
    """
    if not console:
        console = default_console()
    while True:
        str_val = Prompt.ask(f"{name} \\[{unit}]", default=default_str, console=console)
        if default_str and str_val == default_str:
            return (default_str, default)
        try:
            val = validate(str_val)
            return (str_val, val)
        except Exception:
            console.print(f"[prompt.invalid]{err}")


@dataclass
class Parameter(t.Generic[T], ABC):
    """Metadata value prompted for interactively, remembering previous answers."""

    name: str
    default_val: t.Optional[T]

    @abstractmethod
    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> T:
        """Prompt for the value, given the dataset `metadata`."""
        ...


@dataclass
class FloatParam(Parameter[float]):
    """Float prompted in `unit`, multiplied by `conv_factor` if given."""

    unit: t.Optional[str]
    default_str: str
    conv_factor: t.Optional[float] = None

    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> float:
        (val_str, val) = prompt_ask(self.name, self.unit, self.default_val, self.default_str, console, validate=float)
        self.default_str = val_str
        self.default_val = val
        if self.conv_factor:
            return val * self.conv_factor
        return val


@dataclass(init=False)
class CameraParam(Parameter[float]):
    """Diffraction pixel size [mrad/px], defaulted by scaling the configured calibration with camera length."""

    used: t.Dict[t.Tuple[str, t.Optional[str]], t.Tuple[str, float]] = field(default_factory=dict)
    """Maps (f"{voltage:.0f}", f"{camera_length:.3f}") to the (str, value) entered for it"""
    defaults: t.Dict[str, t.Tuple[float, float]]
    """Maps f"{voltage:.0f}" to default (camera_length, diff_step) tuple"""

    def __init__(self, defaults: t.Dict[float, t.Tuple[float, float]]):
        self.name = 'Diffraction pixel spacing'
        self.used = {}
        # kV to V
        self.defaults = { f"{k*1e3:.0f}": v for (k, v) in defaults.items() }

    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> float:
        if not console:
            console = default_console()
        camera_length = metadata.camera_length
        camera_length_str = f"{camera_length:.3f}" if camera_length is not None else None
        voltage_str = f"{metadata.voltage:.0f}"
        if camera_length is not None:
            console.print(f"Camera length: {camera_length * 1e3:.0f} mm")

        (default_str, default) = self.used.get((voltage_str, camera_length_str), ("", None))
        if default is None and camera_length is not None and (t := self.defaults.get(voltage_str)) is not None:
            (default_camera_length, default_val) = t
            # diff_step scales inversely with camera length
            default = default_val * default_camera_length / camera_length
            default_str = f"{default:.03f}"

        (val_str, val) = prompt_ask(self.name, "mrad/px", default, default_str, console)
        self.used[(voltage_str, camera_length_str)] = (val_str, val)
        return val


@dataclass(init=False)
class AduParam(Parameter[float]):
    """Single-electron intensity [ADU], defaulted by scaling the configured value with voltage."""

    used: t.Dict[str, t.Tuple[str, float]]
    """Maps `f"{voltage:.0f}"` to the `(str, value)` entered for it."""
    default_voltage: t.Optional[float]
    """Voltage associated with `default_val` [V]."""

    def __init__(self, default: t.Optional[float] = None, default_voltage: t.Optional[float] = None):
        self.name = 'Single-electron intensity'
        self.default_val = default
        self.used = {}
        self.default_voltage = default_voltage

    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> float:
        voltage = metadata.voltage
        voltage_str = f"{voltage:.0f}"
        (default_str, default) = self.used.get(voltage_str, ("", None))
        if default is None and self.default_val is not None and self.default_voltage is not None:
            # ADU should scale linearly with voltage
            default = self.default_val * voltage / self.default_voltage
            default_str = f"{default:.0f}"

        (val_str, val) = prompt_ask(self.name, "ADU", default, default_str, console)
        self.used[voltage_str] = (val_str, val)
        return val


@dataclass(init=False)
class ConvAngleParam(Parameter[float]):
    """Convergence angle parameter [mrad]."""

    used: t.Dict[t.Optional[str], t.Tuple[str, float]]
    """Maps metadata `f"{conv_angle:.1f}"` (`None` if missing) to the `(str, value)` entered for it."""

    def __init__(self, default: float):
        self.name = 'Convergence angle'
        self.default_val = default
        self.used = {}

    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> float:
        if metadata.conv_angle is None:
            # missing readouts share a key, starting from the config default
            key = None
            (default_str, default) = self.used.get(key, (f"{self.default_val:.1f}", self.default_val))
        else:
            key = f"{metadata.conv_angle:.1f}"
            (default_str, default) = self.used.get(key, (key, metadata.conv_angle))
        (val_str, val) = prompt_ask(self.name, "mrad", default, default_str, console)
        self.used[key] = (val_str, val)
        return val


@dataclass(init=False)
class AuthorParam(Parameter[t.Optional[str]]):
    """Dataset author parameter."""

    used: t.Dict[t.Optional[str], t.Optional[str]]
    """Maps metadata author (`None` if missing) to the author entered for it."""

    def __init__(self):
        self.name = 'Author'
        self.default_val = None
        self.used = {}

    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> t.Optional[str]:
        val = Prompt.ask(self.name, default=self.used.get(metadata.author, metadata.author), console=console)
        self.used[metadata.author] = val
        return val


def make_params(config: ProcessEmpadConfig) -> t.Sequence[t.Tuple[str, Parameter]]:
    """Return the `(metadata_field, Parameter)` pairs to prompt for."""
    return [
        ('defocus', FloatParam("Defocus (CW is +)", 0.0, "nm", "0.0", 1e-9)), # nm to m
        ('conv_angle', ConvAngleParam(config.conv_angle)),
        ('beam_current', FloatParam("Approx. probe current", config.probe_current, "pA", f"{config.probe_current:.1f}", 1e-12)),  # pA to A
        ('diff_step', CameraParam(config.diff_step)),
        ('adu', AduParam(config.adu[1], config.adu[0] * 1e3) if config.adu is not None else AduParam()),
        ('author', AuthorParam()),
    ]


def process_dir(path: t.Union[str, Path], config: ProcessEmpadConfig,
                params: t.Sequence[t.Tuple[str, Parameter]],
                console: Console, prompt: bool = True):
    """Write a `.json` metadata file next to each unprocessed dataset metadata file under `path`.

    Datasets with an existing `.json` file are skipped. If `prompt`, ask for additional metadata.
    """
    n_proc = 0
    path = Path(path)

    raw_dirs = set(f.parent for f in path.rglob('*') if f.suffix == '.xml' or f.name == 'metadata.toml')
    for raw_dir in sorted(raw_dirs, key=natural_key):
        metadata_path = raw_dir / 'metadata.toml'
        # EMPAD2 metadata takes priority (a stub XML is usually present as well)
        empad2 = metadata_path.exists()
        if not empad2:
            try:
                [metadata_path, *extra] = list(filter(lambda f: not f.name.startswith('.'), raw_dir.glob('*.xml')))
            except ValueError:
                continue
            if len(extra):
                console.print(f"[warning]Skipping dir with multiple metadata files: '{raw_dir}'.")
                continue

        output_path = metadata_path.with_suffix('.json')
        if output_path.exists():
            # skip already created
            continue

        try:
            metadata = EmpadMetadata.from_toml(metadata_path) if empad2 else EmpadMetadata.from_xml(metadata_path)
            console.print(f"Loaded '{metadata_path}'.")
        except UnscannedDataset:
            console.print(f"[info]Skipping metadata '{metadata_path}' (not a 4D STEM dataset).")
            continue
        except Exception:
            console.print(f"[error]Couldn't parse metadata '{metadata_path}'. Skipping.")
            console.print_exception()
            continue

        if not (raw_dir / metadata.raw_filename).exists():
            console.print(f"[warning]Couldn't find raw data '{metadata.raw_filename}' in '{raw_dir}'.")

        metadata.det_rotation = config.det_rotation
        # typical orientation on v1 and v2
        metadata.det_flips = config.det_flips or (
            (True, False, False) if metadata.empad_version == 1 else (False, False, False)
        )

        if prompt:
            if metadata.notes is not None:
                console.print(f"Notes: {metadata.notes}", markup=False)

            for (k, param) in params:
                val = param.ask(metadata, console)
                setattr(metadata, k, val)

            metadata.notes = Prompt.ask("Notes", default=metadata.notes, console=console)

        metadata.write_json(output_path, indent=4)
        console.print(f"Wrote metadata to '{output_path}'.")
        n_proc += 1

    if n_proc:
        console.print(f"Processed {n_proc} file(s).")


@click.command()
@click.option('--watch/--no-watch', default=False, help="Watch folder for changes (Linux only)")
@click.option('--prompt/--no-prompt', default=True, help="Prompt for additional metadata.")
@click.option('--config', 'config_name', type=str, default=None,
              help="Named config to use, stored in 'process_empad_{name}.yaml'")
@click.option('--make-config', is_flag=True, default=False, help="Make config file and exit")
@click.argument('folder', type=click.Path(exists=True, file_okay=False), required=False)
def process_empad(
    folder: t.Union[str, Path, None],
    watch: bool = False,
    prompt: bool = True,
    config_name: t.Optional[str] = None,
    make_config: bool = False,
):
    """
    Process EMPAD XML/TOML metadata for all the datasets contained in FOLDER.

    Datasets with a 'metadata.toml' file are processed as EMPAD2. Otherwise,
    XML metadata is used, and processed as EMPAD1.
    """
    console = default_console()
    config = CONFIG
    if config_name is not None:
        try:
            config = config.variant(config_name)
        except ValueError as e:
            console.print(f"[error]{e}")
            raise Exit(1)

    if make_config:
        if not config.write_default():
            # failed to write
            console.print(f"[error]Config file already exists at '{config.path()}'.")
            raise Exit(1)
        console.print("Wrote config file!")
        return

    if config.path().exists():
        console.print(f"Using config '{config.path()}'")
    elif config_name is not None:
        console.print(f"[error]Config '{config_name}' does not exist at '{config.path()}'."
                      " Create it with '--make-config'.")
        raise Exit(1)

    config = config.get()
    params = make_params(config)

    if folder is None:
        folder = Path('.')
    path = Path(folder).resolve()

    console.print(f"Processing dir '{path}'")
    process_dir(path, config, params, console, prompt)

    if not watch:
        return

    while True:
        time.sleep(2.)
        process_dir(path, config, params, console, prompt)

    # TODO: weird bugs with inotify
    if 'linux' not in sys.platform:
        console.print("[error]--watch is supported on Linux only.")
        raise Exit(1)
    try:
        from inotify.adapters import InotifyTree  # type: ignore
        from inotify.constants import IN_CLOSE    # type: ignore
    except ImportError:
        console.print("[error]Couldn't import inotify.\nInstall it for --watch support.")
        raise Exit(1)

    notifier = InotifyTree(str(path), mask=IN_CLOSE)
    console.print("Watching for experiments...")

    gen = notifier.event_gen(yield_nones=False)
    for (_, event_types, _path, filename) in t.cast(t.Iterator[t.Tuple[t.Any, t.List[str], t.Any, str]], gen):
        if not (any(t in event_types for t in ('IN_CLOSE_WRITE', 'IN_CLOSE_NOWRITE'))
                and Path(filename).match('*.xml')):
            continue

        console.print("Found changes, reprocessing.")
        time.sleep(1e-1)
        process_dir(path, config, params, console, prompt)