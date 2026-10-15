"""Reconstruction plan dataclasses."""

import typing as t
from pathlib import Path

from .hooks import (
    EngineHook,
    ObjectHook,
    PostInitHook,
    PostLoadHook,
    ProbeHook,
    RawDataHook,
    ScanHook,
    TiltHook,
)
from .hooks.filter import FilterHook
from .hooks.regularization import (
    CostRegularizerHook,
    GroupConstraintHook,
    IterConstraintHook,
)
from .hooks.schedule import FlagLike, ScheduleLike
from .hooks.solver import (
    ConventionalSolverHook,
    GradientSolverHook,
    NoiseModelHook,
    PositionSolverHook,
)
from .types import (
    BackendName,
    Dataclass,
    EmptyDict,
    IsVersion,
    ReconsVars,
    SimpleFlag,
    Slices,
)

SaveType: t.TypeAlias = t.Literal[
    'probe', 'probe_mag', 'probe_recip', 'probe_recip_mag',
    'object_phase_stack', 'object_phase_sum',
    'object_mag_stack', 'object_mag_sum',
    'scan', 'tilt',
]
"""Image types that can be saved."""


class InitPlan(Dataclass, kw_only=True):
    """Initial state of the reconstruction."""
    state: t.Optional[Path] = None
    """Path to a state file (HDF5) to use as the initial state."""

    scan: t.Union[EmptyDict, ScanHook, None] = None
    """Hook to initialize the scan positions. Merged over the raw data's scan hook, and replaces the scan of `state`."""
    tilt: t.Union[EmptyDict, TiltHook, None] = None
    """Hook to initialize the tilt map. Merged over the raw data's tilt hook, and replaces the tilt of `state`."""
    probe: t.Union[EmptyDict, ProbeHook, None] = None
    """Hook to initialize the probe. Merged over the raw data's probe hook, and replaces the probe of `state`."""
    object: t.Optional[ObjectHook] = None  # ObjectHook('random')
    """Hook to initialize the object. Replaces the object of `state`. Defaults to `'random'`."""


class SaveOptions(Dataclass, kw_only=True):
    """Options for output files."""
    images: t.Tuple[SaveType, ...] = ('probe', 'object_phase_stack')
    """Image types to save."""
    crop_roi: bool = True
    """Crop images to the scan bounding box."""
    unwrap_phase: bool = False
    """Unwrap phase images."""
    img_dtype: t.Literal['float', '8bit', '16bit', '32bit'] = '16bit'
    """Datatype of saved images. `'float'` images are unscaled, others are scaled to saturation."""

    plot_ext: str = "svg"
    """File extension of saved plots."""
    plot_dpi: int = 300
    """Resolution of saved plots [dots per inch]."""

    out_dir: str = ""
    """Format string for the engine output directory, relative to `ReconsPlan.out_dir`. Keys: `{name}`, `{engine_num}`, `{grouping}`, `{niter}`."""
    img_fmt: str = "{type}_iter{iter.total_iter:03}.{ext}"
    """Format string for image filenames. Keys: `{type}`, `{ext}`, `{iter}` (an `IterState`)."""
    hdf5_fmt: str = "iter{iter.total_iter:03}.{ext}"
    """Format string for state filenames. Keys: `{ext}`, `{iter}` (an `IterState`)."""
    img_fmt_init: str = "{type}_initial.{ext}"
    """Format string for initial-state image filenames. Keys: `{type}`, `{ext}`."""
    hdf5_fmt_init: str = "initial.{ext}"
    """Format string for the initial-state filename. Keys: `{ext}`."""


class MtfPlan(Dataclass, kw_only=True):
    """Detector modulation transfer function."""
    filter: FilterHook
    """Filter defining the MTF."""
    domain: t.Literal['real', 'recip'] = 'recip'
    """Space the filter is applied in."""


class EnginePlan(Dataclass, kw_only=True):
    """Options common to all engines."""
    sim_shape: t.Optional[t.Tuple[int, int]] = None
    """Simulation shape `(ny, nx)`. `None` keeps the current shape."""
    resize_method: t.Literal['pad_crop', 'resample'] = 'pad_crop'
    """How to resize to `sim_shape`. `'pad_crop'` keeps the real-space extent; `'resample'` keeps the real-space pixel size."""

    probe_modes: int = 1
    """Number of incoherent probe modes."""
    base_mode_power: float = 0.7
    """Fraction of intensity assigned to the base mode when creating incoherent probe modes."""

    bwlim_frac: t.Optional[float] = 2/3
    """Multislice bandwidth limit, as a fraction of the maximum spatial frequency. `None` disables the limit."""
    obj_pad_px: float = 5.0
    """Object padding beyond the scan region plus half the probe extent [px]."""

    slices: t.Optional[Slices] = None
    """Slices to reslice the object to at the start of the engine."""

    niter: int = 10
    """Number of iterations."""
    grouping: t.Optional[int] = None
    """Number of probe positions per group. `None` uses the default of `64`."""
    compact: bool = False
    """Group neighboring positions (compact) instead of spread-out positions (sparse)."""
    shuffle_groups: t.Optional[FlagLike] = None
    """Whether to shuffle groups each iteration. Defaults to `True` for sparse grouping and `False` for compact grouping."""
    buffer_n_groups: t.Optional[int] = 2
    """Number of pattern groups buffered on the device. `0` disables buffering, `None` preloads the entire dataset."""

    jit_unroll_slices: t.Union[None, bool, int] = None
    """
    Number of slices to unroll during JIT compilation (JAX backend only).

    `True` or `0` unrolls all slices, `False` or `1` disables unrolling.
    The default depends on the engine (`10` for the gradient descent engine).
    """

    update_probe: FlagLike = True
    """Whether to update the probe at each iteration."""
    update_object: FlagLike = True
    """Whether to update the object at each iteration."""
    update_positions: FlagLike = False
    """Whether to update the scan positions at each iteration."""
    update_tilt: FlagLike = False
    """Whether to update the tilt map at each iteration."""

    calc_error: FlagLike = SimpleFlag(every=1)
    """Whether to calculate the detector error at each iteration."""
    calc_error_fraction: float = 0.1
    """Fraction of groups to calculate the error on."""

    save: FlagLike = False
    """Whether to save the state at each iteration. The state is also saved when the engine finishes, if any state output is enabled."""
    save_images: FlagLike = False
    """Whether to save images at each iteration. Images are also saved when the engine finishes, if any image output is enabled."""
    save_initial: bool = False
    """Save the state at the start of the engine."""
    save_initial_images: bool = False
    """Save images at the start of the engine."""
    save_options: SaveOptions = SaveOptions()
    """Output file options."""

    early_termination: t.Optional[int] = None
    """Terminate the engine after this many iterations without improvement. `None` disables."""
    early_termination_smoothing: float = 0.9
    """Smoothing factor of the error moving average. Lower values smooth more; `1` disables smoothing."""

    check_every_group: bool = False
    """Check for a non-finite loss after every group instead of every iteration (gradient engine only)."""
    send_every_group: bool = False
    """Send an update to the web server after every group."""

    send_max_wait_time: t.Optional[float] = 60.0
    """Maximum time between updates sent to the web server [s]. `None` for no limit."""

    mtf: t.Union[MtfPlan, FilterHook, None] = None
    """Detector MTF applied to simulated diffraction patterns."""


class AmplitudeNoisePlan(Dataclass, kw_only=True):
    """Amplitude noise model options."""
    gaussian_variance: float = 0.1
    """Variance of the Gaussian noise."""
    eps: float = 1.0e-3
    """Small constant for numerical stability."""
    offset: float = 0.0
    """Offset added to intensities before the square root."""


class AnscombeNoisePlan(AmplitudeNoisePlan, kw_only=True):
    """Anscombe noise model options."""
    offset: float = 0.375


class PoissonNoisePlan(AmplitudeNoisePlan, kw_only=True):
    """Poisson noise model options."""
    eps: float = 1.0e-3


NoiseModelHook.known['amplitude'] = ('phaser.engines.common.noise_models:AmplitudeNoiseModel', AmplitudeNoisePlan)
NoiseModelHook.known['anscombe'] = ('phaser.engines.common.noise_models:AnscombeNoiseModel', AnscombeNoisePlan)
NoiseModelHook.known['poisson'] = ('phaser.engines.common.noise_models:PoissonNoiseModel', PoissonNoisePlan)


class LSQMLSolverPlan(Dataclass, kw_only=True):
    """LSQML solver options."""
    stochastic: bool = True
    """Unused."""

    beta_object: ScheduleLike = 1.0
    """Object update step size."""
    beta_probe: ScheduleLike = 1.0
    """Probe update step size."""

    illum_reg_object: ScheduleLike = 1e-2
    """Illumination regularization of the object update."""
    illum_reg_probe: ScheduleLike = 1e-2
    """Illumination regularization of the probe update."""

    gamma: ScheduleLike = 1e-4
    """Regularization of the step size calculation."""


class EPIESolverPlan(Dataclass, kw_only=True):
    """ePIE solver options."""
    beta_object: ScheduleLike = 1.0
    """Object update step size."""
    beta_probe: ScheduleLike = 1.0
    """Probe update step size."""


ConventionalSolverHook.known['lsqml'] = ('phaser.engines.conventional.solvers:LSQMLSolver', LSQMLSolverPlan)
ConventionalSolverHook.known['epie'] = ('phaser.engines.conventional.solvers:EPIESolver', EPIESolverPlan)


class ConventionalEnginePlan(EnginePlan, kw_only=True):
    """Conventional (LSQML or ePIE) engine options."""
    noise_model: NoiseModelHook
    """Noise model defining the detector error."""
    solver: ConventionalSolverHook
    """Solver for the object and probe."""
    position_solver: t.Optional[PositionSolverHook] = None
    """Solver for scan positions."""

    group_constraints: t.List[GroupConstraintHook]
    """Constraints applied after every group."""
    iter_constraints: t.List[IterConstraintHook]
    """Constraints applied after every iteration."""


class GradientEnginePlan(EnginePlan, kw_only=True):
    """Gradient descent engine options."""
    noise_model: NoiseModelHook
    """Noise model defining the loss."""
    solvers: t.Dict[ReconsVars, GradientSolverHook]
    """Solver for each group of reconstruction variables."""

    regularizers: t.List[CostRegularizerHook]
    """Cost regularizers added to the loss."""
    group_constraints: t.List[GroupConstraintHook]
    """Constraints applied after every group."""
    iter_constraints: t.List[IterConstraintHook]
    """Constraints applied after every iteration."""


class SGDSolverPlan(Dataclass, kw_only=True):
    """Stochastic gradient descent solver options."""
    learning_rate: ScheduleLike
    """Learning rate."""
    momentum: t.Optional[ScheduleLike] = None
    """Momentum factor. `None` disables momentum."""
    nesterov: bool = True
    """Use Nesterov momentum."""


class AdamSolverPlan(Dataclass, kw_only=True):
    """Adam solver options."""
    learning_rate: ScheduleLike
    """Learning rate."""

    b1: float = 0.9
    """Decay rate of the first moment estimate."""
    b2: float = 0.999
    """Decay rate of the second moment estimate."""
    eps: float = 1.0e-8
    """Small constant added to the second moment denominator."""
    eps_root: float = 0.0
    """Small constant added inside the square root of the second moment."""

    nesterov: bool = False
    """Use Nesterov momentum."""


class PolyakSGDSolverPlan(Dataclass, kw_only=True):
    """Polyak step size SGD solver options."""
    max_learning_rate: ScheduleLike
    """Upper bound on the step size."""
    f_min: float
    """Estimate of the minimum loss value."""
    scaling: ScheduleLike = 1.0
    """Factor applied to the update."""
    eps: float = 0.0
    """Small constant added to the squared gradient norm."""


GradientSolverHook.known['sgd'] = ('phaser.engines.gradient.solvers:SGDSolver', SGDSolverPlan)
GradientSolverHook.known['adam'] = ('phaser.engines.gradient.solvers:AdamSolver', AdamSolverPlan)
GradientSolverHook.known['polyak_sgd'] = ('phaser.engines.gradient.solvers:PolyakSGDSolver', PolyakSGDSolverPlan)

EngineHook.known['conventional'] = ('phaser.engines.conventional.run:run_engine', ConventionalEnginePlan)
EngineHook.known['gradient'] = ('phaser.engines.gradient.run:run_engine', GradientEnginePlan)


class ReconsPlan(Dataclass, kw_only=True):
    """Reconstruction plan."""
    file_type: t.Literal['phaser_plan'] = 'phaser_plan'
    """File type identifier."""
    version: t.Annotated[str, IsVersion(exactly="1.0")] = "1.0"
    """Plan format version."""

    name: str
    """Name of the reconstruction."""
    out_dir: t.Optional[str] = "{name}"
    """Format string for the output directory. Key: `{name}`. `None` outputs to the current directory."""

    backend: t.Optional[BackendName] = None
    """Array backend. `None` selects the default backend."""
    device: t.Optional[str] = None
    """Compute device. `None` selects the first device of the backend."""
    dtype: t.Literal['float32', 'float64'] = 'float32'
    """Floating-point precision."""

    wavelength: t.Optional[float] = None
    """Radiation wavelength [angstrom]. Inferred from the raw data if `None`."""

    raw_data: RawDataHook
    """Hook to load the raw data."""

    post_load: t.Sequence[PostLoadHook] = ()
    """Hooks to run on the raw data after loading."""

    init: InitPlan = InitPlan()
    """Initialization of the probe, object, and scan."""

    post_init: t.Sequence[PostInitHook] = ()
    """Hooks to run after initialization, before the first engine."""

    slices: t.Optional[Slices] = None
    """Initial object slices. `None` for a single-slice object."""

    engines: t.List[EngineHook]
    """Engines to run in sequence."""
    #engines: t.List[t.Annotated[t.Union[ConventionalEngine, GradientEngine], Tagged('type')]]


__all__ = [
    'ReconsPlan', 'InitPlan', 'EnginePlan', 'ConventionalEnginePlan', 'GradientEnginePlan', 'EngineHook'
]