#!/usr/bin/env python3

from pathlib import Path
import functools
import sys
import os
import typing as t

import numpy
from numpy.typing import NDArray
import tifffile
from matplotlib import pyplot
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalRedisBackend
from optuna.samplers import TPESampler
from optuna.pruners import PercentilePruner
import pane

from phaser.utils.num import to_numpy, Sampling, get_backend_module
from phaser.utils.analysis import align_object_to_ground_truth
from phaser.utils.image import affine_transform
from phaser.utils.misc import unwrap
from phaser.plan import ReconsPlan, EngineHook
from phaser.state import PreparedRecons, ReconsState, PartialReconsState
from phaser.observer import Observer
from phaser.execute import execute_engine, initialize_reconstruction

base_dir = Path(__file__).parent.absolute()

STUDY_NAME = "mos2"
MEASURE_EVERY = 10
GROUND_TRUTH_PATH = base_dir / "../ground_truth_mos2_120kV_bwlim.tif"
assert GROUND_TRUTH_PATH.exists()


@functools.cache
def load_ground_truth() -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
    with tifffile.TiffFile(GROUND_TRUTH_PATH) as f:
        ground_truth = f.asarray()
        # grab ground truth pixel size
        ground_truth_sampling: NDArray[numpy.float64] = numpy.array(f.shaped_metadata[0]['spacing'])  # type: ignore

    # rotate ground truth (assumes periodic boundaries)
    theta = 0.0
    if theta:
        c, s = numpy.cos(theta * numpy.pi/180.), numpy.sin(theta * numpy.pi/180.)
        rot = numpy.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        ground_truth = affine_transform(ground_truth, rot, order=1, mode='grid-wrap')

    return (ground_truth, ground_truth_sampling)


def calc_error_mse(state: t.Union[ReconsState, PartialReconsState]) -> t.Tuple[float, NDArray[numpy.floating], NDArray[numpy.floating]]:
    object_state = unwrap(state.object)
    (ground_truth, ground_truth_sampling) = load_ground_truth()

    (upsamp_obj, ground_truth) = map(
        to_numpy, align_object_to_ground_truth(object_state, ground_truth, ground_truth_sampling)
    )

    error = numpy.sqrt(sum(
        float(numpy.nanmean((slice - ground_truth)**2))
        for slice in upsamp_obj
    ) / upsamp_obj.shape[0])

    return error, numpy.mean(upsamp_obj, axis=0), ground_truth


def calc_error_ssim(state: t.Union[ReconsState, PartialReconsState]) -> t.Tuple[float, NDArray[numpy.floating], NDArray[numpy.floating]]:
    from skimage.metrics import structural_similarity
    object_state = unwrap(state.object)
    (ground_truth, ground_truth_sampling) = load_ground_truth()

    (upsamp_obj, ground_truth) = map(
        to_numpy, align_object_to_ground_truth(object_state, ground_truth, ground_truth_sampling)
    )

    data_range = numpy.nanmax(ground_truth) - numpy.nanmin(ground_truth)
    ssim = float(numpy.mean(tuple(
        structural_similarity(
            slice, ground_truth, data_range=data_range,
            gaussian_weights=True, sigma=3.0,
        )
        for slice in upsamp_obj
    )))
    error = 1.0 - ssim

    return error, numpy.mean(upsamp_obj, axis=0), ground_truth


def plot_diff(obj: numpy.ndarray, ground_truth: numpy.ndarray, error: float, fname: t.Union[str, Path, None] = None):
    fig, (ax1, ax2) = pyplot.subplots(ncols=2, sharex=True, sharey=True, constrained_layout=True)
    fig.set_size_inches(8, 4)

    vmin = max(numpy.nanmin(obj).astype(float), numpy.nanmin(ground_truth).astype(float))
    vmax = max(numpy.nanmax(obj).astype(float), numpy.nanmax(ground_truth).astype(float))
    ax1.imshow(obj, cmap='Reds', alpha=0.5, vmin=vmin, vmax=vmax)
    ax1.imshow(ground_truth, cmap='Blues', alpha=0.5, vmin=vmin, vmax=vmax)

    diff = obj - ground_truth
    r = max(-numpy.nanmin(diff).astype(float), numpy.nanmax(diff).astype(float))
    sm = ax2.imshow(diff, cmap='bwr', vmin=-r, vmax=r)
    fig.colorbar(sm, shrink=0.9)

    fig.suptitle(f"Error: {error:.3e}", y=0.96)

    if fname is not None:
        fig.savefig(str(fname), dpi=400)
    else:
        pyplot.show()
    pyplot.close(fig)


class OptunaObserver(Observer):
    def __init__(self, trial: t.Optional[optuna.Trial] = None):
        self.trial: t.Optional[optuna.Trial] = trial
        self.trial_path = base_dir / f"trial{trial.number:04}" if trial is not None else base_dir
        self.last_error: float = 0.0

        if self.trial is not None:
            print(f"Parameters: {self.trial.params}", flush=True)
            self.trial_path.mkdir(exist_ok=True)
            os.chdir(self.trial_path)

        super().__init__()

    def save_json(self, plan: ReconsPlan, engines: t.Iterable[EngineHook]):
        import json

        if self.trial is None:
            return
        plan_json = t.cast(dict, plan.into_data())
        plan_json['engines'] = [pane.into_data(engine, EngineHook) for engine in engines]  # type: ignore

        with open(self.trial_path / 'plan.json', 'w') as f:
            json.dump(plan_json, f, indent=4)

    def update_iteration(self, state: ReconsState, i: int, n: int, error: t.Optional[float] = None):
        i = state.iter.total_iter

        if i % MEASURE_EVERY == 0:
            error, mean_obj, ground_truth = calc_error_ssim(state)
            self.last_error = error
            print(f"Realspace error: {error:.3e}", flush=True)
            if self.trial:
                self.trial.report(error, i)

            plot_diff(mean_obj, ground_truth, error, self.trial_path / f"iter{i:02}_error.png")

        if self.trial and self.trial.should_prune():
            raise optuna.TrialPruned()


# cache the initalization steps for efficiency
@functools.cache
def initialize() -> t.Tuple[PreparedRecons, ReconsPlan]:
    plan = ReconsPlan.from_data({
        "name": "mos2_grad",
        "backend": "jax",
        'dtype': 'float32',
        'raw_data': {
            'type': 'empad',
            'path': '../sample_data/simulated_mos2/mos2_0.00_dstep1.0.json',
        },
        'post_load': [
            {'type': 'poisson', 'scale': 6.09e+6},
        ],
        'post_init': [],
        'engines': [],
    })
    recons = initialize_reconstruction(plan)

    # pad reconstruction
    new_sampling = Sampling((192, 192), extent=tuple(recons.state.probe.sampling.extent))
    print(f"Resampling probe and patterns to shape {new_sampling.shape}...", flush=True)
    recons.state.probe.data = recons.state.probe.sampling.resample(recons.state.probe.data, new_sampling)
    recons.patterns.patterns = recons.state.probe.sampling.resample_recip(recons.patterns.patterns, new_sampling)
    recons.patterns.pattern_mask = recons.state.probe.sampling.resample_recip(recons.patterns.pattern_mask, new_sampling)
    recons.state.probe.sampling = new_sampling

    # store ReconsState on the cpu, we duplicate to GPU for each trial
    return (recons.to_numpy(), plan)


def objective(trial: optuna.Trial):
    (recons, plan) = initialize()
    xp = get_backend_module('jax')

    nesterov = trial.suggest_categorical('nesterov', ['false', 'true']) == 'true'
    grouping = trial.suggest_categorical('grouping', [4, 8, 16, 32, 64, 128, 256, 512])

    engine = pane.convert({
        'type': 'gradient',
        'probe_modes': 4,
        'niter': 150,
        'grouping': grouping,
        'noise_model': {'type': 'amplitude', 'eps': trial.suggest_float('noise_model_eps', 1.0e-4, 1.0e+1, log=True)},
        'solvers': {
            'object': {
                'type': 'adam',
                'learning_rate': trial.suggest_float('obj_learning_rate', 1.0e-4, 1.0e1, log=True),
                'nesterov': nesterov
            },
            'probe': {
                'type': 'adam',
                'learning_rate': trial.suggest_float('probe_learning_rate', 1.0e-4, 1.0e1, log=True),
                'nesterov': nesterov
            },
        },
        'regularizers': [
            {'type': 'obj_l2', 'cost': trial.suggest_float('obj_l2', 1.0e-5, 1.0e+5, log=True)},
            {'type': 'obj_l1', 'cost': trial.suggest_float('obj_l1', 1.0e-5, 1.0e+5, log=True)},
            {'type': 'obj_tikh', 'cost': trial.suggest_float('obj_tikh', 1.0e-5, 1.0e+5, log=True)},
        ],
        'iter_constraints': [],
        'group_constraints': [
            {'type': 'clamp_object_amplitude', 'amplitude': 1.1},
        ],
        'update_probe': {'after': 5},
        'update_object': True,
        'update_positions': False,
        'save': {'every': 10},
        'save_images': {'every': 10},
        'save_options': {
            'images': ['probe', 'probe_recip', 'object_phase_sum', 'object_mag_sum'],
        },
    }, EngineHook)

    observer = OptunaObserver(trial)
    observer.save_json(plan, [engine])

    recons = recons.with_observer(observer)
    recons.state = recons.state.to_xp(xp)
    recons = execute_engine(recons, engine)

    return observer.last_error


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} run|create|recreate|delete STORAGE_URL", file=sys.stderr)
        sys.exit(2)

    verb = sys.argv[1]
    storage = sys.argv[2]

    if storage.startswith("redis"):
        storage = JournalStorage(
            JournalRedisBackend(storage, use_cluster=False)
        )

    # `constant_liar` ensures efficient batch optimization
    sampler = TPESampler(constant_liar=True)
    # don't prune before iteration 20, don't prune at a given step unless we have at least 8 datapoints
    pruner = PercentilePruner(50.0, n_min_trials=8, n_warmup_steps=20)

    if verb in ('recreate', 'create'):
        if verb == 'recreate':
            optuna.delete_study(study_name=STUDY_NAME, storage=storage)
        study = optuna.create_study(
            study_name=STUDY_NAME, storage=storage,
            sampler=sampler, pruner=pruner,
        )
    elif verb == 'delete':
        optuna.delete_study(study_name=STUDY_NAME, storage=storage)
    elif verb == 'run':
        study = optuna.load_study(
            study_name=STUDY_NAME, storage=storage,
            sampler=sampler, pruner=pruner
        )
        study.optimize(objective, n_trials=200)
    else:
        print(f"Unknown command '{verb}'. Expected 'run', 'create', 'recreate', or 'delete'", file=sys.stderr)
        sys.exit(2)
