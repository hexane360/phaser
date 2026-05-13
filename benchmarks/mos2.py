#!/usr/bin/env python3

import itertools
import functools
import sys
import json
import typing as t

import numpy
import pane
import pynvml

from phaser.utils.num import get_backend_devices, get_backend_module, Sampling, set_default_device
from phaser.plan import ReconsPlan, EngineHook, BackendName
from phaser.state import PreparedRecons
from phaser.execute import execute_engine, initialize_reconstruction

N_WARMUP: int = 2


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def print_memory_usage(file=None):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory usage: {sizeof_fmt(info.used)}/{sizeof_fmt(info.total)}", file=file)


#@functools.lru_cache(1)
def initialize(sim_size: int = 128) -> t.Tuple[PreparedRecons, ReconsPlan]:
    plan = ReconsPlan.from_data({
        "name": "mos2_grad",
        "backend": "jax",
        'dtype': 'float32',
        'raw_data': {
            'type': 'empad',
            'path': '~/Downloads/mos2/1/mos2/mos2_0.00_dstep1.0.json',
        },
        'post_load': [
            {'type': 'poisson', 'scale': 1.0e6},
        ],
        'post_init': [],
        'engines': [],
    })

    recons = initialize_reconstruction(plan)

    if sim_size != 128:
        # pad reconstruction
        new_sampling = Sampling((sim_size, sim_size), extent=tuple(recons.state.probe.sampling.extent))
        print(f"Resampling probe and patterns to shape {new_sampling.shape}...", file=sys.stderr, flush=True)
        recons.state.probe.data = recons.state.probe.sampling.resample(recons.state.probe.data, new_sampling)
        recons.patterns.patterns = recons.state.probe.sampling.resample_recip(recons.patterns.patterns, new_sampling)
        recons.patterns.pattern_mask = recons.state.probe.sampling.resample_recip(recons.patterns.pattern_mask, new_sampling)
        recons.state.probe.sampling = new_sampling

    return (recons.to_numpy(), plan)


def benchmark_grad(
    grouping: int, sim_size: int, backend: BackendName,
    unroll: t.Union[int, bool] = 10,
) -> t.List[float]:
    (recons, plan) = initialize(sim_size)
    xp = get_backend_module(backend)
    recons = recons.to_xp(xp)

    devices = get_backend_devices(xp)
    print(f"Available devices: {list(devices)}", file=sys.stderr)
    print(f"Using device '{devices[0]}'", file=sys.stderr)
    set_default_device(devices[0], xp)

    engine = pane.convert({
        'type': 'gradient',
        'buffer_n_groups': 16 if grouping < 256 else 2,
        'jit_unroll_slices': unroll,
        'probe_modes': 4,
        'niter': 15,
        'grouping': grouping,
        'noise_model': {'type': 'amplitude', 'eps': 1.0e-4},
        'solvers': {
            'object': {
                'type': 'adam',
                'learning_rate': 1.0e-3,
                'nesterov': True,
            },
            'probe': {
                'type': 'adam',
                'learning_rate': 1.0e-3,
                'nesterov': True,
            },
        },
        'regularizers': [
        ],
        'iter_constraints': [
            {'type': 'clamp_object_amplitude', 'amplitude': 1.0},
        ],
        'group_constraints': [
        ],
        'update_probe': True,
        'update_object': True,
        'update_positions': False,
        'save': False, 'save_images': False,
    }, EngineHook)

    recons = execute_engine(recons, engine)

    iter_times: t.List[float] = numpy.diff(recons.state.progress['time'].values).tolist()[N_WARMUP:]

    print(f"Iter times: {iter_times}", file=sys.stderr)
    print(f"Mean time: {sum(iter_times) / len(iter_times):.3f} s", file=sys.stderr)
    return iter_times


if __name__ == '__main__':
    pynvml.nvmlInit()
    import jax

    device_name = jax.devices()[0].device_kind
    print(f"device: {device_name}", file=sys.stderr)

    backend = 'jax'

    for sim_size, unroll, grouping in itertools.product((128, 192), (5,), (8, 4, 16, 32, 64, 128, 256, 512, 1024)):
        if backend == 'jax':
            import jax.version
            backend_version = jax.version.__version__
        else:
            raise NotImplementedError()

        print(f"\nRunning grad, sim_size={sim_size} backend={backend!r} grouping={grouping}...", file=sys.stderr)
        print_memory_usage(file=sys.stderr)
        try:
            iter_times = benchmark_grad(grouping, sim_size, backend, unroll=unroll)
        except Exception as e:
            print(f"Failed to run, error:\n{e}", file=sys.stderr)
        else:
            json.dump({
                'engine': 'grad',
                'backend': backend,
                'backend_version': backend_version,
                'sim_size': sim_size,
                'n_positions': 64*64,
                'n_slices': 1,
                'n_modes': 4,
                'grouping': grouping,
                'device': device_name,
                'code': 'v5_unroll5',
                'iter_times': iter_times,
            }, sys.stdout)
            sys.stdout.write("\n")
            sys.stdout.flush()
