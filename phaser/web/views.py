"""View registry: numpy functions of `(Cache, params)` computing the data for each topic view."""
import math
import typing as t

import numpy

import pane

from .pubsub import Cache, View


def _state_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """One job's `JobState` -- status, terminal result, name, timings. Like `_jobs_view`
    below, this reads the owner directly rather than the cache; `'state'` is a synthetic
    dep bumped by `Job.notify_changed`."""
    from .server import server
    return pane.into_data(server.jobs[params['job']].state())


def _progress_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Loss curves, with non-finite values as `null`."""
    if (progress := cache.raw.get('progress')) is None:
        return None
    return {
        k: {**v, 'values': [None if isinstance(x, float) and not math.isfinite(x) else x for x in v['values']]}
        for (k, v) in progress.items()
    }


def _probes_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    # bulk array only -- shape and sampling belong to `probe_meta`.
    probe = cache.raw.get('probe')
    return probe['data'] if probe is not None else None


def _positions_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    # `ScanState.data` (shape (..., 2), in length units); the client flattens the leading axes.
    # `initial` and `tilt` ride along in the same payload, unused by this view.
    scan = cache.raw.get('scan')
    return scan['data'] if scan is not None else None


def _recip_probes(cache: Cache) -> t.Any:
    """The probe modes in reciprocal space. `fft2` un-centers real space before
    transforming (phaser's convention), and `fft2shift` centers the result."""
    from phaser.utils.num import fft2, fft2shift

    return fft2shift(fft2(cache.raw['probe']['data']))


def probes_recip_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Probe modes in reciprocal space, on the grid given by `probe_meta`."""
    return _recip_probes(cache)


def probe_sum_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Total probe intensity `(y, x)`, summed incoherently over modes."""
    from phaser.utils.num import abs2

    return numpy.sum(abs2(cache.raw['probe']['data']), axis=0)


def probe_sum_recip_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Total probe intensity in reciprocal space, summed incoherently over modes."""
    from phaser.utils.num import abs2

    return numpy.sum(abs2(_recip_probes(cache)), axis=0)


# `execute` reshapes a 2D object to a leading axis of length 1 (leaving `thicknesses`
# empty), so an object is normally already (z, y, x); a bare (y, x) one is accepted too.
# `slice_view` and `obj_meta_view` must agree on this, or the client's slice bound and the
# server's clamp disagree.
def _sampling(sampling: t.Mapping[str, t.Any]) -> t.Dict[str, t.Any]:
    """`sampling` with arrays as plain lists, which the client reads directly."""
    return {k: v.tolist() if isinstance(v, numpy.ndarray) else v for (k, v) in sampling.items()}


def _n_slices(shape: t.Sequence[int]) -> int:
    return int(shape[0]) if len(shape) > 2 else 1


def obj_meta_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Object metadata: sampling grid, slice count, and per-slice thicknesses (`None` if not per-slice)."""
    obj = cache.raw.get('object')
    if obj is None:
        return None

    n_slices = _n_slices(obj['data'].shape)
    # `ObjectState.thicknesses` is "length < 2 for single slice, equal to the number of
    # slices otherwise" -- length 0 from `execute`'s 2D normalization, or length 1 from a
    # re-used init state. Both mean "not per-slice", hence null.
    thicknesses = obj.get('thicknesses')
    per_slice = n_slices > 1 and thicknesses is not None and len(thicknesses) == n_slices

    return {
        'sampling': _sampling(obj['sampling']),
        'n_slices': n_slices,
        'thicknesses': [float(t) for t in thicknesses] if per_slice else None,
    }


def probe_meta_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Probe metadata: sampling grid, mode count, and wavelength. `None` until the probe and wavelength arrive."""
    probe = cache.raw.get('probe')
    wavelength = cache.raw.get('wavelength')
    if probe is None or wavelength is None:
        return None

    return {
        'sampling': _sampling(probe['sampling']),
        'nprobes': int(probe['data'].shape[0]),
        'wavelength': float(wavelength),
    }


def project_phase(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Object phase `(y, x)`, summed over all leading (slice) axes."""
    data = cache.raw['object']['data']
    axes = tuple(range(data.ndim - 2))
    return numpy.nansum(numpy.angle(data), axis=axes)


def project_amp_mean(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Object amplitude `(y, x)`, as the geometric mean over all leading (slice) axes."""
    data = cache.raw['object']['data']
    axes = tuple(range(data.ndim - 2))
    with numpy.errstate(divide='ignore'):
        return numpy.exp(numpy.nanmean(numpy.log(numpy.abs(data)), axis=axes))


def slice_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Return object slice `params['slice']`, clamped to the valid range."""
    data = cache.raw['object']['data']
    data = data.reshape((1, *data.shape)) if data.ndim == 2 else data

    idx = min(max(int(params.get('slice', 0)), 0), _n_slices(data.shape) - 1)
    return data[idx]


def _logs_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    # never actually invoked: `logs` is non-retained (no snapshot) and has no deps (so
    # it never matches a dirty-set intersection); live updates are pushed directly by
    # `Job.handle_update` via `Broker.publish_value`. Present for registry completeness.
    return []


def _jobs_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    from .server import server
    return pane.into_data([job.state() for job in server.jobs.values()])


def _workers_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    from .server import server
    return pane.into_data([worker.state() for worker in server.workers.values()])


VIEWS: t.Dict[str, View] = {
    'state':         View(frozenset({'state'}),    True,  'latest', _state_view),
    'progress':      View(frozenset({'progress'}), True,  'latest', _progress_view),
    'probes':        View(frozenset({'probe'}),    True,  'latest', _probes_view),
    'probes_recip':  View(frozenset({'probe'}),    True,  'latest', probes_recip_view),
    'probe_sum':     View(frozenset({'probe'}),    True,  'latest', probe_sum_view),
    'probe_sum_recip': View(frozenset({'probe'}),  True,  'latest', probe_sum_recip_view),
    'obj_phase_sum': View(frozenset({'object'}),   True,  'latest', project_phase),
    'obj_amp_mean':  View(frozenset({'object'}),   True,  'latest', project_amp_mean),
    'obj':           View(frozenset({'object'}),   True,  'latest', slice_view),
    'obj_meta':      View(frozenset({'object'}),   True,  'latest', obj_meta_view),
    'probe_meta':    View(frozenset({'probe', 'wavelength'}), True, 'latest', probe_meta_view),
    'positions':     View(frozenset({'scan'}),     True,  'latest', _positions_view),
    'logs':          View(frozenset(),              False, 'append', _logs_view),
}
"""Per-job views, by name."""

# manager-level owners (`server.jobs` / `server.workers`); not part of `VIEWS` since
# they're resolved by bare string topic (`"jobs"` / `"workers"`), not `{job, view}`.
JOBS_VIEW: View = View(frozenset({'state'}), True, 'latest', _jobs_view)
"""View for the `"jobs"` topic."""
WORKERS_VIEW: View = View(frozenset({'state'}), True, 'latest', _workers_view)
"""View for the `"workers"` topic."""
