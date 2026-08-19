"""
View registry (`phaser/web/pubsub.py` section 2, "View registry"). Each `View.compute`
is a pure, numpy-only function of `(Cache, params) -> wire-ready JSON data`. Server-side
view compute is explicitly numpy-only per the design doc -- no backend-agnostic
requirement here, unlike the rest of `phaser`.
"""
import typing as t

import numpy

import pane

from .pubsub import Cache, View
from .util import decode_obj, encode_obj


def _state_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """One job's `JobState` -- status, terminal result, name, timings. Like `_jobs_view`
    below, this reads the owner directly rather than the cache; `'state'` is a synthetic
    dep bumped by `Job.notify_changed`."""
    from .server import server
    return pane.into_data(server.jobs[params['job']].state())


def _progress_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    return cache.raw.get('progress')


def _probes_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    # already wire-form (encoded by the worker); pass through verbatim, no decode/encode.
    # Bulk array only -- shape and sampling belong to `probe_meta`.
    probe = cache.raw.get('probe')
    return probe['data'] if probe is not None else None


def _positions_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    # `ScanState.data`, wire-form as sent (shape (..., 2), in length units). Passed
    # through verbatim like `_probes_view`; the client flattens the leading axes.
    # `initial` and `tilt` ride along in the same payload, unused by this view.
    scan = cache.raw.get('scan')
    return scan['data'] if scan is not None else None


def _recip_probes(cache: Cache) -> t.Any:
    """The probe modes in reciprocal space. `fft2` un-centers real space before
    transforming (phaser's convention), and `fft2shift` centers the result."""
    from phaser.utils.num import fft2, fft2shift

    return fft2shift(fft2(cache.array('probe')['data']))


def probes_recip_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """The probe modes in reciprocal space, on the (ky, kx) grid `probe_meta`'s
    `wavelength` and `sampling` describe."""
    return encode_obj(_recip_probes(cache))


def probe_sum_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Total probe intensity: `abs2` summed over the modes, giving a real (y, x) image.
    The modes are mutually incoherent, so they add in intensity -- which is also why this
    is the only meaningful reduction over them (a summed amplitude or phase is not)."""
    from phaser.utils.num import abs2

    return encode_obj(numpy.sum(abs2(cache.array('probe')['data']), axis=0))


def probe_sum_recip_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """`probe_sum` in reciprocal space -- the probe's total intensity distribution over
    scattering angle. The sum is taken there rather than transformed from `probe_sum`:
    the transform is per-mode, so the two don't commute."""
    from phaser.utils.num import abs2

    return encode_obj(numpy.sum(abs2(_recip_probes(cache)), axis=0))


# `execute` reshapes a 2D object to a leading axis of length 1 (leaving `thicknesses`
# empty), so an object is normally already (z, y, x); a bare (y, x) one is accepted too.
# `slice_view` and `obj_meta_view` must agree on this, or the client's slice bound and the
# server's clamp disagree.
def _n_slices(shape: t.Sequence[int]) -> int:
    return int(shape[0]) if len(shape) > 2 else 1


def obj_meta_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Everything about the object except its bulk array: the sampling grid, the slice
    count, and the per-slice thicknesses.

    Its own topic rather than a field on the object payloads, because those are per-slice
    (`obj`) and so change topic -- and momentarily lose their value -- whenever a different
    slice is selected. This one is stable for the lifetime of the run.

    Computed from `cache.raw` alone, and deliberately so: `encode_obj` stores an array as
    its `__array_interface__` plus base64 `data`, so the shape is readable without touching
    the payload. Reaching for `cache.array` here would decode the whole object.
    """
    obj = cache.raw.get('object')
    if obj is None:
        return None

    n_slices = _n_slices(obj['data']['shape'])
    # `ObjectState.thicknesses` is "length < 2 for single slice, equal to the number of
    # slices otherwise" -- length 0 from `execute`'s 2D normalization, or length 1 from a
    # re-used init state. Both mean "not per-slice", hence null.
    thicknesses = decode_obj(obj['thicknesses']) if obj.get('thicknesses') is not None else None
    per_slice = n_slices > 1 and thicknesses is not None and len(thicknesses) == n_slices

    return {
        'sampling': obj['sampling'],  # encoded with `to_numpy=False`: already wire-ready
        'n_slices': n_slices,
        'thicknesses': [float(t) for t in thicknesses] if per_slice else None,
    }


def probe_meta_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """The probe's sampling grid, mode count, and wavelength (the last for the reciprocal-
    space view's mrad scales). Split from `probes` for the same reason `obj_meta` is split
    from `obj` -- see there.

    `wavelength` is a field of `ReconsState`, not of the probe, but arrives with it: the
    worker's first update (`WorkerObserver.init_engine`) sends the whole state. Null until
    then, and the client treats that as "no probe yet"."""
    probe = cache.raw.get('probe')
    wavelength = cache.raw.get('wavelength')
    if probe is None or wavelength is None:
        return None

    return {
        'sampling': probe['sampling'],
        'nprobes': int(probe['data']['shape'][0]),
        'wavelength': float(wavelength),
    }


def project_phase(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Projected object phase: `angle` + `nansum` over every leading (slice) axis,
    collapsing an (..., y, x) complex object down to a single real (y, x) phase image.
    Mirrors the (now-retired) client-side `objectPhaseProjected` from `src/array.ts`."""
    data = cache.array('object')['data']
    axes = tuple(range(data.ndim - 2))
    return encode_obj(numpy.nansum(numpy.angle(data), axis=axes))


def project_amp_mean(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Projected object amplitude: `abs`, then the *geometric* mean over every leading
    (slice) axis, giving a real (y, x) image. The conjugate of `project_phase` -- slices
    compose multiplicatively in amplitude where they add in phase, so the geometric mean
    is the amplitude a single slice would need to produce the same total.
    """
    data = cache.array('object')['data']
    axes = tuple(range(data.ndim - 2))
    with numpy.errstate(divide='ignore'):
        return encode_obj(numpy.exp(numpy.nanmean(numpy.log(numpy.abs(data)), axis=axes)))


def slice_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """A single object slice, selected by the `slice` param. v1 decodes the whole object
    blob (accepted per the design doc); byte-range single-slice decode is a later
    optimization.

    `slice` is clamped, not validated, and that clamp is load-bearing: a widget's params
    outlive the run that set them, and the client can only correct an out-of-range one
    after `obj_meta` reaches it. An `IndexError` here would propagate out of the
    `asyncio.gather` in `Broker.publish_dirty` and take down that tick's publish for every
    topic on the job, not just this one.
    """
    data = cache.array('object')['data']
    data = data.reshape((1, *data.shape)) if data.ndim == 2 else data

    idx = min(max(int(params.get('slice', 0)), 0), _n_slices(data.shape) - 1)
    return encode_obj(data[idx])


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

# manager-level owners (`server.jobs` / `server.workers`); not part of `VIEWS` since
# they're resolved by bare string topic (`"jobs"` / `"workers"`), not `{job, view}`.
JOBS_VIEW: View = View(frozenset({'state'}), True, 'latest', _jobs_view)
WORKERS_VIEW: View = View(frozenset({'state'}), True, 'latest', _workers_view)
