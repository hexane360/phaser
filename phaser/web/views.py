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
from .util import encode_obj


def _status_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    return cache.raw.get('status')


def _progress_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    return cache.raw.get('progress')


def _probes_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    # already wire-form (encoded by the worker); pass through verbatim, no decode/encode.
    return cache.raw.get('probe')


def project_phase(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """Projected object phase: `angle` + `nansum` over every leading (slice) axis,
    collapsing an (..., y, x) complex object down to a single real (y, x) phase image.
    Mirrors the (now-retired) client-side `objectPhaseProjected` from `src/array.ts`."""
    obj = cache.array('object')
    data = obj['data']
    axes = tuple(range(data.ndim - 2))
    phase = numpy.nansum(numpy.angle(data), axis=axes)
    return encode_obj({'sampling': obj['sampling'], 'data': phase})


def slice_view(cache: Cache, params: t.Mapping[str, t.Any]) -> t.Any:
    """A single object slice, selected by the `slice` param. v1 decodes the whole object
    blob (accepted per the design doc); byte-range single-slice decode is a later
    optimization."""
    obj = cache.array('object')
    idx = int(params['slice'])
    thicknesses = obj.get('thicknesses')
    thickness = float(thicknesses[idx]) if thicknesses is not None and idx < len(thicknesses) else None
    return encode_obj({'sampling': obj['sampling'], 'data': obj['data'][idx], 'thickness': thickness})


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
    'status':        View(frozenset({'status'}),   True,  'latest', _status_view),
    'progress':      View(frozenset({'progress'}), True,  'latest', _progress_view),
    'probes':        View(frozenset({'probe'}),    True,  'latest', _probes_view),
    'obj_phase_sum': View(frozenset({'object'}),   True,  'latest', project_phase),
    'obj':           View(frozenset({'object'}),   True,  'latest', slice_view),
    'logs':          View(frozenset(),              False, 'append', _logs_view),
}

# manager-level owners (`server.jobs` / `server.workers`); not part of `VIEWS` since
# they're resolved by bare string topic (`"jobs"` / `"workers"`), not `{job, view}`.
JOBS_VIEW: View = View(frozenset({'state'}), True, 'latest', _jobs_view)
WORKERS_VIEW: View = View(frozenset({'state'}), True, 'latest', _workers_view)
