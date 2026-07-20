import asyncio
import json
import typing as t
import sys

from quart import Quart, render_template, request, Response, abort, websocket

import pane

from .types import JobID, ValidationError, WorkerID, WorkerMessage, UpdateMessage
from .types import ClientMessage, TopicUpdate, UpdatesMessage, ErrorMessage, OkResponse
from .pubsub import Session
from .server import server, Job, LocalWorker, ManualWorker, Shutdown, raise_on_shutdown


def serialize(obj: t.Any, ty: t.Any = None) -> bytes:
    return json.dumps(pane.into_data(obj, ty)).encode('utf-8')


def json_response(obj: t.Any, ty: t.Any = None, status: t.Optional[int] = None) -> Response:
    return Response(
        serialize(obj, ty),
        status=status,
        content_type='application/json',
    )


app: Quart = server.app

@app.get("/")
async def index():
    return await render_template("manager.html")

@app.post("/shutdown")
async def shutdown():
    async def shutdown():
        await asyncio.sleep(0.0)
        server.shutdown_event.set()

    server.futs.append(
        asyncio.create_task(shutdown())
    )

    return Response("", status=202)

@app.post("/worker/<string:worker_type>/start")
async def start_worker(worker_type: str):
    _ = await request.get_data()

    if worker_type not in ('manual', 'local', 'slurm'):
        abort(404)

    worker_id = server.make_workerid()

    if worker_type == 'manual':
        worker = ManualWorker(worker_id)
    elif worker_type == 'local':
        worker = LocalWorker(worker_id, server.get_worker_url(worker_id))
    elif worker_type == 'slurm':
        if sys.platform not in ('linux', 'darwin'):
            abort(Response(f"Slurm not supported on platform '{sys.platform}'", 400))
        try:
            await server.slurm_manager.check_slurm_exists()
        except RuntimeError as e:
            abort(Response(f"Slurm not available: {e}", 400))
        # TODO: this is hardcoded
        url = server.get_worker_url(worker_id).replace('localhost', '172.22.254.14')
        worker = await server.slurm_manager.make_worker(worker_id, url)

    await server.workers.add(worker)

    state = t.cast(t.Dict[str, t.Any], pane.into_data(worker.state()))
    if isinstance(worker, ManualWorker):
        state['message'] = f"Start worker using URL: {worker.url}"
    return json_response(state)

@app.post("/job/start")
async def start_job():
    body = await request.get_data()
    d = json.loads(body)
    source = d['source']

    if source == 'path':
        try:
            jobs = await Job.from_path(d['path'])
        except ValidationError as e:
            abort(json_response({'result': 'error', 'msg': e.msg}, status=400))
    elif source == 'yaml':
        try:
            jobs = await Job.from_yaml(d['data'])
        except ValidationError as e:
            abort(json_response({'result': 'error', 'msg': e.msg}, status=400))
    else:
        abort(Response(f"Unknown source type {source}", 400))

    return json_response({
        'result': 'success',
        'jobs': [job.state() for job in jobs],
    }, status=201)

@app.get("/job/<string:job_id>")
async def job_dashboard(job_id: JobID):
    if job_id == "fake":
        return await render_template("dashboard.html")
    if job_id not in server.jobs:
        abort(404)
    return await render_template("dashboard.html")

@app.post("/job/<string:job_id>/cancel")
async def cancel_job(job_id: JobID):
    try:
        job = server.jobs[job_id]
        await job.cancel()
    except KeyError:
        pass

    return json_response(OkResponse())

@app.post("/job/<string:job_id>/delete")
async def delete_job(job_id: JobID):
    try:
        job = server.jobs[job_id]
    except KeyError:
        abort(404)

    if job.status not in ('queued', 'stopped'):
        abort(Response("Cannot delete a running job", 400))
    await job.delete()

    return json_response(OkResponse())

@app.get("/job/<string:job_id>/logs")
async def job_logs(job_id: JobID):
    try:
        job = server.jobs[job_id]
    except KeyError:
        abort(404)

    limit = min(request.args.get('limit', 100, type=int), 100)
    before = request.args.get('before', len(job.logs), type=int)

    first = max(before-limit, 0)
    last = before - 1
    logs = job.logs[first:before]

    return json_response({
        'first': first,
        'last': last,
        'length': len(logs),
        'total_length': len(job.logs),
        'logs': logs,
    })

@app.post("/worker/<string:worker_id>/shutdown")
async def shutdown_worker(worker_id: WorkerID):
    try:
        worker = server.workers[worker_id]
        await worker.cancel()
    except KeyError:
        pass

    return json_response(OkResponse())

@app.post("/worker/<string:worker_id>/reload")
async def reload_worker(worker_id: WorkerID):
    try:
        worker = server.workers[worker_id]
        await worker.reload()
    except KeyError:
        pass

    return json_response(OkResponse())

@app.websocket("/listen")
async def listen():
    """Unified pub/sub endpoint, replacing the old per-manager/-dashboard websockets.
    An optional `?job=<id>` scopes the session: `default_topic={"job": id}` is merged
    into every dict topic the client subscribes to (default fills missing keys, an
    explicit client key wins). The dashboard opens `/listen?job=<id>` and subs
    abbreviated `{view: ...}` topics; the manager opens `/listen` and subs `"jobs"` /
    `"workers"`."""
    if (job_id := websocket.args.get('job')):
        try:
            _ = server.jobs[job_id]
        except KeyError:
            abort(Response("Invalid job ID", 400))
    await websocket.accept()

    session = Session(default_topic={"job": job_id} if job_id else None)

    async def recv():
        while True:
            data = await websocket.receive_json()
            msg = pane.convert(data, ClientMessage)  # type: ignore
            if msg.msg == 'sub':
                for topic in msg.topics:
                    await session.subscribe(topic)
            elif msg.msg == 'unsub':
                for topic in msg.topics:
                    session.unsubscribe(topic)

    async def send():
        while True:
            items = await session.mailbox.drain()
            updates = [item for item in items if isinstance(item, TopicUpdate)]
            errors = [item for item in items if isinstance(item, ErrorMessage)]
            if updates:
                await websocket.send(serialize(UpdatesMessage(updates)))
            for error in errors:
                await websocket.send(serialize(error))

    try:
        await asyncio.gather(send(), recv(), raise_on_shutdown())
    except Shutdown:
        pass
    finally:
        session.close()

@app.post("/worker/<string:worker_id>/update")
async def worker_update(worker_id: WorkerID):
    try:
        worker = server.workers[worker_id]
    except KeyError:
        abort(404)

    data = await request.json
    if data.get('msg') == 'job_update':
        # Bypass `ReconsStateConverter`'s eager `decode_obj`: keep `state` in wire-form
        # (still base64-encoded) so array fields are only ever decoded lazily, by
        # `Cache.array()`, for views that are actually subscribed (see pubsub.py).
        msg: WorkerMessage = UpdateMessage.make_unchecked(data['state'], data['job_id'])
    else:
        msg = pane.convert(data, WorkerMessage)  # type: ignore
    return json_response(await worker.handle_message(msg))