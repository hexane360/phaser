import asyncio
import json
import typing as t
import sys

from quart import Quart, render_template, request, Response, abort, websocket

import pane

from .types import JobID, ValidationError, WorkerID, WorkerMessage
from .types import ManagerConnected, DashboardConnected, OkResponse
from .server import server, Job, LocalWorker, ManualWorker, Shutdown, raise_on_shutdown
from .util import merge_streams


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
        return await render_template("dashboard.html", title="Fake Job")
    if job_id not in server.jobs:
        abort(404)
    return await render_template("dashboard.html",  title=job_id)

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
        abort(Response(f"Cannot delete a running job", 400))
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
async def manager_websocket():
    await websocket.accept()

    await websocket.send(serialize(ManagerConnected(
        server.workers.state(), server.jobs.state()
    )))

    async def send():
        async for msg in merge_streams(
            server.workers.subscribe(),
            server.jobs.subscribe(),
        ):
            await websocket.send(serialize(msg))

    async def recv():
        while True:
            data = await websocket.receive_json()

    try:
        await asyncio.gather(send(), recv(), raise_on_shutdown())
    except Shutdown:
        pass

@app.websocket("/job/<string:job_id>/listen")
async def dashboard_websocket(job_id: JobID):
    try:
        job = server.jobs[job_id]
    except KeyError:
        abort(404)

    await websocket.accept()

    await websocket.send(serialize(DashboardConnected(
        job.state(full=True)
    )))

    async def send():
        async for msg in job.subscribe():
            await websocket.send(serialize(msg))

    try:
        await asyncio.gather(send(), raise_on_shutdown())
    except Shutdown:
        pass

@app.post("/worker/<string:worker_id>/update")
async def worker_update(worker_id: WorkerID):
    try:
        worker = server.workers[worker_id]
    except KeyError:
        abort(404)

    msg: WorkerMessage = pane.convert(await request.json, WorkerMessage)  # type: ignore
    return json_response(await worker.handle_message(msg))