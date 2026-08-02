import asyncio
import json
import logging
import sys
import typing as t

from quart import Quart, Response, abort, render_template, request, websocket
from quart.typing import ResponseReturnValue
from quart.utils import run_sync
from werkzeug.exceptions import HTTPException

import pane

from ..version import version_info
from .pubsub import Session
from .server import (
    Job,
    Kicked,
    LocalWorker,
    ManualWorker,
    Shutdown,
    raise_on_shutdown,
    server,
)
from .types import (
    ClientMessage,
    ErrorMessage,
    HeartbeatAckMessage,
    JobID,
    LogPage,
    LogRecord,
    OkResponse,
    ServerShutdownMessage,
    TopicUpdate,
    UpdateMessage,
    UpdatesMessage,
    ValidationError,
    WorkerID,
    WorkerMessage,
)


def serialize(obj: t.Any, ty: t.Any = None) -> bytes:
    return json.dumps(pane.into_data(obj, ty)).encode('utf-8')


def json_response(obj: t.Any, ty: t.Any = None, status: t.Optional[int] = None) -> Response:
    return Response(
        serialize(obj, ty),
        status=status,
        content_type='application/json',
    )


app: Quart = server.app


def wants_html() -> bool:
    """Whether the client is a browser navigating, rather than an API/XHR caller."""
    accept = request.accept_mimetypes
    return accept["text/html"] > accept["application/json"]


@app.errorhandler(HTTPException)
async def handle_http_error(e: HTTPException) -> ResponseReturnValue:
    # `abort(Response(...))`/`abort(json_response(...))` attach an explicit response;
    # pass those through so existing error paths keep their own bodies.
    if e.response is not None:
        # typed as the sansio base, but under Quart it's always a Quart `Response`
        return t.cast(Response, e.response)
    code = e.code or 500
    if not wants_html():
        return json_response({'result': 'error', 'msg': e.description}, status=code)
    return await render_template(
        "error.html", code=code, name=e.name, description=e.description
    ), code


@app.get("/")
async def index():
    return await render_template("manager.html")

@app.get("/version")
async def version():
    return json_response(version_info())

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
        abort(404, description=f"Unknown worker type '{worker_type}'")

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
    if job_id not in server.jobs and job_id != "fake":
        abort(404, description=(
            f"Job '{job_id}' not found. It may have been deleted, "
            "or the link may be out of date."
        ))
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
        abort(404, description=f"Job '{job_id}' not found")

    if job.status not in ('queued', 'stopped'):
        abort(Response("Cannot delete a running job", 400))
    await job.delete()

    return json_response(OkResponse())

@app.get("/job/<string:job_id>/logs")
async def job_logs(job_id: JobID):
    """A page of log records. `before`/`after` are exclusive cursors on a record's `i`,
    bounding the window on either side: both give the range between them (filled forwards
    from `after`), neither gives the newest `limit` records."""
    try:
        job = server.jobs[job_id]
    except KeyError:
        abort(404, description=f"Job '{job_id}' not found")

    before = request.args.get('before', type=int)
    after = request.args.get('after', type=int)
    limit = min(request.args.get('limit', 100, type=int), 1000)
    min_level = request.args.get('min_level', 0, type=int)

    if any(val is not None and val < 0 for val in (before, after, min_level)) or limit < 1:
        abort(Response("Invalid query parameter", 400))

    return json_response(job.logs.page(before, after, limit, min_level), LogPage)

@app.get("/job/<string:job_id>/logs.txt")
async def job_logs_text(job_id: JobID):
    """The job's complete log as plaintext, for download."""
    try:
        job = server.jobs[job_id]
    except KeyError:
        abort(404, description=f"Job '{job_id}' not found")

    min_level = request.args.get('min_level', 0, type=int)
    if min_level < 0:
        abort(Response("Invalid query parameter", 400))

    # formatting the whole log can be slow, so keep it off the event loop
    text = await run_sync(format_logs)(job.logs.filtered(min_level))

    return Response(text, content_type='text/plain; charset=utf-8', headers={
        'Content-Disposition': f'attachment; filename="phaser-{job_id}.log"',
    })


def format_logs(records: t.Iterable[LogRecord]) -> str:
    def format_log(record: LogRecord) -> str:
        level = logging.getLevelName(record.log_level)
        line = f"{record.timestamp:%Y-%m-%d %H:%M:%S} {level:<8} {record.logger_name}: {record.log}"
        return f"{line}\n{record.stack_info}" if record.stack_info else line

    return "".join(f"{format_log(record)}\n" for record in records)

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
    server.sessions.add(session)

    async def raise_on_kick():
        await session.kicked.wait()
        raise Kicked()

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
            elif msg.msg == 'ping':
                await websocket.send(serialize(HeartbeatAckMessage()))

    async def send():
        while True:
            items = await session.mailbox.drain()
            updates = [item for item in items if isinstance(item, TopicUpdate)]
            errors = [item for item in items if isinstance(item, ErrorMessage)]
            if updates:
                # serialize large objects off the event loop
                data = await run_sync(serialize)(UpdatesMessage(updates))
                await websocket.send(data)
            for error in errors:
                await websocket.send(serialize(error))

    try:
        await asyncio.gather(send(), recv(), raise_on_shutdown(), raise_on_kick())
    except Shutdown:
        await websocket.send(serialize(ServerShutdownMessage()))
    except Kicked:
        # deliberately silent: the client should see this as a dropped connection
        logging.info("Kicking websocket connection")
    finally:
        server.sessions.discard(session)
        session.close()

@app.post("/worker/<string:worker_id>/update")
async def worker_update(worker_id: WorkerID):
    try:
        worker = server.workers[worker_id]
    except KeyError:
        abort(404, description=f"Worker '{worker_id}' not found")

    data = await request.json
    if data.get('msg') == 'job_update':
        # Bypass `ReconsStateConverter`'s eager `decode_obj`: keep `state` in wire-form
        # (still base64-encoded) so array fields are only ever decoded lazily, by
        # `Cache.array()`, for views that are actually subscribed (see pubsub.py).
        msg: WorkerMessage = UpdateMessage.make_unchecked(data['state'], data['job_id'])
    else:
        msg = pane.convert(data, WorkerMessage)  # type: ignore
    return json_response(await worker.handle_message(msg))