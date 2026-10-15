"""HTTP and websocket routes."""
import asyncio
import json
import logging
import os
import shlex
import sys
import typing as t
from pathlib import Path

from markupsafe import Markup, escape
from quart import Quart, Response, abort, render_template, request, websocket
from quart.typing import ResponseReturnValue
from quart.utils import run_sync
from werkzeug.exceptions import HTTPException

import pane

from ..version import version_info
from . import frames
from .config import (
    SERVER_CONFIG,
    ServerConfig,
    apply_overrides,
    remote_worker_netloc,
    resolve_profile,
)
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
    SlurmInfo,
    SlurmLaunch,
    SlurmProfileInfo,
    TopicUpdate,
    UpdateMessage,
    UpdatesMessage,
    ValidationError,
    WorkerID,
    WorkerMessage,
)


def serialize(obj: t.Any, ty: t.Any = None) -> bytes:
    """Return `obj` as UTF-8 JSON, converted as type `ty` if given."""
    return json.dumps(pane.into_data(obj, ty)).encode('utf-8')


def frame(msg: t.Any) -> bytes:
    """Return `msg` as a binary frame for the browser (see `frames.py`), with strict JSON."""
    return frames.pack_bytes(pane.into_data(msg), allow_nan=False)


def json_response(obj: t.Any, ty: t.Any = None, status: t.Optional[int] = None) -> Response:
    """Return `obj` as a JSON `Response`."""
    return Response(
        serialize(obj, ty),
        status=status,
        content_type='application/json',
    )


app: Quart = server.app
"""The Quart app that routes register on."""


def wants_html() -> bool:
    """Return whether the client is a browser navigating, rather than an API/XHR caller."""
    if not request:  # not in request, websocket context
        return False
    accept = request.accept_mimetypes
    return accept["text/html"] > accept["application/json"]


@app.errorhandler(HTTPException)
async def handle_http_error(e: HTTPException) -> ResponseReturnValue:
    """Render an HTTP error as an HTML page for browsers, otherwise as JSON."""
    # `abort(Response(...))`/`abort(json_response(...))` attach an explicit response;
    # pass those through so existing error paths keep their own bodies.
    if e.response is not None:
        # typed as the sansio base, but under Quart it's always a Quart `Response`
        return t.cast(Response, e.response)
    code = e.code or 500
    # `Markup` renders as-is, anything else is escaped
    description = escape(e.description) if e.description else None
    if not wants_html():
        msg = description.striptags() if description else None
        return json_response({'result': 'error', 'msg': msg}, status=code)
    return await render_template(
        "error.html", code=code, name=e.name, description=description
    ), code


@app.get("/")
async def index():
    """Serve the job manager page."""
    return await render_template("manager.html")

@app.get("/version")
async def version():
    """Return the server version info."""
    return json_response(version_info())

# Entries of a single directory, for path completion in the manager. Names only: the client
# re-prefixes them with whatever it typed, so an absolute and a relative spelling of the same
# directory both complete without the server having to echo one back.
MAX_LS_ENTRIES = 500

def _ls_dir(dir: Path) -> t.List[str]:
    entries: t.List[str] = []
    try:
        with os.scandir(dir) as it:
            for entry in it:
                if entry.name.startswith('.'):
                    continue
                try:
                    if entry.is_dir():
                        entries.append(f"{entry.name}/")
                    elif entry.name.endswith(('.yaml', '.yml')):
                        entries.append(entry.name)
                except OSError:
                    # a broken symlink, or something we can't stat
                    continue
    except OSError:
        # missing, not a directory, or unreadable -- the usual state mid-word
        return []

    # directories first, so completing a tree doesn't mean hunting through the files
    entries.sort(key=lambda name: (not name.endswith('/'), name.lower()))
    return entries[:MAX_LS_ENTRIES]

@app.get("/ls_path")
async def ls_path():
    """List directories and YAML files in `?path=` for path completion.

    Responds 403 if the path is outside the server root.
    """
    dir = server.resolve_in_root(request.args.get('path', ''))
    if dir is None:
        abort(json_response({'result': 'error', 'msg': "Path is outside the server root"}, status=403))

    # `os.scandir` can block on a network mount, and this sits on the request path
    return json_response({'entries': await run_sync(_ls_dir)(dir)})

@app.post("/shutdown")
async def shutdown():
    """Shut down the server."""
    async def shutdown():
        await asyncio.sleep(0.0)
        server.shutdown_event.set()

    server.futs.append(
        asyncio.create_task(shutdown())
    )

    return Response("", status=202)

def _server_config() -> t.Tuple[ServerConfig, t.Optional[str]]:
    """The server config, plus a message if it couldn't be read.

    A broken config file leaves the rest of the server usable, so the message is reported to
    whoever asked rather than raised.
    """
    try:
        return (SERVER_CONFIG.get(), None)
    except Exception as e:
        return (ServerConfig(), f"Couldn't read config file '{SERVER_CONFIG.path()}': {e}")


def _remote_worker_url(worker_id: WorkerID, config: ServerConfig) -> str:
    """Worker URL reachable from another machine. Raises `ValueError` if there isn't one."""
    return server.get_worker_url(worker_id, remote_worker_netloc(server.host, config.worker_host))


@app.get("/slurm/profiles")
async def slurm_profiles():
    """Slurm availability and the configured worker profiles, for the launch form."""
    (config, error) = await run_sync(_server_config)()

    try:
        await server.slurm_manager.check_slurm_exists()
        available = True
    except RuntimeError as e:
        (available, error) = (False, error or str(e))

    if available and error is None:
        # surfaced before launching, since it's a server-config problem rather than a
        # problem with the request
        try:
            remote_worker_netloc(server.host, config.worker_host)
        except ValueError as e:
            error = str(e)

    if not (profiles := config.slurm_profiles):
        # with nothing configured, the built-in default profile is what a launch would use
        (name, profile) = resolve_profile(config)
        profiles = {name: profile}

    return json_response(SlurmInfo(
        available=available,
        version=server.slurm_manager.version,
        error=error,
        default_profile=resolve_profile(config)[0],
        profiles=[
            SlurmProfileInfo(
                name=name, description=profile.description,
                working_dir=profile.working_dir, output=profile.output,
                sbatch_args=shlex.join(profile.args()),
                preamble=profile.preamble, python=profile.python,
            )
            for (name, profile) in profiles.items()
        ],
    ))


async def _start_slurm_worker(worker_id: WorkerID, body: t.Union[str, bytes]):
    from .slurm import SlurmError

    if sys.platform not in ('linux', 'darwin'):
        abort(Response(f"Slurm not supported on platform '{sys.platform}'", 400))
    try:
        await server.slurm_manager.check_slurm_exists()
    except RuntimeError as e:
        abort(Response(f"Slurm not available: {e}", 400))

    try:
        launch = pane.from_data(json.loads(body), SlurmLaunch) if body.strip() else SlurmLaunch()
    except (ValueError, pane.ConvertError) as e:
        abort(Response(f"Invalid request: {e}", 400))

    (config, error) = await run_sync(_server_config)()
    if error is not None:
        abort(Response(error, 400))

    try:
        (name, profile) = resolve_profile(config, launch.profile)
        profile = apply_overrides(
            profile, launch.sbatch_args, launch.preamble, launch.python,
            launch.working_dir, launch.output,
        )
        url = _remote_worker_url(worker_id, config)
    except ValueError as e:
        abort(Response(str(e), 400))

    try:
        return await server.slurm_manager.make_worker(worker_id, url, profile, name)
    except SlurmError as e:
        logging.warning(f"Slurm rejected worker {worker_id}: {e}")
        abort(Response(f"Slurm rejected the job:\n{e}", 400))


@app.post("/worker/<string:worker_type>/start")
async def start_worker(worker_type: str):
    """Start a `manual`, `local`, or `slurm` worker and return its state.

    The body of a `slurm` request is an optional `SlurmLaunch` JSON object.
    """
    body = await request.get_data()

    if worker_type not in ('manual', 'local', 'slurm'):
        abort(404, description=f"Unknown worker type '{worker_type}'")

    worker_id = server.make_workerid()

    if worker_type == 'manual':
        # a manual worker may be started anywhere, so it gets a remotely-reachable URL when
        # one exists, and the local one otherwise
        (config, _) = await run_sync(_server_config)()
        try:
            url = _remote_worker_url(worker_id, config)
        except ValueError as e:
            logging.warning(f"Manual worker URL is local-only: {e}")
            url = server.get_worker_url(worker_id)
        worker = ManualWorker(worker_id, url)
    elif worker_type == 'local':
        worker = LocalWorker(worker_id, server.get_worker_url(worker_id))
    elif worker_type == 'slurm':
        worker = await _start_slurm_worker(worker_id, body)

    await server.workers.add(worker)

    state = t.cast(t.Dict[str, t.Any], pane.into_data(worker.state()))
    if isinstance(worker, ManualWorker):
        state['message'] = f"Start worker using URL: {worker.url}"
    return json_response(state)

@app.post("/job/start")
async def start_job():
    """Queue jobs from a plan, given as `{"source": "path", "path": ...}` or `{"source": "yaml", "data": ...}`.

    Responds 400 if the plan is invalid.
    """
    body = await request.get_data()
    d = json.loads(body)
    source = d['source']

    if source == 'path':
        path = server.file_root / Path(d['path']).expanduser()
        try:
            jobs = await Job.from_path(path)
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
    """Serve the dashboard page for job `job_id`."""
    if job_id not in server.jobs and job_id != "fake":
        abort(404, description=Markup(
            "Job <code>{}</code> not found. It may have been deleted, "
            "or the link may be out of date."
        ).format(job_id))
    return await render_template("dashboard.html")

@app.post("/job/<string:job_id>/cancel")
async def cancel_job(job_id: JobID):
    """Cancel job `job_id`. Ignores unknown jobs."""
    try:
        job = server.jobs[job_id]
        await job.cancel()
    except KeyError:
        pass

    return json_response(OkResponse())

@app.post("/job/<string:job_id>/delete")
async def delete_job(job_id: JobID):
    """Delete job `job_id`. Responds 400 if it is running."""
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
    """Return a page of job `job_id`'s log records (see `LogBuffer.page`).

    Query parameters: `before`, `after`, `limit` (at most 1000), `min_level`.
    """
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
    """Return job `job_id`'s log, filtered by `?min_level=`, as a plaintext download."""
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
    """Format `records` as plaintext, one per line, with stack info following."""
    def format_log(record: LogRecord) -> str:
        level = logging.getLevelName(record.log_level)
        line = f"{record.timestamp:%Y-%m-%d %H:%M:%S} {level:<8} {record.logger_name}: {record.log}"
        return f"{line}\n{record.stack_info}" if record.stack_info else line

    return "".join(f"{format_log(record)}\n" for record in records)

@app.post("/worker/<string:worker_id>/shutdown")
async def shutdown_worker(worker_id: WorkerID):
    """Shut down worker `worker_id`. Ignores unknown workers."""
    try:
        worker = server.workers[worker_id]
        await worker.cancel()
    except KeyError:
        pass

    return json_response(OkResponse())

@app.post("/worker/<string:worker_id>/reload")
async def reload_worker(worker_id: WorkerID):
    """Reload worker `worker_id`. Ignores unknown workers."""
    try:
        worker = server.workers[worker_id]
        await worker.reload()
    except KeyError:
        pass

    return json_response(OkResponse())

@app.websocket("/listen")
async def listen():
    """Pub/sub websocket.

    `?job=<id>` sets the session's default topic `{"job": id}`, which fills missing keys of
    dict topics. Clients send `sub`, `unsub`, and `ping` messages. Responds 400 for an
    unknown job.
    """
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
                await websocket.send(frame(HeartbeatAckMessage()))

    async def send():
        while True:
            items = await session.mailbox.drain()
            updates = [item for item in items if isinstance(item, TopicUpdate)]
            errors = [item for item in items if isinstance(item, ErrorMessage)]
            if updates:
                # serialize large objects off the event loop
                try:
                    data = await run_sync(frame)(UpdatesMessage.make_unchecked(updates))
                except (TypeError, ValueError) as e:
                    logging.exception("Failed to encode topic updates")
                    errors.extend(ErrorMessage(update.topic, f"Failed to encode update: {e}") for update in updates)
                else:
                    await websocket.send(data)
            for error in errors:
                await websocket.send(frame(error))

    # `gather` propagates the first exception but leaves its siblings running, so the four
    # are held explicitly and cancelled together -- otherwise a kicked connection strands
    # `send`, blocked on a mailbox nothing will fill again, for the life of the process.
    tasks = [asyncio.ensure_future(coro) for coro in
             (send(), recv(), raise_on_shutdown(), raise_on_kick())]
    try:
        await asyncio.gather(*tasks)
    except Shutdown:
        await websocket.send(frame(ServerShutdownMessage()))
    except Kicked:
        # deliberately silent: the client should see this as a dropped connection
        logging.info("Kicking websocket connection")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        server.sessions.discard(session)
        session.close()

async def _unpack(body: bytes) -> t.Dict[str, t.Any]:
    """Unpack a worker message frame (off the event loop), aborting with a 400 if malformed."""
    try:
        data = await run_sync(frames.unpack)(body)
    except ValueError as e:
        abort(400, description=f"Invalid frame: {e}")
    if not isinstance(data, dict):
        abort(400, description="Expected a message object")
    return t.cast(t.Dict[str, t.Any], data)


@app.post("/worker/<string:worker_id>/update")
async def worker_update(worker_id: WorkerID):
    """Handle a worker message frame and return the server's response.

    With `?upload=`, `?index=`, and `?count=`, the body is one chunk of a larger message.
    """
    try:
        worker = server.workers[worker_id]
    except KeyError:
        abort(404, description=f"Worker '{worker_id}' not found")

    body = t.cast(bytes, await request.get_data(as_text=False))
    if (upload_id := request.args.get('upload')) is not None:
        # one chunk of an `UpdateMessage` body
        try:
            index, count = int(request.args['index']), int(request.args['count'])
        except (KeyError, ValueError):
            abort(400, description="Chunk requires integer 'index' and 'count'")
        try:
            chunks = worker.receive_chunk(upload_id, index, count, body)
        except ValidationError as e:
            abort(400, description=e.msg)
        if chunks is None:
            return json_response(OkResponse())
        data = await _unpack(b''.join(chunks))
    else:
        data = await _unpack(body)

    if data.get('msg') == 'job_update':
        # skip validation, which would convert arrays to lists
        msg: WorkerMessage = UpdateMessage.make_unchecked(data['state'], data['job_id'])
    else:
        msg = pane.convert(data, WorkerMessage)  # type: ignore
    return json_response(await worker.handle_message(msg))