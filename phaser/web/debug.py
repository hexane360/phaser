"""
Testing aids for the web interface, registered only under `phaser serve --debug`.

Fake jobs feed synthetic log records through the same path a real worker uses
(`Job.handle_update`), so the `LogBuffer`, the broker and the `logs` topic all behave
exactly as in production -- which is the point: they exist to exercise the dashboard's
log pane (follow-the-tail, scroll-back paging, reconnect gap-fill) without a
reconstruction to generate the logs.
"""
import asyncio
import datetime
import logging
import random
import typing as t

from quart import Quart, Response, abort, request, url_for
from werkzeug.exceptions import HTTPException, default_exceptions

from .server import Job, server
from .types import JobID, JobStartMessage, LogMessage

_MESSAGES: t.Sequence[t.Tuple[int, str]] = (
    (logging.DEBUG, "gradient norm {i}: {value:.6f}"),
    (logging.INFO, "iteration {i} complete, error {value:.4f}"),
    (logging.INFO, "a deliberately long line to exercise wrapping: " + "lorem ipsum dolor sit amet " * 8),
    (logging.WARNING, "probe intensity drifted by {value:.2f}% at iteration {i}"),
    (logging.ERROR, "failed to converge at iteration {i}"),
)

_generators: t.Dict[JobID, asyncio.Task[None]] = {}


def make_log(job_id: JobID, i: int) -> LogMessage:
    """A synthetic log message; every ~50th carries a fake traceback"""
    level, template = _MESSAGES[i % len(_MESSAGES)]
    stack_info = "Traceback (most recent call last):\n  File \"fake.py\", line 1\n    raise RuntimeError" if i % 50 == 49 else None
    return LogMessage(
        # aware UTC, matching what a real worker's `from_logrecord` sends
        job_id, datetime.datetime.now(datetime.timezone.utc), template.format(i=i, value=random.random() * 100.),
        'phaser.debug', level, line_number=i, func_name='fake_job', stack_info=stack_info,
    )


async def emit_logs(job: Job, count: int, start: int = 0) -> None:
    for i in range(start, start + count):
        await job.handle_update(make_log(job.id, i))


def register_debug_routes(app: Quart) -> None:
    @app.post("/debug/job")
    async def debug_job():
        """Create a fake job emitting `rate` log records/second, stopping after `count`
        of them (0 for unbounded)."""
        rate = request.args.get('rate', 5., type=float)
        count = request.args.get('count', 0, type=int)
        if rate <= 0. or count < 0:
            abort(Response("Invalid query parameter", 400))

        job = Job(server.make_jobid(), plan='{}', name='fake job')
        # not queued: no worker should ever pick this up, it drives itself
        await server.jobs.add(job, queue=False)
        # stands in for the `job_start` a real worker sends, so records get a real `elapsed`
        await job.handle_update(JobStartMessage(job.id, datetime.datetime.now(datetime.timezone.utc)))
        await job.set_status('running')

        async def generate():
            # nobody awaits this task, so an exception would otherwise vanish silently
            try:
                i = 0
                while count == 0 or i < count:
                    await asyncio.sleep(1. / rate)
                    await job.handle_update(make_log(job.id, i))
                    i += 1
                await job.set_status('stopped')
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception(f"Log generator for job '{job.id}' failed")

        _generators[job.id] = asyncio.create_task(generate())

        return {'job_id': job.id, 'dashboard': url_for('job_dashboard', job_id=job.id)}

    @app.post("/debug/job/<string:job_id>/logs")
    async def debug_logs(job_id: JobID):
        """Append `count` records to a job as fast as possible, to test large prepends."""
        try:
            job = server.jobs[job_id]
        except KeyError:
            abort(404)

        count = request.args.get('count', 1000, type=int)
        if count < 0:
            abort(Response("Invalid query parameter", 400))

        await emit_logs(job, count, start=len(job.logs))
        return {'job_id': job.id, 'total': len(job.logs)}

    @app.post("/debug/job/<string:job_id>/stop")
    async def debug_stop(job_id: JobID):
        try:
            job = server.jobs[job_id]
        except KeyError:
            abort(404)

        if (task := _generators.pop(job_id, None)) is not None:
            task.cancel()
        await job.set_status('stopped')
        return {'job_id': job.id}

    @app.get("/debug/error/exception")
    async def debug_exception():
        """An unhandled exception, exercising the 500 path end to end."""
        raise RuntimeError("Deliberate exception from /debug/error/exception")

    @app.get("/debug/error/<int:code>")
    async def debug_error(code: int):
        """Raise an arbitrary HTTP error, to check how the error page renders it.
        `?description=` overrides the status's default text."""
        if not 400 <= code <= 599:
            abort(Response("Code must be between 400 and 599", 400))

        description = request.args.get('description')
        if code in default_exceptions:
            abort(code, description=description)

        e = HTTPException(description=description)
        e.code = code
        raise e

    @app.post("/debug/disconnect")
    async def debug_disconnect():
        """Drop every live `/listen` connection, simulating a network failure. Clients
        should reconnect, re-subscribe, and fill in whatever they missed."""
        sessions = list(server.sessions)
        for session in sessions:
            session.kicked.set()
        return {'disconnected': len(sessions)}

    logging.warning("Debug routes enabled (/debug/*)")
