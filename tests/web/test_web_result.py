import asyncio
import datetime
import logging
import typing as t

import pytest

from phaser.web.server import Job, Worker, _exc_summary, server
from phaser.web.types import (
    RELOAD_EXIT_CODE,
    JobResponse,
    JobResultMessage,
    JobStartMessage,
    OkResponse,
    PollMessage,
    WorkerShutdownMessage,
)
from phaser.web.views import VIEWS
from phaser.web.worker import failure_log

pytestmark = pytest.mark.web

TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "recon.py", line 12, in run\n'
    "    solve(obj)\n"
    "ValueError: negative dimension in object sampling\n"
)


def make_job(job_id: str = 'result-test-job') -> Job:
    return Job(job_id, 'plan-json')


def drive(job: Job, *msgs: t.Any) -> None:
    async def run():
        for msg in msgs:
            await job.handle_update(msg)
    asyncio.run(run())


def raised(exc: BaseException) -> BaseException:
    """`exc` with a real `__traceback__`, so `failure_log` has frames to walk."""
    try:
        raise exc
    except BaseException as e:
        return e


def errored(job_id: str) -> JobResultMessage:
    log = failure_log(job_id, raised(ValueError("negative dimension in object sampling")), TRACEBACK)
    return JobResultMessage(job_id, 'errored', TRACEBACK, log)


def test_errored_result_is_retained_on_the_job():
    job = make_job()
    drive(job, errored(job.id))

    assert job.status == 'stopped'
    assert job.result == 'errored'
    assert job.error_summary == "ValueError: negative dimension in object sampling"


def test_traceback_lands_in_the_log_as_an_error_record():
    # the worker logs the traceback locally only, so the record it attaches to the result is
    # the only copy that reaches the server -- and what makes it reachable at all
    job = make_job()
    drive(job, errored(job.id))

    page = job.logs.page(min_level=logging.ERROR)
    assert page.count == 1
    record = page.logs[0]
    assert record.log_level == logging.ERROR
    assert record.stack_info == TRACEBACK
    assert record.log == "Job failed: ValueError: negative dimension in object sampling"


def test_failure_log_points_at_the_raising_frame():
    # not at the handler that caught it: the record should read like a `logger.error` placed
    # where the exception was raised
    def inner():
        raise ValueError("boom")

    try:
        inner()
    except ValueError as e:
        log = failure_log('job', e, TRACEBACK)

    assert log.func_name == 'inner'
    assert log.logger_name == __name__
    assert log.line_number == inner.__code__.co_firstlineno + 1
    assert log.log_level == logging.ERROR
    assert log.stack_info == TRACEBACK


def test_failure_log_survives_an_exception_with_no_traceback():
    log = failure_log('job', ValueError("never raised"), TRACEBACK)
    assert log.line_number == 0
    assert log.func_name is None
    assert log.log == "Job failed: ValueError: never raised"


def test_result_log_record_is_anchored_to_the_workers_clock():
    # `elapsed` is measured against the worker's clock, so the record's own timestamp -- not
    # the server's arrival time -- is what places it in the log
    job = make_job()
    msg = errored(job.id)
    assert msg.log is not None
    drive(job, JobStartMessage(job.id, msg.log.timestamp - datetime.timedelta(seconds=30)), msg)

    assert job.logs.records[-1].elapsed == 30.


def test_clean_results_record_no_traceback():
    for result in ('finished', 'cancelled', 'interrupted'):
        job = make_job()
        drive(job, JobResultMessage(job.id, result))

        assert job.result == result
        assert job.error_summary is None
        assert len(job.logs) == 0


def test_result_lands_even_when_the_job_already_stopped():
    # regression: `set_status` early-returns on an unchanged status, so a job whose worker
    # went away first (`Worker.set_status('stopped')` -> `Job.set_status('stopped')`) used
    # to swallow the `job_result` that followed, traceback and all
    job = make_job()
    asyncio.run(job.set_status('stopped'))
    assert job.result is None
    published = job.broker.cache.generation('state')

    drive(job, errored(job.id))

    assert job.result == 'errored'
    assert job.error_summary == "ValueError: negative dimension in object sampling"
    assert job.logs.page(min_level=logging.ERROR).count == 1
    # and it republished: the status didn't move, so only the `cause` makes this a change
    assert job.broker.cache.generation('state') > published


def test_state_view_reports_status_and_result():
    from phaser.web.server import Jobs

    job = make_job('state-view-job')

    async def run():
        server.jobs = Jobs()  # `server` is a process-wide singleton; don't inherit its state
        await server.jobs.add(job, queue=False)
        # `Job.state()` builds links with `url_for`, which needs a request context
        async with server.app.test_request_context('/'):
            view = VIEWS['state']
            assert view.deps == frozenset({'state'})

            queued = view.compute(job.broker.cache, {'job': job.id})
            assert queued['status'] == 'queued'
            assert queued['result'] is None

            await job.handle_update(errored(job.id))
            failed = view.compute(job.broker.cache, {'job': job.id})
            assert failed['status'] == 'stopped'
            assert failed['result'] == 'errored'
            assert failed['error_summary'] == "ValueError: negative dimension in object sampling"

    asyncio.run(run())


def test_queued_job_can_be_snapshotted_on_subscribe():
    # `'state'` is seeded in `Job.__init__`: a queued job has made no transition at all, and
    # without the seed `has_deps()` is false and a new subscriber gets no retained snapshot
    from phaser.web.pubsub import Topic

    job = make_job()
    topic = Topic('key', VIEWS['state'], {'job': job.id}, job.broker.cache)
    assert topic.has_deps()


@pytest.mark.parametrize(('error', 'expected'), [
    (None, None),
    ("", None),
    ("   \n  \n", None),
    (TRACEBACK, "ValueError: negative dimension in object sampling"),
    ("RuntimeError: boom\n\n\n", "RuntimeError: boom"),
])
def test_exc_summary(error: t.Optional[str], expected: t.Optional[str]):
    assert _exc_summary(error) == expected


def test_exc_summary_truncates():
    summary = _exc_summary("ValueError: " + "x" * 500, max_len=50)
    assert summary is not None
    assert len(summary) == 50
    assert summary.endswith('…')


# --- a job whose worker went away ------------------------------------------------------

class FakeWorker(Worker):
    """`Worker` is abstract only in `worker_type`; everything under test is on the base."""
    def worker_type(self) -> str:
        return 'fake'


def running_job_with_worker(job_id: str = 'lost-worker-job') -> t.Tuple[Job, FakeWorker]:
    import weakref
    job = make_job(job_id)
    worker = FakeWorker('w-1')
    worker.current_job = weakref.ref(job)
    asyncio.run(job.set_status('running'))
    return job, worker


def test_worker_lost_stops_the_job_and_records_why():
    # the job used to sit at `running` forever, with nothing said about why
    job = make_job()
    asyncio.run(job.set_status('running'))
    asyncio.run(job.worker_lost("Worker w-1 stopped unexpectedly (exit code -9)"))

    assert job.status == 'stopped'
    assert job.result == 'interrupted'
    assert job.error_summary == "Worker w-1 stopped unexpectedly (exit code -9)"

    page = job.logs.page(min_level=logging.ERROR)
    assert page.count == 1
    assert page.logs[0].log == "Worker w-1 stopped unexpectedly (exit code -9)"
    assert page.logs[0].func_name is None  # no source location; the UI omits the span


def test_worker_lost_does_not_overwrite_a_reported_result():
    # the normal order is `job_result` first, worker stop second
    job = make_job()
    drive(job, JobResultMessage(job.id, 'finished'))
    asyncio.run(job.worker_lost("Worker w-1 stopped"))

    assert job.result == 'finished'
    assert job.error_summary is None
    assert len(job.logs) == 0


def test_worker_stopping_carries_its_reason_onto_the_job():
    job, worker = running_job_with_worker()
    asyncio.run(worker.set_status('stopped', reason="Worker w-1 stopped unexpectedly (exit code -9)"))

    assert job.status == 'stopped'
    assert job.result == 'interrupted'
    assert job.error_summary == "Worker w-1 stopped unexpectedly (exit code -9)"


def test_shutdown_message_traceback_reaches_the_job():
    # `WorkerShutdownMessage.error` used to be discarded outright
    job, worker = running_job_with_worker('shutdown-msg-job')
    asyncio.run(worker.handle_message(WorkerShutdownMessage('errored', error=TRACEBACK)))

    assert worker.status == 'stopped'
    assert job.result == 'interrupted'
    assert job.error_summary == "ValueError: negative dimension in object sampling"
    assert job.logs.page(min_level=logging.ERROR).logs[0].stack_info == TRACEBACK


# --- what a worker is allowed to be handed ---------------------------------------------

def poll(worker: 'FakeWorker') -> t.Any:
    """A fresh worker's poll for work, driven the way the real one arrives."""
    return asyncio.run(worker.handle_message(PollMessage()))


def queued_jobs(*job_ids: str) -> t.List[Job]:
    """`job_ids` registered and queued against a clean `server`, in order."""
    from collections import deque

    from phaser.web.server import Jobs

    async def run():
        server.jobs = Jobs()  # `server` is a process-wide singleton; don't inherit its state
        server.job_queue = deque()
        jobs = [make_job(job_id) for job_id in job_ids]
        for job in jobs:
            await server.jobs.add(job)
        return jobs

    return asyncio.run(run())


def test_cancelled_job_is_not_handed_to_a_worker():
    # cancelling only moved the status; the job stayed in `server.job_queue` and the next
    # poll ran it anyway, putting a job the user stopped back to 'starting'
    (job,) = queued_jobs('cancelled-queued-job')
    asyncio.run(job.cancel())
    assert job.status == 'stopped'

    response = poll(FakeWorker('w-poll'))

    assert isinstance(response, OkResponse)
    assert job.status == 'stopped'
    assert len(server.job_queue) == 0


def test_deleted_job_is_skipped_but_the_one_behind_it_still_runs():
    # a deleted job left the queue holding a job no longer in `server.jobs`, which the
    # worker would then report against -- a `KeyError` in `Worker.handle_message`
    deleted, live = queued_jobs('deleted-queued-job', 'live-queued-job')
    asyncio.run(deleted.delete())
    assert deleted.id not in server.jobs

    worker = FakeWorker('w-poll')
    response = poll(worker)

    assert isinstance(response, JobResponse)
    assert response.job_id == live.id
    assert live.status == 'starting'
    assert worker.status == 'running'


def test_worker_goes_idle_when_every_queued_job_is_dead():
    (job,) = queued_jobs('only-cancelled-job')
    asyncio.run(job.cancel())

    worker = FakeWorker('w-poll')
    assert isinstance(poll(worker), OkResponse)
    assert worker.status == 'idle'
    assert worker.current_job is None


def test_reload_exit_code_matches_what_the_worker_exits_with():
    import inspect

    from phaser.web import worker as worker_mod

    assert RELOAD_EXIT_CODE == 128 + 1  # 128 + SIGHUP
    assert 'sys.exit(RELOAD_EXIT_CODE)' in inspect.getsource(worker_mod.run_worker)
