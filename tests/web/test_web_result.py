import asyncio
import datetime
import logging
import typing as t

import pytest

from phaser.web.server import Job, _exc_summary, server
from phaser.web.types import JobResultMessage, JobStartMessage
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

    drive(job, errored(job.id))

    assert job.result == 'errored'
    assert job.error_summary == "ValueError: negative dimension in object sampling"
    assert job.logs.page(min_level=logging.ERROR).count == 1


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
