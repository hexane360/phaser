import asyncio
import datetime
import logging
import typing as t

import pytest

from phaser.web.server import Job, LogBuffer, snap_level
from phaser.web.types import JobStartMessage, LogMessage, LogRecord

pytestmark = pytest.mark.web
LEVELS = (logging.DEBUG, logging.INFO, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)


def make_buffer(n: int = 50) -> LogBuffer:
    buf = LogBuffer()
    for i in range(n):
        buf.append(LogRecord(
            i, datetime.datetime(2026, 1, 1) + datetime.timedelta(seconds=i),
            f"message {i}", 'phaser.test', LEVELS[i % len(LEVELS)], line_number=i,
        ))
    return buf


def page_indices(buf: LogBuffer, **kwargs: t.Any) -> t.List[int]:
    return [record.i for record in buf.page(**kwargs).logs]


def matching(buf: LogBuffer, min_level: int) -> t.List[int]:
    return [record.i for record in buf.records if record.log_level >= min_level]


def test_tail_page():
    buf = make_buffer(50)
    page = buf.page(limit=10)

    assert [record.i for record in page.logs] == list(range(40, 50))
    assert (page.first, page.last, page.count) == (40, 49, 10)
    assert (page.total, page.total_all, page.oldest) == (50, 50, 0)
    assert (page.has_before, page.has_after) == (True, False)


def test_empty_buffer():
    page = LogBuffer().page()

    assert page.logs == []
    assert (page.first, page.last, page.count, page.total) == (None, None, 0, 0)
    assert (page.has_before, page.has_after) == (False, False)


def test_cursors_are_exclusive_and_tile_the_log():
    buf = make_buffer(50)

    assert page_indices(buf, before=10, limit=4) == [6, 7, 8, 9]
    assert page_indices(buf, after=10, limit=4) == [11, 12, 13, 14]

    # page backwards from the tail; the pages should tile the log exactly
    seen: t.List[int] = []
    page = buf.page(limit=7)
    while True:
        seen = [record.i for record in page.logs] + seen
        if not page.has_before:
            break
        page = buf.page(before=page.first, limit=7)

    assert seen == list(range(50))


def test_forward_paging_tiles_the_log():
    buf = make_buffer(50)

    seen: t.List[int] = []
    page = buf.page(after=-1, limit=7)
    while True:
        seen.extend(record.i for record in page.logs)
        if not page.has_after:
            break
        page = buf.page(after=page.last, limit=7)

    assert seen == list(range(50))


def test_both_cursors_give_the_range_between_them():
    buf = make_buffer(50)

    # exclusive on both ends
    page = buf.page(before=15, after=10, limit=100)
    assert [record.i for record in page.logs] == [11, 12, 13, 14]
    assert (page.has_before, page.has_after) == (False, False)

    # `limit` truncates the newer end, so the client can keep filling forwards
    page = buf.page(before=20, after=10, limit=3)
    assert [record.i for record in page.logs] == [11, 12, 13]
    assert (page.has_before, page.has_after) == (False, True)

    # an empty or inverted range yields nothing
    assert buf.page(before=11, after=10).logs == []
    assert buf.page(before=5, after=40).logs == []


def test_gap_fill_closes_exactly():
    buf = make_buffer(50)
    held_last, newer_first = 10, 40  # what the client holds either side of an outage

    seen: t.List[int] = []
    while True:
        page = buf.page(before=newer_first, after=seen[-1] if seen else held_last, limit=7)
        seen.extend(record.i for record in page.logs)
        if not page.has_after:
            break

    assert seen == list(range(11, 40))


def test_end_flags():
    """`has_before`/`has_after` describe the range the cursors asked for, not the whole log"""
    buf = make_buffer(50)

    # `before=1` asks for everything older than record 1; that range is exhausted
    assert (buf.page(before=1).has_before, buf.page(before=1).has_after) == (False, False)
    assert (buf.page(after=48).has_before, buf.page(after=48).has_after) == (False, False)
    assert (buf.page(limit=100).has_before, buf.page(limit=100).has_after) == (False, False)

    # truncated by `limit`, so the un-anchored end continues
    assert (buf.page(limit=10).has_before, buf.page(limit=10).has_after) == (True, False)
    assert (buf.page(after=0, limit=10).has_before, buf.page(after=0, limit=10).has_after) == (False, True)
    assert (buf.page(before=40, limit=10).has_before, buf.page(before=40, limit=10).has_after) == (True, False)


@pytest.mark.parametrize('min_level', (0, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL))
def test_filter_matches_reference(min_level: int):
    buf = make_buffer(50)
    expected = matching(buf, min_level)

    page = buf.page(limit=1000, min_level=min_level)
    assert [record.i for record in page.logs] == expected
    assert (page.total, page.total_all) == (len(expected), 50)
    assert (page.has_before, page.has_after) == (False, False)
    assert [record.i for record in buf.filtered(min_level)] == expected


def test_filtered_paging_tiles_matching_records():
    buf = make_buffer(50)
    expected = matching(buf, logging.WARNING)

    seen: t.List[int] = []
    page = buf.page(limit=2, min_level=logging.WARNING)
    while True:
        seen = [record.i for record in page.logs] + seen
        if not page.has_before:
            break
        page = buf.page(before=page.first, limit=2, min_level=logging.WARNING)

    assert seen == expected


UTC = datetime.timezone.utc


def log_message(job_id: str, timestamp: datetime.datetime) -> LogMessage:
    return LogMessage(job_id, timestamp, 'message', 'phaser.test', logging.INFO, line_number=1)


async def elapsed_for(start: t.Optional[datetime.datetime], stamps: t.Sequence[datetime.datetime]) -> t.List[float]:
    """`elapsed` stamped onto records logged at `stamps`, for a job whose worker reported
    `start` (or never reported one, when `None`)"""
    job = Job('elapsed-test-job', 'plan-json')
    if start is not None:
        await job.handle_update(JobStartMessage(job.id, start))
    for stamp in stamps:
        await job.handle_update(log_message(job.id, stamp))
    return [record.elapsed for record in job.logs]


def test_elapsed_measured_from_worker_start():
    start = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    stamps = [start + datetime.timedelta(seconds=s) for s in (0., 0.5, 90.25)]

    assert asyncio.run(elapsed_for(start, stamps)) == [0., 0.5, 90.25]


def test_elapsed_falls_back_to_the_first_record():
    """No reported start (a failed `job_start`): the first record becomes t=0"""
    base = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    stamps = [base + datetime.timedelta(seconds=s) for s in (10., 11.5, 100.)]

    assert asyncio.run(elapsed_for(None, stamps)) == [0., 1.5, 90.]


def test_elapsed_falls_back_on_naive_aware_mismatch():
    """A worker predating timezone-aware log timestamps: anchored on its own first
    record (which shares its clock), never a `TypeError`"""
    aware = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    naive = [datetime.datetime(2026, 1, 1, 12, 0, s) for s in (0, 30)]

    assert asyncio.run(elapsed_for(aware, naive)) == [0., 30.]


def test_level_snapping():
    buf = make_buffer(50)

    assert snap_level(0) == 0
    assert snap_level(5) == 0
    assert snap_level(25) == logging.INFO
    assert snap_level(100) == logging.CRITICAL

    # a non-standard level filters as the standard level below it, and says so
    page = buf.page(limit=1000, min_level=25)
    assert page.min_level == logging.INFO
    assert [record.i for record in page.logs] == matching(buf, logging.INFO)
