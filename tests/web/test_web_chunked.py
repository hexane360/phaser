import asyncio
import json
import typing as t

import pytest

from phaser.web.server import Job, Worker, server
from phaser.web.types import ValidationError

pytestmark = pytest.mark.web

CHUNK: int = 64


class FakeWorker(Worker):
    def worker_type(self) -> str:
        return 'fake'


def split(body: bytes) -> t.List[bytes]:
    return [body[i:i + CHUNK] for i in range(0, len(body), CHUNK)]


def test_chunks_reassemble_in_order_regardless_of_arrival():
    worker = FakeWorker('w-chunk')
    chunks = split(bytes(range(200)))

    assert worker.receive_chunk('u', 2, 4, chunks[2]) is None
    assert worker.receive_chunk('u', 0, 4, chunks[0]) is None
    assert worker.receive_chunk('u', 0, 4, chunks[0]) is None  # retried chunk
    assert worker.receive_chunk('u', 3, 4, chunks[3]) is None
    assert worker.receive_chunk('u', 1, 4, chunks[1]) == chunks
    assert worker.upload is None


def test_new_upload_replaces_an_abandoned_one():
    worker = FakeWorker('w-chunk')
    worker.receive_chunk('old', 0, 3, b'stale')
    assert worker.receive_chunk('new', 0, 2, b'a') is None
    assert worker.receive_chunk('new', 1, 2, b'b') == [b'a', b'b']


@pytest.mark.parametrize(('index', 'count'), [(-1, 2), (2, 2), (0, 0)])
def test_invalid_chunks_are_rejected(index: int, count: int):
    with pytest.raises(ValidationError):
        FakeWorker('w-chunk').receive_chunk('u', index, count, b'')


def test_count_cannot_change_mid_upload():
    worker = FakeWorker('w-chunk')
    worker.receive_chunk('u', 0, 3, b'a')
    with pytest.raises(ValidationError):
        worker.receive_chunk('u', 1, 4, b'b')


def test_chunked_update_reaches_the_job():
    job = Job('chunked-job', 'plan-json')
    worker = FakeWorker('w-route')
    server.jobs.inner[job.id] = job
    server.workers.inner[worker.id] = worker

    state = {'wavelength': 0.0197, 'padding': 'x' * 300}
    body = json.dumps({'msg': 'job_update', 'job_id': job.id, 'state': state}).encode('utf-8')
    chunks = split(body)
    assert len(chunks) > 2

    async def run() -> t.List[t.Any]:
        client = server.app.test_client()
        url = f'/worker/{worker.id}/update'
        return [
            await (await client.post(url, data=chunk, query_string={'upload': 'u', 'index': i, 'count': len(chunks)})).get_json()
            for (i, chunk) in enumerate(chunks)
        ]

    try:
        responses = asyncio.run(run())
    finally:
        del server.jobs.inner[job.id]
        del server.workers.inner[worker.id]

    assert all(resp == {'msg': 'ok'} for resp in responses)
    assert job.broker.cache.array('padding') == state['padding']
    assert job.broker.cache.array('wavelength') == state['wavelength']


def test_update_is_acknowledged_before_views_are_computed():
    import threading

    from phaser.web.pubsub import Session, View
    from phaser.web.types import UpdateMessage

    started, release = threading.Event(), threading.Event()
    seen: t.List[t.Any] = []

    def compute(cache: t.Any, params: t.Any) -> t.Any:
        seen.append(cache.raw['x'])
        started.set()
        release.wait(5.)
        return cache.raw['x']

    view = View(frozenset({'x'}), False, 'latest', compute)
    job = Job('publish-job', 'plan-json')

    async def run() -> None:
        await job.broker.subscribe(Session(), 'k', view, {})
        await job.handle_update(UpdateMessage.make_unchecked({'x': 1}, job.id))
        await asyncio.to_thread(started.wait, 5.)
        for x in (2, 3):
            # returns while the first publish is still blocked in `compute`
            await asyncio.wait_for(job.handle_update(UpdateMessage.make_unchecked({'x': x}, job.id)), 1.)
        release.set()
        while job._publisher is not None and not job._publisher.done():
            await asyncio.sleep(0.01)

    asyncio.run(run())
    # updates 2 and 3 arrive mid-publish, and are coalesced into one
    assert seen == [1, 3]


@pytest.mark.parametrize('body', [b'{not json', b'[1, 2]', b'\xff'])
def test_malformed_body_is_a_bad_request(body: bytes):
    worker = FakeWorker('w-route')
    server.workers.inner[worker.id] = worker

    async def run() -> int:
        return (await server.app.test_client().post(f'/worker/{worker.id}/update', data=body)).status_code

    try:
        assert asyncio.run(run()) == 400
    finally:
        del server.workers.inner[worker.id]
