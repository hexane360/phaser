"""
Broker/pub-sub unit tests (plan section 5, "Broker unit tests" + "Canonicalization
round-trip"). These exercise `Broker`/`Topic`/`Mailbox`/`Session` directly, without
going through `resolve()` or the Quart `server` singleton -- see
`test_resolve_and_job_wiring` at the bottom for an end-to-end check that does.
"""
import asyncio
import typing as t
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import pytest

from phaser.web.pubsub import Broker, Cache, Mailbox, Session, View
from phaser.web.types import ErrorMessage, TopicUpdate, canonical_topic

pytestmark = pytest.mark.web


def run(coro):
    return asyncio.run(coro)


def counting_view(**overrides) -> tuple[View, list[int]]:
    calls: list[int] = []

    def compute(cache: Cache, params: t.Mapping) -> int:
        calls.append(1)
        return cache.raw['x'] * 2

    defaults = dict(deps=frozenset({'x'}), retained=True, conflation='latest', compute=compute)
    defaults.update(overrides)
    return View(**defaults), calls  # type: ignore


def field_counting_view(field: str) -> tuple[View, list[int]]:
    """A retained view that reads `field` verbatim, counting how many times it's computed."""
    calls: list[int] = []

    def compute(cache: Cache, params: t.Mapping) -> t.Any:
        calls.append(1)
        return cache.raw[field]

    return View(frozenset({field}), True, 'latest', compute), calls


# --- canonicalization ---------------------------------------------------------------

def test_canonical_topic_sorts_keys_and_has_no_whitespace():
    assert canonical_topic('jobs') == '"jobs"'
    assert canonical_topic({'view': 'status', 'job': 'abc'}) == '{"job":"abc","view":"status"}'
    # key order in the input dict must not affect the canonical form
    assert canonical_topic({'job': 'abc', 'view': 'status'}) == canonical_topic({'view': 'status', 'job': 'abc'})


# --- subscribe -> snapshot, laziness, sharing ----------------------------------------

def test_subscribe_computes_snapshot_when_deps_present():
    view, calls = counting_view()
    broker = Broker()
    broker.cache.update_raw({'x': 5})
    session = Session()

    async def scenario():
        await broker.subscribe(session, 'k', view, {})
        return await session.mailbox.drain()

    items = run(scenario())
    assert items == [TopicUpdate('k', 10, None)]
    assert calls == [1]


def test_subscribe_no_snapshot_when_deps_missing():
    # retained view, but its dep field has never been populated -- no snapshot to send
    view, calls = counting_view()
    broker = Broker()
    session = Session()

    async def scenario():
        await broker.subscribe(session, 'k', view, {})
        assert session.mailbox.pending == {}
        assert calls == []

    run(scenario())


def test_unwatched_view_never_computed():
    _, calls = counting_view()
    broker = Broker()
    broker.cache.update_raw({'x': 1})
    # no subscribers at all -- publish_dirty should see no active topics to recompute
    run(broker.publish_dirty(frozenset({'x'})))
    assert calls == []


def test_two_sessions_share_one_compute():
    view, calls = counting_view()
    broker = Broker()
    broker.cache.update_raw({'x': 5})
    s1, s2 = Session(), Session()

    async def scenario():
        await broker.subscribe(s1, 'k', view, {})
        await broker.subscribe(s2, 'k', view, {})
        return await s1.mailbox.drain(), await s2.mailbox.drain()

    items1, items2 = run(scenario())
    assert items1 == items2 == [TopicUpdate('k', 10, None)]
    assert calls == [1]  # not [1, 1] -- computed once, shared


def test_update_propagates_only_to_dirty_topics():
    view_x, calls_x = field_counting_view('x')
    view_y, calls_y = field_counting_view('y')
    broker = Broker()
    broker.cache.update_raw({'x': 1, 'y': 1})
    sx, sy = Session(), Session()

    async def scenario():
        await broker.subscribe(sx, 'kx', view_x, {})
        await broker.subscribe(sy, 'ky', view_y, {})
        await sx.mailbox.drain()
        await sy.mailbox.drain()

        broker.cache.update_raw({'x': 2})
        await broker.publish_dirty(frozenset({'x'}))

        return await sx.mailbox.drain(), sy.mailbox.pending

    items_x, pending_y = run(scenario())
    assert items_x == [TopicUpdate('kx', 2, None)]
    assert pending_y == {}  # y-topic untouched: its dep didn't change
    assert calls_x == [1, 1]
    assert calls_y == [1]


def test_unsubscribe_mid_flight_stops_future_updates():
    view, _ = counting_view()
    broker = Broker()
    broker.cache.update_raw({'x': 1})
    s1, s2 = Session(), Session()

    async def scenario():
        await broker.subscribe(s1, 'k', view, {})
        await broker.subscribe(s2, 'k', view, {})
        await s1.mailbox.drain()
        await s2.mailbox.drain()

        broker.unsubscribe(s1, 'k')

        broker.cache.update_raw({'x': 2})
        await broker.publish_dirty(frozenset({'x'}))

    run(scenario())
    assert s1.mailbox.pending == {}
    assert s2.mailbox.pending != {}
    # the topic still exists (s2 still subscribed) so it wasn't dropped from active_topics
    assert 'k' in broker.active_topics


def test_unsubscribe_last_session_drops_the_topic():
    view, _ = counting_view()
    broker = Broker()
    broker.cache.update_raw({'x': 1})
    session = Session()

    async def scenario():
        await broker.subscribe(session, 'k', view, {})

    run(scenario())
    broker.unsubscribe(session, 'k')
    assert 'k' not in broker.active_topics


def test_broker_close_sends_error_and_clears_subscriptions():
    view, _ = counting_view()
    broker = Broker()
    broker.cache.update_raw({'x': 1})
    session = Session()

    async def scenario():
        await broker.subscribe(session, 'k', view, {})
        await session.mailbox.drain()
        broker.close("job 'j1' removed")
        return await session.mailbox.drain()

    items = run(scenario())
    assert items == [ErrorMessage('k', "job 'j1' removed")]
    assert 'k' not in session.subscriptions
    assert broker.active_topics == {}


def test_one_worker_update_yields_one_batched_drain():
    view_x, _ = counting_view(deps=frozenset({'x'}))
    view_y, _ = counting_view(deps=frozenset({'y'}), compute=lambda cache, params: cache.raw['y'])
    broker = Broker()
    broker.cache.update_raw({'x': 1, 'y': 1})
    session = Session()

    async def scenario():
        await broker.subscribe(session, 'kx', view_x, {})
        await broker.subscribe(session, 'ky', view_y, {})
        await session.mailbox.drain()

        # one "worker update" touches both x and y
        broker.cache.update_raw({'x': 2, 'y': 2})
        await broker.publish_dirty(frozenset({'x', 'y'}))
        return await session.mailbox.drain()

    items = run(scenario())
    # both dirtied topics land in a single drain() -> a single `update` message upstream
    assert {t.cast(str, u.topic) for u in items} == {'kx', 'ky'}
    assert len(items) == 2


# --- conflation -----------------------------------------------------------------------

def test_mailbox_latest_conflation_coalesces():
    mb = Mailbox()
    mb.put_update(TopicUpdate('status', 'a', None), 'latest')
    mb.put_update(TopicUpdate('status', 'b', None), 'latest')
    items = run(mb.drain())
    assert items == [TopicUpdate('status', 'b', None)]


def test_mailbox_append_conflation_never_drops_records():
    mb = Mailbox()
    mb.put_update(TopicUpdate('logs', [1, 2], None), 'append')
    mb.put_update(TopicUpdate('logs', [3], None), 'append')
    mb.put_update(TopicUpdate('logs', [4, 5], None), 'append')
    items = run(mb.drain())
    assert items == [TopicUpdate('logs', [1, 2, 3, 4, 5], None)]


def test_mailbox_drain_blocks_until_event_set():
    mb = Mailbox()

    async def scenario():
        drained: list = []

        async def drainer():
            drained.append(await mb.drain())

        task = asyncio.create_task(drainer())
        await asyncio.sleep(0)  # let drainer start waiting
        assert not task.done()
        mb.put_update(TopicUpdate('k', 1, None), 'latest')
        await task
        return drained

    [items] = run(scenario())
    assert items == [TopicUpdate('k', 1, None)]


# --- Session default-topic merge -------------------------------------------------------

def test_session_merge_default_fills_missing_keys():
    session = Session(default_topic={'job': 'A'})
    assert session._merge({'view': 'status'}) == {'job': 'A', 'view': 'status'}


def test_session_merge_explicit_client_key_wins():
    session = Session(default_topic={'job': 'A'})
    assert session._merge({'job': 'B', 'view': 'status'}) == {'job': 'B', 'view': 'status'}


def test_session_merge_string_topic_passes_through():
    session = Session(default_topic={'job': 'A'})
    assert session._merge('jobs') == 'jobs'


def test_session_merge_no_default_is_noop():
    session = Session()
    assert session._merge({'view': 'status'}) == {'view': 'status'}


def test_session_client_topic_for_echoes_original_form():
    from phaser.web.pubsub import _Subscription

    broker = Broker()
    session = Session(default_topic={'job': 'A'})

    # register a subscription the way `Session.subscribe` would (bypassing `resolve()`,
    # which needs the `server` singleton -- covered separately by `test_resolve_and_job_wiring`)
    key = canonical_topic({'job': 'A', 'view': 'status'})
    session.subscriptions[key] = _Subscription(broker, {'view': 'status'})

    assert session.client_topic_for(key) == {'view': 'status'}
    assert session.client_topic_for('nonexistent') == 'nonexistent'


# --- end-to-end: resolve() + Job/Jobs wiring through the real `server` singleton -------

def test_resolve_and_job_wiring():
    """Exercises the full path a real websocket session takes: `Session.subscribe`
    (which calls `resolve()`) -> `Broker.subscribe` -> a worker update ->
    `Job.handle_update` -> `Broker.publish_dirty` -> mailbox. Also confirms the
    `jobs`/`workers` manager views (which call `url_for`) work when computed off the
    threadpool via `quart.utils.run_sync` (see `Topic.value_async`)."""
    import numpy
    from phaser.web.server import server, Job, Jobs, Workers
    from phaser.web.util import encode_obj, decode_obj

    async def scenario():
        server.compute_pool = ThreadPoolExecutor()
        asyncio.get_running_loop().set_default_executor(server.compute_pool)
        server.workers = Workers()
        server.jobs = Jobs()
        server.job_queue = deque()
        server.app.config['SERVER_NAME'] = 'localhost'

        async with server.app.app_context():
            job = Job('pubsub-test-job', 'plan-json', 'test job')
            await server.jobs.add(job)
            try:
                data = numpy.zeros((2, 3, 4), dtype=numpy.complex64)
                data[1] = 1j  # angle = pi/2 on the second slice
                sampling = {'shape': [3, 4], 'sampling': [1.0, 1.0], 'corner': [0.0, 0.0], 'region_min': None, 'region_max': None}
                wire_state = {
                    'iter': {'engine_num': 1, 'engine_iter': 1, 'total_iter': 1, 'n_engine_iters': None, 'n_total_iters': None},
                    'object': encode_obj({'sampling': sampling, 'data': data, 'thicknesses': numpy.array([1.0, 1.0], dtype=numpy.float32)}),
                }

                class FakeMsg:
                    msg = 'job_update'
                    state = wire_state
                    job_id = job.id

                session = Session(default_topic={'job': job.id})
                await session.subscribe({'view': 'obj_phase_sum'})
                await session.subscribe({'view': 'obj_meta'})
                assert session.mailbox.pending == {}  # no snapshot yet: 'object' dep not populated

                await job.handle_update(FakeMsg())  # type: ignore

                # a meta topic and its bulk topic share the 'object' dep, so one
                # `publish_dirty` recomputes both and they drain in a single batch -- which
                # is what lets the client treat them as one consistent value
                drained = [t.cast(TopicUpdate, item) for item in await session.mailbox.drain()]
                items = {item.topic['view']: item for item in drained}
                assert set(items) == {'obj_phase_sum', 'obj_meta'}

                numpy.testing.assert_allclose(decode_obj(items['obj_phase_sum'].data), numpy.pi / 2, atol=1e-5)
                assert items['obj_meta'].data == {'sampling': sampling, 'n_slices': 2, 'thicknesses': [1.0, 1.0]}

                jobs_session = Session()
                await jobs_session.subscribe('jobs')
                jobs_items = await jobs_session.mailbox.drain()
                assert t.cast(TopicUpdate, jobs_items[0]).data[0]['job_id'] == job.id
                # `server.jobs` is a process-wide singleton, so a subscription left behind
                # here would have every later test's `notify_changed` recompute this view
                jobs_session.close()

                await server.jobs.remove(job.id)
                removed_items = await session.mailbox.drain()
                assert isinstance(removed_items[0], ErrorMessage)
            finally:
                server.compute_pool.shutdown(wait=False)

    asyncio.run(scenario())
