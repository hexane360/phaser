"""
Topic-based pub/sub broker.

A `Broker` owns one `Cache` (a job's or manager's raw wire-form state) plus the set of
currently-subscribed `Topic`s derived from it. Views are pure functions of the cache
(`phaser/web/views.py`), computed lazily (only while subscribed), memoized per tick by
dependency generation, and distributed to subscribing `Session`s' conflating `Mailbox`es.

See the design doc for the full picture; this module implements section 2 ("Server
architecture") minus the view registry (`views.py`) and the wiring into `Job`/`Jobs`/
`Worker`/`Workers` (`server.py`).
"""
from __future__ import annotations

import asyncio
import threading
import typing as t
from dataclasses import dataclass

from quart.utils import run_sync

from .types import ErrorMessage, JobID, TopicUpdate, canonical_topic
from .types import Topic as WireTopic
from .util import decode_obj

Conflation: t.TypeAlias = t.Literal['latest', 'append']


class Cache:
    """Accessor over an owner's (a `Job`'s, or a manager singleton's) raw wire-form
    state. `raw[field]` is the value as last received (wire-form, i.e. still encoded --
    see `phaser/web/util.py`'s `encode_obj`/`decode_obj`). `array(field)` decodes it,
    memoized by the field's generation, so a given update is only ever decoded once no
    matter how many views depend on it."""

    def __init__(self) -> None:
        self._raw: t.Dict[str, t.Any] = {}
        self._generations: t.Dict[str, int] = {}
        self._decoded: t.Dict[str, t.Tuple[int, t.Any]] = {}
        # `View.compute` runs in a worker thread and `Broker.publish_dirty` gathers every
        # dirty topic at once, so several threads can reach `array()` for one field
        # concurrently. Without this they'd each decode the same (potentially large)
        # blob, which is exactly what the memo exists to avoid.
        self._lock: threading.Lock = threading.Lock()

    @property
    def raw(self) -> t.Mapping[str, t.Any]:
        return self._raw

    def generation(self, field: str) -> int:
        return self._generations.get(field, 0)

    def update_raw(self, fields: t.Mapping[str, t.Any]) -> None:
        """Store new wire-form values and bump each field's generation. Values may be
        synthetic (e.g. `status`), not just worker-sent fields."""
        for k, v in fields.items():
            self._raw[k] = v
            self._generations[k] = self._generations.get(k, 0) + 1

    def array(self, field: str) -> t.Any:
        """Decode `raw[field]` (via `decode_obj`), memoized by generation. Thread-safe:
        concurrent callers for the same field wait, then hit the memo."""
        gen = self.generation(field)
        cached = self._decoded.get(field)
        if cached is not None and cached[0] == gen:
            return cached[1]

        with self._lock:
            # another thread may have decoded this generation while we waited
            cached = self._decoded.get(field)
            if cached is not None and cached[0] == gen:
                return cached[1]

            value = decode_obj(self._raw[field])
            self._decoded[field] = (gen, value)
            return value


class View(t.NamedTuple):
    deps: t.FrozenSet[str]
    """Raw cache fields this view reads. A worker/manager update only recomputes views
    whose deps intersect the changed field set."""
    retained: bool
    """If true, a snapshot is computed and sent on subscribe (replaces `Connected`)."""
    conflation: Conflation
    """`'latest'`: a slow session only ever sees the newest value for this topic.
    `'append'`: pending `data` lists are concatenated instead of replaced (logs)."""
    compute: t.Callable[[Cache, t.Mapping[str, t.Any]], t.Any]
    """Pure function of the cache + resolved params -> fully wire-ready JSON data (i.e.
    already run through `encode_obj` where needed -- `TopicUpdate.data` is never
    re-encoded)."""


class Topic:
    """One distinct subscribed topic on one owner (`Broker`). Holds the resolved `View`
    + params and memoizes `(dep-generation-tuple, value)` so recomputation only happens
    when a dependency actually changed."""

    def __init__(self, key: str, view: View, params: t.Mapping[str, t.Any], cache: Cache):
        self.key: str = key
        self.view: View = view
        self.params: t.Mapping[str, t.Any] = params
        self.cache: Cache = cache
        self.subscribers: t.Set[Session] = set()
        self._memo: t.Optional[t.Tuple[t.Tuple[int, ...], t.Any]] = None

    def _dep_generations(self) -> t.Tuple[int, ...]:
        return tuple(self.cache.generation(d) for d in sorted(self.view.deps))

    def has_deps(self) -> bool:
        """Whether every dependency has been populated at least once -- gates whether a
        snapshot can be computed on subscribe."""
        return all(d in self.cache.raw for d in self.view.deps)

    async def value_async(self) -> t.Any:
        gens = self._dep_generations()
        if self._memo is not None and self._memo[0] == gens:
            return self._memo[1]
        value = await run_sync(self.view.compute)(self.cache, self.params)
        self._memo = (gens, value)
        return value


MailboxItem: t.TypeAlias = t.Union[TopicUpdate, ErrorMessage]


class Mailbox:
    """A session's conflating outbox. Bounded by the number of distinct subscribed
    topics: `latest` overwrites pending data for that topic, `append` concatenates it.
    While a slow `ws.send` awaits, producers keep conflating into `pending` instead of
    growing an unbounded queue."""

    def __init__(self) -> None:
        self.pending: t.Dict[str, MailboxItem] = {}
        self.event: asyncio.Event = asyncio.Event()

    def put_update(self, update: TopicUpdate, conflation: Conflation) -> None:
        key = canonical_topic(update.topic)
        prev = self.pending.get(key)
        if conflation == 'append' and isinstance(prev, TopicUpdate):
            update = TopicUpdate(
                update.topic,
                [*prev.data, *update.data],
                update.cause if update.cause is not None else prev.cause,
            )
        self.pending[key] = update
        self.event.set()

    def put_error(self, topic: WireTopic, reason: str) -> None:
        self.pending[canonical_topic(topic)] = ErrorMessage(topic, reason)
        self.event.set()

    async def drain(self) -> t.List[MailboxItem]:
        await self.event.wait()
        items = list(self.pending.values())
        self.pending.clear()
        self.event.clear()
        return items


@dataclass
class _Subscription:
    owner: Broker
    client_topic: WireTopic


class Session:
    """One websocket connection. Owns a conflating `Mailbox`, an optional
    `default_topic` (from `?job=<id>`), and its subscriptions keyed by broker-canonical
    topic (for cleanup + rewriting outgoing topics back to the client's sent form)."""

    def __init__(self, default_topic: t.Optional[t.Mapping[str, t.Any]] = None):
        self.default_topic: t.Optional[t.Mapping[str, t.Any]] = default_topic
        self.mailbox: Mailbox = Mailbox()
        self.subscriptions: t.Dict[str, _Subscription] = {}
        self.kicked: asyncio.Event = asyncio.Event()
        """Set to drop this connection (`phaser/web/debug.py`), as an unexpected close"""

    def _merge(self, client_topic: WireTopic) -> WireTopic:
        """Merge rule: default fills missing keys, an explicit client key wins."""
        if self.default_topic is None or isinstance(client_topic, str):
            return client_topic
        return {**self.default_topic, **client_topic}

    def client_topic_for(self, key: str) -> WireTopic:
        sub = self.subscriptions.get(key)
        return sub.client_topic if sub is not None else key

    async def subscribe(self, client_topic: WireTopic) -> None:
        merged = self._merge(client_topic)
        try:
            owner, key, view, params = resolve(merged)
        except ResolveError as e:
            self.mailbox.put_error(client_topic, e.reason)
            return
        if key in self.subscriptions:
            return  # already subscribed; sub is idempotent
        self.subscriptions[key] = _Subscription(owner, client_topic)
        await owner.subscribe(self, key, view, params)

    def unsubscribe(self, client_topic: WireTopic) -> None:
        merged = self._merge(client_topic)
        key = canonical_topic(merged)
        sub = self.subscriptions.pop(key, None)
        if sub is not None:
            sub.owner.unsubscribe(self, key)

    def close(self) -> None:
        """Unsubscribe from everything (called when the websocket disconnects)."""
        for key, sub in list(self.subscriptions.items()):
            sub.owner.unsubscribe(self, key)
        self.subscriptions.clear()


class Broker:
    """Owns one `Cache` + the set of currently-subscribed `Topic`s for one publishable
    owner (a `Job`, or the `Jobs`/`Workers` manager singletons)."""

    def __init__(self) -> None:
        self.cache: Cache = Cache()
        self.active_topics: t.Dict[str, Topic] = {}

    async def subscribe(self, session: Session, key: str, view: View, params: t.Mapping[str, t.Any]) -> None:
        topic = self.active_topics.get(key)
        if topic is None:
            topic = Topic(key, view, params, self.cache)
            self.active_topics[key] = topic
        topic.subscribers.add(session)

        if view.retained and topic.has_deps():
            value = await topic.value_async()
            session.mailbox.put_update(TopicUpdate(session.client_topic_for(key), value, None), view.conflation)

    def unsubscribe(self, session: Session, key: str) -> None:
        topic = self.active_topics.get(key)
        if topic is None:
            return
        topic.subscribers.discard(session)
        if not topic.subscribers:
            del self.active_topics[key]

    def _distribute(self, topic: Topic, value: t.Any, cause: t.Any) -> None:
        for session in list(topic.subscribers):
            session.mailbox.put_update(
                TopicUpdate(session.client_topic_for(topic.key), value, cause), topic.view.conflation
            )

    async def publish_dirty(self, changed: t.FrozenSet[str], cause: t.Any = None) -> None:
        """Recompute + distribute every active topic whose deps intersect `changed`.
        Views are computed at most once per tick (memoized on `Topic`, shared across
        every subscribing session) and every dirtied topic for a given session lands in
        one conflating `Mailbox.put_update` burst -> one `update` message per tick."""
        dirty = [tp for tp in self.active_topics.values() if tp.view.deps & changed]
        if not dirty:
            return
        values = await asyncio.gather(*(tp.value_async() for tp in dirty))
        for tp, v in zip(dirty, values):
            self._distribute(tp, v, cause)

    def publish_value(self, key: str, value: t.Any, cause: t.Any = None) -> None:
        """Push an already-computed value directly to a topic's subscribers, bypassing
        the dep/generation memoization (used for `logs`, which is an append-only event
        stream rather than a function of generation-memoizable cache state)."""
        topic = self.active_topics.get(key)
        if topic is None:
            return
        self._distribute(topic, value, cause)

    def close(self, reason: str) -> None:
        """Notify every subscriber of every active topic that this owner is gone (e.g. a
        job was removed), then drop them."""
        for key, topic in list(self.active_topics.items()):
            for session in list(topic.subscribers):
                session.mailbox.put_error(session.client_topic_for(key), reason)
                session.subscriptions.pop(key, None)
        self.active_topics.clear()


class ResolveError(Exception):
    def __init__(self, reason: str):
        self.reason: str = reason
        super().__init__(reason)


def resolve(topic: WireTopic) -> t.Tuple[Broker, str, View, t.Mapping[str, t.Any]]:
    """Resolve a (already default-merged) topic to `(owner, canonical key, view,
    params)`. Imports `server`/`views` lazily to avoid a module-level import cycle
    (`server.py` imports `Broker` from this module at load time)."""
    from .server import server
    from .views import JOBS_VIEW, VIEWS, WORKERS_VIEW

    key = canonical_topic(topic)

    if topic == 'jobs':
        return server.jobs.broker, key, JOBS_VIEW, {}
    if topic == 'workers':
        return server.workers.broker, key, WORKERS_VIEW, {}
    if isinstance(topic, str):
        raise ResolveError(f"Unknown topic '{topic}'")

    if 'job' not in topic:
        raise ResolveError("Topic missing 'job' key")
    if 'view' not in topic:
        raise ResolveError("Topic missing 'view' key")

    job_id = t.cast(JobID, topic['job'])
    view_name = topic['view']

    if job_id not in server.jobs:
        raise ResolveError(f"Unknown job '{job_id}'")
    job = server.jobs[job_id]

    view = VIEWS.get(t.cast(str, view_name))
    if view is None:
        raise ResolveError(f"Unknown view '{view_name}'")

    # `job` is kept: a view over the owner itself (`state`) needs to know which job it is,
    # the same way the manager views reach into `server.jobs`/`server.workers`.
    params = {k: v for (k, v) in topic.items() if k != 'view'}
    return job.broker, key, view, params
