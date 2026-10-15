"""Topic-based pub/sub broker: `Broker`, `Session`, and the views they serve (`views.py`)."""
from __future__ import annotations

import asyncio
import typing as t
from dataclasses import dataclass

from quart.utils import run_sync

from .types import ErrorMessage, JobID, TopicUpdate, canonical_topic
from .types import Topic as WireTopic

Conflation: t.TypeAlias = t.Literal['latest', 'append']
"""How pending updates to one topic combine in a `Mailbox`."""


class Cache:
    """Latest state of a `Job` or manager, as fields with generation counters."""

    def __init__(self) -> None:
        self._raw: t.Dict[str, t.Any] = {}
        self._generations: t.Dict[str, int] = {}

    @property
    def raw(self) -> t.Mapping[str, t.Any]:
        """Field values as last received (arrays are read-only views into the received frame)."""
        return self._raw

    def generation(self, field: str) -> int:
        """Return the number of updates to `field`."""
        return self._generations.get(field, 0)

    def update_raw(self, fields: t.Mapping[str, t.Any]) -> None:
        """Store `fields` and bump each field's generation."""
        for k, v in fields.items():
            self._raw[k] = v
            self._generations[k] = self._generations.get(k, 0) + 1


class View(t.NamedTuple):
    """A topic's computation from cache fields to data."""
    deps: t.FrozenSet[str]
    """Cache fields this view reads. An update recomputes it only if one of these changed."""
    retained: bool
    """Whether a snapshot is sent on subscribe."""
    conflation: Conflation
    """`'latest'` keeps only the newest value; `'append'` concatenates pending `data` lists."""
    compute: t.Callable[[Cache, t.Mapping[str, t.Any]], t.Any]
    """Pure function of the cache and topic params, returning JSON-like data with numpy arrays as leaves."""


class Topic:
    """A subscribed topic on one `Broker`, with its memoized view value."""

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
        """Return whether every dependency of the view has been populated."""
        return all(d in self.cache.raw for d in self.view.deps)

    async def value_async(self) -> t.Any:
        """Return the view value, recomputed (in a thread) only if a dependency changed."""
        gens = self._dep_generations()
        if self._memo is not None and self._memo[0] == gens:
            return self._memo[1]
        value = await run_sync(self.view.compute)(self.cache, self.params)
        self._memo = (gens, value)
        return value


MailboxItem: t.TypeAlias = t.Union[TopicUpdate, ErrorMessage]
"""Message queued for a session."""


class Mailbox:
    """A session's outbox, holding at most one pending item per topic."""

    def __init__(self) -> None:
        self.pending: t.Dict[str, MailboxItem] = {}
        self.event: asyncio.Event = asyncio.Event()

    def put_update(self, update: TopicUpdate, conflation: Conflation) -> None:
        """Queue `update`, combining it with any pending update to the topic per `conflation`."""
        key = canonical_topic(update.topic)
        prev = self.pending.get(key)
        if conflation == 'append' and isinstance(prev, TopicUpdate):
            update = TopicUpdate.make_unchecked(
                update.topic,
                [*prev.data, *update.data],
                update.cause if update.cause is not None else prev.cause,
            )
        self.pending[key] = update
        self.event.set()

    def put_error(self, topic: WireTopic, reason: str) -> None:
        """Queue an error for `topic`, replacing any pending item."""
        self.pending[canonical_topic(topic)] = ErrorMessage(topic, reason)
        self.event.set()

    async def drain(self) -> t.List[MailboxItem]:
        """Wait for pending items, then remove and return them."""
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
    """One websocket connection, with its `Mailbox` and subscriptions.

    `default_topic` fills missing keys of dict topics the client subscribes to.
    """

    def __init__(self, default_topic: t.Optional[t.Mapping[str, t.Any]] = None):
        self.default_topic: t.Optional[t.Mapping[str, t.Any]] = default_topic
        self.mailbox: Mailbox = Mailbox()
        self.subscriptions: t.Dict[str, _Subscription] = {}
        self.kicked: asyncio.Event = asyncio.Event()
        """Set to drop this connection (`phaser/web/debug.py`), as an unexpected close"""

    def _merge(self, client_topic: WireTopic) -> WireTopic:
        """Fill missing keys of `client_topic` from `default_topic`; explicit keys win."""
        if self.default_topic is None or isinstance(client_topic, str):
            return client_topic
        return {**self.default_topic, **client_topic}

    def client_topic_for(self, key: str) -> WireTopic:
        """Return the topic as the client sent it, given its canonical `key`."""
        sub = self.subscriptions.get(key)
        return sub.client_topic if sub is not None else key

    async def subscribe(self, client_topic: WireTopic) -> None:
        """Subscribe to `client_topic`. Idempotent. An unresolvable topic queues an error."""
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
        """Unsubscribe from `client_topic`, if subscribed."""
        merged = self._merge(client_topic)
        key = canonical_topic(merged)
        sub = self.subscriptions.pop(key, None)
        if sub is not None:
            sub.owner.unsubscribe(self, key)

    def close(self) -> None:
        """Unsubscribe from all topics."""
        for key, sub in list(self.subscriptions.items()):
            sub.owner.unsubscribe(self, key)
        self.subscriptions.clear()


class Broker:
    """A `Cache` and its subscribed `Topic`s, for a `Job` or the `Jobs`/`Workers` managers."""

    def __init__(self) -> None:
        self.cache: Cache = Cache()
        self.active_topics: t.Dict[str, Topic] = {}

    async def subscribe(self, session: Session, key: str, view: View, params: t.Mapping[str, t.Any]) -> None:
        """Subscribe `session` to topic `key`, sending a snapshot if the view is `retained` and populated."""
        topic = self.active_topics.get(key)
        if topic is None:
            topic = Topic(key, view, params, self.cache)
            self.active_topics[key] = topic
        topic.subscribers.add(session)

        if view.retained and topic.has_deps():
            value = await topic.value_async()
            session.mailbox.put_update(TopicUpdate.make_unchecked(session.client_topic_for(key), value, None), view.conflation)

    def unsubscribe(self, session: Session, key: str) -> None:
        """Unsubscribe `session` from topic `key`, dropping the topic if it has no subscribers."""
        topic = self.active_topics.get(key)
        if topic is None:
            return
        topic.subscribers.discard(session)
        if not topic.subscribers:
            del self.active_topics[key]

    def _distribute(self, topic: Topic, value: t.Any, cause: t.Any) -> None:
        for session in list(topic.subscribers):
            session.mailbox.put_update(
                TopicUpdate.make_unchecked(session.client_topic_for(topic.key), value, cause), topic.view.conflation
            )

    async def publish_dirty(self, changed: t.FrozenSet[str], cause: t.Any = None) -> None:
        """Recompute and distribute every active topic whose deps intersect `changed`.

        Each view is computed once, shared across subscribers. `cause` is attached to the updates.
        """
        dirty = [tp for tp in self.active_topics.values() if tp.view.deps & changed]
        if not dirty:
            return
        values = await asyncio.gather(*(tp.value_async() for tp in dirty))
        for tp, v in zip(dirty, values):
            self._distribute(tp, v, cause)

    def publish_value(self, key: str, value: t.Any, cause: t.Any = None) -> None:
        """Send `value` directly to the subscribers of topic `key`, bypassing view computation."""
        topic = self.active_topics.get(key)
        if topic is None:
            return
        self._distribute(topic, value, cause)

    def close(self, reason: str) -> None:
        """Send an error with `reason` to every subscriber, then drop all topics."""
        for key, topic in list(self.active_topics.items()):
            for session in list(topic.subscribers):
                session.mailbox.put_error(session.client_topic_for(key), reason)
                session.subscriptions.pop(key, None)
        self.active_topics.clear()


class ResolveError(Exception):
    """A topic could not be resolved to a view."""
    def __init__(self, reason: str):
        self.reason: str = reason
        super().__init__(reason)


def resolve(topic: WireTopic) -> t.Tuple[Broker, str, View, t.Mapping[str, t.Any]]:
    """Resolve a default-merged topic to `(owner, canonical key, view, params)`.

    Raises:
        ResolveError: If the topic, job, or view is unknown or the topic is malformed.
    """
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
