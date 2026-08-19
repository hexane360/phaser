from __future__ import annotations

import abc
import asyncio
import bisect
import datetime
import logging
import multiprocessing
import os
import random
import signal
import sys
import threading
import time
import typing as t
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from quart import Quart, g, request, url_for
from typing_extensions import Self

import pane

from ..version import version_info
from .pubsub import Broker, Session
from .types import (
    RELOAD_EXIT_CODE,
    JobID,
    JobResponse,
    JobState,
    JobStatus,
    LogMessage,
    LogPage,
    LogRecord,
    OkResponse,
    Result,
    ServerResponse,
    Signal,
    SignalResponse,
    ValidationError,
    WorkerID,
    # worker - server communication
    WorkerMessage,
    WorkerState,
    # server - client communication
    WorkerStatus,
    canonical_topic,
)
from .util import timeout

T = t.TypeVar('T')


class Shutdown(Exception):
    pass


class Kicked(Exception):
    """A websocket connection was dropped on purpose (`phaser/web/debug.py`)"""


async def raise_on_shutdown():
    await server.shutdown_event.wait()
    raise Shutdown()


class Worker(abc.ABC):
    def __init__(self, worker_id: WorkerID):
        super().__init__()
        self.status: WorkerStatus = 'queued'
        self.id: WorkerID = worker_id

        self.current_job: t.Optional[weakref.ref[Job]] = None
        self.start_time: t.Optional[datetime.datetime] = None
        """UTC time worker started running at"""
        self.hostname: t.Optional[str] = None
        """Hostname worker is running on"""
        self.backends: t.Optional[t.Sequence[t.Tuple[str, str]]] = None
        """Computational backends available to worker"""

    @abc.abstractmethod
    def worker_type(self) -> str:
        ...

    def state(self) -> WorkerState:
        links = {
            'shutdown': url_for('shutdown_worker', worker_id=self.id),
            'reload': url_for('reload_worker', worker_id=self.id),
        }
        current_job = job.id if self.current_job and (job := self.current_job()) else None
        return WorkerState(
            self.id, self.worker_type(), self.status, links=links,
            current_job=current_job, start_time=self.start_time,
            hostname=self.hostname, backends=self.backends,
        )

    async def cancel(self):
        if self.status == 'queued':
            await self.set_status('stopped')
        elif self.status not in ('stopping', 'stopped'):
            await self.set_status('stopping')

    async def reload(self):
        if self.status not in ('stopping', 'stopped'):
            await self.set_status('reloading')

    def action(self) -> t.Optional[Signal]:
        if self.status == 'stopping':
            return 'shutdown'
        elif self.status == 'reloading':
            return 'reload'
        return None

    def on_connected(self):
        """Hook for a worker that has just reported in (`LocalWorker` clears its restart
        count here)."""

    async def set_status(self, status: WorkerStatus, reason: t.Optional[str] = None,
                         error: t.Optional[str] = None):
        """`reason`/`error` describe why the worker stopped, and are carried onto whatever
        job it was running -- otherwise that job has no account of why it ended."""
        self.status = status
        await server.workers.notify_changed({'worker_id': self.id, 'status': status})

        if status == 'stopped':
            if self.current_job and (job := self.current_job()):
                await job.worker_lost(reason or f"Worker {self.id} stopped", error)
            server.workers.schedule_for_removal(self.id, 5.0)

    async def handle_message(self, msg: WorkerMessage) -> ServerResponse:
        if msg.msg == 'shutdown':
            reason = _exc_summary(msg.error) or msg.detail or f"Worker {self.id} shut down"
            await self.set_status('stopped', reason=reason, error=msg.error)
            return OkResponse()

        if msg.msg == 'connect':
            self.start_time = datetime.datetime.now(datetime.timezone.utc)
            self.hostname = 'localhost' if self.worker_type() == 'local' else msg.hostname
            self.backends = msg.backends
            self.on_connected()
            await self.set_status('idle')

        if (job_id := getattr(msg, 'job_id', None)):
            job = server.jobs[job_id]
            await job.handle_update(msg)

            if job.should_cancel():
                return SignalResponse(self.action() or 'cancel')

        if (action := self.action()):
            return SignalResponse(action)

        if msg.msg in ('poll', 'job_result'):
            if (job := server.take_queued_job()) is not None:
                # send a new job if available
                self.current_job = weakref.ref(job)
                await self.set_status('running')
                job.start_time = datetime.datetime.now(datetime.timezone.utc)
                await job.set_status('starting')
                return JobResponse(job.id, job.plan)
            else:
                # otherwise don't
                self.current_job = None
                await self.set_status('idle')

        return OkResponse()

    async def finalize(self):
        if self.status not in ('stopped', 'unknown'):
            logging.error(f"Job {self.id} finalized before completion")


MAX_WORKER_RESTARTS: int = 5
"""Consecutive restarts allowed without the worker connecting in between."""


class LocalWorker(Worker):
    def __init__(self, worker_id: WorkerID, url: str):
        super().__init__(worker_id)
        self.url = url
        self._restarts: int = 0

        self._start()
        self._fut: asyncio.Task[None] = asyncio.create_task(self._watch())

    def worker_type(self) -> str:
        return 'local'

    def on_connected(self):
        self._restarts = 0

    def _start(self):
        from phaser.web.worker import run_worker

        quiet = False
        self.process = multiprocessing.Process(target=run_worker, args=[self.url, quiet], daemon=True)
        self.status = 'starting'
        self.process.start()

    async def _watch(self):
        """Await the process, restarting it on a reload and reporting anything else as a stop."""
        while True:
            await asyncio.to_thread(self.process.join)
            code = self.process.exitcode

            if code == RELOAD_EXIT_CODE:
                if self._restarts >= MAX_WORKER_RESTARTS:
                    reason = f"Worker {self.id} failed to start {self._restarts} times"
                    logging.error(reason)
                    await self.set_status('stopped', reason=reason)
                    break
                self._restarts += 1
                self._start()
                continue

            # negative for a signal-killed child (-9 for SIGKILL), so report it verbatim
            await self.set_status('stopped', reason=(
                f"Worker {self.id} stopped unexpectedly (exit code {code})"
                if code else f"Worker {self.id} stopped"
            ))
            break

    async def finalize(self):
        self.process.terminate()
        await self._fut


class ManualWorker(Worker):
    def __init__(self, worker_id: WorkerID):
        self.url = server.get_worker_url(worker_id)
        logging.warning(f"Worker command: python -m phaser worker {self.url}")
        super().__init__(worker_id)

    def worker_type(self) -> str:
        return 'manual'


class Workers:
    def __init__(self):
        self.inner: t.Dict[WorkerID, Worker] = {}
        self._futs: t.List[asyncio.Task[None]] = []
        self.broker: Broker = Broker()
        """Broker for the manager-level `"workers"` topic."""

    def state(self) -> t.List[WorkerState]:
        return [worker.state() for worker in self.inner.values()]

    async def notify_changed(self, cause: t.Optional[t.Any] = None):
        # 'state' is a synthetic dep: the `workers` view ignores the cache and reads
        # `server.workers` directly, so all that matters here is bumping its generation.
        self.broker.cache.update_raw({'state': None})
        await self.broker.publish_dirty(frozenset({'state'}), cause=cause)

    async def add(self, worker: Worker):
        self.inner[worker.id] = worker
        await self.notify_changed({'worker_id': worker.id, 'status': worker.status})

    def schedule_for_removal(self, worker_id: WorkerID, delay: float = 30.0):
        if worker_id not in self:
            return

        async def task():
            async with server.app.app_context():
                await asyncio.sleep(delay)
                await self.remove(worker_id)

        self._futs.append(asyncio.create_task(task()))

    async def remove(self, worker_id: WorkerID):
        try:
            worker = self.inner.pop(worker_id)
        except KeyError:
            return

        await worker.finalize()
        await self.notify_changed({'worker_id': worker_id, 'status': 'removed'})

    def __contains__(self, item: WorkerID) -> bool:
        return self.inner.__contains__(item)

    def __getitem__(self, item: WorkerID) -> Worker:
        return self.inner[item]

    def items(self) -> t.ItemsView[WorkerID, Worker]:
        return self.inner.items()

    def keys(self) -> t.KeysView[WorkerID]:
        return self.inner.keys()

    def values(self) -> t.ValuesView[Worker]:
        return self.inner.values()

    async def finalize(self):
        await asyncio.gather(*(worker.finalize() for worker in self.inner.values()))
        for fut in self._futs:
            fut.cancel()
        await asyncio.gather(*self._futs, return_exceptions=True)


LOG_LEVELS: t.Tuple[int, ...] = (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)


class LogBuffer:
    """Append-only store of a job's log records, supporting paging in either direction.

    Cumulative per-level indices (`_index[level]` holds the indices of every record at or
    above `level`) make a filtered page cost O(page) rather than a scan of every record.
    """
    def __init__(self) -> None:
        self.records: t.List[LogRecord] = []
        self._index: t.Dict[int, t.List[int]] = {level: [] for level in LOG_LEVELS}
        self.start: t.Optional[datetime.datetime] = None
        """Timestamp of the first record appended. Survives eviction, unlike `records[0]`."""

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> t.Iterator[LogRecord]:
        return iter(self.records)

    def append(self, record: LogRecord) -> None:
        i = len(self.records)
        if self.start is None:
            self.start = record.timestamp
        self.records.append(record)
        for level in LOG_LEVELS:
            if record.log_level >= level:
                self._index[level].append(i)

    def filtered(self, min_level: int = 0) -> t.Sequence[LogRecord]:
        """Records at or above `min_level` (snapped down, see `snap_level`)"""
        if not (level := snap_level(min_level)):
            return self.records
        return [self.records[i] for i in self._index[level]]

    def page(
        self, before: t.Optional[int] = None, after: t.Optional[int] = None,
        limit: int = 100, min_level: int = 0,
    ) -> LogPage:
        """A window of at most `limit` records at or above `min_level`, ascending by `i`.

        `before`/`after` are exclusive cursors on `i`, bounding the window on either side:
        given both, the window is the range between them (a reconnect gap); given neither,
        it's the newest `limit` records. `limit` truncates the end away from `after` when
        given (filling a gap forwards from what the client holds), and the older end
        otherwise.
        """
        level = snap_level(min_level)
        # indices of matching records, ascending. unfiltered, that's just each record's own index
        indices: t.Sequence[int] = self._index[level] if level else range(len(self.records))

        # the range the cursors ask for, which `limit` then truncates at one end
        lo = bisect.bisect_right(indices, after) if after is not None else 0
        hi = max(bisect.bisect_left(indices, before) if before is not None else len(indices), lo)
        (start, stop) = (lo, min(lo + limit, hi)) if after is not None else (max(hi - limit, lo), hi)

        window = indices[start:stop]
        return LogPage(
            [self.records[i] for i in window],
            first=window[0] if len(window) else None,
            last=window[-1] if len(window) else None,
            count=len(window),
            total=len(indices),
            total_all=len(self.records),
            # relative to the requested range, so with both cursors these answer
            # "is the gap closed?" rather than "does the log continue?"
            has_before=start > lo,
            has_after=stop < hi,
            min_level=level,
        )


def _comparable(time: datetime.datetime, start: t.Optional[datetime.datetime]) -> bool:
    """Whether `time - start` is well-defined (both present, and neither mixes a naive
    timestamp with an aware one)"""
    return start is not None and (time.tzinfo is None) == (start.tzinfo is None)


def _exc_summary(error: t.Optional[str], max_len: int = 200) -> t.Optional[str]:
    """Last non-empty line of a formatted traceback -- the exception line itself
    (`ValueError: ...`). Truncated, since this rides along on every `jobs` update."""
    if error is None:
        return None
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1] if len(lines[-1]) <= max_len else lines[-1][:max_len - 1] + '…'


def snap_level(min_level: int) -> int:
    """`min_level` snapped down to a standard logging level, or 0 (no filtering)"""
    return max((level for level in LOG_LEVELS if level <= min_level), default=0)


class Job:
    def __init__(self, id: JobID, plan: str, name: t.Optional[str] = None):
        self.id: JobID = id
        self.plan: str = plan
        self.job_name: t.Optional[str] = name
        """Name of job"""
        self.status: JobStatus = 'queued'
        self.result: t.Optional[Result] = None
        """Terminal outcome, once the job stops. `status` is the lifecycle; this is how it
        ended."""
        self.error_summary: t.Optional[str] = None
        """Final line of the traceback, for `result == 'errored'`. The traceback itself is
        a record in `logs`."""
        self.broker: Broker = Broker()
        """Pub/sub broker for this job's views (`state`, `progress`, `obj_phase_sum`, ...).
        `broker.cache.raw` is the wire-form (still-encoded) view of the latest worker
        state -- the single source of truth `Job.state()` also reads from."""
        # synthetic dep for the `state` view, which reads this `Job` rather than the cache
        # (like `Jobs`/`Workers` do for the manager topics). Seeded here so `has_deps()`
        # holds for a subscriber arriving before the first transition -- a queued job has
        # made none.
        self.broker.cache.update_raw({'state': None})
        self.logs: LogBuffer = LogBuffer()
        """Cache of recorded messages"""
        self.start_time: t.Optional[datetime.datetime] = None
        """Time job was started at (server clock, when the job was dispatched)"""
        self.worker_start_time: t.Optional[datetime.datetime] = None
        """Time the worker took up the job, by its own clock. Log records are timestamped
        by that same clock, so this is what `LogRecord.elapsed` is measured from."""

    @classmethod
    async def from_path(cls, path: t.Union[str, Path]) -> t.List[Self]:
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-m', 'phaser', 'validate', '--json', str(path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await process.communicate()
        return await cls._process_validate_result(stdout)

    @classmethod
    async def from_yaml(cls, plan: t.Union[str, bytes]) -> t.List[Self]:
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-m', 'phaser', 'validate', '--json',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert process.stdin is not None
        process.stdin.write(plan if isinstance(plan, (bytes, bytearray, memoryview)) else plan.encode('utf-8'))
        stdout, _ = await process.communicate()
        return await cls._process_validate_result(stdout)

    @classmethod
    async def _process_validate_result(cls, stdout: bytes) -> t.List[Self]:
        import json
        result = json.loads(stdout)
        if result['result'] == 'error':
            raise ValidationError(result['error'])

        assert result['result'] == 'success'
        jobs = [
            cls(server.make_jobid(), json.dumps(plan), name)
            for (name, plan) in result['plans']
        ]
        for job in jobs:
            await server.jobs.add(job)
        return jobs

    async def notify_changed(self, cause: t.Optional[t.Any] = None):
        """Republish this job's `state` topic. Mirrors `Jobs.notify_changed`: `'state'` is
        synthetic, so all this does is bump the generation the view is memoized against."""
        self.broker.cache.update_raw({'state': None})
        await self.broker.publish_dirty(frozenset({'state'}), cause=cause)

    async def set_status(self, status: JobStatus, cause: t.Optional[t.Any] = None):
        # a `cause` is new information even when the status itself hasn't moved, e.g. a
        # result arriving for a job already stopped
        if self.status == status and cause is None:
            return
        self.status = status
        await self.notify_changed(cause)
        await server.jobs.notify_changed({'job_id': self.id, 'status': status})

    async def worker_lost(self, reason: str, error: t.Optional[str] = None):
        """Terminal transition for a job whose worker went away without reporting a result.

        Returns without touching anything if the worker did report -- the normal sequence is
        `job_result` first, worker stop second, and a real outcome must not be overwritten.
        """
        if self.result is not None:
            return
        self.result = 'interrupted'
        self.error_summary = reason
        self._append_log(LogMessage(
            job_id=self.id, timestamp=datetime.datetime.now(datetime.timezone.utc),
            log=reason, logger_name=__name__, log_level=logging.ERROR,
            line_number=0, stack_info=error,
        ))

        await self.set_status('stopped', cause={'result': self.result})

    async def cancel(self):
        if self.status == 'queued':
            await self.set_status('stopped')
        elif self.status not in ('stopping', 'stopped'):
            await self.set_status('stopping')

    def should_cancel(self) -> bool:
        return self.status == 'stopping'

    async def delete(self):
        if self.status not in ('queued', 'stopped'):
            raise RuntimeError("Cannot delete a running job")
        if self.status != 'stopped':
            await self.set_status('stopped')
        await server.jobs.remove(self.id)

    def _elapsed(self, timestamp: datetime.datetime) -> float:
        """Seconds from the start of the job to `timestamp`.

        Both candidate anchors share the record's own clock, so no cross-machine
        subtraction ever happens: the worker's reported start, or -- when it never
        reported one (a fake job, a failed `job_start`, or a worker predating that
        message) -- the job's first log record, making that record t=0.
        """
        if _comparable(timestamp, self.worker_start_time):
            return (timestamp - t.cast(datetime.datetime, self.worker_start_time)).total_seconds()
        if _comparable(timestamp, self.logs.start):
            return (timestamp - t.cast(datetime.datetime, self.logs.start)).total_seconds()
        return 0.

    def _total_iter(self) -> t.Optional[int]:
        iter_raw = self.broker.cache.raw.get('iter')
        return iter_raw.get('total_iter') if isinstance(iter_raw, dict) else None

    def state(self) -> JobState:
        raw = self.broker.cache.raw
        state = {k: v for (k, v) in raw.items() if k == 'iter'}
        links = {
            'dashboard': url_for('job_dashboard', job_id=self.id),
            'cancel': url_for('cancel_job', job_id=self.id),
            'delete': url_for('delete_job', job_id=self.id),
            'logs': url_for('job_logs', job_id=self.id),
            'logs_txt': url_for('job_logs_text', job_id=self.id),
        }
        return JobState.make_unchecked(
            self.id, self.status, links, job_name=self.job_name, start_time=self.start_time, state=state,
            result=self.result, error_summary=self.error_summary,
        )

    async def handle_update(self, msg: WorkerMessage):
        if msg.msg == 'job_update':
            if self.status in ('queued', 'starting'):
                await self.set_status('running')

            old_total_iter = self._total_iter()
            changed = frozenset(msg.state.keys())
            self.broker.cache.update_raw(msg.state)
            await self.broker.publish_dirty(changed)

            if self._total_iter() != old_total_iter:
                await self.notify_changed()
                await server.jobs.notify_changed({'job_id': self.id})
        elif msg.msg == 'job_start':
            self.worker_start_time = msg.start_time
        elif msg.msg == 'log':
            self._append_log(msg)
        elif msg.msg == 'job_result':
            # assigned before `set_status`, which early-returns on an unchanged status: a
            # job already stopped (its worker went away first) would otherwise swallow the
            # result and its traceback entirely
            self.result = msg.result
            self.error_summary = _exc_summary(msg.error)
            # the worker's own crash log is `local`-only, so this record is the only way the
            # traceback reaches the log view, `min_level` filtering, or the plaintext export
            if msg.log is not None:
                self._append_log(msg.log)

            cause = {'result': msg.result}
            already_stopped = self.status == 'stopped'
            await self.set_status('stopped', cause=cause)
            if already_stopped:
                await self.notify_changed(cause)
                await server.jobs.notify_changed({'job_id': self.id, 'result': msg.result})

    def _append_log(self, msg: LogMessage) -> None:
        record = msg.into_record(len(self.logs), self._elapsed(msg.timestamp))
        self.logs.append(record)
        key = canonical_topic({'job': self.id, 'view': 'logs'})
        self.broker.publish_value(key, [pane.into_data(record)])

    async def finalize(self):
        if self.status != 'stopped':
            logging.error(f"Job {self.id} finalized before completion")


class Jobs:
    def __init__(self):
        self.inner: t.Dict[JobID, Job] = {}
        self.broker: Broker = Broker()
        """Broker for the manager-level `"jobs"` topic."""

    def state(self) -> t.List[JobState]:
        return [job.state() for job in self.inner.values()]

    async def notify_changed(self, cause: t.Optional[t.Any] = None):
        # 'state' is a synthetic dep: the `jobs` view ignores the cache and reads
        # `server.jobs` directly, so all that matters here is bumping its generation.
        self.broker.cache.update_raw({'state': None})
        await self.broker.publish_dirty(frozenset({'state'}), cause=cause)

    async def add(self, job: Job, queue: bool = True):
        """Register `job`, queueing it for a worker unless `queue` is false (a job which
        drives itself, e.g. `phaser/web/debug.py`'s fake jobs)"""
        self.inner[job.id] = job
        if queue:
            server.job_queue.append(job)
        await self.notify_changed({'job_id': job.id, 'status': job.status})

    async def remove(self, job_id: JobID):
        try:
            job = self.inner.pop(job_id)
        except KeyError:
            return

        await job.finalize()
        job.broker.close(f"job '{job_id}' removed")
        await self.notify_changed({'job_id': job_id, 'status': 'removed'})

    def __contains__(self, item: JobID) -> bool:
        return self.inner.__contains__(item)

    def __getitem__(self, item: JobID) -> Job:
        return self.inner[item]

    def items(self) -> t.ItemsView[JobID, Job]:
        return self.inner.items()

    def keys(self) -> t.KeysView[JobID]:
        return self.inner.keys()

    def values(self) -> t.ValuesView[Job]:
        return self.inner.values()

    async def finalize(self):
        await asyncio.gather(*(job.finalize() for job in self.inner.values()))


_ID_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class Server:
    def __init__(self):
        self.app: Quart = Quart(
            __name__,
            static_url_path="/static",
            static_folder="static",
        )
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 5
        self.app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512 MiB

        self.sessions: t.Set[Session] = set()
        """Live `/listen` connections, so they can be dropped en masse (`debug.py`)"""

        # plain containers, no loop required -- built here rather than in `run()` so a
        # `Server` is well-formed on construction (`Job.set_status` reaches `server.jobs`)
        self.workers: Workers = Workers()
        self.jobs: Jobs = Jobs()
        self.job_queue: deque[Job] = deque()

        self.file_root: Path = Path.cwd().resolve()
        """Root for path completion. Resolved, so a symlinked cwd compares correctly."""

        @self.app.after_serving
        async def shutdown():
            logging.info("Shutting down...")
            for fut in self.futs:
                fut.cancel()
            try:
                async with timeout(5):
                    await asyncio.gather(
                        self.jobs.finalize(), self.workers.finalize(), self.slurm_manager.finalize()
                    )
                    await asyncio.gather(*self.futs, return_exceptions=True)
            except TimeoutError:
                logging.warning("Cleanup didn't finish in time")
            finally:
                self.compute_pool.shutdown(wait=False, cancel_futures=True)

    def take_queued_job(self) -> t.Optional[Job]:
        """The next job actually waiting to run, discarding any that no longer are.

        `Job.cancel` and `Job.delete` only move a job's status -- a queued job stays in the
        deque -- so a job is checked when it comes out, not when it stops being wanted.
        """
        while self.job_queue:
            job = self.job_queue.popleft()
            if job.status == 'queued' and job.id in self.jobs:
                return job
        return None

    def resolve_in_root(self, path: str) -> t.Optional[Path]:
        """`path`, relative to the file root or absolute, or None if it escapes the root.

        Normalized lexically rather than resolved, so a symlinked directory under the root
        can be traversed. `..` is still collapsed textually, and the caller must use the
        path returned here -- listing the un-normalized spelling would follow a symlink
        first and land outside the root.
        """
        full = Path(os.path.normpath(self.file_root / Path(path).expanduser()))
        if full.is_relative_to(self.file_root):
            return full
        # a root reached through a symlink (a cwd of `/tmp/x`, really `/private/tmp/x`)
        # fails the lexical test for a path spelled the other way
        resolved = full.resolve()
        return resolved if resolved.is_relative_to(self.file_root) else None

    def get_worker_url(self, worker_id: WorkerID) -> str:
        assert self.host is not None
        url_adapter = self.app.url_map.bind(self.host, self.root_path, url_scheme='http')
        url = url_adapter.build('worker_update', dict(worker_id=worker_id), method='POST', force_external=True)
        return url

    def make_workerid(self) -> WorkerID:
        while True:
            id = "".join(_ID_CHARS[random.randrange(0, len(_ID_CHARS))] for _ in range(10))
            if id not in self.workers:
                return id

    def make_jobid(self) -> JobID:
        while True:
            id = "".join(_ID_CHARS[random.randrange(0, len(_ID_CHARS))] for _ in range(10))
            if id not in self.jobs:
                return id

    def _set_signals(self, loop: asyncio.AbstractEventLoop):
        last_time: t.Optional[float] = None

        def _signal_handler(signal: str) -> None:
            if signal != 'SIGINT':
                logging.warning(f"Received {signal}. Stopping...")
                self.shutdown_event.set()
                return

            if not loop.is_running():
                return

            nonlocal last_time
            t = time.monotonic()

            if last_time is not None and t - last_time < 2:
                self.shutdown_event.set()
                return

            logging.warning("Workers interrupted. Press CTRL + C twice to quit server")
            last_time = t

        for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGQUIT"):
            if hasattr(signal, signal_name):
                try:
                    loop.add_signal_handler(getattr(signal, signal_name), _signal_handler, signal_name)
                except NotImplementedError:
                    # Add signal handler may not be implemented on Windows
                    signal.signal(getattr(signal, signal_name), lambda _sig, _frame, name=signal_name: _signal_handler(name))

    def run(
        self,
        hostname: str = 'localhost',
        port: t.Optional[int] = None,
        root_path: t.Optional[str] = None,
        verbosity: int = 0,
        serving_cb: t.Optional[t.Callable[[], t.Any]] = None,
        debug: bool = False,
    ):
        self.compute_pool: ThreadPoolExecutor = ThreadPoolExecutor()
        """Shared threadpool for pub/sub view compute + array decode (`phaser/web/pubsub.py`)."""
        self.futs: t.List[asyncio.Task[t.Any]] = []
        """Long-lived background tasks, cancelled on shutdown."""

        self.shutdown_event: asyncio.Event = asyncio.Event()

        from .slurm import SlurmManager

        self.slurm_manager: SlurmManager = SlurmManager()

        self.host = f"{hostname}:{port or 5050}"
        self.root_path = root_path or os.environ.get("SCRIPT_NAME")

        if serving_cb:
            self.app.before_serving(serving_cb)

        if debug:
            from .debug import register_debug_routes

            register_debug_routes(self.app)

        logging.basicConfig(level=logging.INFO if verbosity == 0 else logging.DEBUG)

        @self.app.before_request
        async def _time_request():
            g.start_time = time.monotonic()

        @self.app.after_request
        async def _log_request_time(response):
            elapsed = time.monotonic() - g.start_time
            msg = f"{request.method} {request.path} {response.status_code} {elapsed * 1000.:.1f}ms"
            if elapsed > 0.5:  # 500 ms
                logging.warning(msg)
            else:
                logging.debug(msg)
            return response

        @self.app.before_serving
        async def _start_watchdog():
            # tracked, so shutdown cancels it rather than abandoning it mid-sleep
            self.futs.append(asyncio.create_task(_watch_event_loop_lag()))

        @self.app.before_serving
        async def _log_version():
            # resolves & caches `version_info()`, keeping git subprocesses off the request path
            info = version_info()
            logging.info(str(info))

        if verbosity > 0:
            @self.app.before_request
            async def log_request():
                logging.debug(f"{request.method} {request.path} {request.user_agent}")

            self.app.config['DEBUG'] = True

        multiprocessing.set_start_method('spawn', True)

        loop = asyncio.new_event_loop()
        loop.set_debug(verbosity > 1)
        asyncio.set_event_loop(loop)
        # route pub/sub view compute (`phaser/web/pubsub.py`, via `quart.utils.run_sync`)
        # onto our shared threadpool instead of asyncio's own default executor.
        loop.set_default_executor(self.compute_pool)

        if threading.current_thread() is threading.main_thread():
            self._set_signals(loop)

        from hypercorn.asyncio import serve
        from hypercorn.config import Config

        _disable_ws_compression()

        try:
            loop.run_until_complete(
                serve(self.app, Config.from_mapping(
                    bind=self.host,
                    root_path=self.root_path,
                    #websocket_max_message_size="512MiB",
                    #wsgi_max_body_size="512MiB",
                ), shutdown_trigger=self.shutdown_event.wait)
            )
        finally:
            #loop.run_until_complete(self.app.shutdown())
            try:
                _cancel_all_tasks(loop)
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()


async def _watch_event_loop_lag(interval: float = 0.5, threshold: float = 0.2) -> None:
    """Background watchdog. Sleeps in a loop and compares actual elapsed time to the
    requested interval; any excess is scheduling lag caused by something blocking the
    event loop (sync work not offloaded to a thread, a blocking call, etc.)."""
    loop = asyncio.get_running_loop()
    last = loop.time()
    while True:
        await asyncio.sleep(interval)
        now = loop.time()
        lag = now - last - interval
        if lag > threshold:
            logging.warning(f"Event loop stalled for {lag:.3f}s")
        last = now


def _disable_ws_compression() -> None:
    """
    Monkey-patches hypercorn's Handshake.accept function to disable websocket compression
    """
    from hypercorn.protocol.ws_stream import Handshake
    from wsproto.connection import Connection, ConnectionType
    from wsproto.extensions import Extension
    from wsproto.handshake import server_extensions_handshake
    from wsproto.utilities import generate_accept_token

    def accept(
        self: Handshake,
        subprotocol: t.Optional[str],
        additional_headers: t.Iterable[t.Tuple[bytes, bytes]],
    ) -> t.Tuple[int, t.List[t.Tuple[bytes, bytes]], Connection]:
        headers = []
        if subprotocol is not None:
            if self.subprotocols is None or subprotocol not in self.subprotocols:
                raise Exception("Invalid Subprotocol")
            else:
                headers.append((b"sec-websocket-protocol", subprotocol.encode()))

        extensions: t.List[Extension] = []  # permessage-deflate disabled, see above
        accepts = None
        if self.extensions is not None:
            accepts = server_extensions_handshake(self.extensions, extensions)

        if accepts:
            headers.append((b"sec-websocket-extensions", accepts))

        if self.key is not None:
            headers.append((b"sec-websocket-accept", generate_accept_token(self.key)))

        status_code = 200
        if self.http_version == "1.1":
            headers.extend([(b"upgrade", b"WebSocket"), (b"connection", b"Upgrade")])
            status_code = 101

        for name, value in additional_headers:
            if b"sec-websocket-protocol" == name or name.startswith(b":"):
                raise Exception(f"Invalid additional header, {name.decode()}")
            headers.append((name, value))

        self.accepted = True
        return status_code, headers, Connection(ConnectionType.SERVER, extensions)

    Handshake.accept = accept


def _cancel_all_tasks(loop: asyncio.AbstractEventLoop) -> None:
    tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
    if not tasks:
        return

    for task in tasks:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

    for task in tasks:
        if not task.cancelled() and task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )


server: Server = Server()

from . import routes  # noqa: E402, F401
