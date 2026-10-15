"""Wire-format message types for the browser <-> server and worker <-> server protocols."""
import datetime
import json
import logging
import typing as t

from typing_extensions import Self

import pane
from pane.annotations import Tagged

from .frames import ArrayDataConverter

JobID: t.TypeAlias = str
WorkerID: t.TypeAlias = str

WorkerStatus: t.TypeAlias = t.Literal['queued', 'starting', 'reloading', 'idle', 'running', 'stopping', 'stopped', 'unknown']
"""
State of a worker.

- `'queued'`: Queued/scheduled to start.
- `'starting'`: Being started.
- `'reloading'`: Being reloaded.
- `'idle'`: No job is running.
- `'running'`: A job is running.
- `'stopping'`: Being stopped.
- `'stopped'`: Stopped, normally or abnormally.
- `'unknown'`: Has not been heard from in a while.
"""

JobStatus: t.TypeAlias = t.Literal['queued', 'starting', 'running', 'stopping', 'stopped']
"""State of a reconstruction job."""

Result: t.TypeAlias = t.Literal['finished', 'errored', 'cancelled', 'interrupted']
"""
Outcome of a job or worker.

- `'finished'`: Finished normally.
- `'errored'`: Raised an error.
- `'cancelled'`: Cancelled or shut down from the server side.
- `'interrupted'`: Cancelled or shut down by the worker's client or the OS.
"""

Signal: t.TypeAlias = t.Literal['shutdown', 'cancel', 'reload']
"""Signal sent to a job or worker."""

RELOAD_EXIT_CODE: int = 129
"""Exit code a worker exits with to request a restart (128 + SIGHUP)."""

UPDATE_CHUNK_SIZE: int = 16 * 1024 * 1024
"""Maximum size of one worker -> server request body. Larger `UpdateMessage`s are sent in chunks of this size."""

class ValidationError(Exception):
    """Invalid request or plan, with a user-facing message `msg`."""
    def __init__(self, msg: str):
        self.msg: str = msg

class SignalException(Exception):
    """Raised in a worker when the server sends a signal. `urgent` signals request an immediate stop."""
    def __init__(self, signal: Signal, urgent: bool = False):
        self.signal: Signal = signal
        self.urgent: bool = urgent

# pub/sub wire protocol (browser <-> server)

TopicKey: t.TypeAlias = t.Union[str, int, float]
"""Value in a dict `Topic`."""
Topic: t.TypeAlias = t.Union[str, t.Dict[str, TopicKey]]
"""Subscribable topic: a bare string (`"jobs"`) or a flat dict (`{"job": <id>, "view": "status"}`)."""


def canonical_topic(topic: Topic) -> str:
    """Return the canonical JSON key for a `Topic` (sorted keys, no whitespace)."""
    return json.dumps(topic, sort_keys=True, separators=(',', ':'))


class SubscribeMessage(pane.PaneBase):
    """Client -> server: subscribe to one or more topics."""
    topics: t.List[Topic]
    msg: t.Literal['sub'] = 'sub'

class UnsubscribeMessage(pane.PaneBase):
    """Client -> server: unsubscribe from one or more topics."""
    topics: t.List[Topic]
    msg: t.Literal['unsub'] = 'unsub'

class HeartbeatMessage(pane.PaneBase):
    """Client -> server: liveness check."""
    msg: t.Literal['ping'] = 'ping'

ClientMessage: t.TypeAlias = t.Annotated[t.Union[
    SubscribeMessage, UnsubscribeMessage, HeartbeatMessage,
], Tagged('msg')]
"""Any browser -> server message, tagged by `msg`."""

class TopicUpdate(pane.PaneBase):
    """New value of one topic."""
    topic: Topic
    data: t.Any = pane.field(converter=ArrayDataConverter())
    """Topic value. Arrays are sent as frame buffers."""
    cause: t.Optional[t.Any] = None
    """Opaque value identifying what triggered the update, if any."""

class UpdatesMessage(pane.PaneBase):
    """Server -> client: a batch of topic updates, sent at most once per broker tick."""
    updates: t.List[TopicUpdate]
    msg: t.Literal['update'] = 'update'

class ErrorMessage(pane.PaneBase):
    """Server -> client: error on a subscription (unknown job, view, or params; or job removed)."""
    topic: Topic
    reason: str
    msg: t.Literal['error'] = 'error'

class HeartbeatAckMessage(pane.PaneBase):
    """Server -> client: immediate reply to a `HeartbeatMessage`."""
    msg: t.Literal['pong'] = 'pong'

class ServerShutdownMessage(pane.PaneBase):
    """Server -> client: the server is shutting down."""
    msg: t.Literal['shutdown'] = 'shutdown'

ServerMessage: t.TypeAlias = t.Annotated[t.Union[
    UpdatesMessage, ErrorMessage, HeartbeatAckMessage, ServerShutdownMessage,
], Tagged('msg')]
"""Any server -> browser message, tagged by `msg`."""

# server -> client messages

class WorkerState(pane.PaneBase):
    """Server -> client: state of a worker."""
    worker_id: WorkerID
    worker_type: str
    status: WorkerStatus
    links: t.Dict[str, str] = pane.field(default_factory=dict)
    current_job: t.Optional[JobID] = None
    start_time: t.Optional[datetime.datetime] = None
    hostname: t.Optional[str] = None
    backends: t.Optional[t.Sequence[t.Tuple[str, str]]] = None
    """Available `(backend, device)` pairs."""
    url: t.Optional[str] = None
    """URL the worker reports to. A manual worker is started with it by hand."""

class SlurmProfileInfo(pane.PaneBase):
    """Server -> client: a slurm profile, in the form the launch form edits."""
    name: str
    description: t.Optional[str] = None
    working_dir: t.Optional[str] = None
    output: t.Optional[str] = None
    sbatch_args: str = ''
    preamble: str = ''
    python: t.Optional[str] = None

class SlurmInfo(pane.PaneBase):
    """Server -> client: slurm availability and the configured profiles."""
    available: bool
    version: t.Optional[str] = None
    error: t.Optional[str] = None
    """Why slurm is unavailable, or why the server config couldn't be read."""
    default_profile: t.Optional[str] = None
    profiles: t.List[SlurmProfileInfo] = pane.field(default_factory=list)

class SlurmLaunch(pane.PaneBase):
    """Client -> server: launch a slurm worker, overriding the profile for this launch."""
    profile: t.Optional[str] = None
    working_dir: t.Optional[str] = None
    output: t.Optional[str] = None
    sbatch_args: t.Optional[str] = None
    preamble: t.Optional[str] = None
    python: t.Optional[str] = None

class JobState(pane.PaneBase):
    """Server -> client: state of a job."""
    job_id: JobID
    status: JobStatus
    links: t.Dict[str, str] = pane.field(default_factory=dict)
    job_name: t.Optional[str] = None
    start_time: t.Optional[datetime.datetime] = None
    state: t.Dict[str, t.Any] = pane.field(default_factory=dict)
    """Latest reconstruction state reported by the worker."""
    result: t.Optional[Result] = None
    """Terminal outcome, set once the job stops."""
    error_summary: t.Optional[str] = None
    """The exception's final line, for `result == 'errored'`. The traceback is in the log."""

class LogRecord(pane.PaneBase):
    """A stored job log record."""
    i: int
    """Index of the record within the job's log."""

    timestamp: datetime.datetime

    log: str

    logger_name: str
    log_level: int

    line_number: int
    func_name: t.Optional[str] = None
    stack_info: t.Optional[str] = None
    elapsed: float = 0.
    """Seconds since the job started."""

class LogPage(pane.PaneBase):
    """A window of log records, ascending by `i`."""
    logs: t.List[LogRecord] = pane.field(default_factory=list)
    first: t.Optional[int] = None
    """`i` of the first record returned, or `None` if the page is empty."""
    last: t.Optional[int] = None
    """`i` of the last record returned, or `None` if the page is empty."""
    count: int = 0
    """Number of records returned."""
    total: int = 0
    """Records matching `min_level`."""
    total_all: int = 0
    """Records regardless of `min_level`."""
    oldest: int = 0
    """Lowest `i` still retained."""
    has_before: bool = False
    """Whether matching records exist before `first`."""
    has_after: bool = False
    """Whether matching records exist after `last`."""
    min_level: int = 0
    """Level filter applied (the request's `min_level`, snapped down)."""

# worker -> server messages

class ConnectMessage(pane.PaneBase):
    """Worker -> server: sent when a worker starts up."""
    hostname: t.Optional[str] = None
    """Hostname the worker is running on."""
    backends: t.Optional[t.Sequence[t.Tuple[str, str]]] = None
    """Available `(backend, device)` pairs."""
    msg: t.Literal['connect'] = 'connect'

class PollMessage(pane.PaneBase):
    """Worker -> server: request a job."""
    msg: t.Literal['poll'] = 'poll'

class PingMessage(pane.PaneBase):
    """Worker -> server: liveness ping. The response may carry a signal."""
    msg: t.Literal['ping'] = 'ping'

class UpdateMessage(pane.PaneBase):
    """Worker -> server: a reconstruction state update."""
    state: t.Dict[str, t.Any] = pane.field(converter=ArrayDataConverter())
    """State fields to merge into the job's state. Arrays are sent as frame buffers."""
    job_id: JobID
    msg: t.Literal['job_update'] = 'job_update'

class JobStartMessage(pane.PaneBase):
    """Worker -> server: sent when a worker begins a job."""
    job_id: JobID
    start_time: datetime.datetime
    """Start time by the worker's clock."""
    msg: t.Literal['job_start'] = 'job_start'

class LogMessage(pane.PaneBase):
    """Worker -> server: a log entry."""
    job_id: t.Optional[JobID]

    timestamp: datetime.datetime

    log: str

    logger_name: str
    log_level: int

    line_number: int
    func_name: t.Optional[str] = None
    stack_info: t.Optional[str] = None

    msg: t.Literal['log'] = 'log'

    @classmethod
    def from_logrecord(cls, job_id: t.Optional[JobID], record: logging.LogRecord) -> Self:
        """Create a message from a `logging.LogRecord`."""
        timestamp = datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc)
        return cls(
            timestamp=timestamp,
            log=record.getMessage(),
            job_id=job_id,
            logger_name=record.name,
            log_level=record.levelno,
            line_number=record.lineno,
            func_name=record.funcName,
            stack_info=record.stack_info,
        )

    def into_record(self, i: int, elapsed: float = 0.) -> LogRecord:
        """Convert to a `LogRecord` with index `i`."""
        return LogRecord.make_unchecked(
            i, self.timestamp, self.log, self.logger_name, self.log_level,
            self.line_number, self.func_name, self.stack_info, elapsed
        )

class JobResultMessage(pane.PaneBase):
    """Worker -> server: a job has stopped, normally or abnormally."""
    job_id: JobID
    result: Result
    error: t.Optional[str] = None
    log: t.Optional[LogMessage] = None
    """Failure record for `result == 'errored'`, with the traceback in `stack_info`."""

    msg: t.Literal['job_result'] = 'job_result'

class WorkerShutdownMessage(pane.PaneBase):
    """Worker -> server: the worker has stopped."""
    result: Result
    error: t.Optional[str] = None
    detail: t.Optional[str] = None

    msg: t.Literal['shutdown'] = 'shutdown'


# server -> worker responses

class JobResponse(pane.PaneBase):
    """Server -> worker: a job to run."""
    job_id: JobID
    plan: str
    """Serialized reconstruction plan."""
    msg: t.Literal['job'] = 'job'

class OkResponse(pane.PaneBase):
    """Server -> worker: nothing to do."""
    msg: t.Literal['ok'] = 'ok'

class SignalResponse(pane.PaneBase):
    """Server -> worker: a signal to cancel the job, shut down, or reload."""
    signal: t.Literal['shutdown', 'cancel', 'reload'] = 'cancel'
    urgent: bool = False
    """Whether to stop immediately rather than at the next opportunity."""
    msg: t.Literal['signal'] = 'signal'

WorkerMessage: t.TypeAlias = t.Annotated[t.Union[
    PollMessage, ConnectMessage, PingMessage, UpdateMessage,
    LogMessage, JobStartMessage, JobResultMessage, WorkerShutdownMessage,
], Tagged('msg')]
"""Any worker -> server message, tagged by `msg`."""

ServerResponse: t.TypeAlias = t.Annotated[t.Union[
    JobResponse, SignalResponse, OkResponse
], Tagged('msg')]
"""Any server -> worker response, tagged by `msg`."""