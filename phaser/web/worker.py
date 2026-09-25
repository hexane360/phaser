import dataclasses
import datetime
import logging
import socket
import sys
import time
import traceback
import typing as t
import uuid

import backoff
import requests

import pane
from phaser.execute import EnginePlan, Observer, ReconsPlan, execute_plan
from phaser.state import PartialReconsState, ReconsState
from phaser.utils.num import get_devices, repr_device

from . import frames
from .types import (
    RELOAD_EXIT_CODE,
    UPDATE_CHUNK_SIZE,
    ConnectMessage,
    JobID,
    JobResultMessage,
    JobStartMessage,
    LogMessage,
    OkResponse,
    PingMessage,
    PollMessage,
    ServerResponse,
    SignalException,
    UpdateMessage,
    WorkerMessage,
    WorkerShutdownMessage,
)


def failure_log(job_id: JobID, e: BaseException, formatted: str) -> LogMessage:
    """The failure's log record, pointing at the frame that *raised* rather than the one
    that caught -- the same module/function/line a `logger.error` at the raise site would
    have carried. The server appends this to the job's log as-is."""
    frame, lineno = None, 0
    tb = e.__traceback__
    while tb is not None:
        frame, lineno = tb.tb_frame, tb.tb_lineno
        tb = tb.tb_next

    return LogMessage(
        job_id=job_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        log=f"Job failed: {type(e).__name__}: {e}",
        logger_name=frame.f_globals.get('__name__', '?') if frame is not None else __name__,
        log_level=logging.ERROR,
        line_number=lineno,
        func_name=frame.f_code.co_name if frame is not None else None,
        stack_info=formatted,
    )


class LogHandler(logging.Handler):
    def __init__(self, send_message: t.Callable[[WorkerMessage], ServerResponse]):
        self.send_message = send_message
        self.job_id: t.Optional[JobID] = None

        super().__init__(logging.DEBUG)

    def emit(self, record: logging.LogRecord):
        if getattr(record, 'local', False):
            # local-only logging event
            return
        try:
            self.send_message(LogMessage.from_logrecord(self.job_id, record))
        except Exception:  # noqa: BLE001
            self.handleError(record)


class WorkerObserver(Observer):
    def __init__(self, job_id: JobID, send_message: t.Callable[[WorkerMessage], ServerResponse]):
        super().__init__()

        self._send_message: t.Callable[[WorkerMessage], ServerResponse] = send_message
        self.job_id = job_id
        self.msg_time = time.monotonic()

        self.send_max_wait_time: t.Optional[float] = None
        self.send_every_group: bool = False

    def send_message(self, msg: WorkerMessage) -> t.Optional[ServerResponse]:
        try:
            resp = self._send_message(msg)
        except (requests.RequestException, pane.ConvertError):
            logging.exception("Failed to update server", extra={'local': True})
            return

        self.msg_time = time.monotonic()
        if resp.msg == 'signal':
            raise SignalException(resp.signal, resp.urgent)
        return resp

    def send_update(self, state: t.Union[ReconsState, PartialReconsState], exclude: t.AbstractSet[str] = frozenset()):
        self.send_message(UpdateMessage.make_unchecked(
            {k: v for (k, v) in dataclasses.asdict(state.to_numpy()).items() if v is not None and k not in exclude},
            self.job_id
        ))

    def init_engine(
        self, init_state: ReconsState, *, recons_name: str,
        plan: EnginePlan, **kwargs: t.Any
    ):
        self.send_max_wait_time = plan.send_max_wait_time
        self.send_every_group = plan.send_every_group
        self.send_update(init_state)

    def heartbeat(self):
        if (time.monotonic() - self.msg_time) > 5:
            self.send_message(PingMessage())

    def update_group(self, state: t.Union[ReconsState, PartialReconsState], force: bool = False):
        if self.send_every_group or (self.send_max_wait_time is not None and (time.monotonic() - self.msg_time) > self.send_max_wait_time):
            # `progress` is sent per iteration only
            self.send_update(state, exclude={'progress'})

    def update_iteration(self, state: ReconsState, i: int, n: int, errors: t.Dict[str, float]):
        self.send_update(state)


REQUEST_TIMEOUT: t.Tuple[float, float] = (10., 60.)
"""(connect, read) timeout for requests to the server"""


def _error_detail(resp: requests.Response) -> str:
    try:
        return str(resp.json()['msg'])
    except (ValueError, KeyError, TypeError):
        return resp.text[:500]


def run_worker(url: str, quiet: bool = False):
    connect_message = ConnectMessage(
        hostname=socket.gethostname(),
        backends=tuple((backend, repr_device(device)) for (backend, device) in get_devices())
    )

    def post(body: bytes, params: t.Optional[t.Dict[str, t.Any]] = None,
             content_type: str = frames.CONTENT_TYPE,
             session: t.Optional[requests.Session] = None) -> ServerResponse:
        resp = (session or requests).post(url, data=body, params=params, timeout=REQUEST_TIMEOUT,
                                          headers={'Content-Type': content_type})
        if not resp.ok:
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} ({len(body)} bytes sent): {_error_detail(resp)}",
                response=resp
            )
        return pane.convert(resp.json(), ServerResponse)  # type: ignore

    # retry dropped chunks, rather than dropping the whole update
    @backoff.on_exception(backoff.fibo, (requests.ConnectionError, requests.Timeout),
                          max_tries=5, max_time=60)
    def post_chunk(session: requests.Session, body: bytes, upload_id: str, index: int, count: int) -> ServerResponse:
        return post(body, {'upload': upload_id, 'index': index, 'count': count}, 'application/octet-stream', session)

    def send_message(msg: WorkerMessage) -> ServerResponse:
        body = frames.pack_bytes(msg.into_data())
        if not isinstance(msg, UpdateMessage) or len(body) <= UPDATE_CHUNK_SIZE:
            return post(body)

        upload_id = uuid.uuid4().hex
        count = -(-len(body) // UPDATE_CHUNK_SIZE)
        resp: ServerResponse = OkResponse()
        # one connection for the whole upload
        with requests.Session() as session:
            for i in range(count):
                resp = post_chunk(session, body[i * UPDATE_CHUNK_SIZE:(i + 1) * UPDATE_CHUNK_SIZE], upload_id, i, count)
        return resp

    # make inital connection to server
    # this has a relatively short backoff, so we can give up early
    @backoff.on_exception(backoff.fibo, requests.RequestException,
                          max_tries=10, max_time=30,
                          giveup=lambda e: isinstance(e, requests.HTTPError))  # giveup on 404, etc.
    def startup() -> ServerResponse:
        return send_message(connect_message)

    # poll for a job from the server
    # for timeouts, we will eventualy fail and exit the loop
    # if we receive a response, however, we loop forever
    @backoff.on_predicate(backoff.fibo, lambda resp: resp.msg == 'ok',
                          max_value=10)
    @backoff.on_exception(backoff.fibo, requests.RequestException,
                          max_value=10, max_tries=60)
    @backoff.on_exception(backoff.fibo, requests.Timeout,
                          max_tries=10, max_time=60)
    def poll() -> ServerResponse:
        return send_message(PollMessage())

    # send job result
    @backoff.on_exception(backoff.fibo, requests.RequestException,
                          max_tries=10, max_time=30)
    def send_result(msg: JobResultMessage) -> ServerResponse:
        return send_message(msg)

    # send the parting message; it's the server's only account of why we went away
    @backoff.on_exception(backoff.fibo, requests.RequestException,
                          max_tries=10, max_time=30)
    def send_shutdown(msg: WorkerShutdownMessage) -> ServerResponse:
        return send_message(msg)

    log_handler = LogHandler(send_message)
    logging.basicConfig(level=logging.INFO,
        handlers=[log_handler] if quiet else [logging.StreamHandler(), log_handler]
    )
    logger = logging.getLogger('worker')

    try:
        action: t.Optional[t.Literal['shutdown', 'reload']] = None
        resp = startup()
        logger.info("Worker connected to server, response: %r", resp, extra={'local': True})

        while not action:
            if resp.msg == 'ok':
                # poll for a new job
                resp = poll()
                logger.info("Worker polled server, response: %r", resp, extra={'local': True})

            if resp.msg == 'signal' and resp.signal != 'cancel':
                action = resp.signal
                continue

            assert resp.msg == 'job'

            try:
                # run job
                log_handler.job_id = resp.job_id

                # report the start time by our own clock -- the same one log records are
                # stamped with, and the anchor the server measures `elapsed` from
                try:
                    send_message(JobStartMessage(resp.job_id, datetime.datetime.now(datetime.timezone.utc)))
                except requests.RequestException:
                    logger.exception("Failed to report job start time", extra={'local': True})

                plan = ReconsPlan.from_jsons(resp.plan)
                execute_plan(plan, observers=WorkerObserver(resp.job_id, send_message))

            except SignalException as e:
                logger.info("Job cancelled", extra={'local': True})
                msg = JobResultMessage(resp.job_id, 'cancelled')
                if e.signal != 'cancel':
                    action = e.signal
            except KeyboardInterrupt:
                logger.info("Job interrupted", extra={'local': True})
                msg = JobResultMessage(resp.job_id, 'interrupted')
                resp = send_result(msg)
                raise
            except BaseException as e:
                logger.info("Job stopped due to error", exc_info=True, stack_info=True, extra={'local': True})
                s = traceback.format_exc()
                msg = JobResultMessage(resp.job_id, 'errored', s, failure_log(resp.job_id, e, s))
            else:
                logger.info("Job finished successfully", extra={'local': True})
                msg = JobResultMessage(resp.job_id, 'finished')

            resp = send_result(msg)
            log_handler.job_id = None

    except BaseException as e:
        # disconnect message
        logger.error(
            "Worker interrupted" if isinstance(e, KeyboardInterrupt) else "Worker shutting down due to error",  # noqa: TRY401
            exc_info=None if isinstance(e, KeyboardInterrupt) else True,
            extra={'local': True}
        )
        s = traceback.format_exc()
        msg = WorkerShutdownMessage('interrupted' if isinstance(e, KeyboardInterrupt) else 'errored', error=s)
    else:
        if action == 'reload':
            logger.info("Worker reloading", extra={'local': True})
            # instead of sending disconnect message, signal here
            sys.exit(RELOAD_EXIT_CODE)

        # disconnect message
        logger.info("Worker shutting down normally", extra={'local': True})
        msg = WorkerShutdownMessage('finished')

    try:
        send_shutdown(msg)
    except Exception:
        logger.exception("Failed to send shutdown message", extra={'local': True})