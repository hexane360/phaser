import asyncio
import logging
import weakref
import json
import re
import shlex
import importlib.resources
import typing as t

from .config import SlurmProfile
from .server import WorkerID, Worker


SlurmID: t.TypeAlias = int
SlurmState: t.TypeAlias = t.Literal[
    'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'CONFIGURING', 'COMPLETING',
    'DEADLINE', 'FAILED', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PENDING',
    'PREEMPTED', 'RUNNING', 'RESV_DEL_HOLD', 'REQUEUE_FED', 'REQUEUE_HOLD',
    'REQUEUED', 'RESIZING', 'REVOKED', 'SIGNALING', 'SPECIAL_EXIT',
    'STAGE_OUT', 'STOPPED', 'SUSPENDED', 'TIMEOUT',
]


class SlurmJobInfo(t.TypedDict):
    job_id: SlurmID
    job_state: SlurmState
    name: str
    nodes: str
    user_name: str

    # path to stdout and stderr
    standard_output: str
    standard_error: str

    submit_time: int
    start_time: int
    eligible_time: int
    end_time: int


class SqueueResult(t.TypedDict):
    meta: t.Dict[str, t.Any]
    jobs: t.List[SlurmJobInfo]
    warnings: t.List[t.Any]
    errors: t.List[t.Any]


SLURM_JSON_VERSION: t.Tuple[int, int] = (21, 8)
"""First slurm version supporting `squeue --json`"""


def parse_slurm_version(output: str) -> t.Optional[t.Tuple[int, int]]:
    """Major/minor version from `sbatch --version` output (`slurm 23.02.7`, `slurm-wlm 21.08.5`)"""
    if (match := re.search(r'(\d+)\.(\d+)', output)) is None:
        return None
    return (int(match[1]), int(match[2]))


def _parse_job_id(job_id: t.Any) -> t.Optional[SlurmID]:
    """Job id as an int, or `None` for anything else (an array task, `123_4`, included)"""
    if isinstance(job_id, int):
        return job_id
    if isinstance(job_id, str) and (job_id := job_id.strip()).isdigit():
        return int(job_id)
    return None


def parse_squeue_json(output: t.Union[str, bytes]) -> t.Dict[SlurmID, t.List[str]]:
    result = t.cast(SqueueResult, json.loads(output))

    jobs: t.Dict[SlurmID, t.List[str]] = {}
    for job in result['jobs']:
        if (job_id := _parse_job_id(job.get('job_id'))) is None:
            continue
        state = job.get('job_state')
        jobs[job_id] = [state] if isinstance(state, str) else list(state or ())
    return jobs


def parse_squeue_tabular(output: str) -> t.Dict[SlurmID, t.List[str]]:
    """Parse `squeue --noheader --format=%i|%T` output. Array tasks (`123_4`) are skipped."""
    jobs: t.Dict[SlurmID, t.List[str]] = {}
    for line in output.splitlines():
        if not (line := line.strip()):
            continue
        (job_id_s, _, state) = line.partition('|')
        if (job_id := _parse_job_id(job_id_s)) is None:
            continue
        if (state := state.strip()):
            jobs[job_id] = [state]
    return jobs


_MARKER_RE: re.Pattern[str] = re.compile(r'@(PREAMBLE|PYTHON|URL)@')


def render_worker_script(profile: SlurmProfile, url: str) -> str:
    """The batch script for a worker running `profile` and connecting to `url`"""
    template = importlib.resources.files(__package__).joinpath('slurm_worker.sh').read_text()

    # substituted in one pass, so a marker inside the preamble is left alone
    values = {
        'PREAMBLE': profile.preamble.strip('\n'),
        'PYTHON': shlex.quote(profile.python_exec()),
        'URL': shlex.quote(url),
    }
    return _MARKER_RE.sub(lambda match: values[match[1]], template)


class SlurmError(RuntimeError):
    """A submission slurm refused. The message is slurm's own, meant to be shown to whoever
    asked for the worker."""


class SlurmWorker(Worker):
    def __init__(self, worker_id: WorkerID, slurm_job_id: SlurmID,
                 profile_name: t.Optional[str] = None, url: t.Optional[str] = None):
        super().__init__(worker_id, url)
        self.slurm_job_id: SlurmID = slurm_job_id
        self.profile_name: t.Optional[str] = profile_name

    def worker_type(self) -> str:
        return 'slurm'

    async def cancel(self):
        from .server import server
        if self.status == 'queued':
            await server.slurm_manager.cancel_queued_worker(self.slurm_job_id)

        await self.set_status('stopping')


class SlurmManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._slurm_exists: t.Optional[bool] = None
        self._slurm_version: t.Optional[t.Tuple[int, int]] = None
        self.version: t.Optional[str] = None
        """Version string reported by `sbatch --version`"""
        self._use_json: bool = False
        self._slurm_workers: weakref.WeakValueDictionary[SlurmID, SlurmWorker] = weakref.WeakValueDictionary()

        self._poll_task: t.Optional[asyncio.Task[None]] = None
        self._new_worker_event: asyncio.Event = asyncio.Event()

    async def make_worker(
        self, worker_id: WorkerID, url: str,
        profile: t.Optional[SlurmProfile] = None, profile_name: t.Optional[str] = None,
    ) -> SlurmWorker:
        await self.check_slurm_exists()

        profile = profile if profile is not None else SlurmProfile()
        script = render_worker_script(profile, url)
        args = [
            'sbatch', '--parsable', f"--job-name=phaser_{worker_id}",
            *profile.sbatch_options(worker_id), *profile.args(),
        ]

        self.logger.info(f"Submitting slurm worker {worker_id}: {shlex.join(args)}")

        # the script is passed on stdin, so it needs no temporary file
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        (stdout, stderr) = await proc.communicate(script.encode())

        if proc.returncode != 0:
            # sbatch explains itself ("error: invalid partition specified"), so its own
            # message is what's worth reporting
            detail = (stderr.decode().strip() or stdout.decode().strip()
                      or f"sbatch exited with code {proc.returncode}")
            raise SlurmError(detail)

        # `--parsable` prints 'jobid' or 'jobid;cluster'
        try:
            job_id: SlurmID = int(stdout.decode().strip().split(';')[0])
        except ValueError:
            raise SlurmError(f"Couldn't parse the job id sbatch printed: '{stdout.decode().strip()}'") from None

        worker = SlurmWorker(worker_id, job_id, profile_name, url)
        self._slurm_workers[job_id] = worker

        if self._poll_task is None:
            self._poll_task = asyncio.create_task(self._poll_slurm())
        else:
            self._new_worker_event.set()

        return worker

    async def cancel_queued_worker(self, slurm_job_id: SlurmID):
        (returncode, _, stderr) = await _run('scancel', '--state=PENDING', str(slurm_job_id))

        if returncode != 0:
            self.logger.warning(f"Failed to cancel slurm job: '{stderr.decode()}'")

    async def _poll_slurm(self):
        while True:
            while True:
                try:
                    await self._poll_slurm_status()
                except Exception as e:
                    self.logger.warning(f"Failed to poll slurm worker statuses: {e}")

                if any(worker.status == 'queued' for worker in self._slurm_workers.values()):
                    # fast wait cycle
                    await asyncio.sleep(5.0)
                elif any(worker.status != 'stopped' for worker in self._slurm_workers.values()):
                    # if there's any workers to wait for, slow wait cycle
                    await asyncio.sleep(30.0)
                else:
                    # no workers waiting, wait until one is started
                    self._new_worker_event.clear()
                    break
            await self._new_worker_event.wait()

    async def _poll_slurm_status(self):
        jobs = await self._squeue([str(worker.slurm_job_id) for worker in self._slurm_workers.values()])

        for (job_id, job_state) in jobs.items():
            try:
                worker = self._slurm_workers[job_id]
            except KeyError:
                continue

            if any(j in ('CONFIGURING', 'RUNNING') for j in job_state):
                if worker.status == 'queued':
                    await worker.set_status('starting')
            elif not any(j == 'PENDING' for j in job_state):
                self.logger.debug(f"Stopping worker {job_id}, squeue got job_state: {job_state}")
                if worker.status != 'stopped':
                    # TODO grab some exit information here
                    await worker.set_status('stopped')

    async def _squeue(self, job_ids: t.Sequence[str]) -> t.Dict[SlurmID, t.List[str]]:
        """Job states from `squeue`, keyed by job id.

        Uses `--json` where supported, and the tabular format otherwise. A `--json` which
        fails at runtime (a build without a serializer plugin) latches off for the session.
        """
        args = ['squeue', f"--job={','.join(job_ids)}", '--states=all']

        if self._use_json:
            (returncode, stdout, stderr) = await _run(*args, '--json')
            if returncode == 0:
                return parse_squeue_json(stdout)

            self._use_json = False
            self.logger.warning(
                f"'squeue --json' failed, falling back to tabular output. stderr: '{stderr.decode().strip()}'"
            )

        (returncode, stdout, stderr) = await _run(*args, '--noheader', '--format=%i|%T')
        if returncode != 0:
            self.logger.warning(f"Failed to poll slurm worker statuses, stderr: '{stderr.decode().strip()}'")
            return {}

        return parse_squeue_tabular(stdout.decode())

    async def check_slurm_exists(self):
        if self._slurm_exists is None:
            (returncode, stdout, stderr) = await _run('sbatch', '--version')

            if returncode == 0:
                version = self.version = stdout.decode().strip()
                self._slurm_exists = True
                self._slurm_version = parse_slurm_version(version)
                self._use_json = self._slurm_version is not None and self._slurm_version >= SLURM_JSON_VERSION
                self.logger.info(
                    f"Slurm found, version '{version}'{' (has --json)' if self._use_json else ''}"
                )
            else:
                self._slurm_exists = False
                self.logger.warning(f"Slurm not found, stderr '{stderr.decode()}'")

        if not self._slurm_exists:
            raise RuntimeError("Slurm not found, is it installed and on PATH?")

    async def finalize(self):
        if self._poll_task is not None:
            self._poll_task.cancel()


async def _run(*args: str) -> t.Tuple[t.Optional[int], bytes, bytes]:
    """Run a command, returning `(returncode, stdout, stderr)`.

    A command which can't be run at all reports itself the way a shell would, as exit code
    127, rather than raising.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        return (127, b'', f"{args[0]}: {e.strerror}".encode())

    (stdout, stderr) = await proc.communicate()
    return (proc.returncode, stdout, stderr)
