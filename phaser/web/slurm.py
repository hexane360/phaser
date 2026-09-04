import asyncio
import logging
import weakref
import json
import os
import re
import importlib.resources
import typing as t

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


class SlurmWorker(Worker):
    def __init__(self, worker_id: WorkerID, slurm_job_id: SlurmID):
        self.slurm_job_id: SlurmID = slurm_job_id
        super().__init__(worker_id)

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
        self._slurm_workers: weakref.WeakValueDictionary[SlurmID, SlurmWorker] = weakref.WeakValueDictionary()

        self._poll_task: t.Optional[asyncio.Task[None]] = None
        self._new_worker_event: asyncio.Event = asyncio.Event()

    async def make_worker(self, worker_id: WorkerID, url: str) -> SlurmWorker:
        await self.check_slurm_exists()

        job_name = f"phaser_{worker_id}"

        from .. import web
        with importlib.resources.path(web, 'slurm_worker.sh') as script_path:
            slurm_args = "--qos=high --time=4-0 --partition=xeon-g6-volta --cpus-per-task=20 --gres=gpu:volta:1 --signal=SIGINT@120"
            #slurm_args = "--qos=high --time=01:00:00 --partition=debug-cpu --cpus-per-task=20"
            shell = os.environ.get('SHELL', '/bin/bash')
            proc = await asyncio.create_subprocess_exec(
                shell, "-c", f"sbatch --job-name={job_name} {slurm_args} \"{script_path.absolute()}\" \"{url}\"",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(f"Failed to start slurm worker, stderr: '{stderr}'")

            try:
                match = re.fullmatch(r'Submitted batch job (\d+)', stdout.decode().strip())
                job_id: SlurmID = int(match[1])  # type: ignore
            except (ValueError, TypeError):
                raise ValueError(f"Failed to parse slurm return: '{stdout.decode()}'")

            worker = SlurmWorker(worker_id, job_id)
            self._slurm_workers[job_id] = worker

            if self._poll_task is None:
                self._poll_task = asyncio.create_task(self._poll_slurm())
            else:
                self._new_worker_event.set()

            return worker

    async def cancel_queued_worker(self, slurm_job_id: SlurmID):
        proc = await asyncio.create_subprocess_shell(
            f"scancel --state=PENDING {slurm_job_id}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            self.logger.warning(f"Failed to cancel slurm job: '{stderr.decode()}'")

    async def _poll_slurm(self):
        while True:
            while True:
                try:
                    await self._poll_slurm_status()
                except Exception as e:
                    self.logger.warning(f"Failed to poll slurm worker statuses: {e}")

                if any(worker.state == 'queued' for worker in self._slurm_workers.values()):
                    # fast wait cycle
                    await asyncio.sleep(5.0)
                elif any(worker.state != 'stopped' for worker in self._slurm_workers.values()):
                    # if there's any workers to wait for, slow wait cycle
                    await asyncio.sleep(30.0)
                else:
                    # no workers waiting, wait until one is started
                    self._new_worker_event.clear()
                    break
            await self._new_worker_event.wait()

    async def _poll_slurm_status(self):
        jobs = ",".join(str(worker.slurm_job_id) for worker in self._slurm_workers.values())

        proc = await asyncio.create_subprocess_shell(
            f"squeue --job={jobs} --states=all --json",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            self.logger.warning(f"Failed to poll slurm worker statuses, stderr: '{stderr}'")
            return

        d = t.cast(SqueueResult, json.loads(stdout))

        for job in d["jobs"]:
            try:
                job_id = job['job_id']
                job_state = job['job_state']
                worker = self._slurm_workers[job_id]
            except KeyError:
                continue

            if isinstance(job_state, str):
                job_state = [job_state]

            if any(j in ('CONFIGURING', 'RUNNING') for j in job_state):
                if worker.status == 'queued':
                    await worker.set_status('starting')
            elif not any(j == 'PENDING' for j in job_state):
                self.logger.debug(f"Stopping worker {job_id}, squeue got job_state: {job_state}")
                if worker.status != 'stopped':
                    # TODO grab some exit information here
                    await worker.set_status('stopped')

    async def check_slurm_exists(self):
        if self._slurm_exists is None:
            proc = await asyncio.create_subprocess_shell("sbatch --version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                self._slurm_exists = True
                self.logger.info(f"Slurm found, version '{stdout.decode().strip()}'")
            else:
                self._slurm_exists = False
                self.logger.warning(f"Slurm not found, stderr '{stderr.decode()}'")

        if not self._slurm_exists:
            raise RuntimeError("Slurm not found, is it installed and on PATH?")

    async def finalize(self):
        if self._poll_task is not None:
            self._poll_task.cancel()
