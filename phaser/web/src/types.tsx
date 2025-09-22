import { NArray } from 'wasm-array';

export type WorkerStatus = "queued" | "starting" | "idle" | "running" | "stopping" | "stopped" | "unknown";
export type JobStatus = "queued" | "starting" | "running" | "stopping" | "stopped";
export type Result = "finished" | "errored" | "cancelled" | "interrupted";

export interface WorkerState {
    worker_id: string;
    worker_type: string;
    status: WorkerStatus;
    links: Record<string, string>;
    current_job: string | null;
    start_time: string | null;
    hostname: string | null;
    backends: Array<[string, string]> | null;
}

export interface WorkerUpdate {
    worker_id: string;
    status: WorkerStatus;

    msg: "status_change";
}

export interface JobState {
    job_id: string;
    status: JobStatus;
    job_name: string | null;
    links: Record<string, string>;
    start_time: string | null;
    state: PartialReconsData;
};

export interface JobStatusChange {
    status: JobStatus;
    job_id: string;

    msg: "status_change";
}

export interface JobUpdate {
    state: PartialReconsData;
    job_id: string;

    msg: "job_update";
}

export interface LogUpdate {
    new_logs: Array<LogRecord>;
    msg: "log";
}

export interface JobStopped {
    result: Result;
    error: string | null;

    msg: "job_stopped";
}

export type JobMessage = JobStatusChange | JobUpdate | LogUpdate | JobStopped;

export interface DashboardConnected {
    state: JobState;
    msg: "connected";
}

export type DashboardMessage = JobMessage | DashboardConnected;

export interface JobsUpdate {
    event: JobMessage | null;
    state: Array<JobState>;

    msg: "jobs_update";
}

export interface WorkersUpdate {
    event: WorkerUpdate | null;
    state: Array<WorkerState>;

    msg: "workers_update";
}

export interface ManagerConnected {
    workers: Array<WorkerState>;
    jobs: Array<JobState>;
    msg: "connected";
}

export type ManagerMessage = JobsUpdate | WorkersUpdate | ManagerConnected;

export interface LogRecord {
    i: number;
    timestamp: string;  // ISO 8601 format

    log: string;
    logger_name: string;
    log_level: number;

    line_number: number;
    func_name: string | null;
    stack_info: string | null;
}

export interface LogsData {
    first: number;
    last: number;
    length: number;
    total_length: number;
    logs: ReadonlyArray<LogRecord>;
}

export interface ReconsData {
    iter: IterData;
    probe: ProbeData;
    object: ObjectData;
    scan: NArray;
    progress: Record<string, ProgressData>;
}

export type PartialReconsData = { [P in keyof ReconsData]?: ReconsData[P] | null | undefined };

export interface IterData {
    engine_num: number;
    engine_iter: number;
    total_iter: number;
    n_engine_iters: number | null;
    n_total_iters: number | null;
}

export interface Sampling {
    shape: [number, number];
    extent: [number, number];
    sampling: [number, number];
}

export interface ObjectSampling {
    shape: [number, number];
    sampling: [number, number];
    corner: [number, number];
    region_min: [number, number] | null;
    region_max: [number, number] | null;
}

export interface ProbeData {
    sampling: Sampling;
    data: NArray;
};

export interface ObjectData {
    sampling: ObjectSampling;
    data: NArray;
    thicknesses: NArray;
};

export interface ProgressData {
    iters: [number];
    values: [number];
}