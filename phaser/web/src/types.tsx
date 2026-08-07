import { DecodedArray } from './array';

// pub/sub wire protocol (browser <-> server). Mirrors `phaser/web/types.py`.

export type TopicKey = string | number;
export type Topic = string | Record<string, TopicKey>;

// Canonical JSON key for a `Topic`: sorted keys, no whitespace. Must agree with the
// Python `canonical_topic` implementation in `phaser/web/types.py`.
export function canonicalTopic(topic: Topic): string {
    if (typeof topic === 'string') return JSON.stringify(topic);
    const sorted: Record<string, TopicKey> = {};
    for (const k of Object.keys(topic).sort()) sorted[k] = topic[k];
    return JSON.stringify(sorted);
}

export interface SubscribeMessage {
    topics: Array<Topic>;
    msg: "sub";
}

export interface UnsubscribeMessage {
    topics: Array<Topic>;
    msg: "unsub";
}

export interface HeartbeatMessage {
    msg: "ping";
}

export type ClientMessage = SubscribeMessage | UnsubscribeMessage | HeartbeatMessage;

export interface TopicUpdate {
    topic: Topic;
    data: any;
    cause?: any;
}

export interface UpdatesMessage {
    updates: Array<TopicUpdate>;
    msg: "update";
}

export interface ErrorMessage {
    topic: Topic;
    reason: string;
    msg: "error";
}

export interface HeartbeatAckMessage {
    msg: "pong";
}

export interface ServerShutdownMessage {
    msg: "shutdown";
}

export type ServerMessage = UpdatesMessage | ErrorMessage | HeartbeatAckMessage | ServerShutdownMessage;

export type WorkerStatus = "queued" | "starting" | "reloading" | "idle" | "running" | "stopping" | "stopped" | "unknown";
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

export interface JobState {
    job_id: string;
    status: JobStatus;
    job_name: string | null;
    links: Record<string, string>;
    start_time: string | null;
    state: PartialReconsData;
    // terminal outcome, null while the job is still live. The traceback behind an
    // `errored` result isn't here -- it's an ERROR record in the job's log.
    result: Result | null;
    error_summary: string | null;
};

export interface LogRecord {
    i: number;
    timestamp: string;  // ISO 8601 format

    log: string;
    logger_name: string;
    log_level: number;

    line_number: number;
    func_name: string | null;
    stack_info: string | null;
    elapsed: number;  // seconds since the job started
}

// A window of log records from `/job/<id>/logs`, ascending by `i`. `first`/`last` are
// null when the page is empty; `has_before`/`has_after` say whether more matching
// records exist on either side.
export interface LogsData {
    logs: ReadonlyArray<LogRecord>;
    first: number | null;
    last: number | null;
    count: number;
    total: number;      // records matching `min_level`
    total_all: number;  // records regardless of `min_level`
    oldest: number;
    has_before: boolean;
    has_after: boolean;
    min_level: number;
}

export interface ReconsData {
    iter: IterData;
    probe: ProbeData;
    object: ObjectData;
    scan: DecodedArray;
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

// payload of the `probe_meta` view. `wavelength` is in the same length units as
// `sampling` (Angstrom); it sets the reciprocal-space view's mrad scales.
export interface ProbeMeta {
    sampling: Sampling;
    nprobes: number;
    wavelength: number;
}

// payload of the `obj_meta` view. `thicknesses` is null unless the object is genuinely
// multislice (mirroring `ObjectState.thicknesses`, "length < 2 for single slice"), and is
// otherwise exactly `n_slices` long. `n_slices` is 1 for a 2D object.
export interface ObjMeta {
    sampling: ObjectSampling;
    n_slices: number;
    thicknesses: Array<number> | null;
}

// `ObjectState`/`ProbeState` as the worker sends them (see `ReconsData`), not as any view
// publishes them -- the views above split these into metadata and a bare array.
export interface ObjectData {
    sampling: ObjectSampling;
    data: DecodedArray;
    thicknesses: DecodedArray;
};

export interface ProbeData {
    sampling: Sampling;
    data: DecodedArray;
};

export interface ProgressData {
    iters: [number];
    values: [number];
}