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

export interface JobState {
    job_id: string;
    status: JobStatus;
    job_name: string | null;
    links: Record<string, string>;
    start_time: string | null;
    state: PartialReconsData;
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

export interface ProbeData {
    sampling: Sampling;
    data: DecodedArray;
};

// client-side view of a probe's rarely-changing shape (as opposed to its per-tick array data)
export interface ProbeMeta {
    sampling: Sampling;
    nprobes: number;
}

export interface ObjectData {
    sampling: ObjectSampling;
    data: DecodedArray;
    thicknesses: DecodedArray;
};

// payload of the `obj_phase_sum` view: the object phase, already projected (angle +
// nansum over slices) server-side -- see `phaser/web/views.py::project_phase`.
export interface ObjPhaseSumData {
    sampling: ObjectSampling;
    data: DecodedArray; // real-valued, 2D
}

// payload of the `obj` view (a single slice, selected by the `slice` param). Wired but
// unused in v1 (no slice slider yet) -- see `phaser/web/views.py::slice_view`.
export interface ObjSliceData {
    sampling: ObjectSampling;
    data: DecodedArray;
    thickness: number | null;
}

export interface ProgressData {
    iters: [number];
    values: [number];
}