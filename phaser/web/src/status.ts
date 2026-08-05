import { JobState, JobStatus, LogsData, Result, WorkerStatus } from './types';

const LEVEL_ERROR = 40;

// How a job or worker reads in the UI. `status` is a lifecycle and `result` is how the job
// ended, so a stopped job needs both to say anything useful -- "stopped" alone can't tell
// a crash from a clean finish. Shared by the manager list and the dashboard header so the
// two can't drift.
export interface StatusDisplay {
    label: string;
    color: string;
}

// semantic status colors, read from Mantine color-scheme-aware tokens
const GREEN = 'var(--mantine-color-green-text)';
const ORANGE = 'var(--mantine-color-orange-text)';
const RED = 'var(--mantine-color-red-text)';
const GRAY = 'var(--mantine-color-gray-text)';

const RESULTS: Record<Result, StatusDisplay> = {
    finished: { label: 'completed', color: GREEN },
    errored: { label: 'failed', color: RED },
    cancelled: { label: 'cancelled', color: GRAY },
    interrupted: { label: 'interrupted', color: GRAY },
};

const LIFECYCLE: Record<JobStatus, StatusDisplay> = {
    queued: { label: 'queued', color: ORANGE },
    starting: { label: 'starting', color: ORANGE },
    running: { label: 'running', color: GREEN },
    stopping: { label: 'stopping', color: RED },
    // only reached when a job stopped without reporting a result -- its worker went away
    stopped: { label: 'stopped', color: GRAY },
};

export function jobStatus(job: Pick<JobState, 'status' | 'result'>): StatusDisplay {
    if (job.status === 'stopped' && job.result) return RESULTS[job.result];
    return LIFECYCLE[job.status] ?? { label: job.status, color: GRAY };
}

// Whether the job ended badly enough to account for itself. Keys on `error_summary` rather
// than `result`, so it covers both a reconstruction that raised (`errored`) and a job whose
// worker went away (`interrupted`, with a reason) -- but not an ordinary cancel.
export function jobHasFailure(job: Pick<JobState, 'error_summary'>): boolean {
    return job.error_summary !== null;
}

// The detail behind a failure. Deliberately not in `JobState` -- it's an ERROR record in the
// job's log, so it costs nothing until a failure is actually being looked at. `logsUrl` is
// `JobState.links.logs`.
export async function fetchTraceback(logsUrl: string): Promise<string | null> {
    const response = await fetch(`${logsUrl}?min_level=${LEVEL_ERROR}&limit=1`);
    if (!response.ok) {
        throw new Error(`Failed to fetch traceback: ${response.status} ${response.statusText}`);
    }
    const page = await response.json() as LogsData;
    return page.logs[page.logs.length - 1]?.stack_info ?? null;
}

export function workerStatusColor(status: WorkerStatus): string {
    switch (status) {
        case 'running': return GREEN;
        case 'queued':
        case 'starting':
        case 'reloading': return ORANGE;
        case 'stopping': return RED;
        default: return GRAY;  // idle, stopped, unknown
    }
}
