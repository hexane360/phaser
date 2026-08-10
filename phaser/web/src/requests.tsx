// The one way the UI talks to the server. `usePostAction` and `useGetAction` differ only in
// what they do around the request -- a mutation waits for a live socket first, a read doesn't --
// and share the pending flag, the error normalization, and the reporting.

import React from 'react';
import ky, { HTTPError, TimeoutError } from 'ky';

import { NotConnectedError } from './connection';
import { usePubSubConnection } from './pubsub';
import { reportError, ReportOptions } from './notify';

const REQUEST_TIMEOUT_MS = 5_000;

export interface ActionOptions extends ReportOptions {
    // Suppresses the notification, leaving only the returned `error`. For a fetch the user
    // didn't ask for, where a toast would be noise.
    quiet?: boolean;
}

// `[run, pending, error]`. `run` never throws: a failure is reported and returned as `null`.
export type Action<A extends Array<any>, T> = [
    (...args: A) => Promise<T | null>, boolean, string | null,
];

// What the server said, in a form worth showing someone. Every HTTP error from `routes.py`
// carries `{'result': 'error', 'msg'}`, which is far more use than its status line.
export async function errorMessage(error: unknown): Promise<string> {
    if (error instanceof HTTPError) {
        const status = `HTTP ${error.response.status} ${error.response.statusText}`;
        let text: string;
        try {
            text = await error.response.text();
        } catch {
            return status;
        }
        try {
            const json = JSON.parse(text);
            return json.result === 'error' && json.msg ? json.msg : JSON.stringify(json);
        } catch {
            return text || status;
        }
    }
    if (error instanceof TimeoutError) return "Request timed out";
    // includes `NotConnectedError`, whose message is already the thing to show
    if (error instanceof Error) return error.message;
    return String(error);
}

async function parse(response: Response): Promise<any> {
    const type = response.headers.get('content-type') ?? '';
    return type.includes('application/json') ? await response.json() : await response.text();
}

// `run` is re-read through a ref rather than captured, so the returned callback stays stable
// across renders even though every caller passes a fresh closure.
function useAction<A extends Array<any>, T>(
    title: string, {quiet, block}: ActionOptions, run: (...args: A) => Promise<T>,
): Action<A, T> {
    const [pending, setPending] = React.useState(0);
    const [error, setError] = React.useState<string | null>(null);
    const latest = React.useRef(run);
    latest.current = run;

    const call = React.useCallback(async (...args: A) => {
        setPending((n) => n + 1);
        try {
            const result = await latest.current(...args);
            setError(null);
            return result;
        } catch (e) {
            const msg = await errorMessage(e);
            setError(msg);
            if (quiet) console.error(`${title}:`, e);
            else reportError(title, msg, {block});
            return null;
        } finally {
            setPending((n) => n - 1);
        }
    }, [title, quiet, block]);

    return [call, pending > 0, error];
}

// A mutation. Gated on a live websocket: its result arrives as a pub/sub update rather than in
// the response, so one POSTed while the socket is down succeeds *invisibly* -- the server acts
// on it and the page never hears. `title` names the failed action ("Couldn't cancel job").
//
// `body` is sent as-is if a string, as JSON otherwise. Mutations are never retried (`ky`
// excludes POST by default), so a cancel can't fire twice.
export function usePostAction(title: string, options: ActionOptions = {}): Action<[url: string, body?: unknown], any> {
    const conn = usePubSubConnection();

    return useAction(title, options, async (url: string, body?: unknown) => {
        if (!conn) throw new NotConnectedError();
        await conn.ensureConnected();

        const payload = body === undefined ? {body: ""} : typeof body === 'string' ? {body} : {json: body};
        return await parse(await ky(url, {method: "post", timeout: REQUEST_TIMEOUT_MS, ...payload}));
    });
}

// A read. No connectivity gate -- a GET carries its own answer, so a dead socket doesn't make
// it invisible the way it does a mutation.
export function useGetAction<T>(title: string, options: ActionOptions = {}): Action<[url: string], T> {
    return useAction(title, options, async (url: string) => await ky(url, {timeout: REQUEST_TIMEOUT_MS}).json<T>());
}
