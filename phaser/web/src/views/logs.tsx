import React from 'react';
import { useAtomValue, useStore } from 'jotai';

import { LogRecord, LogsData } from '../types';
import { usePubSubView } from '../pubsub';
import { ViewProps } from './types';

// `logs` is the one append-conflated topic: each message is new records, never a whole
// value (see `phaser/web/views.py`). It's also non-retained, so subscribing yields no
// history -- `hydrate` fetches that over REST, on first subscribe and after each reconnect.
// Both live in the shared per-topic entry, so a second Log widget costs neither.
const LOGS_OPTIONS = {
    initial: [] as Array<LogRecord>,
    reduce: (prev: Array<LogRecord>, delta: Array<LogRecord>) => [...prev, ...delta],
    hydrate: async () => (await getLogs()).logs as Array<LogRecord>,
    // a reconnect: keep what we have and append only what's newer. Bounded by the `limit`
    // (100) records the endpoint returns, so a longer outage still leaves a gap.
    merge: (prev: Array<LogRecord> | undefined, fetched: Array<LogRecord>) => {
        if (!prev?.length) return [...fetched];
        const lastKnown = prev[prev.length - 1].i;
        const missed = fetched.filter((log) => log.i > lastKnown);
        return missed.length ? [...prev, ...missed] : prev;
    },
};

export function LogsView(_props: ViewProps) {
    const state = usePubSubView<Array<LogRecord>>({view: 'logs'}, LOGS_OPTIONS);
    const store = useStore();
    const view = useAtomValue(state);
    const ref = React.useRef<HTMLDivElement | null>(null);
    const fetching = React.useRef(false);

    const logs = view.status === 'ok' ? view.data : [];

    // scrolled to the top: page backwards from the oldest record held. Prepends into the
    // shared atom, so a second Log widget gets that history without refetching it.
    const handleScroll = (_event: React.UIEvent) => {
        const first = logs.length ? logs[0].i : undefined;
        if (ref.current!.scrollTop !== 0 || first === undefined || first === 0 || fetching.current) return;

        fetching.current = true;
        getLogs(first)
            .then((data) => store.set(state, (cur) => {
                const held = cur.status === 'ok' ? cur.data : [];
                const oldest = held.length ? held[0].i : Infinity;
                const older = data.logs.filter((log) => log.i < oldest);
                return older.length ? {status: 'ok', data: [...older, ...held]} : cur;
            }))
            .finally(() => { fetching.current = false; });
    };

    return <div ref={ref} onScroll={handleScroll} className='log-cont'>
        {logs.map((log) => <div className="log" key={log.i}>
            {log.log}
        </div>)}
    </div>
}

// `document.URL` is the job dashboard's own URL, so this resolves to `.../job/<id>/logs`.
async function getLogs(before?: number): Promise<LogsData> {
    const params = new URLSearchParams();
    if (before) {
        params.set("before", before.toString());
    }
    const response = await fetch(document.URL + '/logs?' + params);
    if (!response.ok) {
        throw new Error(`Failed to fetch logs: ${response.status} ${response.statusText}`);
    }
    return await response.json() as LogsData;
}
