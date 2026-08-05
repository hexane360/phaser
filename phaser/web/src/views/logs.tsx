import React from 'react';
import { useAtomValue, useStore } from 'jotai';
import { Badge, Code, Collapse, Group, Loader, Text } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { Virtuoso } from 'react-virtuoso';

import { LogRecord, LogsData } from '../types';
import { usePubSubView } from '../pubsub';
import { Mono } from '../components';
import { ViewProps } from './types';
import { formatElapsed } from '../utils';

const PAGE = 200;       // records fetched per scroll-back page
const MAX_GAP = 1000;   // largest reconnect gap we'll stitch before resyncing to the tail
// `firstItemIndex` must stay positive as older records are prepended, so record `i` is
// offset by a constant larger than any log we'd hold.
const FIRST_BASE = 1e7;

// Both runs ascend by `i` and are individually contiguous; the result is too. Overlapping
// or touching runs are unioned (dropping duplicates), disjoint ones can't both be kept
// without leaving a hole, so the newer run wins and the older is dropped.
function combine(held: ReadonlyArray<LogRecord> | undefined, incoming: ReadonlyArray<LogRecord>): Array<LogRecord> {
    if (!held?.length) return [...incoming];
    if (!incoming.length) return [...held];

    const [a, b] = held[0].i <= incoming[0].i ? [held, incoming] : [incoming, held];
    const aLast = a[a.length - 1].i;
    const [bFirst, bLast] = [b[0].i, b[b.length - 1].i];

    if (bFirst > aLast + 1) return [...b];  // disjoint
    if (bLast <= aLast) return [...a];      // b adds nothing
    return [...a, ...b.slice(aLast + 1 - bFirst)];
}

// `logs` is the one append-conflated topic: each message is new records, never a whole
// value (see `phaser/web/views.py`). It's also non-retained, so subscribing yields no
// history -- `hydrate` fetches that over REST, on first subscribe and after each reconnect.
// Both live in the shared per-topic entry, so a second Log widget costs neither.
const LOGS_OPTIONS = {
    initial: [] as Array<LogRecord>,
    reduce: combine,
    // a reconnect: fetch exactly what was missed, in one request. A gap wider than
    // `MAX_GAP` isn't worth stitching -- resync to the tail instead, and let the user
    // scroll back into the rest.
    hydrate: async (prev: ReadonlyArray<LogRecord> | undefined) => {
        const last = prev?.length ? prev[prev.length - 1].i : undefined;
        if (last === undefined) return (await getLogs({limit: PAGE})).logs as Array<LogRecord>;

        const page = await getLogs({after: last, limit: MAX_GAP});
        if (!page.has_after) return page.logs as Array<LogRecord>;
        return (await getLogs({limit: PAGE})).logs as Array<LogRecord>;
    },
    merge: combine,
};

// severity presentation, keyed by the standard `logging` levels. `color` is a Mantine
// palette name (the pill), `accent` the color-scheme-aware token the row's left border and
// the pill both read -- same pattern as the manager's `--status` rows.
function levelInfo(level: number): {name: string, color: string, accent: string} {
    if (level >= 50) return {name: 'CRIT', color: 'red', accent: 'var(--mantine-color-red-text)'};
    if (level >= 40) return {name: 'ERROR', color: 'red', accent: 'var(--mantine-color-red-text)'};
    if (level >= 30) return {name: 'WARN', color: 'yellow', accent: 'var(--mantine-color-yellow-text)'};
    if (level >= 20) return {name: 'INFO', color: 'blue', accent: 'var(--mantine-color-blue-text)'};
    return {name: 'DEBUG', color: 'gray', accent: 'var(--mantine-color-gray-text)'};
}



// One record: a timestamp, a severity pill, and the message itself in monospace. Clicking
// expands the metadata that doesn't earn a column (logger, source line, stack trace) --
// the same click-to-expand card the manager's worker/job rows use, rather than a hover or
// flip, which would fight with scrolling and text selection.
function LogLine({log}: {log: LogRecord}) {
    const [expanded, {toggle}] = useDisclosure(false);
    const level = levelInfo(log.log_level);

    return <div className="log-row" style={{'--level': level.accent} as React.CSSProperties}>
        <div className="log-line" onClick={toggle}>
            {/* time since the run started; the wall clock is in the expanded detail */}
            <time className="log-time" dateTime={log.timestamp}>{formatElapsed(log.elapsed)}</time>
            <div>
                <Badge className="log-level" size="xs" variant="light" color={level.color} radius="sm">{level.name}</Badge>
                <span className="log-msg mono">{log.log}</span>
            </div>
        </div>
        <Collapse className="log-detail" expanded={expanded} keepMounted={false}>
            <Group gap="lg" className="log-meta">
                <span>{new Date(log.timestamp).toLocaleString()}</span>
                <span><Mono>{log.logger_name}</Mono></span>
                {log.func_name && <span><Mono>{log.func_name}:{log.line_number}</Mono></span>}
            </Group>
            {log.stack_info ? <Code block className="log-stack">{log.stack_info}</Code> : null}
        </Collapse>
    </div>
}

export function LogsView(_props: ViewProps) {
    const state = usePubSubView<Array<LogRecord>>({view: 'logs'}, LOGS_OPTIONS);
    const store = useStore();
    const view = useAtomValue(state);
    const fetching = React.useRef(false);
    // lowest `i` the server still retains, per the last response we saw
    const oldest = React.useRef(0);

    const logs = view.status === 'ok' ? view.data : [];
    const first = logs.length ? logs[0].i : 0;
    const hasBefore = logs.length > 0 && first > oldest.current;

    // Virtuoso handles a *decreasing* `firstItemIndex` (a prepend), but not an increasing
    // one -- which is what a resync looks like, since it drops records off the top. Remount
    // the list in that case, landing the user at the bottom of the fresh window.
    const [epoch, setEpoch] = React.useState(0);
    const prevFirst = React.useRef(first);
    React.useEffect(() => {
        if (first > prevFirst.current) setEpoch((epoch) => epoch + 1);
        prevFirst.current = first;
    }, [first]);

    // scrolled to the top: page backwards from the oldest record held. Prepends into the
    // shared atom, so a second Log widget gets that history without refetching it.
    const fetchOlder = () => {
        if (!hasBefore || fetching.current) return;

        fetching.current = true;
        getLogs({before: first, limit: PAGE})
            .then((data) => {
                oldest.current = data.oldest;
                store.set(state, (cur) => ({
                    status: 'ok', data: combine(cur.status === 'ok' ? cur.data : [], data.logs),
                }));
            })
            .catch((e) => console.error("Couldn't fetch older logs:", e))
            .finally(() => { fetching.current = false; });
    };

    // marks the top of the list: either more history to load, or the start of the log
    const Header = () => <div className="log-edge">
        {hasBefore ? <Loader size="xs" type="dots"/> : <Text size="xs" c="dimmed">start of log</Text>}
    </div>;

    return <div className="log-cont">
        <Virtuoso
            key={epoch}
            className="log-scroll"
            // Virtuoso's default root style is `height: 100%`, which collapses to nothing
            // inside the auto-height widget body -- the height has to be a definite value
            // here, where it beats that default (a CSS class does not).
            style={{height: '100ex'}}
            data={logs}
            firstItemIndex={FIRST_BASE + first}
            initialTopMostItemIndex={Math.max(logs.length - 1, 0)}
            // stick to the tail only when already there, so scrolling back stays put
            followOutput={(atBottom) => atBottom ? 'auto' : false}
            startReached={fetchOlder}
            itemContent={(_index, log) => <LogLine log={log}/>}
            components={{Header: logs.length ? Header : undefined}}
        />
    </div>
}

// exclusive cursors on a record's `i`. Both bound the window on either side (a reconnect
// gap, filled forwards from `after`); neither gives the newest `limit` records.
interface LogQuery {
    before?: number,
    after?: number,
    limit?: number,
    min_level?: number,
}

// `document.URL` is the job dashboard's own URL, so this resolves to `.../job/<id>/logs`.
// With no cursor, the endpoint returns the newest `limit` records.
async function getLogs(query: LogQuery = {}): Promise<LogsData> {
    const params = new URLSearchParams();
    for (const [key, val] of Object.entries(query)) {
        if (val !== undefined) params.set(key, val.toString());
    }
    const response = await fetch(document.URL + '/logs?' + params);
    if (!response.ok) {
        throw new Error(`Failed to fetch logs: ${response.status} ${response.statusText}`);
    }
    return await response.json() as LogsData;
}
