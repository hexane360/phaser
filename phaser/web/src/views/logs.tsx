import React from 'react';
import { useAtomValue, useStore } from 'jotai';
import { Badge, Button, Code, Collapse, Group, Loader, Text } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { Virtuoso, VirtuosoHandle } from 'react-virtuoso';

import { LogRecord, LogsData } from '../types';
import { usePubSubView } from '../pubsub';
import { useGetAction } from '../requests';
import { Mono } from '../components';
import { ViewProps } from './types';
import { formatElapsed } from '../utils';

const PAGE = 200;       // records fetched per scroll-back page
const MAX_GAP = 1000;   // largest reconnect gap we'll stitch before resyncing to the tail
// `firstItemIndex` must stay positive as older records are prepended, so record `i` is
// offset by a constant larger than any log we'd hold.
const FIRST_BASE = 1e7;
// px from the bottom still counted as "at the bottom" for autoscroll
const BOTTOM_SLACK = 30;

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
    const [fetchOlderPage, fetching, fetchError] = useGetAction<LogsData>("Couldn't load older logs");
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

    // Whether to stick to the tail. We keep track of this ourselves
    const follow = React.useRef(true);
    const lastTop = React.useRef(0);
    const [following, setFollowing] = React.useState(true);

    const onScroll = (el: HTMLElement) => {
        const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
        if (dist <= BOTTOM_SLACK) follow.current = true;
        else if (el.scrollTop < lastTop.current) follow.current = false;
        lastTop.current = el.scrollTop;
        setFollowing(follow.current);
    };

    // Virtuoso owns the scroller element, so the listener is attached through its ref
    // callback -- which also fires with `null` on the `epoch` remount, detaching the old one.
    const detach = React.useRef<(() => void) | null>(null);
    const scrollerRef = (el: HTMLElement | Window | null) => {
        detach.current?.();
        detach.current = null;
        if (!(el instanceof HTMLElement)) return;

        const handler = () => onScroll(el);
        el.addEventListener('scroll', handler, {passive: true});
        detach.current = () => el.removeEventListener('scroll', handler);
    };
    React.useEffect(() => () => detach.current?.(), []);

    // scrolled to the top: page backwards from the oldest record held. Prepends into the
    // shared atom, so a second Log widget gets that history without refetching it.
    const fetchOlder = () => {
        if (!hasBefore || fetching) return;

        fetchOlderPage(logsUrl({before: first, limit: PAGE})).then((data) => {
            if (!data) return;
            oldest.current = data.oldest;
            store.set(state, (cur) => ({
                status: 'ok', data: combine(cur.status === 'ok' ? cur.data : [], data.logs),
            }));
        });
    };

    // marks the top of the list: either more history to load, or the start of the log. A
    // failed page reports itself here as well as in a notification -- the spinner alone would
    // just keep spinning.
    const Header = () => <div className="log-edge">
        {!hasBefore ? <Text size="xs" c="dimmed">start of log</Text>
            : fetchError ? <Text size="xs" c="red">{fetchError}</Text>
                : <Loader size="xs" type="dots"/>}
    </div>;

    // jumps to the tail and re-arms the latch, for when the log has been scrolled away from
    const listRef = React.useRef<VirtuosoHandle>(null);
    const toBottom = () => {
        follow.current = true;
        setFollowing(true);
        listRef.current?.scrollToIndex({index: 'LAST', align: 'end'});
    };

    return <>
        <div className="log-cont">
            <Virtuoso
                ref={listRef}
                key={epoch}
                className="log-scroll"
                data={logs}
                firstItemIndex={FIRST_BASE + first}
                initialTopMostItemIndex={Math.max(logs.length - 1, 0)}
                // stick to the tail until scrolled away from it, so scrolling back stays put
                followOutput={() => follow.current ? 'auto' : false}
                atBottomThreshold={BOTTOM_SLACK}
                scrollerRef={scrollerRef}
                startReached={fetchOlder}
                itemContent={(_index, log) => <LogLine log={log}/>}
                components={{Header: logs.length ? Header : undefined}}
            />
        </div>
        <Group justify="center" pt={6}>
            <Button size="compact-xs" variant="default" onClick={toBottom} disabled={following}>
                scroll to bottom
            </Button>
        </Group>
    </>
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
function logsUrl(query: LogQuery = {}): string {
    const params = new URLSearchParams();
    for (const [key, val] of Object.entries(query)) {
        if (val !== undefined) params.set(key, val.toString());
    }
    return document.URL + '/logs?' + params;
}

// Used by `hydrate`, which runs outside a component and reports through the view's own state.
// Scrollback goes through `useGetAction` instead -- see `LogsView`.
async function getLogs(query: LogQuery = {}): Promise<LogsData> {
    const response = await fetch(logsUrl(query));
    if (!response.ok) {
        throw new Error(`Failed to fetch logs: ${response.status} ${response.statusText}`);
    }
    return await response.json() as LogsData;
}
