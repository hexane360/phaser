
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { atom, PrimitiveAtom, useAtom, Provider, useStore } from 'jotai';

import '@mantine/core/styles.css';
import '@hexane/plotlib/styles.css';
import { AppShell, Container, createTheme, MantineProvider } from '@mantine/core';
import * as plotlib from '@hexane/plotlib';
import { DecodedArray } from './array';

import './styles.css';
import { LogRecord, LogsData, ProbeData, ProbeMeta, ObjectSampling, ObjPhaseSumData, ProgressData } from './types';
import { Section } from './components';
import { ProbePlot, ObjectPlot, ProgressPlot } from './plots';
import Header from './header';
import { PubSubProvider, usePubSubConnection, usePublishedAtom } from './pubsub';
import { ConnectionStatus } from './connection';
import { isClose } from './utils';

function Logs({state}: {state: PrimitiveAtom<Array<LogRecord>>}) {
    const conn = usePubSubConnection();
    const store = useStore();
    const [logs, setLogs] = useAtom(state);
    const ref = React.useRef<HTMLDivElement | null>(null);

    const handleNewLogs = (data: LogsData) => {
        if (logs.length && data.last !== logs[0].i - 1) {
            throw new Error("Non-contiguous logs fetched");
        }

        setLogs([...data.logs, ...logs]);
    }

    React.useEffect(() => {
        getLogs().then(handleNewLogs);
    }, []);

    // live tail: the REST fetch above supplies history, this appends new records as they
    // arrive. `logs` is a non-retained, append-conflated view (see `phaser/web/views.py`),
    // so every message here is new records to append, never a full replacement.
    React.useEffect(() => {
        if (!conn) return;
        return conn.subscribe({view: 'logs'}, (msg) => {
            if ('error' in msg) {
                console.error(`logs subscription error: ${msg.error}`);
                return;
            }
            store.set(state, (logs) => [...logs, ...(msg.data as Array<LogRecord>)]);
        });
    }, [conn, store, state]);

    // catches whatever was missed while disconnected -- limited to the last `limit`
    // (100) records the server returns; a wider gap would still be missed (no "logs
    // since index N" endpoint exists).
    React.useEffect(() => {
        if (!conn) return;
        return conn.onReconnect(() => {
            getLogs().then((data) => {
                store.set(state, (prev) => {
                    const lastKnown = prev.length ? prev[prev.length - 1].i : -1;
                    const newRecords = data.logs.filter((log) => log.i > lastKnown);
                    return newRecords.length ? [...prev, ...newRecords] : prev;
                });
            });
        });
    }, [conn, store, state]);

    const handleScroll = (event: React.UIEvent) => {
        const first = (logs[0]) ? logs[0].i : undefined;

        if (ref.current!.scrollTop === 0 && first !== 0) {
            getLogs(first).then(handleNewLogs);
        }
    }

    return <div ref={ref} onScroll={handleScroll} className='log-cont'>
        {logs.map((log) => <div className="log" key={log.i}>
            {log.log}
        </div>)}
    </div>
}

// Subscribes to `obj_phase_sum`, splitting it into a rarely-changing `sampling` atom and a
// per-tick `data` atom -- see the comment on `ObjectPlotSub` in `plots.tsx` for why this
// meta/data split matters. Mirrors the pre-pub/sub `App.updateState`'s object handling.
function useObjectPhaseAtoms(): {metaState: PrimitiveAtom<ObjectSampling | null>, dataState: PrimitiveAtom<DecodedArray | null>} {
    const conn = usePubSubConnection();
    const store = useStore();
    const [metaState] = React.useState(() => atom(null as ObjectSampling | null));
    const [dataState] = React.useState(() => atom(null as DecodedArray | null));

    React.useEffect(() => {
        if (!conn) return;
        return conn.subscribe({view: 'obj_phase_sum'}, (msg) => {
            if ('error' in msg) {
                console.error(`obj_phase_sum subscription error: ${msg.error}`);
                return;
            }
            const phase = msg.data as ObjPhaseSumData;
            store.set(dataState, (_) => phase.data);
            store.set(metaState, (prev) => samplingEqual(prev, phase.sampling) ? prev : phase.sampling);
        });
    }, [conn, store, metaState, dataState]);

    return {metaState, dataState};
}

// Subscribes to `probes`, splitting it the same way as `useObjectPhaseAtoms` above.
function useProbeAtoms(): {metaState: PrimitiveAtom<ProbeMeta | null>, dataState: PrimitiveAtom<DecodedArray | null>} {
    const conn = usePubSubConnection();
    const store = useStore();
    const [metaState] = React.useState(() => atom(null as ProbeMeta | null));
    const [dataState] = React.useState(() => atom(null as DecodedArray | null));

    React.useEffect(() => {
        if (!conn) return;
        return conn.subscribe({view: 'probes'}, (msg) => {
            if ('error' in msg) {
                console.error(`probes subscription error: ${msg.error}`);
                return;
            }
            const probe = msg.data as ProbeData;
            store.set(dataState, (_) => probe.data);
            store.set(metaState, (prev) => {
                const meta: ProbeMeta = { sampling: probe.sampling, nprobes: probe.data.shape[0] };
                return probeMetaEqual(prev, meta) ? prev! : meta;
            });
        });
    }, [conn, store, metaState, dataState]);

    return {metaState, dataState};
}

function Dashboard(props: {}) {
    const conn = usePubSubConnection();
    const [fallbackStatus] = React.useState(() => atom<ConnectionStatus>({ type: 'connecting' }));

    const progressState = usePublishedAtom<Record<string, ProgressData>>({view: 'progress'});
    const {metaState: probeMetaState, dataState: probeDataState} = useProbeAtoms();
    const {metaState: objectMetaState, dataState: objectDataState} = useObjectPhaseAtoms();
    const [logsState] = React.useState(() => atom([] as Array<LogRecord>));

    return <AppShell header={{ height: 80 }} padding="md">
        <AppShell.Header><Header serverStatus={conn?.status ?? fallbackStatus}/></AppShell.Header>
        <AppShell.Main><Container size="lg">
            <Section name="Progress"><ProgressPlot state={progressState as PrimitiveAtom<Record<string, ProgressData>>}/></Section>
            <Section name="Probe"><ProbePlot metaState={probeMetaState} dataState={probeDataState}/></Section>
            <Section name="Object"><ObjectPlot metaState={objectMetaState} dataState={objectDataState}/></Section>
            <Section name="Logs"><Logs state={logsState}/></Section>
        </Container></AppShell.Main>
        {/*<StatusBar state={statusState}/>*/}
    </AppShell>
}

function App(props: {}) {
    const store = useStore();

    // job id from the URL path (`.../job/<id>`) -- no `job` threaded through the
    // component tree beyond this: the `?job=<id>` scoping happens once, at the websocket
    // URL, and every descendant subscribes with abbreviated `{view: ...}` topics that the
    // server fills in against this connection's default topic.
    //
    // `/listen` lives at the app root, not nested under `/job/<id>`, so (unlike the old
    // per-job `<pathname>/listen` endpoint) we can't just append to `pathname` -- instead
    // strip the trailing `job/<id>` segments to recover any `root_path`/SCRIPT_NAME mount
    // prefix (e.g. `/prefix/job/abc` -> `/prefix`), same as today's deployments expect.
    const pathParts = window.location.pathname.split('/');
    const jobId = pathParts.pop()!;
    pathParts.pop();  // 'job'
    const prefix = pathParts.join('/');
    const protocol = window.location.protocol == 'https:' ? "wss:" : "ws:";
    const address = `${protocol}//${window.location.host}${prefix}/listen?job=${encodeURIComponent(jobId)}`;

    return <Provider store={store}>
        <PubSubProvider address={address}>
            <Dashboard/>
        </PubSubProvider>
    </Provider>
}

const theme = createTheme({

});

const root = createRoot(document.getElementById('app')!);
root.render(
    <StrictMode>
        <MantineProvider theme={theme}>
            <plotlib.ThemeProvider>
                <App/>
            </plotlib.ThemeProvider>
        </MantineProvider>
    </StrictMode>
);

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

function probeMetaEqual(a: ProbeMeta | null, b: ProbeMeta): boolean {
    return a !== null && a.nprobes === b.nprobes &&
        isClose(a.sampling.shape, b.sampling.shape) && isClose(a.sampling.extent, b.sampling.extent)
            && isClose(a.sampling.sampling, b.sampling.sampling);
}

function samplingEqual(a: ObjectSampling | null, b: ObjectSampling): boolean {
    return a !== null &&
        isClose(a.shape, b.shape) && isClose(a.sampling, b.sampling) && isClose(a.corner, b.corner)
            && (a.region_min === null && b.region_min === null || a.region_min !== null && b.region_min !== null && isClose(a.region_min, b.region_min))
            && (a.region_max === null && b.region_max === null || a.region_max !== null && b.region_max !== null && isClose(a.region_max, b.region_max))
}
