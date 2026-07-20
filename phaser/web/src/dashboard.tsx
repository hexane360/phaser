
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { atom, PrimitiveAtom, useAtom, useAtomValue, Provider, useStore } from 'jotai';

import '@mantine/core/styles.css';
import '@hexane/plotlib/styles.css';
import { AppShell, Container, createTheme, MantineProvider } from '@mantine/core';
import * as plotlib from '@hexane/plotlib';
import { decodeInterchange, ArrayInterchange, DecodedArray } from './array';

import './styles.css';
import { JobStatus, DashboardMessage, LogRecord, LogsData, ProbeMeta, ObjectSampling, ProgressData, PartialReconsData } from './types';
import { Section } from './components';
import { ProbePlot, ObjectPlot, ProgressPlot } from './plots';
import Header from './header';
import websocket from './websocket';
import { isClose } from './utils';

function Logs({state}: {state: PrimitiveAtom<Array<LogRecord>>}) {
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

function App(props: {}) {
    const store = useStore();

    const statusState: PrimitiveAtom<JobStatus | null> = atom(null as JobStatus | null);
    // split so that consumers can subscribe to shape/sampling (which changes rarely)
    // separately from the raw array data (which changes every update)
    const probeMetaState: PrimitiveAtom<ProbeMeta | null> = atom(null as ProbeMeta | null);
    const probeDataState: PrimitiveAtom<DecodedArray | null> = atom(null as DecodedArray | null);
    const objectMetaState: PrimitiveAtom<ObjectSampling | null> = atom(null as ObjectSampling | null);
    const objectDataState: PrimitiveAtom<DecodedArray | null> = atom(null as DecodedArray | null);
    const progressState: PrimitiveAtom<Record<string, ProgressData>> = atom({});
    const logsState: PrimitiveAtom<Array<LogRecord>> = atom([] as Array<LogRecord>);

    function updateState(raw_state: Record<string, any>) {
        const state = decodeState(raw_state) as PartialReconsData;
        console.log(`state: ${JSON.stringify(state)}`);

        if (state.probe) {
            const probe = state.probe;
            store.set(probeDataState, (_: any) => probe.data);
            store.set(probeMetaState, (prev) => {
                const meta: ProbeMeta = { sampling: probe.sampling, nprobes: probe.data.shape[0] };
                return probeMetaEqual(prev, meta) ? prev! : meta;
            });
        }
        if (state.object) {
            const object = state.object;
            store.set(objectDataState, (_: any) => object.data);
            store.set(objectMetaState, (prev) => samplingEqual(prev, object.sampling) ? prev : object.sampling);
        }
        if (state.progress) {
            const progress = state.progress;
            store.set(progressState, (_: any) => progress);
        }
    }

    function onMessage(event: MessageEvent<any>) {
        let text: string;
        if (event.data instanceof ArrayBuffer) {
            let utf8decoder = new TextDecoder();
            text = utf8decoder.decode(event.data);
        } else {
            text = event.data;
        }

        //console.log(`Socket event: ${text}`)
        let data: DashboardMessage = JSON.parse(text);

        if (data.msg === 'job_update') {
            updateState(data.state);
        } else if (data.msg == 'log') {
            store.set(logsState, (logs) => [...logs, ...data.new_logs]);
        } else if (data.msg === 'status_change') {
            store.set(statusState, (_: any) => data.status);
        } else if (data.msg === 'job_stopped') {
            store.set(statusState, (_: any) => 'stopped');
        } else if (data.msg === 'connected') {
            store.set(statusState, (_: any) => data.state.status);
            updateState(data.state.state);
        } else {
            console.warn(`Unknown message type: ${data}`);
        }
    };

    const protocol = window.location.protocol == 'https:' ? "wss:" : "ws:";
    const { status, lastSeen } = websocket({
        address: `${protocol}//${window.location.host}${window.location.pathname}/listen`,
        onMessage,
    });

    return <Provider store={store}>
        <AppShell header={{ height: 80 }} padding="md">
            <AppShell.Header><Header serverStatus={status}/></AppShell.Header>
            <AppShell.Main><Container size="lg">
                <Section name="Progress"><ProgressPlot state={progressState}/></Section>
                <Section name="Probe"><ProbePlot metaState={probeMetaState} dataState={probeDataState}/></Section>
                <Section name="Object"><ObjectPlot metaState={objectMetaState} dataState={objectDataState}/></Section>
                <Section name="Logs"><Logs state={logsState}/></Section>
            </Container></AppShell.Main>
            {/*<StatusBar state={statusState}/>*/}
        </AppShell>
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

function decodeState(state: any): any {
    if (typeof state !== 'object' || state === null) {
        return state;
    }

    if (state instanceof Array) {
        return state.map(decodeState);
    }

    if (state._ty !== undefined) {
        if (state._ty === 'numpy') {
            return decodeInterchange(state as ArrayInterchange);
        }
        throw new Error(`Unknown custom type '${state._ty}'`);
    }

    let out: Record<string, any> = {};
    for (const [k, v] of Object.entries(state)) {
        out[k] = (typeof v === 'object' && v !== null) ? decodeState(v) : v;
    }
    return out;
}

