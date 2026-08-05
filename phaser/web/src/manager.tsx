
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { atom, PrimitiveAtom, useAtomValue, Provider, useStore } from 'jotai';

import '@mantine/core/styles.css';
import { AppShell, MantineProvider, Container, Group, Button, Collapse, Title, LoadingOverlay, Box, Modal, Tabs, Stack, Code, Progress, Text } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import ky from 'ky';
import TimeAgo from 'react-timeago';

import './styles.css';
import { JobState, WorkerState } from './types';
import { fetchTraceback, jobStatus, workerStatusColor } from './status';
import { makeTheme, cssVariableResolver } from './theme';
import { Section, Mono } from './components';
import Header from './header';
import { PubSubProvider, usePubSubConnection, usePubSubView, ViewState } from './pubsub';
import { ConnectionStatus } from './connection';
import { handleRequest } from './requests';
import { rootPrefix } from './utils';


export function Worker({state}: {state: WorkerState}) {
    const [opened, {toggle}] = useDisclosure(false);

    function procBackends(backends: Array<[string, string]>): string {
        // TODO: refactor
        let arr: Array<string> = [];
        for (const [backend, device] of backends) {
            arr.push(`${backend}[${device}]`);
        }
        return arr.join(', ');
    }

    return <div className="row-group" style={{'--status': workerStatusColor(state.status)} as React.CSSProperties}>
        <div className="card" onClick={toggle}>
            <div><Mono>{state.worker_id}</Mono></div>
            <div>{state.worker_type}</div>
            <div className="status">{state.status}</div>
            <Group justify='right'>
                <Button variant="default" onClick={(e) => signal_worker(e, state, 'reload')}>Reload</Button>
                <Button variant="default" mod={{danger: true}} onClick={(e) => signal_worker(e, state, 'shutdown')}>Shutdown</Button>
            </Group>
        </div>
        <Collapse className="card-body" expanded={opened}>
            <div className="grid" style={{gridTemplateColumns: "1fr 1fr"}}>
                <div>{state.hostname ? <>Hostname: <Mono>{state.hostname}</Mono></> : <></>}</div>
                <div>{state.current_job ? `Running job: ${state.current_job}` : ""}</div>
                <div style={{gridColumn: "1/-1"}}>{state.backends ? <>Backends: <Mono>{procBackends(state.backends)}</Mono></> : <></>}</div>
                <div style={{gridColumn: "1/-1"}}>{state.start_time ? <>Running since <TimeAgo date={state.start_time}/></> : <></>}</div>
            </div>
        </Collapse>
    </div>
}

export function Workers({workers}: {workers: PrimitiveAtom<ViewState<Array<WorkerState>>>}) {
    const state = useAtomValue(workers);
    const workers_val = state.status === 'ok' ? state.data : [];

    if (!workers_val.length) {
        return <Title order={4}>No workers have been started</Title>
    }

    return <div className="card-list workers-table">
        <div>
            <div>Worker ID</div>
            <div>Type</div>
            <div>Status</div>
            <div></div>
        </div>
        {...workers_val.map((worker) => <Worker state={worker} key={worker.worker_id}/> )}
    </div>;
}

function IterProgress({value, max}: {value: number, max: number | null}) {
    const pct = max ? Math.min(100, Math.max(0, (value / max) * 100)) : 0;
    return <div className="progress-wrap">
        <Mono>{value}/{max ?? '?'}</Mono>
        <Progress value={pct} w={44} size={4}/>
    </div>;
}

// The traceback for a failed job, fetched on expand rather than carried in `JobState` --
// `jobs` republishes on every iteration, and a traceback per failed job in that payload
// would be paid for continuously. Mounted only while the row is open, so the fetch happens
// once someone actually looks.
function JobFailure({state}: {state: JobState}) {
    const [traceback, setTraceback] = React.useState<string | null | 'pending'>('pending');

    React.useEffect(() => {
        let live = true;
        fetchTraceback(state.links.logs)
            .then((tb) => { if (live) setTraceback(tb); })
            .catch((e) => {
                console.error("Couldn't fetch traceback:", e);
                if (live) setTraceback(null);
            });
        return () => { live = false; };
    }, [state.links.logs]);

    return <Stack gap="xs" style={{gridColumn: "1/-1"}}>
        <Text>Job failed</Text>
        {traceback === 'pending'
            ? <Text size="sm" c="dimmed">Loading traceback…</Text>
            : traceback
                ? <Code block>{traceback}</Code>
                : <Text size="sm" c="dimmed">No traceback recorded. <a href={state.links.logs_txt}>Full log</a></Text>}
    </Stack>;
}

export function Job({state}: {state: JobState}) {
    const [opened, {toggle}] = useDisclosure(false);

    const iter_state = state.state.iter;
    const status = jobStatus(state);

    return <div className="row-group" style={{'--status': status.color} as React.CSSProperties}>
        <div className="card" onClick={toggle}>
            <div>{state.job_id}</div>
            <div>{state.job_name ?? ""}</div>
            <Group className="status">{status.label}</Group>
            <Group visibleFrom="md">
                {iter_state && <IterProgress value={iter_state.engine_iter} max={iter_state.n_engine_iters}/>}
            </Group>
            <Group>
                <Group hiddenFrom="md" style={{paddingLeft: "20px"}}>Total iter:</Group>
                {iter_state && <IterProgress value={iter_state.total_iter} max={iter_state.n_total_iters}/>}
            </Group>
            <Group justify='right'>
                <Button variant="default" component='a' href={state.links.dashboard}>Watch</Button>
                <Button variant="default" mod={{danger: true}} onClick={(e) => cancel_job(state, e)}>Cancel</Button>
            </Group>
        </div>
        <Collapse className="card-body" expanded={opened} keepMounted={false}>
            <div className="grid" style={{gridTemplateColumns: "1fr 1fr"}}>
                {state.result === 'errored' && <JobFailure state={state}/>}
            </div>
        </Collapse>
    </div>
}

export function Jobs({jobs}: {jobs: PrimitiveAtom<ViewState<Array<JobState>>>}) {
    const state = useAtomValue(jobs);
    const jobs_val = state.status === 'ok' ? state.data : [];

    if (!jobs_val.length) {
        return <Title order={4}>No jobs are running</Title>
    }

    return <div className="card-list jobs-table">
        <div>
            <div>Job ID</div>
            <div>Name</div>
            <div>Status</div>
            <div>Engine iter</div>
            <div>Total iter</div>
            <div></div>
        </div>
        {...jobs_val.map((job) => <Job state={job} key={job.job_id}/> )}
    </div>;
}

export function StartWorkers(props: {}) {
    const [submitting, { open: setSubmitting, close: finishSubmitting }] = useDisclosure(false);
    const [message, setMessage] = React.useState<[string, string] | null>(null);

    function start_worker(worker_type: string): (e: React.MouseEvent) => void {
        return async function(e: React.MouseEvent) {
            setSubmitting();
            const result = await handleRequest('submitting worker', () => ky(`worker/${worker_type}/start`, {
                method: "post",
                body: "",
                timeout: 5000,
            }));
            if (result.type == 'error') {
                setMessage(["Error submitting worker", result.msg]);
            } else if (result.body.message) {
                setMessage(["Allocated worker", result.body.message]);
            }
            finishSubmitting();
        }
    }

    const panelStyle = {
        padding: "10px",
        minHeight: "100px",
    };

    return <Box pos="relative" style={{maxWidth: "600px"}}>
        <Modal opened={message !== null} onClose={() => setMessage(null)} title={message ? message[0] : ""}>{message ? message[1] : ""}</Modal>
        <LoadingOverlay visible={submitting} zIndex={1000}/>
        <Tabs variant="pills" defaultValue="local">
            <Tabs.List>
                <Tabs.Tab value="local">Local</Tabs.Tab>
                <Tabs.Tab value="slurm">Slurm</Tabs.Tab>
                <Tabs.Tab value="manual">Manual</Tabs.Tab>
            </Tabs.List>
            <Tabs.Panel value="local" style={panelStyle}>
                <Stack>
                    <div>Starts a worker on the local computer</div>
                    <div><button onClick={start_worker("local")}>Start</button></div>
                </Stack>
            </Tabs.Panel>
            <Tabs.Panel value="slurm" style={panelStyle}>
                <Stack>
                    <div>Starts a remote worker using Slurm (more configuration to come!)</div>
                    <div><button onClick={start_worker("slurm")}>Start</button></div>
                </Stack>
            </Tabs.Panel>
            <Tabs.Panel value="manual" style={panelStyle}>
                <Stack>
                    <div>Create a worker which must be started manually</div>
                    <div>Start with <Code>phaser worker &lt;url&gt;</Code></div>
                    <div><button onClick={start_worker("manual")}>Start</button></div>
                </Stack>
            </Tabs.Panel>
        </Tabs>
    </Box>
}

export function StartJobs(props: {}) {
    const pathRef: React.RefObject<HTMLInputElement | null> = React.useRef(null);

    const [submitting, { open: setSubmitting, close: finishSubmitting }] = useDisclosure(false);
    const [error, setError] = React.useState<string | null>(null);

    async function submit_job(e: React.MouseEvent) {
        const path = pathRef.current!.value;
        setSubmitting();
        const result = await handleRequest('submitting job', () => ky("job/start", {
            method: "post",
            body: JSON.stringify({'source': 'path', path}),
            timeout: 5000,
        }));
        if (result.type == 'error') {
            setError(result.msg)
        }
        finishSubmitting();
    }

    return <Box pos="relative">
        <Modal opened={!!error} onClose={() => setError(null)} title="Error submitting job">{error}</Modal>
        <LoadingOverlay visible={submitting} zIndex={1000}/>
        <input name="path" type="text" size={50} ref={pathRef}/>
        <button type="submit" onClick={submit_job}>Submit</button>
    </Box>;
}

function Manager(props: {}) {
    const conn = usePubSubConnection();
    const [fallbackStatus] = React.useState(() => atom<ConnectionStatus>({ type: 'connecting' }));

    const jobs = usePubSubView<Array<JobState>>('jobs');
    const workers = usePubSubView<Array<WorkerState>>('workers');

    return <AppShell header={{ height: 80 }} padding="md">
        <AppShell.Header><Header serverStatus={conn?.status ?? fallbackStatus} size="lg"/></AppShell.Header>
        <AppShell.Main><Container>
            <Section name="Start workers"><StartWorkers/></Section>
            <Section name="Workers"><Workers workers={workers}/></Section>
            <Section name="Start reconstructions"><StartJobs/></Section>
            <Section name="Jobs"><Jobs jobs={jobs}/></Section>
        </Container></AppShell.Main>
    </AppShell>;
}

export function App(props: {}) {
    const store = useStore();

    const protocol = window.location.protocol == 'https:' ? "wss:" : "ws:";
    const address = `${protocol}//${window.location.host}${rootPrefix()}/listen`;

    return <Provider store={store}>
        <PubSubProvider address={address}>
            <Manager/>
        </PubSubProvider>
    </Provider>;
}

function signal_worker(e: React.MouseEvent<HTMLElement>, worker: WorkerState, signal: string) {
    e.stopPropagation();
    fetch(worker.links[signal], {
        method: "POST",
        body: "",
    })
    .then((response) => response.ok ? response.json() : Promise.reject(response))
    .then((json) => {
        console.log(`Got response: ${JSON.stringify(json)}`);
    })
    .catch((response: Response) => {
        console.error(`Error: HTTP ${response.status} ${response.statusText}`)
    });
};

function cancel_job(job: JobState, e: React.MouseEvent) {
    console.log(`cancelRecons: recons: ${JSON.stringify(job)}`);
    fetch(job.links.cancel, {
        method: "POST",
        body: "",
    })
    .then((response) => response.ok ? response.json() : Promise.reject(response))
    .then((json) => {
        console.log(`Got response: ${JSON.stringify(json)}`);
    })
    .catch((response: Response) => {
        console.error(`Error: HTTP ${response.status} ${response.statusText}`)
    });
};


const root = createRoot(document.getElementById('app')!);
root.render(
    <StrictMode>
        <MantineProvider theme={makeTheme()} cssVariablesResolver={cssVariableResolver}>
            <App/>
        </MantineProvider>
    </StrictMode>
);