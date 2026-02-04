
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { atom, PrimitiveAtom, useAtomValue, Provider, useStore } from 'jotai';

import '@mantine/core/styles.css';
import { AppShell, MantineProvider, Container, Group, Button, Collapse, Title, LoadingOverlay, Box, Modal, Tabs, Stack, Code } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import ky from 'ky';
import TimeAgo from 'react-timeago';

import '../static/styles.css';
import { JobState, WorkerState, ManagerMessage } from './types';
import { makeTheme, cssVariableResolver } from './theme';
import { Section } from './components';
import Header from './header';
import websocket from './websocket';
import { handleRequest } from './requests';


export function Worker({i, state}: {i: number, state: WorkerState}) {
    const [opened, {toggle}] = useDisclosure(false);

    function procBackends(backends: Array<[string, string]>): string {
        // TODO: refactor
        let arr: Array<string> = [];
        for (const [backend, device] of backends) {
            arr.push(`${backend}[${device}]`);
        }
        return arr.join(', ');
    }

    return <>
        <div className="card" style={{gridRow: 2*i + 2}} onClick={toggle}>
            <div style={{gridColumn: 1}}>{state.worker_id}</div>
            <div style={{gridColumn: 2}}>{state.worker_type}</div>
            <div style={{gridColumn: 3}}>{state.status}</div>
            <Group style={{gridColumn: 4}} justify='center'>
                <Button color="yellow" onClick={(e) => signal_worker(e, state, 'reload')}>Reload</Button>
                <Button color="red" onClick={(e) => signal_worker(e, state, 'shutdown')}>Shutdown</Button>
            </Group>
        </div>
        <Collapse className="card-body" style={{gridRow: 2*i + 3}} in={opened}>
            <div className="grid" style={{gridTemplateColumns: "1fr 1fr"}}>
                <div>{state.hostname ? `Hostname: ${state.hostname}` : ""}</div>
                <div>{state.current_job ? `Running job: ${state.current_job}` : ""}</div>
                <div style={{gridColumn: "1/-1"}}>{state.backends ? `Backends: ${procBackends(state.backends)}` : ""}</div>
                <div style={{gridColumn: "1/-1"}}>{state.start_time ? <>Running since <TimeAgo date={state.start_time}/></> : <></>}</div>
            </div>
        </Collapse>
    </>
}

export function Workers({workers}: {workers: PrimitiveAtom<Array<WorkerState>>}) {
    const workers_val = useAtomValue(workers);

    if (!workers_val.length) {
        return <Title order={4}>No workers have been started</Title>
    }

    const style = {
        maxWidth: "800px",
        gridTemplateColumns: "minmax(150px, 1fr) minmax(130px, 1fr) minmax(130px, 2fr) minmax(250px, 1fr)",
    }

    return <div className="card-list" style={style}>
        <div>
            <div>Worker ID</div>
            <div>Type</div>
            <div>Status</div>
            <div></div>
        </div>
        {...workers_val.map((worker, i) => <Worker state={worker} i={i}/> )}
    </div>;
}

export function Job({i, state}: {i: number, state: JobState}) {
    const [opened, {toggle}] = useDisclosure(false);

    const iter_state = state.state.iter;
    const engine_progress = iter_state ? `${iter_state.engine_iter}/${iter_state.n_engine_iters ?? '?'}` : "";
    const total_progress = iter_state ? `${iter_state.total_iter}/${iter_state.n_total_iters ?? '?'}` : "";

    return <>
        <div className="card" style={{gridRow: 2*i + 2}} onClick={toggle}>
            <div style={{gridColumn: 1}}>{state.job_id}</div>
            <div style={{gridColumn: 2}}>{state.job_name ?? ""}</div>
            <div style={{gridColumn: 3}}>{state.status}</div>
            <div style={{gridColumn: 4}}>{engine_progress}</div>
            <div style={{gridColumn: 5}}>{total_progress}</div>
            <Group style={{gridColumn: -1}} justify='center'>
                <Button color="yellow" component='a' href={state.links.dashboard}  target="_blank" rel="noopener noreferrer">Watch</Button>
                <Button color="red" onClick={(e) => cancel_job(state, e)}>Cancel</Button>
            </Group>
        </div>
        <Collapse className="card-body" style={{gridRow: 2*i + 3}} in={opened}>
            <div className="grid" style={{gridTemplateColumns: "1fr 1fr"}}>
            </div>
        </Collapse>
    </>
}

export function Jobs({jobs}: {jobs: PrimitiveAtom<Array<JobState>>}) {
    const jobs_val = useAtomValue(jobs);

    if (!jobs_val.length) {
        return <Title order={4}>No jobs are running</Title>
    }

    const style = {
        maxWidth: "900px", minWidth: "800px",
        gridTemplateColumns: "1fr 2fr 1fr 1fr 1fr 2fr",
    }
    return <div className="card-list" style={style}>
        <div>
            <div>Job ID</div>
            <div>Name</div>
            <div>Status</div>
            <div>Engine iter</div>
            <div>Total iter</div>
            <div></div>
        </div>
        {...jobs_val.map((job, i) => <Job state={job} i={i}/> )}
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
    const pathRef: React.MutableRefObject<HTMLInputElement | null> = React.useRef(null);

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

export function App(props: {}) {
    const store = useStore();

    const jobs: PrimitiveAtom<Array<JobState>> = atom([] as Array<JobState>);
    const workers: PrimitiveAtom<Array<WorkerState>> = atom([] as Array<WorkerState>);

    const onMessage = function(event: MessageEvent<any>) {
        let text: string;
        if (event.data instanceof ArrayBuffer) {
            let utf8decoder = new TextDecoder();
            text = utf8decoder.decode(event.data);
        } else {
            text = event.data;
        }

        console.log(`Socket event: ${text}`)
        let data: ManagerMessage = JSON.parse(text);

        if (data.msg === "jobs_update") {
            store.set(jobs, (prev) => data.state);
        } else if (data.msg === "workers_update") {
            store.set(workers, (prev) => data.state);
        } else if (data.msg === "connected") {
            store.set(workers, (prev) => data.workers);
            store.set(jobs, (prev) => data.jobs);
        }
    }

    const protocol = window.location.protocol == 'https:' ? "wss:" : "ws:";
    const { status, lastSeen } = websocket({
        address: `${protocol}//${window.location.host}${window.location.pathname}/listen`,
        onMessage,
    });

    return <Provider store={store}>
        <AppShell header={{ height: 80 }} padding="md">
            <AppShell.Header><Header serverStatus={status}/></AppShell.Header>
            <AppShell.Main><Container size="lg">
                <Section name="Start workers"><StartWorkers/></Section>
                <Section name="Workers"><Workers workers={workers}/></Section>
                <Section name="Start reconstructions"><StartJobs/></Section>
                <Section name="Jobs"><Jobs jobs={jobs}/></Section>
            </Container></AppShell.Main>
        </AppShell>
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