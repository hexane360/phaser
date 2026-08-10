
import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { atom, PrimitiveAtom, useAtomValue, Provider, useStore } from 'jotai';

import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';
import '@mantine/dropzone/styles.css';
import { AppShell, MantineProvider, Container, Group, Button, Collapse, Title, LoadingOverlay, Box, Tabs, Stack, Code, Progress, Text, ActionIcon, Autocomplete, Modal } from '@mantine/core';
import { Dropzone, FileRejection } from '@mantine/dropzone';
import { Notifications } from '@mantine/notifications';
import { IconUpload, IconFileText, IconX } from '@tabler/icons-react';
import { useDisclosure, useLocalStorage } from '@mantine/hooks';
import TimeAgo from 'react-timeago';

import './styles.css';
import { JobState, WorkerState } from './types';
import { fetchTraceback, jobHasFailure, jobStatus, workerStatusColor } from './status';
import { makeTheme, cssVariableResolver } from './theme';
import { Section, Mono } from './components';
import Header from './header';
import { PubSubProvider, usePubSubConnection, usePubSubView, ViewState } from './pubsub';
import { ConnectionStatus } from './connection';
import { useGetAction, usePostAction } from './requests';
import { reportError } from './notify';
import { rootPrefix } from './utils';


export function Worker({state}: {state: WorkerState}) {
    const [opened, {toggle}] = useDisclosure(false);
    const [reload] = usePostAction("Couldn't reload worker");
    const [shutdown] = usePostAction("Couldn't shut down worker");

    // the row card toggles on click, so a button press must not reach it
    function signal(e: React.MouseEvent, run: (url: string) => void, link: string) {
        e.stopPropagation();
        run(state.links[link]);
    }

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
                <Button variant="default" onClick={(e) => signal(e, reload, 'reload')}>Reload</Button>
                <Button variant="default" mod={{danger: true}} onClick={(e) => signal(e, shutdown, 'shutdown')}>Shutdown</Button>
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

// The detail behind a failure, fetched on expand rather than carried in `JobState` --
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
        <Text c={jobStatus(state).color}>{state.error_summary}</Text>
        {traceback === 'pending'
            ? <Text size="sm" c="dimmed">Loading details…</Text>
            : traceback
                ? <Code block>{traceback}</Code>
                : <Text size="sm" c="dimmed">No traceback recorded. <a href={state.links.logs_txt}>Full log</a></Text>}
    </Stack>;
}

export function Job({state}: {state: JobState}) {
    const [opened, {toggle}] = useDisclosure(false);
    const [cancel] = usePostAction("Couldn't cancel job");

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
                <Button variant="default" mod={{danger: true}} onClick={(e) => {
                    e.stopPropagation();
                    cancel(state.links.cancel);
                }}>Cancel</Button>
            </Group>
        </div>
        <Collapse className="card-body" expanded={opened} keepMounted={false}>
            <div className="grid" style={{gridTemplateColumns: "1fr 1fr"}}>
                {jobHasFailure(state) && <JobFailure state={state}/>}
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
    const [start, pending] = usePostAction("Couldn't start worker");

    // an allocated worker announces itself on the `workers` topic, so there's nothing to
    // report on success
    const start_worker = (worker_type: string) => () => start(`worker/${worker_type}/start`);

    const panelStyle = {
        padding: "10px",
        minHeight: "100px",
    };

    return <Box pos="relative" style={{maxWidth: "600px"}}>
        <LoadingOverlay visible={pending} zIndex={1000}/>
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

// A dropped file and what became of it. Ids come from a counter rather than the name, so a
// row stays addressable while others are removed and the same file can be dropped twice.
interface UploadedFile {
    id: number
    file: File
    status: 'submitting' | 'submitted' | 'failed'
}

let next_upload_id = 0;

const MAX_PLAN_BYTES = 5 * 1024 * 1024;
const PLAN_INPUT_PROPS = {accept: '.yaml,.yml,text/yaml,text/json'};

// A job submitted with no worker to run it just sits in the queue, which looks exactly like
// a stuck one. Returns `[check, modal]`: call `check` once a job is queued, and render the
// modal. Both tabs share a single instance, mounted by `StartJobs`.
function useWorkerPrompt(): [() => void, React.ReactNode] {
    const workers = usePubSubView<Array<WorkerState>>('workers');
    const state = useAtomValue(workers);
    const [opened, {open, close}] = useDisclosure(false);
    const [start, pending] = usePostAction("Couldn't start worker");

    // a stopped worker will never take the job; anything else either is running or is on
    // its way to running
    const available = state.status === 'ok' && state.data.some((w) => w.status !== 'stopped');

    const start_worker = async () => {
        // the new worker announces itself on the `workers` topic, so closing is all that's left
        await start("worker/local/start");
        close();
    };

    const modal = <Modal opened={opened} onClose={close} title="No workers running" centered>
        <Text size="sm">
            The job is queued, but no workers are currently running.
            Start a local worker?
        </Text>
        <Group justify="right" mt="md">
            <Button variant="default" onClick={close}>Not now</Button>
            <Button onClick={start_worker} loading={pending}>Start local worker</Button>
        </Group>
    </Modal>;

    return [() => { if (!available) open(); }, modal];
}

const MAX_COMPLETIONS = 50;

// The directory part of a typed path, i.e. everything through the last separator. `''` is the
// server's root.
function dirOf(path: string): string {
    return path.slice(0, path.lastIndexOf('/') + 1);
}

function SubmitPath({onQueued}: {onQueued: () => void}) {
    // survives a reload: the same plan usually gets submitted several times in a sitting
    const [path, setPath] = useLocalStorage({
        key: 'phaser.manager.path', defaultValue: "", getInitialValueInEffect: false,
    });
    const [entries, setEntries] = React.useState<Array<string>>([]);
    const [submit, pending] = usePostAction("Couldn't submit job", {block: true});
    // quiet: a completion nobody asked for shouldn't raise a toast when it fails
    const [ls] = useGetAction<{entries: Array<string>}>("Couldn't list directory", {quiet: true});

    const dir = dirOf(path);

    // Keyed on the directory rather than the path, so typing within a segment costs nothing
    // and only crossing a `/` fetches. The server restricts what it will list to its root;
    // a path typed past that simply completes to nothing.
    //
    // Nothing is remembered beyond the directory in the field. Someone retyping a path has
    // usually just changed what's on disk, and a stale listing is worse than a re-fetch.
    React.useEffect(() => {
        // cleared first: options are built by prefixing these names with `dir`, so entries
        // held over from the previous directory would render as plausible-looking paths
        // under the new one -- including when the new one is refused as outside the root
        setEntries([]);

        let live = true;
        ls(`ls_path?path=${encodeURIComponent(dir)}`).then((result) => {
            if (result && live) setEntries(result.entries);
        });
        return () => { live = false; };
    }, [dir, ls]);

    // Full paths in the field's own spelling, which is what makes Mantine's default filter
    // (label contains the search string) the right one -- the typed value is always a prefix.
    // A directory keeps its trailing `/`, so picking one is itself the trigger to list it.
    const options = React.useMemo(() => entries.map((entry) => dir + entry), [dir, entries]);

    // a started job announces itself on the `jobs` topic, so there's nothing to report on success
    const submit_job = async () => {
        if (await submit("job/start", {source: 'path', path})) onQueued();
    };

    // Enter submits, except while the dropdown has an option highlighted -- that keypress
    // belongs to the completion. Mantine publishes exactly that state as `aria-activedescendant`
    // on the input, and this handler runs before the combobox's own, so the attribute is the
    // only thing available to tell the two cases apart.
    const key_down = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key !== 'Enter' || e.nativeEvent.isComposing) return;
        if (e.currentTarget.getAttribute('aria-activedescendant')) return;
        submit_job();
    };

    return <Box pos="relative">
        <LoadingOverlay visible={pending} zIndex={1000}/>
        <Stack>
            <div>Runs a plan from a file on the server's filesystem</div>
            <Group align="flex-start">
                <Autocomplete
                    name="path" value={path} onChange={setPath} data={options}
                    onKeyDown={key_down} limit={MAX_COMPLETIONS} w={400}
                    placeholder="path/to/plan.yaml"
                />
                <button type="submit" onClick={submit_job}>Submit</button>
            </Group>
        </Stack>
    </Box>;
}

function SubmitUpload({onQueued}: {onQueued: () => void}) {
    const [uploads, setUploads] = React.useState<Array<UploadedFile>>([]);
    // no overlay: a drop submits immediately, and the per-row status is the progress display
    const [submit] = usePostAction("Couldn't submit job", {block: true});

    const setStatus = (id: number, status: UploadedFile['status']) =>
        setUploads((s) => s.map((f) => f.id === id ? {...f, status} : f));

    // Dropping submits. Each plan is its own job, posted one at a time so a failure in one
    // doesn't stop the rest; the row carries the outcome, since the error toast names the
    // action rather than the file.
    const drop = async (files: Array<File>) => {
        const added = files.map((file): UploadedFile => ({id: next_upload_id++, file, status: 'submitting'}));
        setUploads((s) => [...s, ...added]);

        let queued = false;
        for (const {id, file} of added) {
            const result = await submit("job/start", {source: 'yaml', data: await file.text()});
            setStatus(id, result ? 'submitted' : 'failed');
            queued ||= !!result;
        }
        // once for the batch, not once per file
        if (queued) onQueued();
    };

    const reject = (rejections: Array<FileRejection>) => {
        for (const {file, errors} of rejections)
            reportError("Couldn't accept file", `${file.name}: ${errors.map((e) => e.message).join(', ')}`);
    };

    const retry = async ({id, file}: UploadedFile) => {
        setStatus(id, 'submitting');
        const result = await submit("job/start", {source: 'yaml', data: await file.text()});
        setStatus(id, result ? 'submitted' : 'failed');
    };

    const status_of = (upload: UploadedFile) => {
        switch (upload.status) {
        case 'submitting':
            return <Text size="sm" c="dimmed">submitting…</Text>;
        case 'submitted':
            return <Text size="sm" c="green">submitted</Text>;
        case 'failed':
            return <Group gap="xs">
                <Text size="sm" c="red">failed</Text>
                <Button size="compact-xs" variant="default" onClick={() => retry(upload)}>Retry</Button>
            </Group>;
        }
    };

    return <Stack>
        <Dropzone
            onDrop={drop} onReject={reject} maxSize={MAX_PLAN_BYTES}
            inputProps={PLAN_INPUT_PROPS} activateOnClick
        >
            <Group justify="center" gap="lg" mih={100} style={{pointerEvents: 'none'}}>
                <Dropzone.Accept><IconUpload size={40}/></Dropzone.Accept>
                <Dropzone.Reject><IconX size={40}/></Dropzone.Reject>
                <Dropzone.Idle><IconFileText size={40}/></Dropzone.Idle>
                <Stack gap={0}>
                    <Text size="lg">Drop reconstruction plans here</Text>
                    <Text size="sm" c="dimmed">or click to browse.</Text>
                </Stack>
            </Group>
        </Dropzone>
        {uploads.length > 0 && <Stack gap="xs">
            {...uploads.map((upload) =>
                <Group key={upload.id} justify="space-between">
                    <Group gap="xs">
                        <Mono>{upload.file.name}</Mono>
                        <Text size="sm" c="dimmed">{(upload.file.size / 1024).toFixed(1)} kB</Text>
                        {status_of(upload)}
                    </Group>
                    <ActionIcon
                        variant="subtle" onClick={() => setUploads((s) => s.filter((f) => f.id !== upload.id))}
                    ><IconX size={16}/></ActionIcon>
                </Group>
            )}
            <Group><Button variant="default" onClick={() => setUploads([])}>Clear</Button></Group>
        </Stack>}
    </Stack>;
}

export function StartJobs(props: {}) {
    const [checkWorkers, workerModal] = useWorkerPrompt();

    const panelStyle = {
        padding: "10px",
        minHeight: "100px",
    };

    return <Box style={{maxWidth: "600px"}}>
        {workerModal}
        <Tabs variant="pills" defaultValue="upload">
            <Tabs.List>
                <Tabs.Tab value="upload">Upload</Tabs.Tab>
                <Tabs.Tab value="path">Server path</Tabs.Tab>
            </Tabs.List>
            <Tabs.Panel value="upload" style={panelStyle}><SubmitUpload onQueued={checkWorkers}/></Tabs.Panel>
            <Tabs.Panel value="path" style={panelStyle}><SubmitPath onQueued={checkWorkers}/></Tabs.Panel>
        </Tabs>
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


const root = createRoot(document.getElementById('app')!);
root.render(
    <StrictMode>
        <MantineProvider theme={makeTheme()} cssVariablesResolver={cssVariableResolver}>
            <Notifications position="top-center"/>
            <App/>
        </MantineProvider>
    </StrictMode>
);