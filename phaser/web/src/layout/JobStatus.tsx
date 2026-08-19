import React from 'react';
import { useAtomValue } from 'jotai';

import { Alert, Button, Code, Collapse, Group, Text, UnstyledButton } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { IconAlertTriangle, IconChevronRight } from '@tabler/icons-react';

import { JobState } from '../types';
import { fetchTraceback, jobHasFailure, jobStatus } from '../status';
import { usePubSubView } from '../pubsub';
import { usePostAction } from '../requests';
import classes from './Layout.module.css';

// The dashboard's own job. Both consumers below read this one topic; the pub/sub layer
// shares a single atom per topic, so this is one subscription regardless.
function useJobState(): JobState | null {
    const state = useAtomValue(usePubSubView<JobState>({view: 'state'}));
    return state.status === 'ok' ? state.data : null;
}

// Header slot: what job this dashboard is showing, and how it's going. The dashboard had no
// job status at all before -- every other pane is a user-configurable widget that may not
// be in the layout.
export function JobStatusText() {
    const job = useJobState();
    if (!job) return null;

    const status = jobStatus(job);
    return <Text size="lg" c="dimmed">
        {job.job_name ?? job.job_id} · <span style={{color: status.color}}>{status.label}</span>
    </Text>;
}

// Header slot: the manager list's cancel button, for the job this dashboard is watching.
export function JobCancelButton() {
    const job = useJobState();
    const [cancel] = usePostAction("Couldn't cancel job");
    if (!job) return null;

    return <Button
        variant="default" mod={{danger: true}}
        onClick={() => cancel(job.links.cancel)}
    >Cancel</Button>;
}

// Fixed chrome rather than a palette widget, deliberately: a widget can be absent from a
// stored layout, and a silently-failed job is the thing this exists to prevent.
export function JobErrorBanner() {
    const job = useJobState();
    const [dismissed, setDismissed] = React.useState(false);
    const [expanded, {toggle}] = useDisclosure(false);
    const [traceback, setTraceback] = React.useState<string | null>(null);

    const failed = !!job && jobHasFailure(job);
    const logsUrl = job?.links.logs;

    React.useEffect(() => {
        if (!failed || !logsUrl) return;
        let live = true;
        fetchTraceback(logsUrl)
            .then((tb) => { if (live) setTraceback(tb); })
            .catch((e) => console.error("Couldn't fetch traceback:", e));
        return () => { live = false; };
    }, [failed, logsUrl]);

    if (!job || !failed || dismissed) return null;

    // titled from the shared label, so a job whose worker died reads "Job interrupted"
    // rather than claiming the reconstruction itself failed
    return <Alert
        className={classes.errorBanner} color="red" icon={<IconAlertTriangle/>}
        title={`Job ${jobStatus(job).label}`} withCloseButton onClose={() => setDismissed(true)}
    >
        <Text size="sm">{job.error_summary}</Text>
        {traceback && <>
            <UnstyledButton onClick={toggle} mt="xs">
                <Group gap={4}>
                    <IconChevronRight size={14} style={{transform: expanded ? 'rotate(90deg)' : undefined}}/>
                    <Text size="sm">{expanded ? "Hide" : "Show"} details</Text>
                </Group>
            </UnstyledButton>
            <Collapse expanded={expanded} keepMounted={false}>
                <Code block mt="xs">{traceback}</Code>
            </Collapse>
        </>}
    </Alert>;
}
