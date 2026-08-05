import React from 'react';
import { useAtomValue } from 'jotai';

import { Alert, Code, Collapse, Group, Text, UnstyledButton } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { IconAlertTriangle, IconChevronRight } from '@tabler/icons-react';

import { JobState } from '../types';
import { fetchTraceback, jobStatus } from '../status';
import { usePubSubView } from '../pubsub';
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

// Fixed chrome rather than a palette widget, deliberately: a widget can be absent from a
// stored layout, and a silently-failed job is the thing this exists to prevent.
export function JobErrorBanner() {
    const job = useJobState();
    const [dismissed, setDismissed] = React.useState(false);
    const [expanded, {toggle}] = useDisclosure(false);
    const [traceback, setTraceback] = React.useState<string | null>(null);

    const failed = job?.result === 'errored';
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

    return <Alert
        className={classes.errorBanner} color="red" icon={<IconAlertTriangle/>}
        title="Job failed" withCloseButton onClose={() => setDismissed(true)}
    >
        <Text size="sm">{job.error_summary ?? "The reconstruction stopped with an error."}</Text>
        {traceback && <>
            <UnstyledButton onClick={toggle} mt="xs">
                <Group gap={4}>
                    <IconChevronRight size={14} style={{transform: expanded ? 'rotate(90deg)' : undefined}}/>
                    <Text size="sm">{expanded ? "Hide" : "Show"} traceback</Text>
                </Group>
            </UnstyledButton>
            <Collapse expanded={expanded} keepMounted={false}>
                <Code block mt="xs">{traceback}</Code>
            </Collapse>
        </>}
    </Alert>;
}
