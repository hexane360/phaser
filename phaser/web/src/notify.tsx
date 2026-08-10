import React from 'react';
import { PrimitiveAtom, useAtomValue } from 'jotai';

import { Button, Code, Group, Modal, Text } from '@mantine/core';
import { notifications } from '@mantine/notifications';

import { ConnectionStatus } from './connection';

const AUTO_CLOSE_MS = 8_000;
// A block error is there to be read, not glanced at
const BLOCK_AUTO_CLOSE_MS = 30_000;
// How long the socket must stay down before the modal appears. Reconnects are usually far
// quicker than this, and a blocking dialog for a blip nobody noticed is worse than the blip.
const DISCONNECT_GRACE_MS = 5_000;

export interface ReportOptions {
    // Renders the message as a monospace block, keeping its line structure. For errors the
    // server composes across several lines -- a plan validation report -- which collapse into
    // an unreadable paragraph otherwise.
    block?: boolean;
}

// Shows a failure as a red notification. `message` comes from `errorMessage` (`requests.tsx`).
export function reportError(title: string, message: string, {block}: ReportOptions = {}): void {
    console.error(`${title}: ${message}`);
    notifications.show({
        color: 'red', title, autoClose: block ? BLOCK_AUTO_CLOSE_MS : AUTO_CLOSE_MS,
        message: block
            ? <Code block style={{whiteSpace: 'pre-wrap', maxHeight: '20em', overflow: 'auto'}}>{message}</Code>
            : message,
    });
}

// A page whose socket is down is *silently* stale: every view keeps rendering its last value
// with nothing to say it stopped updating. That warrants blocking the UI rather than a toast
// the user can miss -- what's on screen can no longer be trusted.
//
// Rendered by `PubSubProvider`, so every page with a connection has it without opting in.
export function ConnectionModal({status}: {status: PrimitiveAtom<ConnectionStatus>}) {
    const conn = useAtomValue(status);
    const [dismissed, setDismissed] = React.useState(false);
    const [expired, setExpired] = React.useState(false);

    const connected = conn.type === 'connected';
    const stopped = conn.type === 'stopped';

    // Keyed on these two alone, deliberately: a reconnect cycles through
    // connecting/reconnecting/disconnected repeatedly, and keying on `conn.type` would
    // restart the grace timer on every one of those, so it would never expire.
    React.useEffect(() => {
        if (connected) {
            setDismissed(false);
            setExpired(false);
            return;
        }
        // the server said it was going away -- there's nothing to wait out
        if (stopped) {
            setExpired(true);
            return;
        }
        const timer = setTimeout(() => setExpired(true), DISCONNECT_GRACE_MS);
        return () => clearTimeout(timer);
    }, [connected, stopped]);

    return <Modal
        opened={expired && !connected && !dismissed} onClose={() => setDismissed(true)}
        title={stopped ? "Server has been shutdown" : "Server disconnected"}
        withCloseButton={false} closeOnClickOutside={false} closeOnEscape={false} centered
    >
        <Text size="sm">
            {stopped
                ? "The server shut down. This page is no longer live."
                : "Lost contact with the server. This page is no longer live."}
        </Text>
        <Group justify="right" mt="md">
            <Button variant="default" onClick={() => setDismissed(true)}>Dismiss</Button>
        </Group>
    </Modal>;
}
