
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Provider, useStore } from 'jotai';

import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';
import '@hexane/plotlib/styles.css';
import { MantineProvider } from '@mantine/core';
import { Notifications } from '@mantine/notifications';
import * as plotlib from '@hexane/plotlib';

import './styles.css';
import { PubSubProvider } from './pubsub';
import { makeTheme, cssVariableResolver } from './theme';
import { rootPrefix } from './utils';
import { Dashboard } from './layout/Dashboard';

function App(props: {}) {
    const store = useStore();

    const jobId = window.location.pathname.replace(/\/$/, '').split('/').pop()!;
    const prefix = rootPrefix();
    const protocol = window.location.protocol == 'https:' ? "wss:" : "ws:";
    const address = `${protocol}//${window.location.host}${prefix}/listen?job=${encodeURIComponent(jobId)}`;

    return <Provider store={store}>
        <PubSubProvider address={address}>
            <Dashboard/>
        </PubSubProvider>
    </Provider>
}

const root = createRoot(document.getElementById('app')!);
root.render(
    <StrictMode>
        <MantineProvider theme={makeTheme()} cssVariablesResolver={cssVariableResolver}>
            <Notifications position="top-center"/>
            <plotlib.ThemeProvider>
                <App/>
            </plotlib.ThemeProvider>
        </MantineProvider>
    </StrictMode>
);
