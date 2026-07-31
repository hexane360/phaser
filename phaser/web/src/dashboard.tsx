
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Provider, useStore } from 'jotai';

import '@mantine/core/styles.css';
import '@hexane/plotlib/styles.css';
import { createTheme, MantineProvider } from '@mantine/core';
import * as plotlib from '@hexane/plotlib';

import './styles.css';
import { PubSubProvider } from './pubsub';
import { Dashboard } from './layout/Dashboard';

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
