
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import '@mantine/core/styles.css';
import { AppShell, Button, Container, MantineProvider, Stack, Text, Title } from '@mantine/core';

import './styles.css';
import { makeTheme, cssVariableResolver } from './theme';
import Header from './header';
import { rootPrefix } from './utils';

interface ErrorData {
    code: number,
    name: string,
    description: string | null,
}

// server-rendered, from `templates/error.html`
function errorData(): ErrorData {
    const el = document.getElementById('error-data');
    if (el?.textContent) {
        try {
            return JSON.parse(el.textContent);
        } catch (e) {
            console.error("Couldn't parse error data:", e);
        }
    }
    return {code: 500, name: "Error", description: null};
}

function ErrorPage({code, name, description}: ErrorData) {
    return <AppShell header={{height: 80}} padding="md">
        <AppShell.Header><Header/></AppShell.Header>
        <AppShell.Main>
            <Container size="sm" pt="xl">
                <Stack align="center" gap="md">
                    <Title order={1} c="dimmed" fw="normal">{code}</Title>
                    <Title order={2} fw="normal">{name}</Title>
                    {/* escaped server-side */}
                    {description && <Text ta="center" dangerouslySetInnerHTML={{__html: description}}/>}
                    <Button component="a" href={`${rootPrefix()}/`} mt="md">Back to jobs</Button>
                </Stack>
            </Container>
        </AppShell.Main>
    </AppShell>;
}

const root = createRoot(document.getElementById('app')!);
root.render(
    <StrictMode>
        <MantineProvider theme={makeTheme()} cssVariablesResolver={cssVariableResolver}>
            <ErrorPage {...errorData()}/>
        </MantineProvider>
    </StrictMode>
);
