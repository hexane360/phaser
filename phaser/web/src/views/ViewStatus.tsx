import React from 'react';
import { Atom, useAtomValue } from 'jotai';
import { selectAtom } from 'jotai/utils';

import { Text } from '@mantine/core';

import { ViewState } from '../pubsub';

// Placeholder for a view whose topic has no value yet, or failed. Only ever rendered for a
// non-`ok` state, so it says nothing about *why* data is absent beyond what the state carries.
export function ViewStatus({state, pending}: {state: ViewState<unknown>, pending?: string}) {
    if (state.status === 'error') return <Text size="sm" c="red">{state.reason}</Text>;
    return <Text size="sm" c="dimmed">{pending ?? 'Waiting for data…'}</Text>;
}

// Renders `children` once the topic has a value. Reads a status-only atom rather than the
// view state itself: a plot's wrapper must not re-render every tick, or the meta/data split
// its `PlotImage` relies on buys nothing.
export function ViewGate({state, children}: React.PropsWithChildren<{state: Atom<ViewState<unknown> | null>}>) {
    const status = useAtomValue(state);
    return status === null ? <>{children}</> : <ViewStatus state={status}/>;
}

// null once the topic is `ok`, so the gate above only re-renders on a status change.
export function gateAtom<T>(state: Atom<ViewState<T>>): Atom<ViewState<unknown> | null> {
    return selectAtom(state, (s) => s.status === 'ok' ? null : s);
}
