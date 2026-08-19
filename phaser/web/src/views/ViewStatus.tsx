import React from 'react';
import { Atom, atom, useAtomValue } from 'jotai';
import { selectAtom } from 'jotai/utils';

import { Text } from '@mantine/core';

import { ViewState } from '../pubsub';

// Never settles. Returned in place of a `PlotImage` draw/autoscale function when the image
// isn't available yet, which holds it in its loading state (see `Plot`'s `suspense`): the
// previously drawn canvas stays up, and the figure is never unmounted. The counterpart to
// `ViewGate` for the gap a plot covers itself rather than being gated on.
export const PENDING: Promise<never> = new Promise(() => {});

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
//
// A plot view gates on its *metadata* topic, which is stable across param changes, and
// passes its bulk array topics as `also`: those contribute only their errors, never their
// pending state. Gating on a pending bulk topic would unmount the plot (and reflow the
// layout) every time another slice is selected, where the plot can instead hold its last
// image and cover the gap itself -- see `PENDING` above.
export function gateAtom(
    state: Atom<ViewState<unknown>>, ...also: Array<Atom<ViewState<unknown>>>
): Atom<ViewState<unknown> | null> {
    if (also.length === 0) return selectAtom(state, (s) => s.status === 'ok' ? null : s);

    return atom((get) => {
        const s = get(state);
        if (s.status !== 'ok') return s;

        for (const other of also) {
            const o = get(other);
            if (o.status === 'error') return o;
        }
        return null;
    });
}
