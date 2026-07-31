import React, { useMemo } from 'react';
import { Atom, useAtomValue } from 'jotai';
import { selectAtom } from 'jotai/utils';

import { Group, NumberInput, Text } from '@mantine/core';

import { objectPhaseProjected } from '../array';
import { ObjSliceData } from '../types';
import { usePubSubView } from '../pubsub';
import { PhaseImage, usePhaseAtoms } from './objectPhaseSum';
import { ViewProps } from './types';
import { ViewGate, gateAtom } from './ViewStatus';

// `obj` sends the raw complex slice (unlike `obj_phase_sum`, which is projected
// server-side), so the phase image is derived here.
const slicePhase = (data: ObjSliceData) => objectPhaseProjected(data.data);

export function ObjectPhaseStackView({params, setParams}: ViewProps) {
    const slice = Math.max(0, Math.trunc(Number(params.slice ?? 0)));
    // a different `slice` is a different topic, so this re-points at another shared atom
    const state = usePubSubView<ObjSliceData>({view: 'obj', slice});
    const {metaState, phaseState} = usePhaseAtoms(state, slicePhase);
    const gate = useMemo(() => gateAtom(state), [state]);
    const thicknessState = useMemo(() => selectAtom(
        state, (s) => s.status === 'ok' ? s.data.thickness : null,
    ), [state]);

    return <>
        <Group gap="lg" mb="sm">
            <NumberInput
                label="Slice" size="xs" w={100} min={0} step={1} allowDecimal={false}
                value={slice} onChange={(value) => setParams({...params, slice: Math.max(0, Math.trunc(Number(value) || 0))})}
            />
            <Thickness state={thicknessState}/>
        </Group>
        <ViewGate state={gate}>
            <PhaseImage metaState={metaState} phaseState={phaseState} label="Slice Phase"/>
        </ViewGate>
    </>;
}

function Thickness({state}: {state: Atom<number | null>}) {
    const thickness = useAtomValue(state);
    if (thickness === null) return null;
    return <Text size="sm" c="dimmed">Thickness: {thickness.toPrecision(4)} Å</Text>;
}
