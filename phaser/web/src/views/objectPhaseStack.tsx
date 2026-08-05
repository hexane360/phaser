import React, { useMemo } from 'react';
import { useAtomValue } from 'jotai';
import { selectAtom } from 'jotai/utils';

import { Group, Slider, Text } from '@mantine/core';

import { DecodedArray, objectPhaseProjected } from '../array';
import { ObjMeta } from '../types';
import { usePubSubView } from '../pubsub';
import { PhaseImage, useObjMeta, usePhaseData } from './objectPhaseSum';
import { ViewProps } from './types';
import { ViewGate, gateAtom } from './ViewStatus';

// `obj` sends the raw complex slice (unlike `obj_phase_sum`, which is projected
// server-side), so the phase image is derived here.
const slicePhase = objectPhaseProjected;

export function ObjectPhaseStackView({params, setParams}: ViewProps) {
    const slice = Math.max(0, Math.trunc(Number(params.slice ?? 0)));

    // `live` drives the subscription so the image follows the drag; `params` is its
    // persisted mirror, written once per gesture. The layout lives in `localStorage` (see
    // `Dashboard.tsx`), so every `setParams` is a serialize + write + re-render of the whole
    // dashboard -- affordable per gesture, not per pointer move.
    const [live, setLive] = React.useState(slice);
    React.useEffect(() => setLive(slice), [slice]);

    const metaTopic = usePubSubView<ObjMeta>({view: 'obj_meta'});
    // a different `slice` is a different topic, so this re-points at another shared atom.
    // `obj_meta` doesn't, which is what keeps the slider mounted across a change.
    const dataTopic = usePubSubView<DecodedArray>({view: 'obj', slice: live});

    const metaState = useObjMeta(metaTopic);
    const phaseState = usePhaseData(dataTopic, slicePhase);
    const gate = useMemo(() => gateAtom(metaTopic, dataTopic), [metaTopic, dataTopic]);

    const nSlices = useAtomValue(useMemo(() => selectAtom(
        metaState, (m) => m?.n_slices ?? null,
    ), [metaState]));

    // a layout saved against a multislice object can outlive it; the server clamps its own
    // read, this brings the selection (and the topic it names) back in range to match
    React.useEffect(() => {
        if (nSlices === null) return;
        const max = nSlices - 1;
        if (slice > max) setParams({...params, slice: max});
        if (live > max) setLive(max);
    }, [nSlices, slice, live]);

    const multislice = nSlices !== null && nSlices > 1;

    return <>
        <ViewGate state={gate}>
            {/* a single slice is the whole object, so it's the projection too. The label
                deliberately omits the slice index: it feeds the colorbar scale, and
                rebuilding that on every change is the churn this view is avoiding. */}
            <PhaseImage
                metaState={metaState} phaseState={phaseState}
                label={multislice ? 'Slice Phase [rad]' : 'Object Phase [rad]'}
            />
        </ViewGate>
        {multislice && <Group gap="sm" mb="sm" align="center" justify="center" wrap="nowrap">
            <Text size="xs">Slice</Text>
            <Slider
                w={240} size="sm" min={0} max={nSlices - 1} step={1} value={live}
                onChange={setLive} onChangeEnd={(value) => setParams({...params, slice: value})}
                label={(value) => `${value} / ${nSlices - 1}`}
            />
        </Group>}
    </>;
}
