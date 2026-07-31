import React, { useMemo } from 'react';
import { atom, Atom, useAtomValue } from 'jotai';
import { selectAtom } from 'jotai/utils';
import { interpolateMagma } from 'd3-scale-chromatic';

import * as plotlib from '@hexane/plotlib';
import type { ScaleSpec } from '@hexane/plotlib';
import type { NumericScale, ColorLike } from '@hexane/plotlib/scale';
import { useComputedColorScheme } from '@mantine/core';

import { DecodedArray, CropBounds, minmaxNaN, applyMagmaInto } from '../array';
import { ObjectSampling, ObjPhaseSumData } from '../types';
import { usePubSubView, ViewState } from '../pubsub';
import { isClose } from '../utils';
import { ViewProps } from './types';
import { ViewGate, gateAtom } from './ViewStatus';

const projectedPhase = (data: ObjPhaseSumData) => data.data;

export function ObjectPhaseSumView(_props: ViewProps) {
    const state = usePubSubView<ObjPhaseSumData>({view: 'obj_phase_sum'});
    const {metaState, phaseState} = usePhaseAtoms(state, projectedPhase);
    const gate = useMemo(() => gateAtom(state), [state]);

    return <ViewGate state={gate}>
        <PhaseImage metaState={metaState} phaseState={phaseState} label="Object Phase"/>
    </ViewGate>;
}

// Splits a view's state into a rarely-changing `sampling` atom and a per-tick image atom --
// see the comment on `PhaseImageSub` for why that split matters. `selectAtom`'s equality
// check is what holds the sampling reference stable across ticks, replacing the
// compare-then-set that used to live in each subscription callback.
export function usePhaseAtoms<T extends {sampling: ObjectSampling}>(
    state: Atom<ViewState<T>>,
    phase: (data: T) => DecodedArray | null,
): {metaState: Atom<ObjectSampling | null>, phaseState: Atom<DecodedArray | null>} {
    const metaState = useMemo(() => selectAtom(
        state,
        (s) => s.status === 'ok' ? s.data.sampling : null,
        (a, b) => a === b || (a !== null && b !== null && samplingEqual(a, b)),
    ), [state]);

    const phaseState = useMemo(() => selectAtom(
        state,
        (s) => s.status === 'ok' ? phase(s.data) : null,
    ), [state, phase]);

    return {metaState, phaseState};
}

export function samplingEqual(a: ObjectSampling | null, b: ObjectSampling): boolean {
    return a !== null &&
        isClose(a.shape, b.shape) && isClose(a.sampling, b.sampling) && isClose(a.corner, b.corner)
            && (a.region_min === null && b.region_min === null || a.region_min !== null && b.region_min !== null && isClose(a.region_min, b.region_min))
            && (a.region_max === null && b.region_max === null || a.region_max !== null && b.region_max !== null && isClose(a.region_max, b.region_max))
}

// y position = y index * sampling + corner; y index = (y position - corner) / sampling
function cropBoundsForAutoscale(sampling: ObjectSampling, nx: number): CropBounds | undefined {
    if (!sampling.region_min || !sampling.region_max) return undefined;
    return {
        nx,
        yMin: Math.ceil((sampling.region_min[0] - sampling.corner[0]) / sampling.sampling[0]),
        yMax: Math.floor((sampling.region_max[0] - sampling.corner[0]) / sampling.sampling[0]),
        xMin: Math.ceil((sampling.region_min[1] - sampling.corner[1]) / sampling.sampling[1]),
        xMax: Math.floor((sampling.region_max[1] - sampling.corner[1]) / sampling.sampling[1]),
    };
}

interface PhaseImageProps {
    metaState: Atom<ObjectSampling | null>
    // real-valued (y, x) phase image, matching `metaState`'s sampling
    phaseState: Atom<DecodedArray | null>
    label: string
}

// A real (y, x) phase image on an `ObjectSampling` grid; shared by the projected-phase and
// single-slice object views.
export function PhaseImage({metaState, phaseState, label}: PhaseImageProps) {
    const hasObject = useAtomValue(metaState) !== null;
    if (!hasObject) return <div></div>;
    return <PhaseImageSub metaState={metaState} phaseState={phaseState} label={label}/>;
}

// `metaState` only changes when the object's shape/sampling does (rare), so this only
// re-renders (as a React tree) on those occasions. The per-tick array data instead flows
// through `phaseState` straight into atoms passed to `PlotImage`, which subscribes to them
// directly and redraws its canvas without forcing `Figure`/`Plot` (and the whole layout/
// interaction machinery underneath them) to re-render every update.
function PhaseImageSub({metaState, phaseState, label}: PhaseImageProps) {
    const sampling = useAtomValue(metaState)!;

    const [ny, nx] = sampling.shape;
    const xmin = sampling.corner[1], xmax = sampling.corner[1] + nx * sampling.sampling[1];
    const ymin = sampling.corner[0], ymax = sampling.corner[0] + ny * sampling.sampling[0];

    const aspect = nx / ny;
    const size = 500.0;
    // keep area constant
    const [xSize, ySize] = [Math.ceil(size * Math.sqrt(aspect)), Math.ceil(size / Math.sqrt(aspect))];

    const cropBounds = useMemo(() => cropBoundsForAutoscale(sampling, nx), [sampling, nx]);

    const drawFnAtom = useMemo(() => atom((get) => {
        const phase = get(phaseState);
        return (_ctx: CanvasRenderingContext2D, imageData: ImageData, scale: NumericScale<ColorLike>) => {
            if (!phase) return;
            const [vmin, vmax] = scale.domain as [number, number];
            applyMagmaInto(imageData, phase.data, vmin, vmax);
        };
    }), [phaseState]);

    const autoscaleFnAtom = useMemo(() => atom((get) => {
        const phase = get(phaseState);
        return () => {
            if (!phase) return { vmin: null, vmax: null };
            const [vmin, vmax] = minmaxNaN(phase.data, cropBounds);
            return { vmin, vmax };
        };
    }), [phaseState, cropBounds]);

    const scales: Map<string, ScaleSpec> = useMemo(() => new Map([
        ["x", { scale: plotlib.linear([xmin, xmax], undefined, { show: false }), size: `${xSize}px` }],
        ["y", { scale: plotlib.linear([ymin, ymax], undefined, { show: false }), size: `${ySize}px` }],
        ["phase", { scale: plotlib.linear([0, 1], interpolateMagma, { label }) }],
    ] satisfies [string, ScaleSpec][]), [xmin, xmax, ymin, ymax, xSize, ySize, label]);

    return <plotlib.Figure scales={scales} width="100%" colorScheme={useComputedColorScheme('light')}>
        <plotlib.layout.CenteredX>
            <plotlib.Plot xaxis="x" yaxis="y" colorbar="phase" fixedAspect={true}>
                <plotlib.Plot.Clip>
                    <plotlib.PlotImage draw_fn={drawFnAtom} autoscale_fn={autoscaleFnAtom} width={nx} height={ny} scale="phase"/>
                </plotlib.Plot.Clip>
                <plotlib.Scalebar unitScale={1e-10}/>
            </plotlib.Plot>
        </plotlib.layout.CenteredX>
    </plotlib.Figure>;
}
