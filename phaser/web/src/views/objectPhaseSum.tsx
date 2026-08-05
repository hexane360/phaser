import { useMemo } from 'react';
import { atom, Atom, useAtomValue } from 'jotai';
import { selectAtom } from 'jotai/utils';
import { interpolateMagma } from 'd3-scale-chromatic';

import * as plotlib from '@hexane/plotlib';
import type { ScaleSpec } from '@hexane/plotlib';
import type { NumericScale, ColorLike } from '@hexane/plotlib/scale';
import { Group, useComputedColorScheme } from '@mantine/core';

import { DecodedArray, CropBounds, minmaxNaN, applyMagmaInto } from '../array';
import { ObjectSampling, ObjMeta } from '../types';
import { usePubSubView, ViewState } from '../pubsub';
import { isClose } from '../utils';
import { ViewProps } from './types';
import { ViewGate, gateAtom } from './ViewStatus';

// `obj_phase_sum` is projected server-side, so the payload is already the phase image
const projectedPhase = (data: DecodedArray) => data;

export function ObjectPhaseSumView(_props: ViewProps) {
    const metaTopic = usePubSubView<ObjMeta>({view: 'obj_meta'});
    const dataTopic = usePubSubView<DecodedArray>({view: 'obj_phase_sum'});

    const metaState = useObjMeta(metaTopic);
    const phaseState = usePhaseData(dataTopic, projectedPhase);
    const gate = useMemo(() => gateAtom(metaTopic, dataTopic), [metaTopic, dataTopic]);

    return <ViewGate state={gate}>
        <PhaseImage metaState={metaState} phaseState={phaseState} label="Object Phase [rad]"/>
    </ViewGate>;
}

// Never settles. Returned in place of a draw/autoscale function when the phase image isn't
// available yet, which holds `PlotImage` in its loading state (see `Plot`'s `suspense`):
// the previously drawn canvas stays up, and the figure is never unmounted.
export const PENDING: Promise<never> = new Promise(() => {});

// The object's metadata, held stable by value. The equality check is what makes this a
// "rarely-changing" atom: `obj_meta` republishes on every tick (its dep, `object`, changes
// every tick) with identical contents, and without it every tick would hand out a fresh
// object and re-render `PhaseImageSub` -- see the comment there for why that matters.
export function useObjMeta(metaTopic: Atom<ViewState<ObjMeta>>): Atom<ObjMeta | null> {
    return useMemo(() => selectAtom(
        metaTopic,
        (s) => s.status === 'ok' ? s.data : null,
        (a, b) => a === b || (a !== null && b !== null && objMetaEqual(a, b)),
    ), [metaTopic]);
}

// The per-tick image itself, on its own topic and so its own atom. `phase` maps the wire
// array to a real (y, x) image -- identity for `obj_phase_sum`, which is projected
// server-side, and `objectPhaseProjected` for `obj`, which sends the raw complex slice. It
// must be module-level (a stable reference), or the atom is rebuilt on every render.
export function usePhaseData(
    dataTopic: Atom<ViewState<DecodedArray>>,
    phase: (data: DecodedArray) => DecodedArray | null,
): Atom<DecodedArray | null> {
    return useMemo(() => selectAtom(
        dataTopic, (s) => s.status === 'ok' ? phase(s.data) : null,
    ), [dataTopic, phase]);
}

function objMetaEqual(a: ObjMeta, b: ObjMeta): boolean {
    return a.n_slices === b.n_slices && samplingEqual(a.sampling, b.sampling)
        && (a.thicknesses === null ? b.thicknesses === null
            : b.thicknesses !== null && isClose(a.thicknesses, b.thicknesses));
}

function samplingEqual(a: ObjectSampling, b: ObjectSampling): boolean {
    return isClose(a.shape, b.shape) && isClose(a.sampling, b.sampling) && isClose(a.corner, b.corner)
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
    metaState: Atom<ObjMeta | null>
    // real-valued (y, x) phase image, on `metaState`'s sampling grid
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
    const {sampling} = useAtomValue(metaState)!;

    const [ny, nx] = sampling.shape;
    const xmin = sampling.corner[1], xmax = sampling.corner[1] + nx * sampling.sampling[1];
    const ymin = sampling.corner[0], ymax = sampling.corner[0] + ny * sampling.sampling[0];

    //const aspect = nx / ny;
    const x_size = 60.0; //%
    const y_size = ny/nx * x_size;
    // keep area constant
    //const [xSize, ySize] = [Math.ceil(size * Math.sqrt(aspect)), Math.ceil(size / Math.sqrt(aspect))];

    const cropBounds = useMemo(() => cropBoundsForAutoscale(sampling, nx), [sampling, nx]);

    const drawFnAtom = useMemo(() => atom((get) => {
        const phase = get(phaseState);
        if (!phase) return () => PENDING;
        return (_ctx: CanvasRenderingContext2D, imageData: ImageData, scale: NumericScale<ColorLike>) => {
            const [vmin, vmax] = scale.domain as [number, number];
            applyMagmaInto(imageData, phase.data, vmin, vmax);
        };
    }), [phaseState]);

    // also pending while the image is: this is awaited first, so `PlotImage` parks here
    // rather than in the draw, before it has allocated an `ImageData` for the canvas
    const autoscaleFnAtom = useMemo(() => atom((get) => {
        const phase = get(phaseState);
        if (!phase) return () => PENDING;
        return () => {
            const [vmin, vmax] = minmaxNaN(phase.data, cropBounds);
            return { vmin, vmax };
        };
    }), [phaseState, cropBounds]);

    const scales: Map<string, ScaleSpec> = useMemo(() => new Map([
        ["x", { scale: plotlib.linear([xmin, xmax], undefined, { show: false }), size: `${x_size}%` }],
        ["y", { scale: plotlib.linear([ymin, ymax], undefined, { show: false }), size: `${y_size}%` }],
        ["phase", { scale: plotlib.linear([0, 1], interpolateMagma, { label, tickFormat: '.1f' }) }],
    ] satisfies [string, ScaleSpec][]), [xmin, xmax, ymin, ymax, x_size, y_size, label]);

    return <Group justify="center"><plotlib.Figure scales={scales} width="100%" colorScheme={useComputedColorScheme('light')}>
        <plotlib.layout.CenteredX>
            <plotlib.Plot xaxis="x" yaxis="y" colorbar="phase" fixedAspect={true} suspense={true}>
                <plotlib.Plot.Clip>
                    <plotlib.PlotImage draw_fn={drawFnAtom} autoscale_fn={autoscaleFnAtom} width={nx} height={ny} scale="phase"/>
                </plotlib.Plot.Clip>
                <plotlib.Scalebar unitScale={1e-10}/>
            </plotlib.Plot>
        </plotlib.layout.CenteredX>
    </plotlib.Figure></Group>;
}
