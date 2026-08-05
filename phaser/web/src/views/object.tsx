import React, { useMemo } from 'react';
import { atom, Atom, useAtomValue } from 'jotai';
import { selectAtom } from 'jotai/utils';
import { interpolateMagma } from 'd3-scale-chromatic';

import * as plotlib from '@hexane/plotlib';
import type { ScaleSpec } from '@hexane/plotlib';
import type { NumericScale, ColorLike } from '@hexane/plotlib/scale';
import { Group, Slider, Text, useComputedColorScheme } from '@mantine/core';

import { DecodedArray, CropBounds, amplitude, minmaxNaN, applyMagmaInto, objectPhaseProjected } from '../array';
import { ObjectSampling, ObjMeta } from '../types';
import { usePubSubView, ViewState } from '../pubsub';
import { isClose } from '../utils';
import { ViewProps } from './types';
import { PENDING, ViewGate, gateAtom } from './ViewStatus';

// The four object views, as two shells x two quantities. Phase and amplitude are conjugate
// readings of the same complex object: phase composes additively across slices (`nansum`),
// amplitude multiplicatively (a geometric mean) -- both reductions are server-side, in
// `phaser/web/views.py`. The whole-object views read those reduced topics; the stack views
// share the one `obj` topic (a raw complex slice) and differ only in the transform below.

// the whole-object topics arrive already reduced to a real image
const identity = (data: DecodedArray) => data;

export const ObjectPhaseSumView = (_props: ViewProps) =>
    <ObjectProjection view="obj_phase_sum" quantity="Phase sum [rad]"/>;

export const ObjectAmpMeanView = (_props: ViewProps) =>
    <ObjectProjection view="obj_amp_mean" quantity="Amp. (geom. mean)"/>;

export const ObjectPhaseStackView = (props: ViewProps) =>
    <ObjectStack {...props} transform={objectPhaseProjected} quantity="Phase [rad]"/>;

export const ObjectAmpStackView = (props: ViewProps) =>
    <ObjectStack {...props} transform={amplitude} quantity="Amplitude"/>;


// A whole-object image on its own topic: no params, so nothing here changes topic.
function ObjectProjection({view, quantity}: {view: string, quantity: string}) {
    const metaTopic = usePubSubView<ObjMeta>({view: 'obj_meta'});
    const dataTopic = usePubSubView<DecodedArray>({view});

    const metaState = useObjMeta(metaTopic);
    const imageState = useObjectImage(dataTopic, identity);
    const gate = useMemo(() => gateAtom(metaTopic, dataTopic), [metaTopic, dataTopic]);

    return <ViewGate state={gate}>
        <ObjectImage metaState={metaState} imageState={imageState} label={`Object ${quantity}`}/>
    </ViewGate>;
}

interface ObjectStackProps extends ViewProps {
    // maps the raw complex slice to the real image to draw
    transform: (data: DecodedArray) => DecodedArray
    quantity: string
}

// One slice of the object, selected by a slider. `obj_meta` bounds the selector and is a
// topic of its own, so nothing here unmounts when the selection changes -- see `ViewStatus`.
function ObjectStack({params, setParams, transform, quantity}: ObjectStackProps) {
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
    const imageState = useObjectImage(dataTopic, transform);
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
            <ObjectImage
                metaState={metaState} imageState={imageState}
                label={multislice ? `Slice ${quantity}` : `Object ${quantity}`}
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

// The object's metadata, held stable by value. The equality check is what makes this a
// "rarely-changing" atom: `obj_meta` republishes on every tick (its dep, `object`, changes
// every tick) with identical contents, and without it every tick would hand out a fresh
// object and re-render `ObjectImageSub` -- see the comment there for why that matters.
function useObjMeta(metaTopic: Atom<ViewState<ObjMeta>>): Atom<ObjMeta | null> {
    return useMemo(() => selectAtom(
        metaTopic,
        (s) => s.status === 'ok' ? s.data : null,
        (a, b) => a === b || (a !== null && b !== null && objMetaEqual(a, b)),
    ), [metaTopic]);
}

// The per-tick image itself, on its own topic and so its own atom. `transform` must be a
// stable reference (module-level, or a prop that doesn't change), or the atom is rebuilt on
// every render.
function useObjectImage(
    dataTopic: Atom<ViewState<DecodedArray>>,
    transform: (data: DecodedArray) => DecodedArray | null,
): Atom<DecodedArray | null> {
    return useMemo(() => selectAtom(
        dataTopic, (s) => s.status === 'ok' ? transform(s.data) : null,
    ), [dataTopic, transform]);
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

interface ObjectImageProps {
    metaState: Atom<ObjMeta | null>
    // real-valued (y, x) image, on `metaState`'s sampling grid
    imageState: Atom<DecodedArray | null>
    label: string
}

// A real (y, x) image on an `ObjectSampling` grid; shared by all four object views.
function ObjectImage({metaState, imageState, label}: ObjectImageProps) {
    const hasObject = useAtomValue(metaState) !== null;
    if (!hasObject) return <div></div>;
    return <ObjectImageSub metaState={metaState} imageState={imageState} label={label}/>;
}

// `metaState` only changes when the object's shape/sampling does (rare), so this only
// re-renders (as a React tree) on those occasions. The per-tick array data instead flows
// through `imageState` straight into atoms passed to `PlotImage`, which subscribes to them
// directly and redraws its canvas without forcing `Figure`/`Plot` (and the whole layout/
// interaction machinery underneath them) to re-render every update.
function ObjectImageSub({metaState, imageState, label}: ObjectImageProps) {
    const {sampling} = useAtomValue(metaState)!;

    const [ny, nx] = sampling.shape;
    const xmin = sampling.corner[1], xmax = sampling.corner[1] + nx * sampling.sampling[1];
    const ymin = sampling.corner[0], ymax = sampling.corner[0] + ny * sampling.sampling[0];

    const x_size = 60.0; //%
    const y_size = ny/nx * x_size;

    const cropBounds = useMemo(() => cropBoundsForAutoscale(sampling, nx), [sampling, nx]);

    const drawFnAtom = useMemo(() => atom((get) => {
        const image = get(imageState);
        if (!image) return () => PENDING;
        return (_ctx: CanvasRenderingContext2D, imageData: ImageData, scale: NumericScale<ColorLike>) => {
            const [vmin, vmax] = scale.domain as [number, number];
            applyMagmaInto(imageData, image.data, vmin, vmax);
        };
    }), [imageState]);

    // also pending while the image is: this is awaited first, so `PlotImage` parks here
    // rather than in the draw, before it has allocated an `ImageData` for the canvas
    const autoscaleFnAtom = useMemo(() => atom((get) => {
        const image = get(imageState);
        if (!image) return () => PENDING;
        return () => {
            const [vmin, vmax] = minmaxNaN(image.data, cropBounds);
            return { vmin, vmax };
        };
    }), [imageState, cropBounds]);

    const scales: Map<string, ScaleSpec> = useMemo(() => new Map([
        ["x", { scale: plotlib.linear([xmin, xmax], undefined, { show: false }), size: `${x_size}%` }],
        ["y", { scale: plotlib.linear([ymin, ymax], undefined, { show: false }), size: `${y_size}%` }],
        ["value", { scale: plotlib.linear([0, 1], interpolateMagma, { label, tickFormat: '.2f' }) }],
    ] satisfies [string, ScaleSpec][]), [xmin, xmax, ymin, ymax, x_size, y_size, label]);

    return <Group justify="center"><plotlib.Figure scales={scales} width="100%" colorScheme={useComputedColorScheme('light')}>
        <plotlib.layout.CenteredX>
            <plotlib.Plot xaxis="x" yaxis="y" colorbar="value" fixedAspect={true} suspense={true}>
                <plotlib.Plot.Clip>
                    <plotlib.PlotImage draw_fn={drawFnAtom} autoscale_fn={autoscaleFnAtom} width={nx} height={ny} scale="value"/>
                </plotlib.Plot.Clip>
                <plotlib.Scalebar unitScale={1e-10}/>
            </plotlib.Plot>
        </plotlib.layout.CenteredX>
    </plotlib.Figure></Group>;
}
