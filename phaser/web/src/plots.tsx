import React, { useMemo } from 'react';
import { atom, useAtomValue, PrimitiveAtom } from 'jotai';
import { interpolateMagma } from 'd3-scale-chromatic';

import * as plotlib from '@hexane/plotlib';
import type { ScaleSpec } from '@hexane/plotlib';
import type { NumericScale, ColorLike } from '@hexane/plotlib/scale';
import { NArray } from 'wasm-array';
import { np } from './wasm-array';

import { ProbeMeta, ObjectSampling, ProgressData } from './types';
import { useComputedColorScheme } from '@mantine/core';

function objectPhase(data: NArray): NArray {
    let phase = np!.angle(data);
    while (phase.shape.length > 2) {
        phase = np!.nansum(phase, [0]);
    }
    return phase;
}

function croppedPhase(phase: NArray, sampling: ObjectSampling): NArray {
    if (!sampling.region_min || !sampling.region_max) return phase;

    // y position = y index * sampling + corner
    // y index = (y position - corner) / sampling
    const [y_min, y_max, x_min, x_max] = [
        Math.ceil((sampling.region_min[0] - sampling.corner[0]) / sampling.sampling[0]),
        Math.floor((sampling.region_max[0] - sampling.corner[0]) / sampling.sampling[0]),
        Math.ceil((sampling.region_min[1] - sampling.corner[1]) / sampling.sampling[1]),
        Math.floor((sampling.region_max[1] - sampling.corner[1]) / sampling.sampling[1]),
    ];

    return phase.slice(new np!.Slice(y_min, y_max), new np!.Slice(x_min, x_max));
}

interface ObjectPlotProps {
    metaState: PrimitiveAtom<ObjectSampling | null>
    dataState: PrimitiveAtom<NArray | null>
}

export function ObjectPlot({metaState, dataState}: ObjectPlotProps) {
    const hasObject = useAtomValue(metaState) !== null;
    if (!hasObject || !np) return <div></div>;
    return <ObjectPlotSub metaState={metaState} dataState={dataState} />;
}

// `metaState` only changes when the object's shape/sampling does (rare), so this only
// re-renders (as a React tree) on those occasions. The per-tick array data instead flows
// through `dataState` straight into atoms passed to `PlotImage`, which subscribes to them
// directly and redraws its canvas without forcing `Figure`/`Plot` (and the whole layout/
// interaction machinery underneath them) to re-render every update.
function ObjectPlotSub({metaState, dataState}: ObjectPlotProps) {
    const sampling = useAtomValue(metaState)!;

    const [ny, nx] = sampling.shape;
    const xmin = sampling.corner[1], xmax = sampling.corner[1] + nx * sampling.sampling[1];
    const ymin = sampling.corner[0], ymax = sampling.corner[0] + ny * sampling.sampling[0];

    const aspect = nx / ny;
    const size = 500.0;
    // keep area constant
    const [xSize, ySize] = [Math.ceil(size * Math.sqrt(aspect)), Math.ceil(size / Math.sqrt(aspect))];

    const phaseAtom = useMemo(() => atom((get) => {
        const data = get(dataState);
        return data ? objectPhase(data) : null;
    }), [dataState]);

    const drawFnAtom = useMemo(() => atom((get) => {
        const phase = get(phaseAtom);
        return (_ctx: CanvasRenderingContext2D, imageData: ImageData, scale: NumericScale<ColorLike>) => {
            if (!phase || !np) return;
            const [vmin, vmax] = scale.domain as [number, number];
            imageData.data.set(
                np.expr`(${phase} - ${vmin}) / (${vmax} - ${vmin})`.apply_cmap('magma')
            );
        };
    }), [phaseAtom]);

    const autoscaleFnAtom = useMemo(() => atom((get) => {
        const phase = get(phaseAtom);
        return () => {
            if (!phase || !np) return { vmin: null, vmax: null };
            const cropped = croppedPhase(phase, sampling);
            return {
                vmin: np.nanmin(cropped).toNestedArray() as number,
                vmax: np.nanmax(cropped).toNestedArray() as number,
            };
        };
    }), [phaseAtom, sampling]);

    const scales: Map<string, ScaleSpec> = useMemo(() => new Map([
        ["x", { scale: plotlib.linear([xmin, xmax], undefined, { show: false }), size: `${xSize}px` }],
        ["y", { scale: plotlib.linear([ymin, ymax], undefined, { show: false }), size: `${ySize}px` }],
        ["phase", { scale: plotlib.linear([0, 1], interpolateMagma, { label: "Object Phase" }) }],
    ] satisfies [string, ScaleSpec][]), [xmin, xmax, ymin, ymax, xSize, ySize]);

    return <plotlib.Figure scales={scales} colorScheme={useComputedColorScheme('light')}>
        <plotlib.Plot xaxis="x" yaxis="y" colorbar="phase" fixedAspect={true}>
            <plotlib.Plot.Clip>
                <plotlib.PlotImage draw_fn={drawFnAtom} autoscale_fn={autoscaleFnAtom} width={nx} height={ny} scale="phase"/>
            </plotlib.Plot.Clip>
            <plotlib.Scalebar unitScale={1e-10}/>
        </plotlib.Plot>
    </plotlib.Figure>;
}

interface ProbePlotProps {
    metaState: PrimitiveAtom<ProbeMeta | null>
    dataState: PrimitiveAtom<NArray | null>
}

export function ProbePlot({metaState, dataState}: ProbePlotProps) {
    const hasProbe = useAtomValue(metaState) !== null;
    if (!hasProbe || !np) return <div></div>;
    return <ProbePlotSub metaState={metaState} dataState={dataState} />;
}

// see the equivalent comment on `ObjectPlot`/`ObjectPlotSub`: `metaState` only changes
// rarely, `dataState` carries the per-tick array data straight into `PlotImage` atoms.
function ProbePlotSub({metaState, dataState}: ProbePlotProps) {
    const {sampling, nprobes: n_plots} = useAtomValue(metaState)!;
    const [ny, nx] = sampling.shape;

    const intensitiesAtom = useMemo(() => atom((get) => {
        const data = get(dataState);
        return data ? np!.abs2(data) : null;
    }), [dataState]);

    // computed once per tick and shared by every probe mode's draw_fn below
    const intensitySlicesAtom = useMemo(() => atom((get) => {
        const intensities = get(intensitiesAtom);
        return intensities ? np!.split(intensities) : null;
    }), [intensitiesAtom]);

    const autoscaleFnAtom = useMemo(() => atom((get) => {
        const intensities = get(intensitiesAtom);
        return () => {
            if (!intensities || !np) return { vmin: null, vmax: null };
            return {
                vmin: np.nanmin(intensities).toNestedArray() as number,
                vmax: np.nanmax(intensities).toNestedArray() as number,
            };
        };
    }), [intensitiesAtom]);

    const drawFnAtoms = useMemo(() => Array.from({length: n_plots}, (_, i) => atom((get) => {
        const slices = get(intensitySlicesAtom);
        return (_ctx: CanvasRenderingContext2D, imageData: ImageData, scale: NumericScale<ColorLike>) => {
            if (!slices || !np) return;
            const [vmin, vmax] = scale.domain as [number, number];
            imageData.data.set(
                np.expr`(${slices[i]} - ${vmin}) / (${vmax} - ${vmin})`.apply_cmap('magma')
            );
        };
    })), [intensitySlicesAtom, n_plots]);

    const scales: Map<string, ScaleSpec> = useMemo(() => new Map([
        ["x", { scale: plotlib.linear([0, nx * sampling.sampling[1]], undefined, { show: false }), size: '180px' }],
        ["y", { scale: plotlib.linear([0, ny * sampling.sampling[0]], undefined, { show: false }), size: '180px' }],
        ["intensity", { scale: plotlib.linear([0, 1], interpolateMagma, { label: "Probe Intensity" }) }],
    ] satisfies [string, ScaleSpec][]), [nx, ny]);

    return <plotlib.Figure scales={scales} width="100%" colorScheme={useComputedColorScheme('light')}>
        <plotlib.layout.CenteredX>
            <plotlib.layout.Decorated right={<plotlib.Colorbar scale="intensity" shrink={0.8}/>}>
                <plotlib.layout.FlexBox wrap columnGap="12pt" rowGap="12pt">
                    {drawFnAtoms.map((drawFnAtom, i) => (
                        <plotlib.Plot key={i} xaxis="x" yaxis="y" fixedAspect={true}>
                            <plotlib.Plot.Clip>
                                <plotlib.PlotImage draw_fn={drawFnAtom} autoscale_fn={autoscaleFnAtom} width={nx} height={ny} scale="intensity"/>
                            </plotlib.Plot.Clip>
                            {i === n_plots - 1 && <plotlib.Scalebar unitScale={1e-10}/>}
                        </plotlib.Plot>
                    ))}
                </plotlib.layout.FlexBox>
            </plotlib.layout.Decorated>
        </plotlib.layout.CenteredX>
    </plotlib.Figure>;
}

export function ProgressPlot({state}: {state: PrimitiveAtom<Record<string, ProgressData>>}) {
    const progress = useAtomValue(state);
    if (!progress || !np || !progress.total_loss) return <div></div>;

    return <ProgressPlotSub progress={progress.total_loss} />;
}

// pads a [hi, lo] domain by `frac` on each side in log-space (hi upward, lo downward)
function padLogDomain([hi, lo]: [number, number], frac: number): [number, number] {
    const ratio = Math.pow(Math.max(hi / lo, 1.05), frac);
    return [hi * ratio, lo / ratio];
}

function ProgressPlotSub({progress}: {progress: ProgressData}) {
    const markerId = React.useId();
    const markerRef = `url(#${markerId})`;

    const xs = progress.iters;
    const ys = progress.values;
    const x_max = Math.max(10, ...xs.filter(isFinite));
    const ys_filt = ys.filter(isFinite);

    let y_min: number, y_max: number;
    if (ys_filt.length) {
        [y_min, y_max] = [Math.min(...ys_filt), Math.max(...ys_filt)];
    } else {
        [y_min, y_max] = [1.0, 1.0e5]
    }

    // scales are recomputed (and `Figure` re-rendered) every tick here: `iter`'s domain
    // grows with each new datapoint and `error`'s autoscales to the current loss range,
    // so there's no rarely-changing "meta" to split out like there was for Object/Probe.
    const scales: Map<string, ScaleSpec> = useMemo(() => new Map([
        ["iter", { scale: plotlib.linear([0, x_max], undefined, { label: "Iteration" }), size: '500px' }],
        ["error", {
            scale: plotlib.log(padLogDomain([y_max, y_min], 0.1), undefined, 10, {
                label: "Error", labelOffset: 110, tickFormat: ".2e",
            }),
            size: '300px', zoomExtent: [0.7, Infinity],
        }],
    ] satisfies [string, ScaleSpec][]), [x_max, y_max, y_min]);

    return <plotlib.Figure scales={scales} colorScheme={useComputedColorScheme('light')}>
        <plotlib.Plot xaxis="iter" yaxis="error">
            <plotlib.Plot.Clip>
                <marker id={markerId} viewBox="0 0 10 10" refX="5" refY="5" style={{fill: 'var(--mantine-color-bright, black)', stroke: 'none'}}>
                    <circle cx={5} cy={5} r={4}/>
                </marker>
                <plotlib.PlotLine xs={xs} ys={ys} markerStart={markerRef} markerMid={markerRef} markerEnd={markerRef} label="Total loss"/>
            </plotlib.Plot.Clip>
            <plotlib.PlotLegend location='upper right'/>
        </plotlib.Plot>
    </plotlib.Figure>;
}