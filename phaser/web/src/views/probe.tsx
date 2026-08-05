import React, { useMemo } from 'react';
import { atom, Atom, useAtomValue } from 'jotai';
import { selectAtom } from 'jotai/utils';
import { interpolateMagma } from 'd3-scale-chromatic';

import * as plotlib from '@hexane/plotlib';
import type { ScaleSpec } from '@hexane/plotlib';
import type { NumericScale, ColorLike } from '@hexane/plotlib/scale';
import { Group, useComputedColorScheme } from '@mantine/core';

import { DecodedArray, abs2, splitAxis0, minmaxNaN, applyMagmaInto } from '../array';
import { ProbeMeta } from '../types';
import { usePubSubView } from '../pubsub';
import { isClose } from '../utils';
import { PENDING } from './objectPhaseSum';
import { ViewProps } from './types';
import { ViewGate, gateAtom } from './ViewStatus';

export function ProbeModesView(_props: ViewProps) {
    const metaTopic = usePubSubView<ProbeMeta>({view: 'probe_meta'});
    const dataTopic = usePubSubView<DecodedArray>({view: 'probes'});

    // rarely-changing meta vs per-tick data, each on its own topic. The equality check
    // matters: `probe_meta` republishes every tick with identical contents, and without it
    // `ProbeModesPlotSub` would re-render (and rebuild every mode's plot) on each one.
    const metaState = useMemo(() => selectAtom(
        metaTopic,
        (s) => s.status === 'ok' ? s.data : null,
        (a, b) => a === b || (a !== null && b !== null && probeMetaEqual(a, b)),
    ), [metaTopic]);
    const dataState = useMemo(() => selectAtom(
        dataTopic, (s) => s.status === 'ok' ? s.data : null,
    ), [dataTopic]);
    const gate = useMemo(() => gateAtom(metaTopic, dataTopic), [metaTopic, dataTopic]);

    return <ViewGate state={gate}>
        <ProbeModesPlot metaState={metaState} dataState={dataState}/>
    </ViewGate>;
}

function probeMetaEqual(a: ProbeMeta, b: ProbeMeta): boolean {
    return a.nprobes === b.nprobes &&
        isClose(a.sampling.shape, b.sampling.shape) && isClose(a.sampling.extent, b.sampling.extent)
            && isClose(a.sampling.sampling, b.sampling.sampling);
}

interface ProbePlotProps {
    metaState: Atom<ProbeMeta | null>
    dataState: Atom<DecodedArray | null>
}

function ProbeModesPlot({metaState, dataState}: ProbePlotProps) {
    const hasProbe = useAtomValue(metaState) !== null;
    if (!hasProbe) return <div></div>;
    return <ProbeModesPlotSub metaState={metaState} dataState={dataState} />;
}

// see the equivalent comment on `PhaseImageSub`: `metaState` only changes rarely,
// `dataState` carries the per-tick array data straight into `PlotImage` atoms.
function ProbeModesPlotSub({metaState, dataState}: ProbePlotProps) {
    const {sampling, nprobes: n_plots} = useAtomValue(metaState)!;
    const [ny, nx] = sampling.shape;

    const intensitiesAtom = useMemo(() => atom((get) => {
        const data = get(dataState);
        return data ? abs2(data) : null;
    }), [dataState]);

    // computed once per tick and shared by every probe mode's draw_fn below
    const intensitySlicesAtom = useMemo(() => atom((get) => {
        const intensities = get(intensitiesAtom);
        return intensities ? splitAxis0(intensities) : null;
    }), [intensitiesAtom]);

    // meta and data are separate topics now, so this can mount before any array has
    // arrived. Staying pending holds the colorbar rather than autoscaling it to nothing.
    const autoscaleFnAtom = useMemo(() => atom((get) => {
        const intensities = get(intensitiesAtom);
        if (!intensities) return () => PENDING;
        return () => {
            const [vmin, vmax] = minmaxNaN(intensities.data);
            return { vmin, vmax };
        };
    }), [intensitiesAtom]);

    const drawFnAtoms = useMemo(() => Array.from({length: n_plots}, (_, i) => atom((get) => {
        const slices = get(intensitySlicesAtom);
        // `slices[i]` is absent if `nprobes` leads the array by a tick
        if (!slices?.[i]) return () => PENDING;
        return (_ctx: CanvasRenderingContext2D, imageData: ImageData, scale: NumericScale<ColorLike>) => {
            const [vmin, vmax] = scale.domain as [number, number];
            applyMagmaInto(imageData, slices[i].data, vmin, vmax);
        };
    })), [intensitySlicesAtom, n_plots]);

    const scales: Map<string, ScaleSpec> = useMemo(() => new Map([
        ["x", { scale: plotlib.linear([0, nx * sampling.sampling[1]], undefined, { show: false }), size: '180px' }],
        ["y", { scale: plotlib.linear([0, ny * sampling.sampling[0]], undefined, { show: false }), size: '180px' }],
        ["intensity", { scale: plotlib.linear([0, 1], interpolateMagma, { label: "Probe Intensity" }) }],
    ] satisfies [string, ScaleSpec][]), [nx, ny, sampling]);

    return <Group justify="center"><plotlib.Figure scales={scales} width="80%" colorScheme={useComputedColorScheme('light')}>
        <plotlib.layout.CenteredX hug={plotlib.layout.Strength.weak}>
            <plotlib.layout.Decorated right={<plotlib.Colorbar scale="intensity" shrink={0.8}/>}>
                <plotlib.layout.FlexBox wrap columnGap="12pt" rowGap="12pt">
                    {drawFnAtoms.map((drawFnAtom, i) => (
                        <plotlib.Plot key={i} xaxis="x" yaxis="y" fixedAspect={true} suspense={true}>
                            <plotlib.Plot.Clip>
                                <plotlib.PlotImage draw_fn={drawFnAtom} autoscale_fn={autoscaleFnAtom} width={nx} height={ny} scale="intensity"/>
                            </plotlib.Plot.Clip>
                            {i === n_plots - 1 && <plotlib.Scalebar unitScale={1e-10}/>}
                        </plotlib.Plot>
                    ))}
                </plotlib.layout.FlexBox>
            </plotlib.layout.Decorated>
        </plotlib.layout.CenteredX>
    </plotlib.Figure></Group>;
}
