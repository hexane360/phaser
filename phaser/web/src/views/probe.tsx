import React, { useMemo } from 'react';
import { atom, Atom, useAtomValue } from 'jotai';
import { selectAtom } from 'jotai/utils';
import { interpolateMagma } from 'd3-scale-chromatic';

import * as plotlib from '@hexane/plotlib';
import type { ScaleSpec } from '@hexane/plotlib';
import type { NumericScale, ColorLike } from '@hexane/plotlib/scale';
import { useComputedColorScheme } from '@mantine/core';

import { DecodedArray, abs2, splitAxis0, minmaxNaN, applyMagmaInto } from '../array';
import { ProbeData, ProbeMeta } from '../types';
import { usePubSubView, ViewState } from '../pubsub';
import { isClose } from '../utils';
import { ViewProps } from './types';
import { ViewGate, gateAtom } from './ViewStatus';

export function ProbeModesView(_props: ViewProps) {
    const state = usePubSubView<ProbeData>({view: 'probes'});

    // rarely-changing meta vs per-tick data, as in `objectPhase`'s `usePhaseAtoms`
    const metaState = useMemo(() => selectAtom(
        state,
        (s): ProbeMeta | null => s.status === 'ok'
            ? { sampling: s.data.sampling, nprobes: s.data.data.shape[0] } : null,
        (a, b) => a === b || (a !== null && b !== null && probeMetaEqual(a, b)),
    ), [state]);
    const dataState = useMemo(() => selectAtom(
        state, (s) => s.status === 'ok' ? s.data.data : null,
    ), [state]);
    const gate = useMemo(() => gateAtom(state), [state]);

    return <ViewGate state={gate}>
        <ProbeModesPlot metaState={metaState} dataState={dataState}/>
    </ViewGate>;
}

function probeMetaEqual(a: ProbeMeta | null, b: ProbeMeta): boolean {
    return a !== null && a.nprobes === b.nprobes &&
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

    const autoscaleFnAtom = useMemo(() => atom((get) => {
        const intensities = get(intensitiesAtom);
        return () => {
            if (!intensities) return { vmin: null, vmax: null };
            const [vmin, vmax] = minmaxNaN(intensities.data);
            return { vmin, vmax };
        };
    }), [intensitiesAtom]);

    const drawFnAtoms = useMemo(() => Array.from({length: n_plots}, (_, i) => atom((get) => {
        const slices = get(intensitySlicesAtom);
        return (_ctx: CanvasRenderingContext2D, imageData: ImageData, scale: NumericScale<ColorLike>) => {
            if (!slices) return;
            const [vmin, vmax] = scale.domain as [number, number];
            applyMagmaInto(imageData, slices[i].data, vmin, vmax);
        };
    })), [intensitySlicesAtom, n_plots]);

    const scales: Map<string, ScaleSpec> = useMemo(() => new Map([
        ["x", { scale: plotlib.linear([0, nx * sampling.sampling[1]], undefined, { show: false }), size: '180px' }],
        ["y", { scale: plotlib.linear([0, ny * sampling.sampling[0]], undefined, { show: false }), size: '180px' }],
        ["intensity", { scale: plotlib.linear([0, 1], interpolateMagma, { label: "Probe Intensity" }) }],
    ] satisfies [string, ScaleSpec][]), [nx, ny]);

    return <plotlib.Figure scales={scales} width="100%" colorScheme={useComputedColorScheme('light')}>
        <plotlib.layout.CenteredX hug={plotlib.layout.Strength.medium}>
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
