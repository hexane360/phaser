import React, { useMemo } from 'react';
import { atom, Atom, useAtomValue } from 'jotai';
import { selectAtom } from 'jotai/utils';
import { interpolateMagma, interpolateSinebow } from 'd3-scale-chromatic';

import * as plotlib from '@hexane/plotlib';
import type { ScaleSpec } from '@hexane/plotlib';
import type { NumericScale, ColorLike } from '@hexane/plotlib/scale';
import { Group, Select, Text, useComputedColorScheme } from '@mantine/core';

import {
    DecodedArray, abs2, amplitude as abs, angle, splitAxis0, minmaxNaN,
    applyMagmaInto, applySinebowInto, colorizeComplexInto,
} from '../array';
import { ProbeMeta } from '../types';
import { usePubSubView } from '../pubsub';
import { isClose } from '../utils';
import { ViewProps } from './types';
import { PENDING, ViewGate, gateAtom } from './ViewStatus';

// What each probe mode's complex array is displayed as. All four read the one `probes`
// topic, so switching never changes the subscription.
type ProbeMode = 'phase' | 'amplitude' | 'intensity' | 'phaseAmp';

const PROBE_MODES: Array<{value: ProbeMode, label: string}> = [
    {value: 'phaseAmp', label: 'Phase & amplitude'},
    {value: 'phase', label: 'Phase'},
    {value: 'amplitude', label: 'Amplitude'},
    {value: 'intensity', label: 'Intensity'},
];

const DEFAULT_MODE: ProbeMode = 'phaseAmp';

// a persisted layout can name a mode this build doesn't have
function parseMode(value: unknown): ProbeMode {
    return PROBE_MODES.some((m) => m.value === value) ? value as ProbeMode : DEFAULT_MODE;
}

export function ProbeModesView({params, setParams}: ViewProps) {
    const mode = parseMode(params.mode);

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

    return <>
        <ViewGate state={gate}>
            <ProbeModesPlot metaState={metaState} dataState={dataState} mode={mode}/>
        </ViewGate>
        <Group gap="sm" mb="sm" align="center" justify="center" wrap="nowrap">
            <Text size="xs">Display</Text>
            <Select
                w={180} size="xs" allowDeselect={false} data={PROBE_MODES} value={mode}
                onChange={(value) => setParams({...params, mode: parseMode(value)})}
            />
        </Group>
    </>;
}

function probeMetaEqual(a: ProbeMeta, b: ProbeMeta): boolean {
    return a.nprobes === b.nprobes &&
        isClose(a.sampling.shape, b.sampling.shape) && isClose(a.sampling.extent, b.sampling.extent)
            && isClose(a.sampling.sampling, b.sampling.sampling);
}

interface ProbePlotProps {
    metaState: Atom<ProbeMeta | null>
    dataState: Atom<DecodedArray | null>
    mode: ProbeMode
}

function ProbeModesPlot({metaState, dataState, mode}: ProbePlotProps) {
    const hasProbe = useAtomValue(metaState) !== null;
    if (!hasProbe) return <div></div>;
    return <ProbeModesPlotSub metaState={metaState} dataState={dataState} mode={mode} />;
}

// see the equivalent comment on `PhaseImageSub`: `metaState` only changes rarely,
// `dataState` carries the per-tick array data straight into `PlotImage` atoms.
function ProbeModesPlotSub({metaState, dataState, mode}: ProbePlotProps) {
    const {sampling, nprobes} = useAtomValue(metaState)!;
    const [ny, nx] = sampling.shape;

    // the raw complex modes, one per plot. `phaseAmp` draws straight from these; the
    // scalar modes map each through `transform` below.
    const modesAtom = useMemo(() => atom((get) => {
        const data = get(dataState);
        return data ? splitAxis0(data) : null;
    }), [dataState]);

    // shared by every mode's autoscale, so all plots agree on one color scale
    const intensitiesAtom = useMemo(() => atom((get) => {
        const data = get(dataState);
        return data ? abs2(data) : null;
    }), [dataState]);

    const transform = mode === 'phase' ? angle : mode === 'intensity' ? abs2 : abs;

    // computed once per tick and shared by every probe mode's draw_fn below
    const imagesAtom = useMemo(() => atom((get) => {
        const slices = get(modesAtom);
        return slices ? slices.map(transform) : null;
    }), [modesAtom, transform]);

    // meta and data are separate topics now, so this can mount before any array has
    // arrived. Staying pending holds the colorbar rather than autoscaling it to nothing.
    const autoscaleFnAtom = useMemo(() => atom((get) => {
        const intensities = get(intensitiesAtom);
        if (!intensities) return () => PENDING;
        return () => {
            // fixed domain: the cyclic map only lines up with the data at exactly +-pi
            if (mode === 'phase') return { vmin: -Math.PI, vmax: Math.PI };
            const [vmin, vmax] = minmaxNaN(intensities.data);
            if (mode === 'intensity') return [vmin, vmax];
            return { vmin: Math.sqrt(Math.max(vmin, 0)), vmax: Math.sqrt(vmax) };
        };
    }), [intensitiesAtom, mode]);

    const drawFnAtoms = useMemo(() => Array.from({length: nprobes}, (_, i) => atom((get) => {
        const slices = get(mode === 'phaseAmp' ? modesAtom : imagesAtom);
        // `slices[i]` is absent if `nprobes` leads the array by a tick
        if (!slices?.[i]) return () => PENDING;
        return (_ctx: CanvasRenderingContext2D, imageData: ImageData, scale: NumericScale<ColorLike>) => {
            const [vmin, vmax] = scale.domain as [number, number];
            if (mode === 'phaseAmp') colorizeComplexInto(imageData, slices[i], vmax);
            else if (mode === 'phase') applySinebowInto(imageData, slices[i].data, vmin, vmax);
            else applyMagmaInto(imageData, slices[i].data, vmin, vmax);
        };
    })), [modesAtom, imagesAtom, nprobes, mode]);

    const valueScale = useMemo(() => {
        switch (mode) {
            case 'phase': return plotlib.linear([-Math.PI, Math.PI], interpolateSinebow, { label: "Probe Phase [rad]", tickFormat: ".2f" });
            case 'amplitude': return plotlib.linear([0, 1], interpolateMagma, { label: "Probe Amplitude", tickFormat: ".2f" });
            case 'intensity': return plotlib.linear([0, 1], interpolateMagma, { label: "Probe Intensity", tickFormat: ".2f" });
            // a 1D bar can't show hue x value, so nothing renders this one
            case 'phaseAmp': return plotlib.linear([0, 1], interpolateMagma, { show: false });
        }
    }, [mode]);

    const scales: Map<string, ScaleSpec> = useMemo(() => new Map([
        ["x", { scale: plotlib.linear([0, nx * sampling.sampling[1]], undefined, { show: false }), size: '180px' }],
        ["y", { scale: plotlib.linear([0, ny * sampling.sampling[0]], undefined, { show: false }), size: '180px' }],
        ["intensity", { scale: valueScale }],
    ] satisfies [string, ScaleSpec][]), [nx, ny, sampling, valueScale]);

    return <Group justify="center"><plotlib.Figure scales={scales} width="80%" colorScheme={useComputedColorScheme('light')}>
        <plotlib.layout.CenteredX hug={plotlib.layout.Strength.weak}>
            <plotlib.layout.Decorated right={mode === 'phaseAmp' ? [] : [<plotlib.Colorbar key="cbar" scale="intensity" shrink={0.8}/>]}>
                <plotlib.layout.FlexBox wrap columnGap="12pt" rowGap="12pt">
                    {drawFnAtoms.map((drawFnAtom, i) => (
                        <plotlib.Plot key={i} xaxis="x" yaxis="y" fixedAspect={true} suspense={true}>
                            <plotlib.Plot.Clip>
                                <plotlib.PlotImage draw_fn={drawFnAtom} autoscale_fn={autoscaleFnAtom} width={nx} height={ny} scale="intensity"/>
                            </plotlib.Plot.Clip>
                            {i === nprobes - 1 && <plotlib.Scalebar unitScale={1e-10}/>}
                        </plotlib.Plot>
                    ))}
                </plotlib.layout.FlexBox>
            </plotlib.layout.Decorated>
        </plotlib.layout.CenteredX>
    </plotlib.Figure></Group>;
}
