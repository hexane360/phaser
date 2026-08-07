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
import { ProbeMeta, Sampling } from '../types';
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

// Real space, or its `fft2shift(fft2(...))` on the server (`probes_recip`,
// `probe_sum_recip`). The two differ only in the topic they read and the scale they're
// drawn against: a length in real space, a scattering angle in reciprocal space.
type ProbeSpace = 'real' | 'recip';

// a persisted layout can name a mode this build doesn't have
function parseMode(value: unknown): ProbeMode {
    return PROBE_MODES.some((m) => m.value === value) ? value as ProbeMode : DEFAULT_MODE;
}

// `probe_meta` plus one bulk-array topic, the pair every probe view subscribes to.
// The meta equality check matters: `probe_meta` republishes every tick with identical
// contents, and without it the `*PlotSub` components would re-render (and rebuild every
// plot) on each one.
function useProbeTopics(view: string) {
    const metaTopic = usePubSubView<ProbeMeta>({view: 'probe_meta'});
    const dataTopic = usePubSubView<DecodedArray>({view});

    const metaState = useMemo(() => selectAtom(
        metaTopic,
        (s) => s.status === 'ok' ? s.data : null,
        (a, b) => a === b || (a !== null && b !== null && probeMetaEqual(a, b)),
    ), [metaTopic]);
    const dataState = useMemo(() => selectAtom(
        dataTopic, (s) => s.status === 'ok' ? s.data : null,
    ), [dataTopic]);
    const gate = useMemo(() => gateAtom(metaTopic, dataTopic), [metaTopic, dataTopic]);

    return {metaState, dataState, gate};
}

// Real space spans the probe extent from the origin; reciprocal space spans
// +-lambda/(2*sampling), the Nyquist angle of the real-space grid, in mrad. Neither is
// labelled -- `ProbeScalebar` carries the scale instead.
function useSpatialScales(sampling: Sampling, wavelength: number, space: ProbeSpace) {
    return useMemo(() => {
        const [ny, nx] = sampling.shape;
        if (space === 'real') return [
            plotlib.linear([0, nx * sampling.sampling[1]], undefined, { show: false }),
            plotlib.linear([0, ny * sampling.sampling[0]], undefined, { show: false }),
        ];
        const kx = wavelength / (2 * sampling.sampling[1]) * 1e3;
        const ky = wavelength / (2 * sampling.sampling[0]) * 1e3;
        return [
            plotlib.linear([-kx, kx], undefined, { label: "kx [mrad]", show: false }),
            plotlib.linear([-ky, ky], undefined, { label: "ky [mrad]", show: false }),
        ];
    }, [space, sampling, wavelength]);
}

const ProbeScalebar = ({space}: {space: ProbeSpace}) =>
    <plotlib.Scalebar unitScale={space === 'real' ? 1e-10 : 1e-3} unit={space === 'real' ? 'm' : 'rad'}/>;

export const ProbeModesView = (props: ViewProps) => <ProbeModes {...props} space="real"/>;
export const ProbeModesRecipView = (props: ViewProps) => <ProbeModes {...props} space="recip"/>;

function ProbeModes({params, setParams, space}: ViewProps & {space: ProbeSpace}) {
    const mode = parseMode(params.mode);
    const {metaState, dataState, gate} = useProbeTopics(space === 'real' ? 'probes' : 'probes_recip');

    return <>
        <ViewGate state={gate}>
            <ProbeModesPlot metaState={metaState} dataState={dataState} mode={mode} space={space}/>
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

// The modes summed in intensity, on the `probe_sum`/`probe_sum_recip` topics. Incoherent
// modes add in intensity and in nothing else, so there's no quantity to choose here -- no
// `Select`, and the image arrives real-valued and already reduced.
export const ProbeSumView = (props: ViewProps) => <ProbeSum {...props} space="real"/>;
export const ProbeSumRecipView = (props: ViewProps) => <ProbeSum {...props} space="recip"/>;

function ProbeSum({space}: ViewProps & {space: ProbeSpace}) {
    const {metaState, dataState, gate} = useProbeTopics(space === 'real' ? 'probe_sum' : 'probe_sum_recip');

    return <ViewGate state={gate}>
        <ProbeSumPlot metaState={metaState} dataState={dataState} space={space}/>
    </ViewGate>;
}

interface ProbeSumPlotProps {
    metaState: Atom<ProbeMeta | null>
    dataState: Atom<DecodedArray | null>
    space: ProbeSpace
}

function ProbeSumPlot(props: ProbeSumPlotProps) {
    const hasProbe = useAtomValue(props.metaState) !== null;
    if (!hasProbe) return <div></div>;
    return <ProbeSumPlotSub {...props} />;
}

// as `ProbeModesPlotSub`: `metaState` is the rarely-changing half, and the per-tick
// array reaches `PlotImage` through atoms without re-rendering the figure.
function ProbeSumPlotSub({metaState, dataState, space}: ProbeSumPlotProps) {
    const {sampling, wavelength} = useAtomValue(metaState)!;
    const [ny, nx] = sampling.shape;
    const [xScale, yScale] = useSpatialScales(sampling, wavelength, space);

    const drawFnAtom = useMemo(() => atom((get) => {
        const image = get(dataState);
        if (!image) return () => PENDING;
        return (_ctx: CanvasRenderingContext2D, imageData: ImageData, scale: NumericScale<ColorLike>) => {
            const [vmin, vmax] = scale.domain as [number, number];
            applyMagmaInto(imageData, image.data, vmin, vmax);
        };
    }), [dataState]);

    const autoscaleFnAtom = useMemo(() => atom((get) => {
        const image = get(dataState);
        if (!image) return () => PENDING;
        return () => minmaxNaN(image.data);
    }), [dataState]);

    const scales: Map<string, ScaleSpec> = useMemo(() => new Map([
        ["x", { scale: xScale, size: '60%' }],
        ["y", { scale: yScale, size: '60%' }],
        ["intensity", { scale: plotlib.linear([0, 1], interpolateMagma, { label: "Probe Intensity", tickFormat: ".1f" }) }],
    ] satisfies [string, ScaleSpec][]), [xScale, yScale]);

    return <Group justify="center"><plotlib.Figure scales={scales} width="80%" colorScheme={useComputedColorScheme('light')}>
        <plotlib.layout.CenteredX hug={plotlib.layout.Strength.weak}>
            <plotlib.Plot xaxis="x" yaxis="y" colorbar="intensity" fixedAspect={true} suspense={true}>
                <plotlib.Plot.Clip>
                    <plotlib.PlotImage draw_fn={drawFnAtom} autoscale_fn={autoscaleFnAtom} width={nx} height={ny} scale="intensity"/>
                </plotlib.Plot.Clip>
                <ProbeScalebar space={space}/>
            </plotlib.Plot>
        </plotlib.layout.CenteredX>
    </plotlib.Figure></Group>;
}

function probeMetaEqual(a: ProbeMeta, b: ProbeMeta): boolean {
    return a.nprobes === b.nprobes && isClose(a.wavelength, b.wavelength) &&
        isClose(a.sampling.shape, b.sampling.shape) && isClose(a.sampling.extent, b.sampling.extent)
            && isClose(a.sampling.sampling, b.sampling.sampling);
}

interface ProbePlotProps {
    metaState: Atom<ProbeMeta | null>
    dataState: Atom<DecodedArray | null>
    mode: ProbeMode
    space: ProbeSpace
}

function ProbeModesPlot(props: ProbePlotProps) {
    const hasProbe = useAtomValue(props.metaState) !== null;
    if (!hasProbe) return <div></div>;
    return <ProbeModesPlotSub {...props} />;
}

// see the equivalent comment on `PhaseImageSub`: `metaState` only changes rarely,
// `dataState` carries the per-tick array data straight into `PlotImage` atoms.
function ProbeModesPlotSub({metaState, dataState, mode, space}: ProbePlotProps) {
    const {sampling, nprobes, wavelength} = useAtomValue(metaState)!;
    const [ny, nx] = sampling.shape;
    const [xScale, yScale] = useSpatialScales(sampling, wavelength, space);

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
        ["x", { scale: xScale, size: '180px' }],
        ["y", { scale: yScale, size: '180px' }],
        ["intensity", { scale: valueScale }],
    ] satisfies [string, ScaleSpec][]), [xScale, yScale, valueScale]);

    return <Group justify="center"><plotlib.Figure scales={scales} width="80%" colorScheme={useComputedColorScheme('light')}>
        <plotlib.layout.CenteredX hug={plotlib.layout.Strength.weak}>
            <plotlib.layout.Decorated right={mode === 'phaseAmp' ? [] : [<plotlib.Colorbar key="cbar" scale="intensity" shrink={0.8}/>]}>
                <plotlib.layout.FlexBox wrap columnGap="12pt" rowGap="12pt">
                    {drawFnAtoms.map((drawFnAtom, i) => (
                        <plotlib.Plot key={i} xaxis="x" yaxis="y" fixedAspect={true} suspense={true}>
                            <plotlib.Plot.Clip>
                                <plotlib.PlotImage draw_fn={drawFnAtom} autoscale_fn={autoscaleFnAtom} width={nx} height={ny} scale="intensity"/>
                            </plotlib.Plot.Clip>
                            {(i === nprobes - 1) && <ProbeScalebar space={space}/>}
                        </plotlib.Plot>
                    ))}
                </plotlib.layout.FlexBox>
            </plotlib.layout.Decorated>
        </plotlib.layout.CenteredX>
    </plotlib.Figure></Group>;
}
