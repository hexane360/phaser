import React, { useMemo } from 'react';
import { PrimitiveAtom, useAtomValue } from 'jotai';

import * as plotlib from '@hexane/plotlib';
import type { ScaleSpec } from '@hexane/plotlib';
import { Group, useComputedColorScheme } from '@mantine/core';

import { ProgressData } from '../types';
import { usePubSubView, ViewState } from '../pubsub';
import { ViewProps } from './types';
import { ViewStatus } from './ViewStatus';

export function ProgressView(_props: ViewProps) {
    const state = usePubSubView<Record<string, ProgressData>>({view: 'progress'});
    return <ProgressPlot state={state}/>;
}

function ProgressPlot({state}: {state: PrimitiveAtom<ViewState<Record<string, ProgressData>>>}) {
    const progress = useAtomValue(state);
    if (progress.status !== 'ok') return <ViewStatus state={progress}/>;
    if (!progress.data.total_loss) return <ViewStatus state={progress} pending="No loss recorded yet"/>;

    return <ProgressPlotSub progress={progress.data.total_loss} />;
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
            size: '300px', zoomExtent: [1.0, Infinity],
        }],
    ] satisfies [string, ScaleSpec][]), [x_max, y_max, y_min]);

    return <Group justify="center"><plotlib.Figure scales={scales} colorScheme={useComputedColorScheme('light')}>
        <plotlib.Plot xaxis="iter" yaxis="error">
            <plotlib.Plot.Clip>
                <marker id={markerId} viewBox="0 0 10 10" refX="5" refY="5" style={{fill: 'var(--mantine-color-bright, black)', stroke: 'none'}}>
                    <circle cx={5} cy={5} r={4}/>
                </marker>
                <plotlib.PlotLine xs={xs} ys={ys} markerStart={markerRef} markerMid={markerRef} markerEnd={markerRef} label="Total loss"/>
            </plotlib.Plot.Clip>
            <plotlib.PlotLegend location='upper right'/>
        </plotlib.Plot>
    </plotlib.Figure></Group>;
}
