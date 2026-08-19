import React, { useMemo } from 'react';
import { useAtomValue } from 'jotai';

import * as plotlib from '@hexane/plotlib';
import type { ScaleSpec } from '@hexane/plotlib';
import { Group, useComputedColorScheme } from '@mantine/core';

import { DecodedArray } from '../array';
import { usePubSubView } from '../pubsub';
import { ViewProps } from './types';
import { ViewStatus } from './ViewStatus';

// The scan positions, in the order they're scanned -- a `PlotLine` rather than a scatter,
// so the scan path itself is visible.
export function PositionsView(_props: ViewProps) {
    const state = useAtomValue(usePubSubView<DecodedArray>({view: 'positions'}));
    if (state.status !== 'ok') return <ViewStatus state={state}/>;
    return <PositionsPlot positions={state.data}/>;
}

interface Columns {
    xs: Array<number>
    ys: Array<number>
    xlim: [number, number]
    ylim: [number, number]
}

// (..., 2) in any shape -> the flat (N, 2) the plot draws, split into columns, with each
// one's bounds taken in the same pass. The leading axes are the scan grid, which the scan
// order already walks in row-major order, so flattening them is exactly the path.
function toColumns(positions: DecodedArray): Columns {
    if (positions.complex || positions.shape[positions.shape.length - 1] !== 2) {
        throw new Error(`Expected real positions of shape (..., 2), got ${positions.complex ? 'complex ' : ''}(${positions.shape})`);
    }
    const n = positions.data.length / 2;
    const xs = new Array<number>(n), ys = new Array<number>(n);
    const xlim: [number, number] = [Infinity, -Infinity];
    const ylim: [number, number] = [Infinity, -Infinity];

    for (let i = 0; i < n; i++) {
        const y = positions.data[2 * i], x = positions.data[2 * i + 1];
        ys[i] = y; xs[i] = x;
        if (y < ylim[0]) ylim[0] = y;
        if (y > ylim[1]) ylim[1] = y;
        if (x < xlim[0]) xlim[0] = x;
        if (x > xlim[1]) xlim[1] = x;
    }
    return {xs, ys, xlim, ylim};
}

// pads a [min, max] domain by `frac` of its span on each side
function padDomain([min, max]: [number, number], frac: number): [number, number] {
    const pad = Math.max((max - min) * frac, 1e-6);
    return [min - pad, max + pad];
}

// Positions move under position correction, so the domain follows them: every tick
// re-renders the figure, as in `ProgressPlotSub` and unlike the image views, which have a
// rarely-changing `*_meta` topic to split out.
function PositionsPlot({positions}: {positions: DecodedArray}) {
    const {xs, ys, xlim, ylim} = useMemo(() => toColumns(positions), [positions]);

    const scales: Map<string, ScaleSpec> = useMemo(() => {
        const [ymin, ymax] = padDomain(ylim, 0.05);
        return new Map([
            ["x", { scale: plotlib.linear(padDomain(xlim, 0.05), undefined, { label: "x [Å]" }), size: '70%' }],
            // y increases downward, matching the object and probe images
            ["y", { scale: plotlib.linear([ymin, ymax], undefined, { label: "y [Å]" }), size: '70%' }],
        ] satisfies [string, ScaleSpec][]);
    }, [xlim, ylim]);

    const markerId = React.useId();
    const markerRef = `url(#${markerId})`;

    return <Group justify="center"><plotlib.Figure width="100%" scales={scales} colorScheme={useComputedColorScheme('light')}>
        <plotlib.layout.CenteredX hug={plotlib.layout.Strength.weak}>
            <plotlib.Plot xaxis="x" yaxis="y" fixedAspect={true}>
                <marker id={markerId} viewBox="0 0 6 6" refX="3" refY="3" style={{fill: 'var(--mantine-color-bright, black)', stroke: 'none'}}>
                    <circle cx={3} cy={3} r={2}/>
                </marker>
                <plotlib.Plot.Clip>
                    <plotlib.PlotLine style={{stroke: 'none'}} xs={xs} ys={ys} markerStart={markerRef} markerMid={markerRef} markerEnd={markerRef} label="Scan positions"/>
                </plotlib.Plot.Clip>
            </plotlib.Plot>
        </plotlib.layout.CenteredX>
    </plotlib.Figure></Group>;
}
