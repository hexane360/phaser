import React, { useMemo } from 'react';
import { useAtomValue, PrimitiveAtom } from 'jotai';

import { np } from './wasm-array';
import { ProbeData, ObjectData, ProgressData } from './types';
import { PlotScale, LogPlotScale } from './plotting/scale';
import { Figure, PlotGrid, Plot, AxisSpec, ColorScale, PlotImage, PlotLine, makeId } from './plotting/plot';
import { Colorbar } from './plotting/colorbar';
import { HBox } from './components';
import Scalebar from './plotting/scalebar';


interface ObjectPlotProps {
    state: PrimitiveAtom<ObjectData | null>
}

export function ObjectPlot(props: ObjectPlotProps) {
    let object = useAtomValue(props.state);
    if (!object || !np) return <div></div>;
    return <ObjectPlotSub object={object} />;
}

function ObjectPlotSub({object}: {object: ObjectData}) {
    let object_data = object.data;

    let phase = np!.angle(object_data);
    while (phase.shape.length > 2) {
        phase = np!.nansum(phase, [0]);
    }

    const [ny, nx] = phase.shape.values();

    let phase_cropped;
    if (object.sampling.region_min && object.sampling.region_max) {
        // y position = y index * sampling + corner
        // y index = (y position - corner) / sampling
        const [y_min, y_max, x_min, x_max] = [
            Math.ceil((object.sampling.region_min[0] - object.sampling.corner[0]) / object.sampling.sampling[0]),
            Math.floor((object.sampling.region_max[0] - object.sampling.corner[0]) / object.sampling.sampling[0]),
            Math.ceil((object.sampling.region_min[1] - object.sampling.corner[1]) / object.sampling.sampling[1]),
            Math.floor((object.sampling.region_max[1] - object.sampling.corner[1]) / object.sampling.sampling[1]),
        ];

        phase_cropped = phase.slice(new np!.Slice(y_min, y_max), new np!.Slice(x_min, x_max));
    } else {
        phase_cropped = phase;
    }

    const [vmin, vmax]: [number, number] = [
        np!.nanmin(phase_cropped).toNestedArray() as number,
        np!.nanmax(phase_cropped).toNestedArray() as number
    ];

    const aspect = nx / ny;
    const size = 500.0;
    // keep area constant
    const [x_size, y_size] = [Math.ceil(size * Math.sqrt(aspect)), Math.ceil(size / Math.sqrt(aspect))];

    const xmin = object.sampling.corner[1],
          xmax = object.sampling.corner[1] + nx * object.sampling.sampling[1];

    const ymin = object.sampling.corner[0],
          ymax = object.sampling.corner[0] + nx * object.sampling.sampling[0];

    const axes: Map<string, AxisSpec> = useMemo(() => new Map([
        ["x", {
            scale: new PlotScale([xmin, xmax], [0.0, x_size]),
            label: "X",
            show: false,
        }],
        ["y", {
            scale: new PlotScale([ymin, ymax], [0.0, y_size]),
            label: "Y",
            show: false,
        }],
    ]), [xmin, xmax, ymin, ymax, x_size, y_size]);

    const scales: Map<string, ColorScale> = new Map([
        ["phase", {
            cmap: 'magma',
            range: [vmin, vmax],
            label: "Object Phase",
        }]
    ]);

    return <Figure axes={axes} scales={scales}>
        <HBox>
            <Plot xaxis="x" yaxis="y"><PlotImage data={phase} scale="phase"/><Scalebar unitScale={1e-10}/></Plot>
            <Colorbar scale="phase"/>
        </HBox>
    </Figure>;
}

interface ProbePlotProps {
    state: PrimitiveAtom<ProbeData | null>
}

export function ProbePlot(props: ProbePlotProps) {
    const probes = useAtomValue(props.state);
    if (!probes || !np) return <div></div>;
    return <ProbePlotSub probes={probes} />
}

function ProbePlotSub({probes}: {probes: ProbeData}) {
    let probes_data = probes.data;

    const [nprobes, ny, nx] = probes_data.shape.values();

    const intensities = np!.abs2(probes_data);
    const [vmin, vmax]: [number, number] = [
        np!.nanmin(intensities).toNestedArray() as number,
        np!.nanmax(intensities).toNestedArray() as number
    ];

    const axes: Map<string, AxisSpec> = useMemo(() => new Map([
        ["x", {
            scale: new PlotScale([0, nx], [0.0, 180.0]),
            label: "X",
            show: false,
        }],
        ["y", {
            scale: new PlotScale([0, ny], [0.0, 180.0]),
            label: "Y",
            show: false,
        }],
    ]), [nx, ny]);

    const scales: Map<string, ColorScale> = new Map([
        ["intensity", {
            cmap: 'magma',
            range: [vmin, vmax],
            label: "Probe Intensity",
        }]
    ]);

    const n_plots = intensities.shape[0];

    const plots = np!.split(intensities).map((intensity, i) => {
        const scalebar = i + 1 == n_plots ? <Scalebar unitScale={1e-10}/> : null;
        return <Plot key={i}><PlotImage data={intensity} scale="intensity"/>{scalebar}</Plot>;
    });

    return <Figure axes={axes} scales={scales}>
        <HBox>
            <PlotGrid ncols={nprobes} nrows={1} xaxes="x" yaxes="y">
                {plots}
            </PlotGrid>
            <Colorbar scale="intensity" length={100}/>
        </HBox>
    </Figure>;
}

interface ProgressPlotProps {
    state: PrimitiveAtom<ProgressData | null>
}

export function ProgressPlot(props: ProgressPlotProps) {
    const progress = useAtomValue(props.state);
    if (!progress || !np) return <div></div>;

    return <ProgressPlotSub progress={progress} />;
}

function ProgressPlotSub({progress}: {progress: ProgressData}) {
    const markerId = React.useMemo(() => makeId("marker"), []);
    const markerRef = `url(#${markerId})`;

    const xs = progress.iters.toNestedArray() as Array<number>;
    const ys = progress.detector_errors.toNestedArray() as Array<number>;

    const x_max = Math.max(10, ...xs.filter(isFinite));
    const ys_filt = ys.filter(isFinite);

    let y_min, y_max;
    if (ys_filt.length) {
        [y_min, y_max] = [Math.min(...ys_filt), Math.max(...ys_filt)];
    } else {
        [y_min, y_max] = [1.0, 1.0e5]
    }

    const axes: Map<string, AxisSpec> = useMemo(() => new Map([
        ["iter", {
            scale: new PlotScale([0, x_max], [0.0, 500.0]),
            label: "Iteration",
            show: true,
        }],
        ["error", {
            scale: (new LogPlotScale([y_max, y_min], [0.0, 300.0])).pad_frac(0.1),
            label: "Error",
            show: true,
        }],
    ]), [x_max, y_max, y_min]);

    return <Figure axes={axes}>
        <Plot xaxis="iter" yaxis="error">
            <marker id={markerId} viewBox="0 0 10 10" refX="5" refY="5" className="plot-marker">
                <circle cx={5} cy={5} r={4}/>
            </marker>
            <PlotLine xs={xs} ys={ys} markerStart={markerRef} markerMid={markerRef} markerEnd={markerRef}/>
        </Plot>
    </Figure>;
}