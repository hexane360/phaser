// Minimal typed-array numeric helpers, replacing `wasm-array`.
//
// This deliberately isn't a general ndarray library -- it implements exactly
// the operations the dashboard needs: decoding numpy arrays sent from the
// server (always real float32/float64 or complex64/complex128, always
// C-contiguous -- see `phaser/web/util.py`), converting complex probe/object
// data to real-valued displays, and rendering them to canvas via a magma
// colormap.

import { rgb } from 'd3-color';
import { interpolateMagma, interpolateSinebow } from 'd3-scale-chromatic';

export type RealTypedArray = Float32Array | Float64Array;

export interface DecodedArray {
    // interleaved re,im,re,im,... (numpy's native complex memory layout) when `complex` is true
    data: RealTypedArray;
    shape: ReadonlyArray<number>;
    complex: boolean;
}

export interface ArrayInterchange {
    data: string; // base64
    typestr: string; // '<f4' | '<f8' | '<c8' | '<c16'
    shape: ReadonlyArray<number>;
    strides: unknown;
    version: number;
}

function base64ToBytes(b64: string): Uint8Array {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
}

function zerosLike(data: RealTypedArray, n: number): RealTypedArray {
    return data instanceof Float64Array ? new Float64Array(n) : new Float32Array(n);
}

// Only the dtypes actually sent by `phaser/web/util.py` are supported. The
// server always runs little-endian, so only `<` typestrings are handled --
// a `>` (big-endian) typestr throws rather than silently mis-rendering.
export function decodeInterchange(obj: ArrayInterchange): DecodedArray {
    const bytes = base64ToBytes(obj.data);
    switch (obj.typestr) {
        case '<f4': return { data: new Float32Array(bytes.buffer), shape: obj.shape, complex: false };
        case '<f8': return { data: new Float64Array(bytes.buffer), shape: obj.shape, complex: false };
        case '<c8': return { data: new Float32Array(bytes.buffer), shape: obj.shape, complex: true };
        case '<c16': return { data: new Float64Array(bytes.buffer), shape: obj.shape, complex: true };
        default:
            throw new Error(`Unsupported array typestr '${obj.typestr}' (only <f4, <f8, <c8, <c16 are supported)`);
    }
}

// Recursively walks a decoded-JSON value, replacing any `{_ty: "numpy", ...}` interchange
// object with its `decodeInterchange` result. Used to decode pub/sub `TopicUpdate.data`
// payloads (and, previously, whole `job_update` state blobs) into plain JS objects with
// typed-array leaves.
export function decodeState(state: any): any {
    if (typeof state !== 'object' || state === null) {
        return state;
    }

    if (state instanceof Array) {
        return state.map(decodeState);
    }

    if (state._ty !== undefined) {
        if (state._ty === 'numpy') {
            return decodeInterchange(state as ArrayInterchange);
        }
        throw new Error(`Unknown custom type '${state._ty}'`);
    }

    let out: Record<string, any> = {};
    for (const [k, v] of Object.entries(state)) {
        out[k] = (typeof v === 'object' && v !== null) ? decodeState(v) : v;
    }
    return out;
}

// Fused `angle` + `nansum` over every leading axis, collapsing an (..., y, x)
// complex array down to a single real (y, x) phase image in one pass --
// avoids materializing a full-size intermediate array for multi-slice objects.
export function objectPhaseProjected(arr: DecodedArray): DecodedArray {
    if (!arr.complex) throw new Error("'objectPhaseProjected' requires complex input");
    const [ny, nx] = arr.shape.slice(-2);
    const plane = ny * nx;
    const nLeading = arr.shape.slice(0, -2).reduce((a, b) => a * b, 1);

    const out = zerosLike(arr.data, plane);
    const buf = arr.data;

    for (let l = 0; l < nLeading; l++) {
        const base = l * plane;
        for (let p = 0; p < plane; p++) {
            const idx = base + p;
            const re = buf[2 * idx], im = buf[2 * idx + 1];
            if (!Number.isNaN(re)) out[p] += Math.atan2(im, re);
        }
    }
    return { data: out, shape: [ny, nx], complex: false };
}

// |x|, elementwise. The conjugate of `objectPhaseProjected` for a single object slice --
// no projection, since amplitudes compose multiplicatively and the server does that
// reduction (`project_amp_mean`) for the whole-object view.
export function amplitude(arr: DecodedArray): DecodedArray {
    if (!arr.complex) throw new Error("'amplitude' requires complex input");
    const n = arr.data.length / 2;
    const out = zerosLike(arr.data, n);
    const buf = arr.data;
    for (let i = 0; i < n; i++) {
        out[i] = Math.hypot(buf[2 * i], buf[2 * i + 1]);
    }
    return { data: out, shape: arr.shape, complex: false };
}

// arg(x), elementwise, in [-pi, pi].
export function angle(arr: DecodedArray): DecodedArray {
    if (!arr.complex) throw new Error("'angle' requires complex input");
    const n = arr.data.length / 2;
    const out = zerosLike(arr.data, n);
    const buf = arr.data;
    for (let i = 0; i < n; i++) {
        out[i] = Math.atan2(buf[2 * i + 1], buf[2 * i]);
    }
    return { data: out, shape: arr.shape, complex: false };
}

// |x|^2, elementwise, over an entire complex array (e.g. the full probe mode stack).
export function abs2(arr: DecodedArray): DecodedArray {
    if (!arr.complex) throw new Error("'abs2' requires complex input");
    const n = arr.data.length / 2;
    const out = zerosLike(arr.data, n);
    const buf = arr.data;
    for (let i = 0; i < n; i++) {
        const re = buf[2 * i], im = buf[2 * i + 1];
        out[i] = re * re + im * im;
    }
    return { data: out, shape: arr.shape, complex: false };
}

// Zero-copy split along axis 0 (e.g. probe mode stack -> one 2D array per mode).
export function splitAxis0(arr: DecodedArray): DecodedArray[] {
    const [n, ...rest] = arr.shape;
    // a complex plane occupies two buffer elements per array element
    const planeSize = rest.reduce((a, b) => a * b, 1) * (arr.complex ? 2 : 1);
    const out: DecodedArray[] = [];
    for (let i = 0; i < n; i++) {
        out.push({
            data: arr.data.subarray(i * planeSize, (i + 1) * planeSize) as RealTypedArray,
            shape: rest,
            complex: arr.complex,
        });
    }
    return out;
}

export interface CropBounds {
    nx: number;
    yMin: number; yMax: number;
    xMin: number; xMax: number;
}

// NaN-aware min/max. Without `bounds`, scans the whole flat buffer (e.g. a
// full probe intensity stack). With `bounds`, scans only the given row/col
// window of a 2D array's flat buffer -- used to autoscale off a cropped
// region-of-interest without ever materializing a cropped copy (the crop is
// only ever used for this reduction, never for display).
export function minmaxNaN(data: RealTypedArray, bounds?: CropBounds): [number, number] {
    let mn = Infinity, mx = -Infinity;
    if (!bounds) {
        for (let i = 0; i < data.length; i++) {
            const v = data[i];
            if (Number.isNaN(v)) continue;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
    } else {
        const { nx, yMin, yMax, xMin, xMax } = bounds;
        for (let y = yMin; y < yMax; y++) {
            const rowBase = y * nx;
            for (let x = xMin; x < xMax; x++) {
                const v = data[rowBase + x];
                if (Number.isNaN(v)) continue;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
        }
    }
    return [mn, mx];
}

const LUT_SIZE = 256;
const _luts: Map<(t: number) => string, Uint8ClampedArray> = new Map();

// Precomputes a colormap lookup table from a d3 interpolator (already a
// dependency, used for the colorbar/legend gradient) once per colormap, so the
// hot per-pixel canvas render never parses a CSS color string. This also
// guarantees the raster and the colorbar legend render identical colors,
// since both derive from the same source.
function colorLut(interpolate: (t: number) => string): Uint8ClampedArray {
    const cached = _luts.get(interpolate);
    if (cached) return cached;

    const lut = new Uint8ClampedArray(LUT_SIZE * 4);
    for (let i = 0; i < LUT_SIZE; i++) {
        // interpolators differ in the string they return -- the ramp-based maps (magma)
        // '#rrggbb', the analytic ones (sinebow) 'rgb(r, g, b)'
        const c = rgb(interpolate(i / (LUT_SIZE - 1)));
        lut[4 * i] = c.r; lut[4 * i + 1] = c.g; lut[4 * i + 2] = c.b; lut[4 * i + 3] = 255 * c.opacity;
    }
    _luts.set(interpolate, lut);
    return lut;
}

// Normalizes `data` into [vmin, vmax], maps through the colormap LUT (clamping
// out-of-range values to the endpoint colors), and writes RGBA bytes
// directly into `imageData.data`. NaN pixels render fully transparent,
// matching wasm-array's `apply_cmap` default invalid color ([0,0,0,0]).
export function applyLutInto(
    imageData: ImageData, data: RealTypedArray, vmin: number, vmax: number,
    interpolate: (t: number) => string,
): void {
    const lut = colorLut(interpolate);
    const out = imageData.data;
    const scale = (LUT_SIZE - 1) / (vmax - vmin);

    for (let i = 0; i < data.length; i++) {
        const v = data[i];
        const o = i * 4;
        if (Number.isNaN(v)) {
            out[o] = 0; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0;
            continue;
        }
        let t = (v - vmin) * scale;
        if (t < 0) t = 0; else if (t > LUT_SIZE - 1) t = LUT_SIZE - 1;
        const idx = t | 0;
        out[o] = lut[4 * idx]; out[o + 1] = lut[4 * idx + 1];
        out[o + 2] = lut[4 * idx + 2]; out[o + 3] = lut[4 * idx + 3];
    }
}

export const applyMagmaInto = (imageData: ImageData, data: RealTypedArray, vmin: number, vmax: number) =>
    applyLutInto(imageData, data, vmin, vmax, interpolateMagma);

// cyclic, for phase in [-pi, pi]
export const applySinebowInto = (imageData: ImageData, data: RealTypedArray, vmin: number, vmax: number) =>
    applyLutInto(imageData, data, vmin, vmax, interpolateSinebow);

// Complex-domain coloring: hue from arg(x) through the cyclic sinebow map, brightness
// from |x| normalized to `vmax`. The `colorize_complex` of `phaser/utils/image.py`,
// with sinebow in place of HSV. NaN pixels render fully transparent, as in `applyLutInto`.
export function colorizeComplexInto(imageData: ImageData, arr: DecodedArray, vmax: number): void {
    if (!arr.complex) throw new Error("'colorizeComplexInto' requires complex input");
    const lut = colorLut(interpolateSinebow);
    const out = imageData.data;
    const buf = arr.data;
    const n = buf.length / 2;

    for (let i = 0; i < n; i++) {
        const re = buf[2 * i], im = buf[2 * i + 1];
        const o = i * 4;
        if (Number.isNaN(re) || Number.isNaN(im)) {
            out[o] = 0; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0;
            continue;
        }
        let v = Math.sqrt(re * re + im * im) / vmax;
        if (v < 0) v = 0; else if (v > 1) v = 1;

        let t = (Math.atan2(im, re) / (2 * Math.PI) + 0.5) * (LUT_SIZE - 1);
        if (t < 0) t = 0; else if (t > LUT_SIZE - 1) t = LUT_SIZE - 1;
        const idx = t | 0;
        out[o] = lut[4 * idx] * v; out[o + 1] = lut[4 * idx + 1] * v; out[o + 2] = lut[4 * idx + 2] * v;
        out[o + 3] = lut[4 * idx + 3];
    }
}
