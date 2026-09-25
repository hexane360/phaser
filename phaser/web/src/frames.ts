// Binary frame wire format, mirroring `phaser/web/frames.py`:
//
//   u32 LE   header length H
//   H bytes  UTF-8 JSON header: {"buffers": [nbytes, ...], "data": <message>}
//            zero padding to a multiple of `ALIGN`
//   buf 0, zero padding to `ALIGN`, buf 1, ...
//
// Arrays in `data` are `{"_ty": "numpy", typestr, shape, buf}` placeholders, decoded
// here to zero-copy `DecodedArray` views into the frame.

import { DecodedArray } from './array';

// Buffer alignment. Must match `ALIGN` in `phaser/web/frames.py`.
export const ALIGN = 8;

interface ArrayPlaceholder {
    _ty: 'numpy';
    typestr: string;
    shape: ReadonlyArray<number>;
    buf: number;
}

// `[byte offset, byte length]` of each buffer in the frame
type BufferTable = ReadonlyArray<readonly [number, number]>;

const padding = (n: number) => (ALIGN - n % ALIGN) % ALIGN;

// Only the dtypes the server sends are supported. The server always runs little-endian,
// so a `>` (big-endian) typestr throws rather than silently mis-rendering.
function decodeArray(obj: ArrayPlaceholder, frame: ArrayBuffer, buffers: BufferTable): DecodedArray {
    const entry = buffers[obj.buf];
    if (entry === undefined) throw new Error(`Invalid buffer index ${obj.buf}`);
    const [offset, nbytes] = entry;
    switch (obj.typestr) {
        case '<f4': return { data: new Float32Array(frame, offset, nbytes / 4), shape: obj.shape, complex: false };
        case '<f8': return { data: new Float64Array(frame, offset, nbytes / 8), shape: obj.shape, complex: false };
        case '<c8': return { data: new Float32Array(frame, offset, nbytes / 4), shape: obj.shape, complex: true };
        case '<c16': return { data: new Float64Array(frame, offset, nbytes / 8), shape: obj.shape, complex: true };
        default:
            throw new Error(`Unsupported array typestr '${obj.typestr}' (only <f4, <f8, <c8, <c16 are supported)`);
    }
}

function decode(obj: any, frame: ArrayBuffer, buffers: BufferTable): any {
    if (typeof obj !== 'object' || obj === null) return obj;
    if (Array.isArray(obj)) return obj.map((v) => decode(v, frame, buffers));

    if (obj._ty !== undefined) {
        if (obj._ty === 'numpy') return decodeArray(obj as ArrayPlaceholder, frame, buffers);
        throw new Error(`Unknown custom type '${obj._ty}'`);
    }
    return Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, decode(v, frame, buffers)]));
}

// Decode a frame. Arrays are views into `frame`, which must not be modified afterwards.
export function unpack(frame: ArrayBuffer): any {
    if (frame.byteLength < 4) throw new Error("Frame truncated before header");
    const headerLen = new DataView(frame).getUint32(0, true);
    let pos = 4 + headerLen;
    if (pos > frame.byteLength) throw new Error("Frame truncated in header");

    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(frame, 4, headerLen)));
    if (typeof header !== 'object' || header === null || !Array.isArray(header.buffers)) {
        throw new Error("Invalid frame header");
    }

    const buffers: Array<readonly [number, number]> = [];
    for (const size of header.buffers as Array<unknown>) {
        if (typeof size !== 'number' || !Number.isInteger(size) || size < 0) throw new Error(`Invalid buffer size ${size}`);
        pos += padding(pos);
        if (pos + size > frame.byteLength) throw new Error("Frame truncated in buffers");
        buffers.push([pos, size]);
        pos += size;
    }
    if (pos !== frame.byteLength) throw new Error(`${frame.byteLength - pos} trailing bytes after frame`);

    return decode(header.data, frame, buffers);
}
