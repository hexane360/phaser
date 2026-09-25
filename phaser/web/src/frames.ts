// Binary frame wire format, mirroring `phaser/web/frames.py`:
//
//   u32 LE      JSON length H
//   u32 LE      buffer count N
//   N x u64 LE  buffer sizes
//   H bytes     UTF-8 JSON message
//               zero padding to a multiple of `ALIGN`
//   buf 0, zero padding to `ALIGN`, buf 1, ...
//
// Arrays in the message are `{"_ty": "numpy", typestr, shape, buf}` placeholders, decoded
// here to `DecodedArray`s. `bytes` are `{"_ty": "bytes", buf}` placeholders, decoded to
// `Uint8Array`s. Both are copied out of the frame, so each owns its own `ArrayBuffer`.

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

function copyBuffer(frame: ArrayBuffer, buffers: BufferTable, buf: number): ArrayBuffer {
    const entry = buffers[buf];
    if (entry === undefined) throw new Error(`Invalid buffer index ${buf}`);
    const [offset, nbytes] = entry;
    return frame.slice(offset, offset + nbytes);
}

// Only the dtypes the server sends are supported. The server always runs little-endian,
// so a `>` (big-endian) typestr throws rather than silently mis-rendering.
function decodeArray(obj: ArrayPlaceholder, frame: ArrayBuffer, buffers: BufferTable): DecodedArray {
    switch (obj.typestr) {
        case '<f4': return { data: new Float32Array(copyBuffer(frame, buffers, obj.buf)), shape: obj.shape, complex: false };
        case '<f8': return { data: new Float64Array(copyBuffer(frame, buffers, obj.buf)), shape: obj.shape, complex: false };
        case '<c8': return { data: new Float32Array(copyBuffer(frame, buffers, obj.buf)), shape: obj.shape, complex: true };
        case '<c16': return { data: new Float64Array(copyBuffer(frame, buffers, obj.buf)), shape: obj.shape, complex: true };
        default:
            throw new Error(`Unsupported array typestr '${obj.typestr}' (only <f4, <f8, <c8, <c16 are supported)`);
    }
}

function decode(obj: any, frame: ArrayBuffer, buffers: BufferTable): any {
    if (typeof obj !== 'object' || obj === null) return obj;
    if (Array.isArray(obj)) return obj.map((v) => decode(v, frame, buffers));

    if (obj._ty !== undefined) {
        if (obj._ty === 'numpy') return decodeArray(obj as ArrayPlaceholder, frame, buffers);
        if (obj._ty === 'bytes') return new Uint8Array(copyBuffer(frame, buffers, obj.buf));
        throw new Error(`Unknown custom type '${obj._ty}'`);
    }
    return Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, decode(v, frame, buffers)]));
}

// Decode a frame.
export function unpack(frame: ArrayBuffer): any {
    if (frame.byteLength < 8) throw new Error("Frame truncated before header");
    const view = new DataView(frame);
    const headerLen = view.getUint32(0, true);
    const nBuffers = view.getUint32(4, true);
    const headerStart = 8 + 8 * nBuffers;
    if (headerStart > frame.byteLength) throw new Error("Frame truncated in buffer table");
    let pos = headerStart + headerLen;
    if (pos > frame.byteLength) throw new Error("Frame truncated in header");

    const buffers: Array<readonly [number, number]> = [];
    for (let i = 0; i < nBuffers; i++) {
        const size = Number(view.getBigUint64(8 + 8 * i, true));
        pos += padding(pos);
        if (pos + size > frame.byteLength) throw new Error("Frame truncated in buffers");
        buffers.push([pos, size]);
        pos += size;
    }
    if (pos !== frame.byteLength) throw new Error(`${frame.byteLength - pos} trailing bytes after frame`);

    const message = JSON.parse(new TextDecoder().decode(new Uint8Array(frame, headerStart, headerLen)));
    return decode(message, frame, buffers);
}
