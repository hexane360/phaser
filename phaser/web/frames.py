"""
Binary frame wire format, shared by worker -> server requests and server -> browser
websocket messages (mirrored in `src/frames.ts`):

```
u32 LE   header length H
H bytes  UTF-8 JSON header: {"buffers": [nbytes, ...], "data": <message>}
         zero padding to a multiple of `ALIGN`
buf 0, zero padding to `ALIGN`, buf 1, ...
```

Arrays in `data` are replaced by `{"_ty": "numpy", "typestr", "shape", "buf": i}`, with
their (C-contiguous) bytes stored out-of-band in buffer `i`.
"""

import dataclasses
import json
import struct
import typing as t

import numpy
from pane.converters import Converter

CONTENT_TYPE: str = 'application/x-phaser-frames'

ALIGN: int = 8
"""Buffer alignment. Must be a multiple of the largest element size, for JS typed array views."""

_HEADER_LEN = struct.Struct('<I')


def _padding(n: int) -> int:
    return -n % ALIGN


def _encode(obj: t.Any, buffers: t.List[memoryview], to_numpy: bool = True) -> t.Any:
    if isinstance(obj, numpy.ndarray):
        if not to_numpy:
            return obj.tolist()
        buffers.append(memoryview(numpy.ascontiguousarray(obj).reshape(-1).view(numpy.uint8)))
        return {'_ty': 'numpy', 'typestr': obj.dtype.str, 'shape': obj.shape, 'buf': len(buffers) - 1}

    if isinstance(obj, numpy.generic):
        return obj.item()

    if isinstance(obj, (list, tuple)):
        return [_encode(v, buffers, to_numpy) for v in obj]

    if isinstance(obj, dict):
        d = obj
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = obj.asdict()  # type: ignore
    else:
        return obj

    # `sampling` is small metadata, which the client reads as plain lists
    return {k: _encode(v, buffers, to_numpy and k != 'sampling') for (k, v) in d.items()}


def pack(obj: t.Any) -> t.List[t.Union[bytes, memoryview]]:
    """Encode `obj` as a frame, returned as a list of parts to concatenate."""
    buffers: t.List[memoryview] = []
    data = _encode(obj, buffers)
    header = json.dumps({'buffers': [b.nbytes for b in buffers], 'data': data}, allow_nan=True).encode('utf-8')

    parts: t.List[t.Union[bytes, memoryview]] = [_HEADER_LEN.pack(len(header)), header]
    pos = _HEADER_LEN.size + len(header)
    for buf in buffers:
        parts.append(bytes(_padding(pos)))
        pos += _padding(pos)
        parts.append(buf)
        pos += buf.nbytes
    return parts


def pack_bytes(obj: t.Any) -> bytes:
    """Encode `obj` as a single frame."""
    return b''.join(pack(obj))


def _decode(obj: t.Any, buffers: t.Sequence[memoryview]) -> t.Any:
    if isinstance(obj, list):
        return [_decode(v, buffers) for v in obj]
    if not isinstance(obj, dict):
        return obj

    if (ty := obj.get('_ty')) is None:
        return {k: _decode(v, buffers) for (k, v) in obj.items()}
    if ty != 'numpy':
        raise ValueError(f"Unknown custom type '{ty}'")

    try:
        buf, dtype, shape = buffers[obj['buf']], numpy.dtype(obj['typestr']), tuple(obj['shape'])
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Invalid array in frame: {e}") from None
    # zero-copy (and so read-only) view into the frame
    return numpy.frombuffer(buf, dtype).reshape(shape)


def unpack(body: t.Union[bytes, bytearray, memoryview]) -> t.Any:
    """Decode a frame. Arrays are read-only views into `body`. Raises `ValueError` if malformed."""
    view = memoryview(body)
    if len(view) < _HEADER_LEN.size:
        raise ValueError("Frame truncated before header")
    (n,) = _HEADER_LEN.unpack_from(view)
    pos = _HEADER_LEN.size + n
    if pos > len(view):
        raise ValueError("Frame truncated in header")

    header = json.loads(bytes(view[_HEADER_LEN.size:pos]))
    if not isinstance(header, dict) or not isinstance(sizes := header.get('buffers'), list):
        raise ValueError("Invalid frame header")

    buffers: t.List[memoryview] = []
    for size in t.cast(t.List[t.Any], sizes):
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"Invalid buffer size {size!r}")
        pos += _padding(pos)
        if pos + size > len(view):
            raise ValueError("Frame truncated in buffers")
        buffers.append(view[pos:pos + size])
        pos += size

    if pos != len(view):
        raise ValueError(f"{len(view) - pos} trailing bytes after frame")
    return _decode(header.get('data'), buffers)


class ArrayDataConverter(Converter[t.Any]):
    """Passes values through unchanged, so arrays reach `pack` intact (rather than being
    converted to lists by `pane.into_data`)."""

    def expected(self, plural: bool = False) -> str:
        return "data"

    def into_data(self, val: t.Any) -> t.Any:
        return val

    def try_convert(self, val: t.Any) -> t.Any:
        return val

    def collect_errors(self, val: t.Any) -> None:
        return None
