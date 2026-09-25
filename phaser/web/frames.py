"""
Binary frame wire format, shared by worker -> server requests and server -> browser
websocket messages (mirrored in `src/frames.ts`):

```
u32 LE      JSON length H
u32 LE      buffer count N
N x u64 LE  buffer sizes
H bytes     UTF-8 JSON message
            zero padding to a multiple of `ALIGN`
buf 0, zero padding to `ALIGN`, buf 1, ...
```

Arrays in the message are replaced by `{"_ty": "numpy", "typestr", "shape", "buf": i}`,
with their (C-contiguous) bytes stored out-of-band in buffer `i`. `bytes` are likewise
replaced by `{"_ty": "bytes", "buf": i}`. Containers are walked by `json` itself, through
the `default` and `object_hook` hooks.
"""

import functools
import json
import struct
import typing as t

import numpy
from pane.converters import Converter

CONTENT_TYPE: str = 'application/x-phaser-frames'

ALIGN: int = 8
"""Buffer alignment. Must be a multiple of the largest element size, for JS typed array views."""

_PREFIX = struct.Struct('<II')
_SIZE_LEN = 8


def _padding(n: int) -> int:
    return -n % ALIGN


def _encode(obj: t.Any, buffers: t.List[memoryview]) -> t.Any:
    if isinstance(obj, numpy.ndarray):
        buffers.append(memoryview(numpy.ascontiguousarray(obj).reshape(-1).view(numpy.uint8)))
        return {'_ty': 'numpy', 'typestr': obj.dtype.str, 'shape': obj.shape, 'buf': len(buffers) - 1}

    if isinstance(obj, bytes):
        buffers.append(memoryview(obj))
        return {'_ty': 'bytes', 'buf': len(buffers) - 1}

    if isinstance(obj, numpy.generic):
        return obj.item()

    raise TypeError(f"Object of type {type(obj).__name__} can't be encoded in a frame")


def pack(obj: t.Any, *, allow_nan: bool = True) -> t.List[t.Union[bytes, memoryview]]:
    """Encode `obj` as a frame, returned as a list of parts to concatenate.

    With `allow_nan=False`, non-finite floats raise `ValueError` rather than being written
    as `NaN`/`Infinity`, which `JSON.parse` rejects."""
    buffers: t.List[memoryview] = []
    header = json.dumps(obj, default=functools.partial(_encode, buffers=buffers), allow_nan=allow_nan).encode('utf-8')

    parts: t.List[t.Union[bytes, memoryview]] = [
        _PREFIX.pack(len(header), len(buffers)),
        struct.pack(f'<{len(buffers)}Q', *(b.nbytes for b in buffers)),
        header,
    ]
    pos = _PREFIX.size + _SIZE_LEN * len(buffers) + len(header)
    for buf in buffers:
        parts.append(bytes(_padding(pos)))
        pos += _padding(pos)
        parts.append(buf)
        pos += buf.nbytes
    return parts


def pack_bytes(obj: t.Any, *, allow_nan: bool = True) -> bytes:
    """Encode `obj` as a single frame."""
    return b''.join(pack(obj, allow_nan=allow_nan))


def _is_index(v: t.Any) -> t.TypeGuard[int]:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _decode(obj: t.Dict[str, t.Any], buffers: t.Sequence[memoryview]) -> t.Any:
    if (ty := obj.get('_ty')) is None:
        return obj
    if ty not in ('bytes', 'numpy'):
        raise ValueError(f"Unknown custom type '{ty}'")
    if not _is_index(i := obj.get('buf')) or i >= len(buffers):
        raise ValueError(f"Invalid buffer index {i!r}")
    if ty == 'bytes':
        return bytes(buffers[i])

    if not isinstance(shape := obj.get('shape'), list) or not all(map(_is_index, t.cast(t.List[t.Any], shape))):
        raise ValueError(f"Invalid array shape {shape!r}")
    try:
        # zero-copy (and so read-only) view into the frame
        return numpy.frombuffer(buffers[i], numpy.dtype(obj['typestr'])).reshape(shape)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid array in frame: {e}") from None


def unpack(body: t.Union[bytes, bytearray, memoryview]) -> t.Any:
    """Decode a frame. Arrays are read-only views into `body`. Raises `ValueError` if malformed."""
    view = memoryview(body)
    if len(view) < _PREFIX.size:
        raise ValueError("Frame truncated before header")
    (n, n_buffers) = _PREFIX.unpack_from(view)
    header_start = _PREFIX.size + _SIZE_LEN * n_buffers
    if header_start > len(view):
        raise ValueError("Frame truncated in buffer table")
    pos = header_start + n
    if pos > len(view):
        raise ValueError("Frame truncated in header")

    buffers: t.List[memoryview] = []
    for size in struct.unpack_from(f'<{n_buffers}Q', view, _PREFIX.size):
        pos += _padding(pos)
        if pos + size > len(view):
            raise ValueError("Frame truncated in buffers")
        buffers.append(view[pos:pos + size])
        pos += size

    if pos != len(view):
        raise ValueError(f"{len(view) - pos} trailing bytes after frame")
    return json.loads(bytes(view[header_start:header_start + n]), object_hook=functools.partial(_decode, buffers=buffers))


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
