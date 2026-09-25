import json
import struct
import typing as t

import numpy
import pytest

from phaser.web.frames import ALIGN, pack, pack_bytes, unpack

pytestmark = pytest.mark.web


def _array(dtype: str, shape: tuple = (3, 5), seed: int = 0) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    arr = rng.normal(size=shape)
    return (arr + 1j * rng.normal(size=shape) if dtype in ('<c8', '<c16') else arr).astype(dtype)


@pytest.mark.parametrize('dtype', ['<f4', '<f8', '<c8', '<c16'])
def test_roundtrip(dtype: str):
    arr = _array(dtype)
    out = unpack(pack_bytes(arr))
    assert out.dtype == arr.dtype
    assert out.shape == arr.shape
    numpy.testing.assert_array_equal(out, arr)
    # zero-copy view into the frame
    assert not out.flags.writeable


def test_noncontiguous_and_transposed():
    base = _array('<f8', (6, 8), seed=1)
    for arr in (base[:, ::2], base.T):
        assert not arr.flags['C_CONTIGUOUS']
        numpy.testing.assert_array_equal(unpack(pack_bytes(arr)), arr)


def test_nested_structures():
    a, b = _array('<f4', (2, 3)), _array('<c16', (4,), seed=2)
    obj = {'msg': 'x', 'n': 1, 'x': 0.5, 'none': None, 'nested': {'list': [a, {'b': b}], 'tuple': (1, 2)}}
    out = unpack(pack_bytes(obj))

    assert (out['msg'], out['n'], out['x'], out['none']) == ('x', 1, 0.5, None)
    assert out['nested']['tuple'] == [1, 2]
    numpy.testing.assert_array_equal(out['nested']['list'][0], a)
    numpy.testing.assert_array_equal(out['nested']['list'][1]['b'], b)


def test_scalars():
    out = unpack(pack_bytes({'x': float('nan'), 'y': float('inf'), 'z': numpy.float32(1.5), 'i': numpy.int64(3)}))
    assert numpy.isnan(out['x']) and out['y'] == float('inf')
    assert (out['z'], out['i']) == (1.5, 3)


def test_strict_nan_raises():
    with pytest.raises(ValueError):
        pack_bytes({'x': [1., float('nan')]}, allow_nan=False)
    assert unpack(pack_bytes({'x': 1.}, allow_nan=False)) == {'x': 1.}


def test_bytes():
    data = b'\x00\x01\xff\xfe hello'
    out = unpack(pack_bytes({'b': data, 'arr': numpy.ones(3)}))
    assert out['b'] == data
    numpy.testing.assert_array_equal(out['arr'], numpy.ones(3))


def test_buffers_are_aligned():
    # odd sizes, so every buffer after the first needs padding
    arrs = [numpy.arange(n, dtype='<f4') for n in (1, 3, 5)] + [numpy.arange(3, dtype='<c16')]
    parts = pack(arrs)

    pos = 0
    for part in parts:
        if isinstance(part, memoryview):
            assert pos % ALIGN == 0
        pos += len(part) if isinstance(part, bytes) else part.nbytes

    body = b''.join(parts)
    assert len(body) == pos
    for arr, out in zip(arrs, unpack(body)):
        numpy.testing.assert_array_equal(out, arr)


def test_layout():
    arr = numpy.arange(3, dtype='<f4')
    body = pack_bytes({'a': arr})
    (n, n_buffers) = struct.unpack_from('<II', body)
    assert n_buffers == 1
    assert struct.unpack_from('<Q', body, 8) == (arr.nbytes,)
    assert json.loads(body[16:16 + n]) == {'a': {'_ty': 'numpy', 'typestr': '<f4', 'shape': [3], 'buf': 0}}
    assert body[16 + n + -(16 + n) % ALIGN:] == arr.tobytes()


def test_unsupported_type_raises():
    with pytest.raises(TypeError):
        pack_bytes({'x': object()})


def test_no_buffers():
    body = pack_bytes({'msg': 'pong'})
    assert unpack(body) == {'msg': 'pong'}


def test_empty_and_scalar_arrays():
    out = unpack(pack_bytes([numpy.zeros((0, 3), dtype='<f8'), numpy.array(2., dtype='<f4'), numpy.ones(2)]))
    assert out[0].shape == (0, 3)
    assert out[1].shape == () and out[1] == 2.
    numpy.testing.assert_array_equal(out[2], [1., 1.])


def _frame(header: bytes, sizes: t.Sequence[int] = (), rest: bytes = b'') -> bytes:
    prefix = struct.pack(f'<II{len(sizes)}Q', len(header), len(sizes), *sizes) + header
    return prefix + (bytes(-len(prefix) % ALIGN) if sizes else b'') + rest


@pytest.mark.parametrize('body', [
    b'',
    b'\x01\x00',
    struct.pack('<II', 255, 0) + b'{}',  # header longer than the body
    struct.pack('<II', 2, 4) + b'{}',  # truncated buffer table
    _frame(b'{not json'),
    _frame(b'\xff\xfe\xfa'),  # not UTF-8
    _frame(b'1', [64], bytes(8)),  # truncated buffer
    _frame(b'1', rest=b'extra'),  # trailing bytes
    _frame(b'{"_ty": "foo"}'),
    _frame(b'{"_ty": "bytes"}'),
    _frame(b'{"_ty": "bytes", "buf": -1}', [1], bytes(1)),
    _frame(b'{"_ty": "numpy", "typestr": "<f8", "shape": [1], "buf": 0}'),
    _frame(b'{"_ty": "numpy", "shape": [1], "buf": 0}', [8], bytes(8)),
    _frame(b'{"_ty": "numpy", "typestr": "<f8", "shape": [1], "buf": 0}', [3], bytes(3)),
    *(_frame(b'{"_ty": "numpy", "typestr": "<f8", ' + arr + b'}', [8], bytes(8)) for arr in (
        b'"shape": [1]',  # missing buf
        b'"shape": [1], "buf": -1',
        b'"shape": [1], "buf": true',
        b'"shape": [1], "buf": "0"',
        b'"shape": ["a"], "buf": 0',
        b'"shape": [1.0], "buf": 0',
        b'"shape": [[1]], "buf": 0',
        b'"shape": [-1], "buf": 0',
        b'"shape": [true], "buf": 0',
        b'"shape": [2], "buf": 0',  # size mismatch
        b'"shape": 1, "buf": 0',
    )),
    # nested placeholder
    _frame(b'[{"x": {"_ty": "numpy", "typestr": "<f8", "shape": [2], "buf": 0}}]', [8], bytes(8)),
])
def test_malformed_frames_raise(body: bytes):
    with pytest.raises(ValueError):
        unpack(body)
