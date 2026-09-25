import struct

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


def test_sampling_arrays_are_plain_lists():
    arr = numpy.array([1., 2., 3.])
    out = unpack(pack_bytes({'sampling': {'corner': arr, 'nested': [arr]}, 'data': arr}))
    assert out['sampling'] == {'corner': [1., 2., 3.], 'nested': [[1., 2., 3.]]}
    assert isinstance(out['data'], numpy.ndarray)


def test_buffers_are_aligned():
    # odd sizes, so every buffer after the first needs padding
    arrs = [numpy.arange(n, dtype='<f4') for n in (1, 3, 5)] + [numpy.arange(3, dtype='<c16')]
    parts = pack(arrs)
    (n,) = struct.unpack_from('<I', parts[0])

    pos = 4 + n
    for part in parts[2:]:
        if isinstance(part, memoryview):
            assert pos % ALIGN == 0
        pos += len(part) if isinstance(part, bytes) else part.nbytes

    body = b''.join(parts)
    assert len(body) == pos
    for arr, out in zip(arrs, unpack(body)):
        numpy.testing.assert_array_equal(out, arr)


def test_no_buffers():
    body = pack_bytes({'msg': 'pong'})
    assert unpack(body) == {'msg': 'pong'}


def test_empty_and_scalar_arrays():
    out = unpack(pack_bytes([numpy.zeros((0, 3), dtype='<f8'), numpy.array(2., dtype='<f4'), numpy.ones(2)]))
    assert out[0].shape == (0, 3)
    assert out[1].shape == () and out[1] == 2.
    numpy.testing.assert_array_equal(out[2], [1., 1.])


def _frame(header: bytes, rest: bytes = b'') -> bytes:
    return struct.pack('<I', len(header)) + header + rest


@pytest.mark.parametrize('body', [
    b'',
    b'\x01\x00',
    b'\xff\x00\x00\x00{}',  # header longer than the body
    _frame(b'{not json'),
    _frame(b'[1, 2]'),
    _frame(b'{"data": 1}'),
    _frame(b'{"buffers": [-1], "data": 1}'),
    _frame(b'{"buffers": [1.5], "data": 1}'),
    _frame(b'{"buffers": [64], "data": 1}', bytes(8)),  # truncated buffer
    _frame(b'{"buffers": [], "data": 1}', b'extra'),  # trailing bytes
    _frame(b'{"buffers": [], "data": {"_ty": "bytes"}}'),
    _frame(b'{"buffers": [], "data": {"_ty": "numpy", "typestr": "<f8", "shape": [1], "buf": 0}}'),
    _frame(b'{"buffers": [], "data": {"_ty": "numpy", "shape": [1], "buf": 0}}'),
])
def test_malformed_frames_raise(body: bytes):
    with pytest.raises(ValueError):
        unpack(body)
