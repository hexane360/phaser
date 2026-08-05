import asyncio

import numpy
import pytest

from phaser.web.util import _timeout, decode_obj, encode_obj

pytestmark = pytest.mark.web


@pytest.mark.parametrize('dtype', ['<f4', '<f8', '<c8', '<c16'])
def test_encode_decode_roundtrip(dtype: str):
    rng = numpy.random.default_rng(0)
    if dtype in ('<c8', '<c16'):
        arr = (rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))).astype(dtype)
    else:
        arr = rng.normal(size=(3, 5)).astype(dtype)

    encoded = encode_obj(arr)
    assert encoded['_ty'] == 'numpy'
    assert encoded['typestr'] == dtype
    assert encoded['strides'] is None  # always C-contiguous on the wire

    decoded = decode_obj(encoded)
    assert decoded.dtype == arr.dtype
    assert decoded.shape == arr.shape
    numpy.testing.assert_array_equal(decoded, arr)


def test_encode_decode_noncontiguous_and_transposed():
    rng = numpy.random.default_rng(1)
    base = rng.normal(size=(6, 8)).astype('<f8')

    # non-contiguous view (every other column)
    sliced = base[:, ::2]
    assert not sliced.flags['C_CONTIGUOUS']
    decoded = decode_obj(encode_obj(sliced))
    numpy.testing.assert_array_equal(decoded, sliced)
    assert decoded.shape == sliced.shape

    # transposed view
    transposed = base.T
    assert not transposed.flags['C_CONTIGUOUS']
    decoded_t = decode_obj(encode_obj(transposed))
    numpy.testing.assert_array_equal(decoded_t, transposed)
    assert decoded_t.shape == transposed.shape


def test_encode_decode_bytes():
    data = b'\x00\x01\xff\xfe hello'
    encoded = encode_obj(data)
    assert encoded['_ty'] == 'bytes'
    assert decode_obj(encoded) == data


def test_encode_nested_dict_and_sampling_key_forces_plain_lists():
    arr = numpy.array([1.0, 2.0, 3.0], dtype='<f8')
    obj = {'sampling': {'corner': arr}, 'data': arr}
    encoded = encode_obj(obj)
    # under 'sampling', arrays are encoded as plain lists (no interchange dict)
    assert encoded['sampling']['corner'] == [1.0, 2.0, 3.0]
    # elsewhere, arrays use the full numpy interchange format
    assert encoded['data']['_ty'] == 'numpy'


def test_encode_decode_idempotent_on_already_encoded_dict():
    # `encode_obj` on an already-encoded (no-array) dict is a no-op passthrough --
    # relied on by `Job.state()`'s `iter`-only listing, which reuses wire-form values
    # directly rather than re-decoding + re-encoding them.
    d = {'iter': {'engine_num': 1, 'engine_iter': 2, 'total_iter': 3}}
    assert encode_obj(d) == d


# --- `asyncio.timeout` polyfill (`_timeout`, used on 3.10) ------------------------------

def test_timeout_polyfill_raises_on_expiry():
    async def scenario():
        async with _timeout(0.05):
            await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        asyncio.run(scenario())


def test_timeout_polyfill_passes_through_under_budget():
    async def scenario():
        async with _timeout(5):
            await asyncio.sleep(0)
            return 'ok'

    assert asyncio.run(scenario()) == 'ok'


def test_timeout_polyfill_leaves_foreign_cancellation_alone():
    # a cancel that isn't ours must stay a `CancelledError`, not become a `TimeoutError`
    async def scenario():
        async def inner():
            async with _timeout(10):
                await asyncio.sleep(10)

        task = asyncio.ensure_future(inner())
        await asyncio.sleep(0.01)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())


def test_timeout_polyfill_accepts_no_deadline():
    async def scenario():
        async with _timeout(None):
            return 'ok'

    assert asyncio.run(scenario()) == 'ok'
