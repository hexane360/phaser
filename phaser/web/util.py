import asyncio
import base64
import contextlib
import dataclasses
import sys
import typing as t

import numpy
from pane.converters import Converter
from pane.errors import ProductErrorNode, WrongTypeError, ParseInterrupt

T = t.TypeVar('T')


@contextlib.asynccontextmanager
async def _timeout(delay: t.Optional[float]) -> t.AsyncIterator[None]:
    """`asyncio.timeout` for 3.10: cancel the enclosing task after `delay` seconds and raise
    `TimeoutError` instead.

    Lacking 3.11's `uncancel`, this can't distinguish its own expiry from a cancellation
    arriving at the same moment, and nesting two attributes the timeout to the inner one.
    """
    if delay is None:
        yield
        return

    task = asyncio.current_task()
    assert task is not None
    expired = False

    def on_timeout():
        nonlocal expired
        expired = True
        task.cancel()

    handle = asyncio.get_running_loop().call_later(delay, on_timeout)
    try:
        yield
    except asyncio.CancelledError:
        if expired:
            raise TimeoutError from None
        raise
    finally:
        handle.cancel()


timeout = asyncio.timeout if sys.version_info >= (3, 11) else _timeout
"""`asyncio.timeout`, or `_timeout` on 3.10 where it doesn't exist yet."""

class _array_dummy():
    def __init__(self, array_interface: t.Any):
        self.__array_interface__ = array_interface


def encode_obj(obj: t.Any, to_numpy: bool = True) -> t.Any:
    if isinstance(obj, numpy.ndarray):
        if not to_numpy:
            return obj.tolist()
        # guarantee C-contiguity so `strides` is always `None` and the client
        # never needs to handle strided array data
        obj = numpy.ascontiguousarray(obj)
        d = obj.__array_interface__
        d['data'] = base64.b64encode(obj.tobytes()).decode('ascii')
        d['_ty'] = 'numpy'
        return d

    if isinstance(obj, bytes):
        return {
            '_ty': 'bytes',
            'data': base64.urlsafe_b64encode(obj).decode('ascii')
        }

    if isinstance(obj, dict):
        d = obj
    elif dataclasses.is_dataclass(obj):
        d = obj.asdict()  # type: ignore
    else:
        return obj

    return {k: encode_obj(v, to_numpy and k not in ('sampling',)) for (k, v) in d.items()}


def decode_obj(obj: t.Any) -> t.Any:
    """Inverse of `encode_obj`. Pure: the wire-form passed in is left untouched, so it
    stays decodable. `Cache` holds exactly this wire-form and hands it to several views,
    and some (`_probes_view`) forward it to the client verbatim -- decoding in place
    would corrupt both."""
    if isinstance(obj, dict):
        d = obj
    elif dataclasses.is_dataclass(obj):
        d = obj.asdict()  # type: ignore
    else:
        return obj

    if '_ty' not in d:
        return {k: decode_obj(v) for (k, v) in d.items()}

    if (ty := d['_ty']) == 'numpy':
        interface = {k: v for (k, v) in d.items() if k != '_ty'}
        interface['data'] = base64.b64decode(d['data'].encode('utf-8'))
        interface['shape'] = tuple(d['shape'])
        # copies, so the array doesn't alias `interface['data']`
        return numpy.array(_array_dummy(interface))

    if ty == 'bytes':
        return base64.urlsafe_b64decode(d['data'].encode('utf-8'))

    raise ValueError(f"Unknown custom type '{ty}', while parsing reconstruction state update")


class ReconsStateConverter(Converter[t.Dict[str, t.Any]]):
    def __init__(self):
        ...

    def expected(self, plural: bool = False) -> str:
        return "reconstruction state"

    def into_data(self, val: t.Any) -> t.Any:
        return encode_obj(val)

    def try_convert(self, val: t.Any) -> t.Dict[str, t.Any]:
        try:
            val = decode_obj(val)
        except Exception:
            raise ParseInterrupt()
        #if not isinstance(val, t.Mapping) or ({'iter', 'wavelength'} - val.keys()):
        #    raise ParseInterrupt()

        return val  # type: ignore

    def collect_errors(self, val: t.Any) -> t.Union[ProductErrorNode, WrongTypeError, None]:
        try:
            val = decode_obj(val)
        except Exception as e:
            import traceback

            tb = e.__traceback__.tb_next  # type: ignore
            tb = traceback.TracebackException(type(e), e, tb)
            return WrongTypeError(self.expected(), val, tb)
        if not isinstance(val, t.Mapping):
            return WrongTypeError(self.expected(), val)
        #if (missing := {'iter', 'wavelength'} - val.keys()):
        #    return ProductErrorNode(self.expected(), {}, val, missing)

        return None