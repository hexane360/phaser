import asyncio
import contextlib
import sys
import typing as t

T = t.TypeVar('T')

if sys.version_info >= (3, 11):
    timeout = asyncio.timeout
else:
    @contextlib.asynccontextmanager
    async def timeout(delay: t.Optional[float]) -> t.AsyncIterator[None]:
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