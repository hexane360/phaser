"""Web helper utilities."""
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
        """Backport of `asyncio.timeout`: raise `TimeoutError` if the block runs longer than `delay` seconds.

        Cannot distinguish its own expiry from a simultaneous cancellation, and nested
        timeouts are attributed to the inner one.
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