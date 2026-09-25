import asyncio
import sys
import typing as t
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.web


@pytest.fixture
def timeout() -> t.Callable[[float], asyncio.Timeout]:
    # ensure we grab polyfill
    with patch.object(sys, "version_info", (3, 10, 0, 'final', 0)):
        from phaser.web.util import timeout

    return timeout


def test_timeout_polyfill_raises_on_expiry(timeout):
    async def scenario():
        async with timeout(0.05):
            await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        asyncio.run(scenario())


def test_timeout_polyfill_passes_through_under_budget(timeout):
    async def scenario():
        async with timeout(5):
            await asyncio.sleep(0)
            return 'ok'

    assert asyncio.run(scenario()) == 'ok'


def test_timeout_polyfill_leaves_foreign_cancellation_alone(timeout):
    # a cancel that isn't ours must stay a `CancelledError`, not become a `TimeoutError`
    async def scenario():
        async def inner():
            async with timeout(10):
                await asyncio.sleep(10)

        task = asyncio.ensure_future(inner())
        await asyncio.sleep(0.01)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())


def test_timeout_polyfill_accepts_no_deadline(timeout):
    async def scenario():
        async with timeout(None):
            return 'ok'

    assert asyncio.run(scenario()) == 'ok'
