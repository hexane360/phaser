import asyncio

import pytest

from phaser.web.util import timeout

pytestmark = pytest.mark.web


def test_timeout_raises_on_expiry():
    async def scenario():
        async with timeout(0.05):
            await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        asyncio.run(scenario())


def test_timeout_passes_through_under_budget():
    async def scenario():
        async with timeout(5):
            await asyncio.sleep(0)
            return 'ok'

    assert asyncio.run(scenario()) == 'ok'


def test_timeout_leaves_foreign_cancellation_alone():
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


def test_timeout_accepts_no_deadline():
    async def scenario():
        async with timeout(None):
            return 'ok'

    assert asyncio.run(scenario()) == 'ok'
