import threading
import time
import typing as t

import numpy
import pytest
from numpy.testing import assert_array_equal
from typing_extensions import Self

from phaser.engines.common.simulation import (
    GroupPatterns,
    PatternStream,
    ResidentPatterns,
    iter_patterns,
    resolve_patterns,
)
from phaser.utils.num import get_array_module, get_backend_module, to_numpy

from .utils import with_backends


def _groups(n: int) -> t.List[numpy.ndarray]:
    # groups index the scan dimensions: shape (scan ndim, group size)
    return [numpy.array([[2*i, 2*i + 1]]) for i in range(n)]


@with_backends('numpy', 'jax', 'cupy', 'torch')
@pytest.mark.parametrize('buf_n', (0, 1, 3))
def test_pattern_stream(backend: str, buf_n: int):
    xp = get_backend_module(backend)
    patterns = numpy.arange(10 * 4 * 4, dtype=numpy.float32).reshape(10, 4, 4)
    groups = _groups(5)

    stream = PatternStream(groups, patterns, xp, buf_n)
    n = 0
    for (expected_group, (group, group_patterns)) in zip(groups, stream, strict=True):
        assert_array_equal(group, expected_group)
        # transferred to the default device, even from worker threads
        expected = xp.asarray(patterns)
        assert get_array_module(group_patterns) is get_array_module(expected)
        assert str(group_patterns.device) == str(expected.device)
        assert_array_equal(to_numpy(group_patterns), patterns[tuple(group)])
        # release using an output computed from the group
        stream.release_after(group_patterns * 2.)
        n += 1
    assert n == len(groups)


class _Output:
    """Fake async output, which becomes ready once `ready` is set."""
    def __init__(self):
        self.ready = threading.Event()
        self.waited = False

    def block_until_ready(self) -> Self:
        self.ready.wait()
        self.waited = True
        return self


class _RecordingPatterns:
    """Patterns which record which groups have been loaded."""
    def __init__(self, patterns: numpy.ndarray):
        self.patterns = patterns
        self.loaded: t.List[int] = []

    def __getitem__(self, idx):
        self.loaded.append(int(idx[0][0]) // 2)
        return self.patterns[idx]


def test_pattern_stream_release():
    patterns = _RecordingPatterns(numpy.zeros((10, 4, 4), dtype=numpy.float32))
    stream = PatternStream(_groups(5), patterns, numpy, buf_n=2)  # type: ignore
    it = iter(stream)
    outputs = [_Output() for _ in range(5)]

    def wait_loaded(n: int):
        for _ in range(100):
            if len(patterns.loaded) >= n:
                break
            time.sleep(0.01)

    next(it)
    stream.release_after(outputs[0])
    next(it)
    stream.release_after(outputs[1])

    # group 2 can't be loaded until group 0 is released
    time.sleep(0.05)
    assert patterns.loaded == [0, 1]

    outputs[0].ready.set()
    next(it)
    stream.release_after(outputs[2])
    wait_loaded(3)
    assert patterns.loaded == [0, 1, 2]

    # group 3 waits on group 1
    time.sleep(0.05)
    assert patterns.loaded == [0, 1, 2]

    outputs[1].ready.set()
    next(it)
    # group 3 isn't released explicitly, so it's released once its transfer completes
    outputs[2].ready.set()
    next(it)
    with pytest.raises(StopIteration):
        next(it)
    assert patterns.loaded == [0, 1, 2, 3, 4]


@with_backends('numpy', 'jax')
@pytest.mark.parametrize('buffer_n_groups', (None, 0, 2))
def test_iter_patterns(backend: str, buffer_n_groups: t.Optional[int]):
    xp = get_backend_module(backend)
    patterns = numpy.arange(10 * 4 * 4, dtype=numpy.float32).reshape(10, 4, 4)
    groups = _groups(5)

    source = iter_patterns(groups, xp.asarray(patterns) if buffer_n_groups is None else patterns, xp, buffer_n_groups)
    assert isinstance(source, ResidentPatterns if buffer_n_groups is None else PatternStream)

    with source:
        for (expected_group, (group, group_patterns)) in zip(groups, source, strict=True):
            assert_array_equal(group, expected_group)
            if buffer_n_groups is None:
                assert isinstance(group_patterns, GroupPatterns)
            group_patterns = resolve_patterns(group_patterns)
            assert_array_equal(to_numpy(group_patterns), patterns[tuple(group)])
            source.release_after(group_patterns * 2.)


def test_resident_patterns_release():
    patterns = numpy.zeros((10, 4, 4), dtype=numpy.float32)
    source = ResidentPatterns(_groups(4), patterns, max_in_flight=1)
    it = iter(source)
    outputs = [_Output() for _ in range(4)]
    for output in outputs:
        output.ready.set()

    next(it)
    source.release_after(outputs[0])
    assert not outputs[0].waited
    # group 1 waits for group 0
    next(it)
    assert outputs[0].waited
    # group 1 isn't released, so group 2 doesn't wait
    next(it)
    source.release_after(outputs[2])
    assert not outputs[2].waited
    next(it)
    assert outputs[2].waited
    with pytest.raises(StopIteration):
        next(it)


def test_pattern_stream_close():
    patterns = _RecordingPatterns(numpy.zeros((10, 4, 4), dtype=numpy.float32))
    outputs = [_Output() for _ in range(5)]

    with PatternStream(_groups(5), patterns, numpy, buf_n=2) as stream:  # type: ignore
        for (i, _) in enumerate(stream):
            stream.release_after(outputs[i])
            if i == 1:
                break
    # group 2 is queued behind group 0, and cancelled
    outputs[0].ready.set()
    outputs[1].ready.set()
    time.sleep(0.05)
    assert patterns.loaded == [0, 1]


def test_resident_patterns_max_in_flight():
    patterns = numpy.zeros((10, 4, 4), dtype=numpy.float32)
    source = ResidentPatterns(_groups(4), patterns, max_in_flight=2)
    it = iter(source)
    outputs = [_Output() for _ in range(4)]
    for output in outputs:
        output.ready.set()

    for i in range(2):
        next(it)
        source.release_after(outputs[i])
    assert not any(output.waited for output in outputs)
    # group 2 waits for group 0, but not group 1
    next(it)
    assert outputs[0].waited
    assert not outputs[1].waited


@with_backends('jax')
def test_group_patterns_jit(backend: str):
    import jax
    xp = get_backend_module(backend)
    patterns = numpy.arange(3 * 4 * 2 * 2, dtype=numpy.float32).reshape(3, 4, 2, 2)
    group = numpy.array([[0, 2, 1], [3, 0, 1]])  # 2D scan

    @jax.jit
    def f(group_patterns):
        return resolve_patterns(group_patterns) * 2.

    out = f(GroupPatterns(xp.asarray(patterns), group))
    assert_array_equal(to_numpy(out), 2. * patterns[tuple(group)])
