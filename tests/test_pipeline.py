from queue import Empty
from threading import Event, current_thread

import pytest

from qwen_asr_opt.pipeline import PrefetchChunks


def test_order_bounded_read_ahead_and_producer_ownership():
    observed = []
    closed = []
    third = Event()

    def source():
        try:
            for value in range(20):
                observed.append(value)
                if value == 2:
                    third.set()
                yield value
        finally:
            closed.append(current_thread().name)

    stream = PrefetchChunks(source(), depth=2)
    try:
        assert third.wait(2)
        assert observed == [0, 1, 2]  # Two queued plus one blocked producer item.
        assert list(stream) == list(range(20))
        assert closed == ["qwen-audio-prefetch"]
        assert not stream.thread.is_alive()
    finally:
        stream.close()


def test_producer_failure_preserves_prior_items_and_closes():
    closed = []

    def source():
        try:
            yield "saved"
            raise ValueError("decoder failure")
        finally:
            closed.append(True)

    stream = PrefetchChunks(source(), depth=1)
    assert next(stream) == "saved"
    with pytest.raises(ValueError, match="decoder failure"):
        next(stream)
    assert closed == [True]
    assert not stream.thread.is_alive()
    stream.close()


def test_consumer_failure_cancels_full_queue_without_leaking_worker():
    closed = []
    second = Event()

    def source():
        try:
            for value in range(100):
                if value == 1:
                    second.set()
                yield value
        finally:
            closed.append(True)

    stream = PrefetchChunks(source(), depth=1)
    assert second.wait(2)
    try:
        raise RuntimeError("consumer failed")
    except RuntimeError:
        stream.close()
    assert closed == [True]
    assert not stream.thread.is_alive()
    assert list(stream) == []


def test_empty_source_and_invalid_depth():
    stream = PrefetchChunks(iter([]))
    assert list(stream) == []
    assert not stream.thread.is_alive()
    with pytest.raises(ValueError, match="depth"):
        PrefetchChunks([], 0)


def test_completion_arriving_between_queue_timeout_and_liveness_check():
    stream = PrefetchChunks(iter([]))
    stream.thread.join(timeout=2)
    original = stream.queue

    class TimeoutBeforeCompletion:
        def get(self, timeout):
            raise Empty

        def get_nowait(self):
            return original.get_nowait()

    stream.queue = TimeoutBeforeCompletion()
    assert list(stream) == []
