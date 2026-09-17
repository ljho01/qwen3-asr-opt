from queue import Empty
from threading import Event, current_thread

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

import qwen_asr_opt.encoder_prefetch as module
from qwen_asr_opt.encoder_prefetch import EncoderPrefetch


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.arange(16, dtype=mx.float16).reshape(1, 16)
        self.owner_names = []

    def __call__(self, mel, lengths):
        self.owner_names.append(current_thread().name)
        return mel.reshape(1, 1, 1) @ self.weight, lengths


@pytest.fixture
def tiny_features(monkeypatch):
    monkeypatch.setattr(module, "compute_features", lambda wave: (
        mx.array(wave[:1]).reshape(1, 1, 1), mx.array([1])))


def test_bounded_order_cross_stream_evaluation_and_source_ownership(tiny_features):
    observed, closed = [], []
    third = Event()

    def source():
        try:
            for i in range(8):
                observed.append(i)
                if i == 2:
                    third.set()
                yield np.array([i], dtype=np.float32), i * 30.
        finally:
            closed.append(current_thread().name)

    encoder = TinyEncoder()
    stream = EncoderPrefetch(source(), encoder, depth=2)
    try:
        assert third.wait(2)
        assert observed == [0, 1, 2]
        # Main-thread compiled GPU work consumes only fully evaluated arrays
        # produced on a different thread/stream, while the producer continues.
        consume = mx.compile(lambda value: value + 1)
        for i, item in enumerate(stream):
            assert (item.index, item.offset, item.samples) == (i, i * 30., 1)
            np.testing.assert_array_equal(np.array(consume(item.features)),
                                          (np.arange(16) * i + 1).reshape(1, 1, 16))
        assert closed == ["qwen-encoder-prefetch"]
        assert encoder.owner_names == ["qwen-encoder-prefetch"] * 8
        assert not stream.thread.is_alive()
    finally:
        stream.close()


def test_worker_feature_failure_preserves_prior_item_and_closes_source(monkeypatch):
    closed = []

    def features(wave):
        if wave[0] == 1:
            raise ValueError("feature failure")
        return mx.array(wave).reshape(1, 1, 1), mx.array([1])

    def source():
        try:
            for i in range(3):
                yield np.array([i], dtype=np.float32), float(i)
        finally:
            closed.append(True)

    monkeypatch.setattr(module, "compute_features", features)
    stream = EncoderPrefetch(source(), TinyEncoder(), depth=1)
    try:
        assert next(stream).index == 0
        with pytest.raises(ValueError, match="feature failure"):
            next(stream)
        assert closed == [True] and not stream.thread.is_alive()
    finally:
        stream.close()


def test_consumer_exit_cancels_full_queue_and_releases_source(tiny_features):
    second = Event()
    closed = []

    def source():
        try:
            for i in range(100):
                if i == 1:
                    second.set()
                yield np.array([i], dtype=np.float32), float(i)
        finally:
            closed.append(True)

    stream = EncoderPrefetch(source(), TinyEncoder(), depth=1)
    assert second.wait(2)
    stream.close()
    assert not stream.thread.is_alive() and stream.queue.empty() and closed == [True]
    assert list(stream) == []
    stream.close()


@pytest.mark.parametrize("size", [0, 480001])
def test_invalid_audio_propagates(size, tiny_features):
    stream = EncoderPrefetch(iter([(np.zeros(size, dtype=np.float32), 0.)]), TinyEncoder())
    try:
        with pytest.raises(ValueError, match="audio"):
            next(stream)
    finally:
        stream.close()


def test_empty_and_completion_timeout_race(tiny_features):
    stream = EncoderPrefetch(iter([]), TinyEncoder())
    stream.thread.join(timeout=2)
    original = stream.queue

    class TimeoutBeforeCompletion:
        def get(self, timeout):
            raise Empty

        def get_nowait(self):
            return original.get_nowait()

    stream.queue = TimeoutBeforeCompletion()
    assert list(stream) == [] and not stream.thread.is_alive()


@pytest.mark.parametrize("depth", [0, 3, True])
def test_invalid_depth(depth):
    with pytest.raises(ValueError, match="depth"):
        EncoderPrefetch(iter([]), TinyEncoder(), depth)
