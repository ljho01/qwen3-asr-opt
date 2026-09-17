import numpy as np
import pytest

from qwen_asr_opt import vad as module


class FakeVad:
    def reset(self):
        self.calls = 0

    def probability(self, frame):
        assert frame.shape == (512,)
        self.calls += 1
        return 0.99 if np.max(np.abs(frame)) > 0.1 else 0.01


def chunks_for(monkeypatch, wave, read_size=30 * 16000, **kwargs):
    def source(path):
        for index in range(0, len(wave), read_size):
            yield wave[index:index + read_size], index / 16000
    monkeypatch.setattr(module, "audio_chunks", source)
    result = list(module.speech_chunks("unused", FakeVad(), **kwargs))
    if len(wave):
        np.testing.assert_array_equal(np.concatenate([part for part, _ in result]), wave)
    else:
        assert result == []
    seen = 0
    for part, offset in result:
        assert round(offset * 16000) == seen
        assert 0 < len(part) <= round(kwargs.get("max_chunk_s", 30) * 16000)
        seen += len(part)
    assert seen == len(wave)
    return result


@pytest.mark.parametrize("read_size", [65536, 30 * 16000])
def test_pause_boundaries_preserve_audio_and_partial_frames(monkeypatch, read_size):
    wave = np.concatenate([np.full(4 * 16000, 0.2, dtype=np.float32),
                           np.zeros(16000, dtype=np.float32),
                           np.full(7 * 16000, 0.3, dtype=np.float32),
                           np.zeros(7000, dtype=np.float32),
                           np.full(5 * 16000 + 13, 0.4, dtype=np.float32)])
    result = chunks_for(monkeypatch, wave, read_size)
    assert len(result) == 3
    assert 4 < result[1][1] < 4.5
    assert 12 < result[2][1] < 12.4375


@pytest.mark.parametrize("amplitude", [0, 0.2, 0.00001])
def test_silence_continuous_and_very_quiet_audio_are_never_dropped(monkeypatch, amplitude):
    wave = np.full(69 * 16000 + 19, amplitude, dtype=np.float32)
    result = chunks_for(monkeypatch, wave)
    assert len(result) == 3


def test_empty_short_tail_and_invalid_parameters(monkeypatch):
    chunks_for(monkeypatch, np.empty(0, dtype=np.float32))
    wave = np.concatenate([np.full(5 * 16000, 0.2, dtype=np.float32),
                           np.zeros(5000, dtype=np.float32)])
    assert len(chunks_for(monkeypatch, wave)) == 1
    with pytest.raises(ValueError, match="silence_ms"):
        chunks_for(monkeypatch, wave, silence_ms=0)
    with pytest.raises(ValueError, match="Require"):
        chunks_for(monkeypatch, wave, min_chunk_s=30)


def test_boundaries_are_independent_of_reader_blocks(monkeypatch):
    rng = np.random.default_rng(53)
    wave = rng.uniform(0.15, 0.25, 98 * 16000 + 71).astype(np.float32)
    for start in (9, 26, 44, 72, 90):
        wave[start * 16000:(start + 1) * 16000] = 0
    first = chunks_for(monkeypatch, wave, 65536, silence_ms=500, min_chunk_s=8)
    second = chunks_for(monkeypatch, wave, 30 * 16000, silence_ms=500, min_chunk_s=8)
    assert [(offset, len(part)) for part, offset in first] == [(offset, len(part)) for part, offset in second]


def test_decoder_stream_closes_after_vad_failure(monkeypatch):
    closed = []

    def source(path):
        try:
            yield np.zeros(30 * 16000, dtype=np.float32), 0.0
        finally:
            closed.append(True)

    class FailingVad(FakeVad):
        def probability(self, frame):
            raise RuntimeError("VAD failure")

    monkeypatch.setattr(module, "audio_chunks", source)
    with pytest.raises(RuntimeError, match="VAD failure"):
        list(module.speech_chunks("unused", FailingVad()))
    assert closed == [True]
