import json

import numpy as np
import pytest
import soundfile as sf

from qwen_asr_opt.longform import audio_chunks, split_buffer


def test_chunks_cover_every_sample_once():
    rng = np.random.default_rng(5)
    original = rng.normal(size=16000 * 95 + 317).astype(np.float32)
    remaining = original
    parts = []
    while len(remaining) >= 30 * 16000:
        ready, remaining = split_buffer(remaining)
        assert 28 * 16000 <= len(ready) <= 30 * 16000
        parts.append(ready)
    parts.append(remaining)
    np.testing.assert_array_equal(np.concatenate(parts), original)


def test_ffmpeg_stream_matches_original_and_preserves_tail(tmp_path):
    wave = np.random.default_rng(0).uniform(-0.1, 0.1, 16000 * 7 + 13).astype(np.float32)
    path = tmp_path / "input.wav"
    sf.write(path, wave, 16000, subtype="FLOAT")
    chunks = list(audio_chunks(str(path), chunk_s=3))
    np.testing.assert_array_equal(np.concatenate([r[0] for r in chunks]), wave)
    previous = 0
    for samples, offset in chunks:
        assert round(offset * 16000) == previous
        previous += len(samples)


@pytest.mark.parametrize("fail_after_first", [False, True])
@pytest.mark.parametrize("prefetch", [0, 2])
@pytest.mark.parametrize("kv_policy", ["padded", "growing128"])
def test_continuous_persists_completion_order_and_closes_stream(tmp_path, monkeypatch,
                                                               fail_after_first, prefetch, kv_policy):
    from qwen_asr_opt import continuous, longform

    output = tmp_path / "result.json"
    partial = output.with_suffix(".chunks.jsonl")
    closed = []

    def stream(*args):
        try:
            for index in range(3):
                yield np.zeros(16000, dtype=np.float32), float(index)
        finally:
            closed.append(True)

    def run(session, chunks, language, batch_size, on_complete, *, kv_policy):
        assert kv_policy in ("padded", "growing128")
        # Keep the decoder stream suspended when the first completed row is saved.
        next(chunks)
        next(chunks)
        for index in (1, 0, 2):
            if index == 2:
                next(chunks)
            on_complete({"index": index, "start": float(index), "end": float(index + 1),
                         "text": ["first", "second", "third"][index], "language": "English",
                         "finish_reason": "eos", "truncated": False})
            assert not output.exists()
            assert json.loads(partial.read_text().splitlines()[-1])["index"] == index
            if fail_after_first:
                raise RuntimeError("simulated later decoding failure")
        return {"refills": 1}

    monkeypatch.setattr(longform, "audio_chunks", stream)
    monkeypatch.setattr(continuous, "transcribe_continuous_chunks", run)
    if fail_after_first:
        with pytest.raises(RuntimeError, match="later decoding failure"):
            longform.transcribe_long(None, "input.wav", output, "English", batch_size=2,
                                     scheduler="continuous", audio_prefetch=prefetch, kv_policy=kv_policy)
        assert len(partial.read_text().splitlines()) == 1
        assert not output.exists()
    else:
        result = longform.transcribe_long(None, "input.wav", output, "English", batch_size=2,
                                          scheduler="continuous", audio_prefetch=prefetch, kv_policy=kv_policy)
        assert result["text"] == "first second third"
        assert [r["index"] for r in result["chunks"]] == [0, 1, 2]
        assert [json.loads(line)["index"] for line in partial.read_text().splitlines()] == [1, 0, 2]
        assert result["audio_s"] == 3 and result["scheduler_stats"]["refills"] == 1
        assert result["kv_policy"] == kv_policy
        with pytest.raises(FileExistsError):
            longform.transcribe_long(None, "input.wav", output, scheduler="continuous")
    assert closed == [True]
