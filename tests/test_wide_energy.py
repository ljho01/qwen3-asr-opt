import numpy as np
import pytest

from qwen_asr_opt.wide_energy import rechunk_stream, wide_cut

SR = 16000


def blocks(wave, size):
    for start in range(0, len(wave), size):
        yield wave[start:start + size], start / SR


@pytest.mark.parametrize("minimum", [15, 20])
def test_window_energy_matches_direct_reference(minimum):
    wave = np.random.default_rng(40).normal(size=30 * SR).astype(np.float32)
    starts = np.arange(minimum * SR - 4000, 30 * SR - 8000 + 1, 800)
    energy = [np.sum(np.square(wave[start:start + 8000], dtype=np.float64)) for start in starts]
    expected = starts[len(energy) - 1 - int(np.argmin(energy[::-1]))] + 4000
    assert wide_cut(wave, minimum_s=minimum) == expected
    wave[:] = 0
    assert wide_cut(wave, minimum_s=minimum) == round(29.75 * SR)


def test_quiet_phoneme_does_not_outrank_complete_pause():
    wave = np.ones(30 * SR, dtype=np.float32)
    wave[28 * SR:28 * SR + 320] = 0
    wave[22 * SR:23 * SR] = 0
    assert wide_cut(wave) == round(22.75 * SR)


@pytest.mark.parametrize("minimum", [15, 20])
def test_all_pcm_offsets_and_reader_block_invariance(minimum):
    wave = np.random.default_rng(22).uniform(-0.1, 0.1, 97 * SR + 23).astype(np.float32)
    wave[20 * SR:23 * SR] = 0
    reference = None
    for size in (6553, 29 * SR, 30 * SR):
        parts = list(rechunk_stream(blocks(wave, size), minimum_s=minimum))
        np.testing.assert_array_equal(np.concatenate([p[0] for p in parts]), wave)
        offset = 0
        spans = []
        for part, start in parts:
            assert round(start * SR) == offset and 0 < len(part) <= 30 * SR
            spans.append((offset, len(part)))
            offset += len(part)
        assert offset == len(wave)
        if reference is None:
            reference = spans
        else:
            assert spans == reference


@pytest.mark.parametrize("samples", [0, 13, 30 * SR, 30 * SR + 1])
def test_empty_short_exact_max_and_tail(samples):
    wave = np.random.default_rng(91).uniform(-0.2, 0.2, samples).astype(np.float32)
    parts = list(rechunk_stream(blocks(wave, 30 * SR)))
    assert sum(len(p[0]) for p in parts) == samples
    if samples == 30 * SR:
        assert len(parts) == 1 and len(parts[0][0]) == samples
    if samples == 30 * SR + 1:
        assert len(parts) == 2 and all(len(p[0]) <= 30 * SR for p in parts)


@pytest.mark.parametrize("failure", ["cancel", "offset", "nonfinite"])
def test_closes_upstream_on_cancellation_and_errors(failure):
    closed = []

    def source():
        try:
            for index in range(5):
                wave = np.zeros(30 * SR, dtype=np.float32)
                if failure == "nonfinite" and index == 1:
                    wave[0] = np.nan
                yield wave, index * 30 + (1 if failure == "offset" and index == 1 else 0)
        finally:
            closed.append(True)

    stream = rechunk_stream(source())
    if failure == "cancel":
        next(stream)
        stream.close()
    else:
        with pytest.raises(ValueError):
            list(stream)
    assert closed == [True]


@pytest.mark.parametrize("kwargs", [{"minimum_s": 30}, {"window_s": 0}, {"stride_s": 0},
                                    {"maximum_s": 31}, {"minimum_s": float("nan")}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        wide_cut(np.zeros(30 * SR, dtype=np.float32), **kwargs)
