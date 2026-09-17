"""Experimental bounded-memory chunk boundaries using wider energy windows."""
from __future__ import annotations

import math

import numpy as np

from .longform import SAMPLE_RATE, audio_chunks


def _parameters(minimum_s, maximum_s, window_s, stride_s):
    values = (minimum_s, maximum_s, window_s, stride_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Chunk parameters must be finite")
    minimum, maximum, window, stride = [round(v * SAMPLE_RATE) for v in values]
    if not (1 <= window <= minimum < maximum <= 30 * SAMPLE_RATE and stride >= 1):
        raise ValueError("Require positive window/stride and window <= minimum < maximum <=30s")
    if minimum - window // 2 > maximum - window:
        raise ValueError("No complete energy window fits the search interval")
    return minimum, maximum, window, stride


def wide_cut(samples, *, minimum_s=20.0, maximum_s=30.0,
             window_s=0.5, stride_s=0.05):
    """Select the latest minimum-energy window center between minimum and maximum.

    The whole window must be inside the maximum-duration prefix. Energy affects
    only the boundary; returned PCM is never normalized, gated or discarded.
    """
    minimum, maximum, window, stride = _parameters(minimum_s, maximum_s, window_s, stride_s)
    if samples.ndim != 1 or len(samples) < maximum:
        raise ValueError("Require a mono buffer containing the maximum-duration prefix")
    prefix = samples[:maximum]
    if not np.isfinite(prefix).all():
        raise ValueError("Audio contains non-finite samples")
    starts = np.arange(minimum - window // 2, maximum - window + 1, stride)
    if not len(starts):
        raise ValueError("No complete energy window fits the search interval")
    squares = np.square(prefix, dtype=np.float64)
    cumulative = np.concatenate((np.zeros(1), np.cumsum(squares, dtype=np.float64)))
    energies = cumulative[starts + window] - cumulative[starts]
    index = len(energies) - 1 - int(np.argmin(energies[::-1]))
    return int(starts[index] + window // 2)


def rechunk_stream(source, *, minimum_s=20.0, maximum_s=30.0,
                   window_s=0.5, stride_s=0.05):
    """Rechunk contiguous PCM without reading or retaining a whole recording.

    Input blocks supplied by audio_chunks are <=30s. Pending output, an input
    block and the remaining prefix bound retained PCM independently of file length.
    The source is closed on exhaustion, early cancellation and failure.
    """
    # Validate parameters before consuming the stream, including empty inputs.
    _, maximum, _, _ = _parameters(minimum_s, maximum_s, window_s, stride_s)
    iterator = iter(source)
    buffer = np.empty(0, dtype=np.float32)
    total, base, pending = 0, 0, None
    try:
        for wave, offset in iterator:
            if wave.ndim != 1 or not np.isfinite(wave).all():
                raise ValueError("Expected finite mono PCM")
            if round(offset * SAMPLE_RATE) != total:
                raise ValueError("Noncontiguous decoded source")
            if not len(wave):
                continue
            total += len(wave)
            buffer = np.concatenate((buffer, wave))
            while len(buffer) > maximum:
                cut = wide_cut(buffer, minimum_s=minimum_s, maximum_s=maximum_s,
                               window_s=window_s, stride_s=stride_s)
                current = buffer[:cut].copy(), base / SAMPLE_RATE
                buffer = buffer[cut:]
                base += cut
                if pending is not None:
                    yield pending
                pending = current
        if len(buffer):
            if pending is not None and len(buffer) < SAMPLE_RATE and len(pending[0]) + len(buffer) <= maximum:
                pending = np.concatenate((pending[0], buffer)), pending[1]
            else:
                if pending is not None:
                    yield pending
                pending = buffer.copy(), base / SAMPLE_RATE
        if pending is not None:
            yield pending
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def wide_energy_chunks(path, *, minimum_s=20.0):
    """Experimental 500ms window, 50ms stride,30s maximum; no CLI default change."""
    yield from rechunk_stream(audio_chunks(path), minimum_s=minimum_s)
