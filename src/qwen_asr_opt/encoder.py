"""Batch independent audio-attention windows without cross-window attention."""
from __future__ import annotations

from itertools import pairwise

import mlx.core as mx


def batched_window_layers(x, layers, cu_seqlens):
    """Replace block-diagonal attention with a batch of padded independent windows.

    Positions were added per convolution chunk before this function. Only keys in
    the final short window need masking; padding queries are discarded afterwards.
    The boundary contract is the Apache-2.0 mlx-qwen3-asr encoder helper's contract.
    """
    if (len(cu_seqlens) < 2 or x.shape[0] != 1 or cu_seqlens[0] != 0
            or cu_seqlens[-1] != x.shape[1]):
        raise ValueError("Expected one sequence and complete window boundaries")
    lengths = [end - start for start, end in pairwise(cu_seqlens)]
    if not lengths or min(lengths) <= 0:
        raise ValueError("Attention windows must be nonempty")
    if len(lengths) == 1:
        for layer in layers:
            x = layer(x, mask=None)
        return x
    width = max(lengths)
    windows = mx.concatenate([
        mx.pad(x[:, start:end], [(0, 0), (0, width - length), (0, 0)])
        for start, end, length in zip(cu_seqlens[:-1], cu_seqlens[1:], lengths, strict=True)
    ], axis=0)
    valid = mx.arange(width)[None, :] < mx.array(lengths)[:, None]
    mask = mx.where(valid, mx.array(0, dtype=x.dtype), mx.array(mx.finfo(x.dtype).min, dtype=x.dtype))
    for layer in layers:
        windows = layer(windows, mask=mask[:, None, None, :])
    return mx.concatenate([windows[i:i + 1, :length] for i, length in enumerate(lengths)], axis=1)


def configure_windowed_encoder():
    """Install only within this process; dependency files remain untouched."""
    import mlx_qwen3_asr.encoder as upstream
    upstream._WINDOWED_SEGMENT_MIN_WINDOWS = 1
    upstream._apply_windowed_encoder_layers = batched_window_layers
