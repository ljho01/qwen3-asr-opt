"""Scoped runtime switches, isolated per benchmark process; no dependency edits."""
from __future__ import annotations

import importlib
from functools import partial

import mlx.core as mx


def configure(decoder: str, cache_mb: int | None):
    pipeline = importlib.import_module("mlx_qwen3_asr.transcribe")
    if decoder in ("pipelined", "compiled"):
        from .decode import generate_pipelined
        pipeline.generate = partial(generate_pipelined, compiled=decoder == "compiled")
    if cache_mb is not None:
        if cache_mb < 0:
            raise ValueError("cache_mb must be nonnegative")
        mx.set_cache_limit(cache_mb * 1024 * 1024)
        # Upstream purges the allocator after every chunk. A bounded pool reuses
        # temporary allocations without retaining live per-chunk tensors.
        pipeline._clear_mlx_cache = lambda: None
