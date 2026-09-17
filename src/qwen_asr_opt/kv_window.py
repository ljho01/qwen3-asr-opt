"""Experimental contiguous per-row KV positions and bounded attention reads."""
from __future__ import annotations

import mlx.core as mx

from .speculative import RowCache


def attention_width(last_position: int, capacity: int, quantum: int = 128) -> int:
    """Include the current write and every earlier token, without dropping context."""
    if capacity < 1 or quantum < 1 or not 0 <= last_position < capacity:
        raise ValueError("Attention position must fit the positive cache capacity")
    return min(capacity, ((last_position + 1 + quantum - 1) // quantum) * quantum)


class CompactRowCache(RowCache):
    def __init__(self, keys, values, starts, width):
        super().__init__(keys, values, starts, "scatter")
        self.width = width

    def update(self, key, value, layer_idx):
        keys, values = super().update(key, value, layer_idx)
        # Retain full-capacity arrays for future writes/refills. Only the attention
        # views are narrowed; no tokens are evicted and no history is quantized.
        return keys[:, :, :self.width], values[:, :, :self.width]


def make_compact_step(model, width):
    if width < 1:
        raise ValueError("Attention width must be positive")

    def step(tokens, positions, keys, values):
        if tokens.shape[1] != 1 or width > keys[0].shape[2]:
            raise ValueError("Expected one token per row and in-bounds attention width")
        cache = CompactRowCache(keys, values, positions, width)
        slots = mx.arange(width)[None, None, None, :]
        valid = slots <= positions[:, None, None, None]
        rope = mx.broadcast_to(positions[:, None, None], (tokens.shape[0], 3, 1))
        hidden = model.model(inputs_embeds=model.model.embed_tokens(tokens),
                             position_ids=rope, attention_mask=valid, cache=cache)
        return mx.argmax(model.lm_head(hidden), axis=-1), cache.keys, cache.values

    return mx.compile(step)
