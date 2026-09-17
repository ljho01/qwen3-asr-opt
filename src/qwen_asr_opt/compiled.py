"""Compile the single-token decoder with a fixed-capacity, dynamically indexed KV cache."""
from __future__ import annotations

import mlx.core as mx


class ArrayCache:
    """Pure-array cache update contract consumed by the existing attention layers."""

    def __init__(self, keys, values, position):
        self.keys = list(keys)
        self.values = list(values)
        self.position = position
        # Existing decoder uses offset only to choose single-token causal semantics.
        self.offset = 1

    def update(self, key, value, layer_idx):
        start = self.position.reshape(1).astype(mx.int32)
        self.keys[layer_idx] = mx.slice_update(self.keys[layer_idx], key, start, axes=[2])
        self.values[layer_idx] = mx.slice_update(self.values[layer_idx], value, start, axes=[2])
        return self.keys[layer_idx], self.values[layer_idx]


def make_compiled_step(model):
    def step(token, position, keys, values):
        cache = ArrayCache(keys, values, position)
        mask = mx.arange(keys[0].shape[2]) <= position
        hidden = model.model(
            inputs_embeds=model.model.embed_tokens(token),
            position_ids=mx.broadcast_to(position.reshape(1, 1, 1), (1, 3, 1)),
            attention_mask=mask.reshape(1, 1, 1, -1), cache=cache,
        )
        token = mx.argmax(model.lm_head(hidden).reshape(-1)).reshape(1, 1)
        return token, cache.keys, cache.values
    return mx.compile(step)
