"""Experimental prefill computing only the final layer's final query/output.

All prefix K/V entries are retained at every layer. No weights are changed and
no token is removed from attention. Different matmul/attention shapes can still
change floating-point rounding, so actual recognition requires verification.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_qwen3_asr.attention import _scaled_dot_product_attention
from mlx_qwen3_asr.decoder import _create_causal_mask
from mlx_qwen3_asr.mrope import _rotate_half

from .compiled_prefill import PrefixPrefill


def terminal_prefix_outputs(model, embeds, positions):
    """Fresh prefix only: return final logits plus complete per-layer K/V.

    Layers before the final one process the whole prefix. In the final layer,
    every position needs K/V for later decode, but only the last query, output
    projection, MLP and final norm/head are consumed by the first-token decision.
    Its single query can attend to every supplied prefix key, hence mask=None.
    This is not valid as a replacement for arbitrary full hidden-state output.
    """
    batch, length, _ = embeds.shape
    cache = model.create_cache()
    cos, sin = model.model.rotary_emb(positions, dtype=embeds.dtype)
    mask = _create_causal_mask(length, embeds.dtype)
    hidden = embeds
    for index, layer in enumerate(model.model.layers[:-1]):
        hidden = layer(hidden, cos, sin, mask=mask, cache=cache, layer_idx=index)
    layer = model.model.layers[-1]
    attention = layer.self_attn
    norm = layer.input_layernorm(hidden)
    query = attention.q_norm(attention.q_proj(norm[:, -1:]).reshape(
        batch, 1, attention.num_heads, attention.head_dim)).transpose(0, 2, 1, 3)
    key = attention.k_norm(attention.k_proj(norm).reshape(
        batch, length, attention.num_kv_heads, attention.head_dim)).transpose(0, 2, 1, 3)
    value = attention.v_proj(norm).reshape(
        batch, length, attention.num_kv_heads, attention.head_dim).transpose(0, 2, 1, 3)
    query = query*cos[:, None, -1:] + _rotate_half(query)*sin[:, None, -1:]
    key = key*cos[:, None] + _rotate_half(key)*sin[:, None]
    key, value = cache.update(key, value, len(model.model.layers)-1)
    attended = _scaled_dot_product_attention(query, key, value, mask=None)
    attended = attended.transpose(0, 2, 1, 3).reshape(batch, 1, -1)
    last = hidden[:, -1:] + attention.o_proj(attended)
    last = last + layer.mlp(layer.post_attention_layernorm(last))
    return model.lm_head(model.model.norm(last)), cache.keys, cache.values


class TerminalPrefill(PrefixPrefill):
    """Use existing validated prompt preparation/cache packing with a new graph.

    For immutable inference models only; construct a fresh helper after any
    module/config/dtype/weight change. Exact shape/dtype LRU bounds retained Python
    callables, not every Metal/driver cache. Never installs itself into the model.
    """

    def __init__(self, model, *, compiled=True, max_graphs=32):
        if type(max_graphs) is not int or max_graphs < 1 or not model.model.layers:
            raise ValueError("Require decoder layers and a positive integer cache bound")
        super().__init__(model, compiled=compiled, max_graphs=max_graphs)
        self.evictions = 0

    def _function(self, embeds, positions):
        key = (embeds.shape, embeds.dtype, positions.shape, positions.dtype)
        if self.compiled and key in self.graphs:
            self.cache_hits += 1
            self.graphs.move_to_end(key)
            return self.graphs[key]
        model = self.model

        def function(hidden, position_ids):
            logits, keys, values = terminal_prefix_outputs(model, hidden, position_ids)
            return mx.argmax(logits, axis=-1), keys, values

        if not self.compiled:
            return function
        function = mx.compile(function, shapeless=False)
        if len(self.graphs) == self.max_graphs:
            self.graphs.popitem(last=False)
            self.evictions += 1
        self.graphs[key] = function
        self.graphs_created += 1
        return function

    def statistics(self):
        return {**super().statistics(), "evictions": self.evictions,
                "last_layer_query_output_only": True}
