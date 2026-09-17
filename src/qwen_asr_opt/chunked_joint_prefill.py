"""Experimental balanced prefix chunks sharing projections with ongoing decodes.

Algebraic probe only. The caller must handle EOS/cancellation between chunks in a
real scheduler. This bounded helper runs a fixed number of steps and returns all
consumed ongoing tokens so experiments can reject an early-finished fixture.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_qwen3_asr.attention import _scaled_dot_product_attention
from mlx_qwen3_asr.mrope import apply_rotary_pos_emb

from .speculative import RowCache


def balanced_ranges(length, maximum):
    if type(length) is not int or type(maximum) is not int or min(length, maximum) < 1:
        raise ValueError("Require positive integer lengths")
    count = (length+maximum-1)//maximum
    base, remainder = divmod(length, count)
    start, result = 0, []
    for i in range(count):
        end = start + base + (i < remainder)
        result.append((start, end))
        start = end
    return result


class ChunkedJointPrefill:
    def __init__(self, model, *, maximum_chunk=128):
        balanced_ranges(1, maximum_chunk)
        self.model, self.maximum_chunk = model, maximum_chunk
        self.function = mx.compile(self._forward)

    def _forward(self, fresh, fresh_positions, fresh_start, tokens, positions, keys, values):
        model = self.model
        length, rows, width = fresh.shape[1], tokens.shape[0], keys[0].shape[2]
        hidden = mx.concatenate([fresh, model.model.embed_tokens(tokens).reshape(1, rows, -1)], axis=1)
        decode_positions = mx.broadcast_to(positions[None, None, :], (1, 3, rows))
        cos, sin = model.model.rotary_emb(mx.concatenate([fresh_positions, decode_positions], axis=2), dtype=hidden.dtype)
        slots = mx.arange(width)[None, None, None, :]
        fresh_mask = slots <= (fresh_start+mx.arange(length))[None, None, :, None]
        ongoing_mask = slots <= positions[:, None, None, None]
        cache = RowCache([k[1:] for k in keys], [v[1:] for v in values], positions, "scatter")
        result_keys, result_values = [], []
        for index, layer in enumerate(model.model.layers):
            residual = hidden
            norm = layer.input_layernorm(hidden)
            attention = layer.self_attn
            total = length+rows
            q = attention.q_norm(attention.q_proj(norm).reshape(1, total, attention.num_heads, attention.head_dim))
            k = attention.k_norm(attention.k_proj(norm).reshape(1, total, attention.num_kv_heads, attention.head_dim))
            v = attention.v_proj(norm).reshape(1, total, attention.num_kv_heads, attention.head_dim)
            q, k = apply_rotary_pos_emb(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), cos, sin)
            v = v.transpose(0, 2, 1, 3)
            fk = mx.slice_update(keys[index][:1], k[:, :, :length], fresh_start.reshape(1), axes=[2])
            fv = mx.slice_update(values[index][:1], v[:, :, :length], fresh_start.reshape(1), axes=[2])
            fresh_out = _scaled_dot_product_attention(q[:, :, :length], fk, fv, mask=fresh_mask)
            dk, dv = cache.update(k[:, :, length:].transpose(2, 1, 0, 3), v[:, :, length:].transpose(2, 1, 0, 3), index)
            decode_out = _scaled_dot_product_attention(q[:, :, length:].transpose(2, 1, 0, 3), dk, dv, mask=ongoing_mask)
            result_keys.append(mx.concatenate([fk, dk], axis=0))
            result_values.append(mx.concatenate([fv, dv], axis=0))
            attended = mx.concatenate([fresh_out, decode_out.transpose(2, 1, 0, 3)], axis=2)
            attended = attended.transpose(0, 2, 1, 3).reshape(1, total, -1)
            hidden = residual + attention.o_proj(attended)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        last = mx.concatenate([hidden[:, length-1:length], hidden[:, length:]], axis=1)
        token = mx.argmax(model.lm_head(model.model.norm(last).reshape(rows+1, 1, -1)), axis=-1)
        return token, result_keys, result_values

    def __call__(self, prompt, tokens, positions, keys, values):
        ids, features, rope = prompt
        if not keys or len(keys) != len(values) or len(keys) != len(self.model.model.layers):
            raise ValueError("Require one KV tensor per model layer")
        rows, _, capacity, _ = keys[0].shape
        if rows < 1 or tokens.shape != (rows, 1) or positions.shape != (rows,):
            raise ValueError("Expected nonempty ongoing rows and positions")
        if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= capacity:
            raise ValueError("Fresh prefix must fit capacity")
        if rope.shape != (1, 3, ids.shape[1]) or any(v.shape != keys[0].shape for v in keys+values):
            raise ValueError("Inconsistent position/cache shape")
        hidden = self.model._embed_tokens(ids, validate_input_ids=True)
        hidden = self.model._inject_audio_features(hidden, features, ids == self.model.audio_token_id)
        keys = [mx.concatenate([mx.zeros_like(k[:1]), k], axis=0) for k in keys]
        values = [mx.concatenate([mx.zeros_like(v[:1]), v], axis=0) for v in values]
        consumed = []
        for start, end in balanced_ranges(ids.shape[1], self.maximum_chunk):
            consumed.append(tokens)
            next_tokens, keys, values = self.function(hidden[:, start:end], rope[:, :, start:end],
                mx.array(start, dtype=mx.int32), tokens, positions, keys, values)
            mx.async_eval(next_tokens, keys, values)
            tokens, positions = next_tokens[1:], positions+1
        return next_tokens, keys, values, mx.stack(consumed)
