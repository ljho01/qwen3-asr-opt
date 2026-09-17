"""Experimental shared projections for one fresh prefix and ongoing token rows.

No scheduler or preset installs this helper. Attention stays independent per
recording. Only projections/norm/MLP/head combine their tokens into one matrix.
Different matrix kernels may round differently from independent greedy decoding.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_qwen3_asr.attention import _scaled_dot_product_attention
from mlx_qwen3_asr.decoder import _create_causal_mask
from mlx_qwen3_asr.mrope import apply_rotary_pos_emb

from .speculative import RowCache


class JointPrefillStep:
    """Immutable-weight helper returning fresh row first, then ongoing rows.

    Ongoing positions are logical contiguous KV write positions. Callers must
    provide in-bounds positions and exclude completed/inactive streams. Existing
    KV storage is not mutated: every returned row is a functional array update.
    Prefix capacity and ongoing physical capacity must match. No context is lost.
    """

    def __init__(self, model, *, compiled=False):
        self.model = model
        self.function = mx.compile(self._forward) if compiled else self._forward

    def _forward(self, fresh, fresh_positions, tokens, positions, keys, values):
        model = self.model
        length, rows, width = fresh.shape[1], tokens.shape[0], keys[0].shape[2]
        hidden = mx.concatenate([fresh, model.model.embed_tokens(tokens).reshape(1, rows, -1)], axis=1)
        ongoing_positions = mx.broadcast_to(positions[None, None, :], (1, 3, rows))
        position_ids = mx.concatenate([fresh_positions, ongoing_positions], axis=2)
        cos, sin = model.model.rotary_emb(position_ids, dtype=hidden.dtype)
        fresh_mask = _create_causal_mask(length, hidden.dtype)
        ongoing_mask = mx.arange(width)[None, None, None, :] <= positions[:, None, None, None]
        cache = RowCache(keys, values, positions, "scatter")
        fresh_keys, fresh_values = [], []
        for index, layer in enumerate(model.model.layers):
            residual = hidden
            norm = layer.input_layernorm(hidden)
            attention = layer.self_attn
            total = length + rows
            q = attention.q_norm(attention.q_proj(norm).reshape(1, total, attention.num_heads, attention.head_dim))
            k = attention.k_norm(attention.k_proj(norm).reshape(1, total, attention.num_kv_heads, attention.head_dim))
            v = attention.v_proj(norm).reshape(1, total, attention.num_kv_heads, attention.head_dim)
            q, k = apply_rotary_pos_emb(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), cos, sin)
            v = v.transpose(0, 2, 1, 3)
            fk, fv = k[:, :, :length], v[:, :, :length]
            fresh_keys.append(mx.pad(fk, [(0, 0), (0, 0), (0, width-length), (0, 0)]))
            fresh_values.append(mx.pad(fv, [(0, 0), (0, 0), (0, width-length), (0, 0)]))
            fresh_out = _scaled_dot_product_attention(q[:, :, :length], fk, fv, mask=fresh_mask)
            dk, dv = k[:, :, length:].transpose(2, 1, 0, 3), v[:, :, length:].transpose(2, 1, 0, 3)
            dk, dv = cache.update(dk, dv, index)
            decode_out = _scaled_dot_product_attention(q[:, :, length:].transpose(2, 1, 0, 3), dk, dv, mask=ongoing_mask)
            attended = mx.concatenate([fresh_out, decode_out.transpose(2, 1, 0, 3)], axis=2)
            attended = attended.transpose(0, 2, 1, 3).reshape(1, total, -1)
            hidden = residual + attention.o_proj(attended)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        # Only the fresh prefix's last position and each ongoing token need logits.
        last = mx.concatenate([hidden[:, length-1:length], hidden[:, length:]], axis=1)
        last = model.model.norm(last).reshape(rows+1, 1, -1)
        token = mx.argmax(model.lm_head(last), axis=-1)
        result_keys = [mx.concatenate([f, d], axis=0) for f, d in zip(fresh_keys, cache.keys, strict=True)]
        result_values = [mx.concatenate([f, d], axis=0) for f, d in zip(fresh_values, cache.values, strict=True)]
        return token, result_keys, result_values

    def __call__(self, prompt, tokens, positions, keys, values):
        ids, features, rope = prompt
        if not keys or len(keys) != len(values) or len(keys) != len(self.model.model.layers):
            raise ValueError("Require one key/value tensor per decoder layer")
        rows, _, width, _ = keys[0].shape
        if rows < 1 or tokens.shape != (rows, 1) or positions.shape != (rows,):
            raise ValueError("Require nonempty ongoing token rows and per-row positions")
        if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= width:
            raise ValueError("Fresh prompt must fit the shared KV capacity")
        if rope.shape != (1, 3, ids.shape[1]):
            raise ValueError("Fresh position IDs must match its prompt")
        if any(k.shape != keys[0].shape for k in keys + values):
            raise ValueError("All ongoing KV tensors must share their shape")
        hidden = self.model._embed_tokens(ids, validate_input_ids=True)
        hidden = self.model._inject_audio_features(hidden, features, ids == self.model.audio_token_id)
        return self.function(hidden, rope, tokens, positions, keys, values)
