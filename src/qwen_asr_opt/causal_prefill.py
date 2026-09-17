"""Experimental explicit causal SDPA for fresh, unpadded decoder prefixes."""
from __future__ import annotations

import mlx.core as mx


def prefill_causal(model, prompts, capacity):
    """Match serial prefill while replacing its additive mask with 'causal'.

    Each attention invocation has equal query/key lengths: only after prefill
    finishes is its cache padded to the caller's physical capacity. This is
    essential because MLX's causal mask is aligned to the lower-right corner.
    Preserve upstream embedding/audio validation, weights and the token decoder.
    No monkeypatch, persistent model mutation or default installation occurs here.
    """
    if not prompts or type(capacity) is not int or capacity < 1:
        raise ValueError("Require nonempty prompts and positive integer capacity")
    for ids, features, positions in prompts:
        if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= capacity:
            raise ValueError("Each prompt must have shape (1, positive length) fitting capacity")
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError("Audio features must have shape (1, audio length, width)")
        if positions.shape != (1, 3, ids.shape[1]):
            raise ValueError("Position shape must match the prompt")
    lengths = [ids.shape[1] for ids, _, _ in prompts]
    caches, first_tokens = [], []
    for (ids, features, positions), length in zip(prompts, lengths, strict=True):
        cache = model.create_cache(max_seq_len=length)
        embeds = model._embed_tokens(ids, validate_input_ids=True)
        embeds = model._inject_audio_features(embeds, features, ids == model.audio_token_id)
        hidden = model.model(inputs_embeds=embeds, position_ids=positions,
                             attention_mask="causal", cache=cache)
        token = mx.argmax(model.lm_head(hidden[:, -1:, :]), axis=-1)
        mx.eval(token, cache.keys, cache.values)
        caches.append(cache)
        first_tokens.append(token)

    def pack(name):
        return [mx.concatenate([
            mx.pad(getattr(cache, name)[layer],
                   [(0, 0), (0, 0), (0, capacity - length), (0, 0)])
            for cache, length in zip(caches, lengths, strict=True)
        ], axis=0) for layer in range(len(caches[0].keys))]

    keys, values = pack("keys"), pack("values")
    token = mx.concatenate(first_tokens, axis=0)
    mx.eval(token, keys, values)
    return token, keys, values
