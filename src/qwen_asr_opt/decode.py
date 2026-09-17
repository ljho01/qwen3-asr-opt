"""Greedy decode with one-token GPU/host overlap, preserving termination semantics."""
from __future__ import annotations

import mlx.core as mx
from mlx_qwen3_asr.generate import GenerationConfig, GenerationResult, detect_repetition


def generate_pipelined(model, input_ids, audio_features, position_ids, config=None, *, compiled=False):
    config = config or GenerationConfig()
    if config.temperature != 0:
        raise ValueError("Pipelined decoder is deterministic greedy only")
    if config.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if config.max_new_tokens == 0:
        return GenerationResult([], "length", 0, 0)
    prefix = input_ids.shape[1]
    capacity = prefix + config.max_new_tokens
    if compiled:
        capacity = ((capacity + 255) // 256) * 256
    cache = model.create_cache(max_seq_len=capacity)
    if compiled and not hasattr(model, "_optimized_step"):
        from .compiled import make_compiled_step
        model._optimized_step = make_compiled_step(model)
    logits = model.prefill(input_ids, audio_features, position_ids, cache)
    token = mx.argmax(logits.reshape(-1)).reshape(1, 1)
    mx.async_eval(token)
    positions = mx.broadcast_to(
        mx.arange(prefix, prefix + config.max_new_tokens, dtype=position_ids.dtype)[None, None, :],
        (1, 3, config.max_new_tokens),
    )
    tokens = []
    finish = "length"
    for index in range(config.max_new_tokens):
        # Schedule the following GPU step before synchronizing this token with Python.
        # At EOS this may compute one unused token. Drain it before returning so timing
        # and resource accounting include all work and no work leaks into the next clip.
        next_token = None
        if index + 1 < config.max_new_tokens:
            if compiled:
                next_token, cache.keys, cache.values = model._optimized_step(
                    token, positions[0, 0, index], cache.keys, cache.values,
                )
            else:
                logits = model.step(
                    token, positions[:, :, index:index + 1], cache, validate_input_ids=False,
                )
                next_token = mx.argmax(logits.reshape(-1)).reshape(1, 1)
            mx.async_eval(next_token, *[v for v in cache.keys + cache.values if v is not None])
        value = int(token.item())
        if value in config.eos_token_ids:
            finish = "eos"
            break
        tokens.append(value)
        # Match the reference decoder: at the token cap, length takes precedence.
        if len(tokens) < config.max_new_tokens and detect_repetition(tokens):
            finish = "repetition"
            break
        token = next_token
    mx.synchronize()
    return GenerationResult(tokens, finish, len(tokens), config.max_new_tokens)
