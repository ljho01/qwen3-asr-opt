"""Batch independent decoder streams while retaining each recording's prompt and KV state."""
from __future__ import annotations

import mlx.core as mx
from mlx_qwen3_asr.audio import compute_features, load_audio_np
from mlx_qwen3_asr.generate import (
    GenerationConfig,
    GenerationResult,
    detect_repetition,
    resolve_max_new_tokens,
)
from mlx_qwen3_asr.tokenizer import canonicalize_language, parse_asr_output
from mlx_qwen3_asr.transcribe import TranscriptionResult

from .compiled import ArrayCache


def make_batch_step(model):
    def step(tokens, logical_positions, write_position, prefix_lengths, shared_prefix, keys, values):
        cache = ArrayCache(keys, values, write_position)
        slots = mx.arange(keys[0].shape[2])[None, :]
        valid = ((slots < prefix_lengths[:, None])
                 | ((slots >= shared_prefix) & (slots <= write_position)))
        positions = mx.broadcast_to(logical_positions[:, None, None], (tokens.shape[0], 3, 1))
        hidden = model.model(inputs_embeds=model.model.embed_tokens(tokens),
                             position_ids=positions,
                             attention_mask=valid[:, None, None, :], cache=cache)
        return mx.argmax(model.lm_head(hidden), axis=-1), cache.keys, cache.values
    return mx.compile(step)


def prefill_serial(model, prompts, capacity):
    """Reference path: prefill each prompt independently, then pack its cache."""
    lengths = [ids.shape[1] for ids, _, _ in prompts]
    caches, first_tokens = [], []
    for (ids, features, positions), length in zip(prompts, lengths, strict=True):
        cache = model.create_cache(max_seq_len=length)
        token = mx.argmax(model.prefill(ids, features, positions, cache), axis=-1)
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


def prefill_batched(model, prompts, capacity):
    """Right-pad prompts, inject independent audio, and evaluate the decoder once.

    Causal attention prevents valid prefix queries from seeing right padding. The
    token loop's existing per-row masks exclude the padded KV slots afterwards.
    Gather the final VALID hidden state in each row, never the shared padded end.
    """
    lengths = [ids.shape[1] for ids, _, _ in prompts]
    shared_prefix = max(lengths)
    audio_width = max(features.shape[1] for _, features, _ in prompts)
    if capacity < shared_prefix or min(lengths) < 1:
        raise ValueError("Cache capacity must cover nonempty prompts")
    ids = mx.concatenate([mx.pad(ids, [(0, 0), (0, shared_prefix - length)])
                          for (ids, _, _), length in zip(prompts, lengths, strict=True)], axis=0)
    features = mx.concatenate([mx.pad(features, [(0, 0), (0, audio_width - features.shape[1]), (0, 0)])
                               for _, features, _ in prompts], axis=0)
    positions = mx.concatenate([mx.pad(pos, [(0, 0), (0, 0), (0, shared_prefix - length)])
                                for (_, _, pos), length in zip(prompts, lengths, strict=True)], axis=0)
    embeds = model._embed_tokens(ids, validate_input_ids=True)
    embeds = model._inject_audio_features(embeds, features, ids == model.audio_token_id)
    cache = model.create_cache(max_seq_len=capacity)
    hidden = model.model(inputs_embeds=embeds, position_ids=positions, cache=cache)
    last = hidden[mx.arange(len(prompts)), mx.array(lengths) - 1][:, None, :]
    token = mx.argmax(model.lm_head(last), axis=-1)
    mx.eval(token, cache.keys, cache.values)
    return token, cache.keys, cache.values


def generate_batch(model, prompts, configs, prefill_mode="serial"):
    """Share weight reads during batched greedy token steps and optionally prefill.

    prompts contains (input_ids, audio_features, position_ids) per independent input.
    Gaps after shorter prompts are masked; RoPE uses each input's unpadded position.
    Finished rows remain allocated until the batch completes, never emitting more text.
    """
    if not prompts or len(prompts) != len(configs):
        raise ValueError("Expected one generation config per nonempty batch item")
    if any(c.temperature != 0 or c.max_new_tokens < 1 for c in configs):
        raise ValueError("Batch decoding requires greedy generation and positive token budgets")
    if prefill_mode not in ("serial", "batched"):
        raise ValueError("prefill_mode must be serial or batched")
    lengths = [ids.shape[1] for ids, _, _ in prompts]
    shared_prefix = max(lengths)
    max_tokens = max(c.max_new_tokens for c in configs)
    capacity = ((shared_prefix + max_tokens + 255) // 256) * 256
    prefill = prefill_batched if prefill_mode == "batched" else prefill_serial
    token, keys, values = prefill(model, prompts, capacity)
    if not hasattr(model, "_optimized_batch_step"):
        model._optimized_batch_step = make_batch_step(model)
    prefix_lengths = mx.array(lengths, dtype=mx.int32)
    common = mx.array(shared_prefix, dtype=mx.int32)
    tokens = [[] for _ in configs]
    finishes = [None] * len(configs)
    for index in range(max_tokens):
        next_token = None
        if index + 1 < max_tokens:
            next_token, keys, values = model._optimized_batch_step(
                token, prefix_lengths + index, common + index, prefix_lengths, common,
                keys, values,
            )
            mx.async_eval(next_token, keys, values)
        for row, value in enumerate(token.reshape(-1).tolist()):
            if finishes[row] is not None:
                continue
            config = configs[row]
            if value in config.eos_token_ids:
                finishes[row] = "eos"
            else:
                tokens[row].append(value)
                if len(tokens[row]) >= config.max_new_tokens:
                    finishes[row] = "length"
                elif detect_repetition(tokens[row]):
                    finishes[row] = "repetition"
        if all(reason is not None for reason in finishes):
            break
        token = next_token
    mx.synchronize()
    return [GenerationResult(t, reason, len(t), config.max_new_tokens)
            for t, reason, config in zip(tokens, finishes, configs, strict=True)]


def transcribe_batch(session, audio, languages):
    """Transcribe up to eight independent chunks of at most 30 seconds each."""
    if not 1 <= len(audio) <= 8 or len(audio) != len(languages):
        raise ValueError("Batch size must be 1..8 with one language hint per input")
    prompts, configs, durations, hints = [], [], [], []
    for item, language in zip(audio, languages, strict=True):
        wave = load_audio_np(item)
        if not 0 < len(wave) <= 30 * 16000:
            raise ValueError("Batch inputs must contain 0 < audio <= 30 seconds; use --long")
        duration = len(wave) / 16000
        hint = canonicalize_language(language) if language else None
        mel, lengths = compute_features(wave)
        features, _ = session.model.audio_tower(mel.astype(mx.float16), lengths)
        ids = mx.array([session.tokenizer.build_prompt_tokens(features.shape[1], language=hint)])
        positions = mx.broadcast_to(mx.arange(ids.shape[1])[None, None, :], (1, 3, ids.shape[1]))
        prompts.append((ids, features, positions))
        configs.append(GenerationConfig(max_new_tokens=resolve_max_new_tokens(
            None, audio_duration_sec=duration)))
        durations.append(duration)
        hints.append(hint)
    generated = generate_batch(session.model, prompts, configs,
                               prefill_mode=getattr(session.model, "_batch_prefill_mode", "serial"))
    results = []
    for generation, duration, hint in zip(generated, durations, hints, strict=True):
        language, text = parse_asr_output(session.tokenizer.decode(generation.tokens),
                                         user_language=hint)
        language = canonicalize_language(language) or language
        chunk = {"text": text, "start": 0.0, "end": duration, "chunk_index": 0,
                 "language": language, "finish_reason": generation.finish_reason,
                 "truncated": generation.truncated, "generated_tokens": generation.generated_tokens,
                 "max_new_tokens": generation.max_new_tokens}
        results.append(TranscriptionResult(text=text, language=language, chunks=[chunk],
                                          finish_reason=generation.finish_reason,
                                          truncated=generation.truncated))
    return results
