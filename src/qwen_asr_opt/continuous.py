"""Experimental bounded-slot decoding that refills each completed row immediately."""
from __future__ import annotations

import time

import mlx.core as mx
from mlx_qwen3_asr.audio import compute_features
from mlx_qwen3_asr.generate import (
    GenerationConfig,
    GenerationResult,
    detect_repetition,
    resolve_max_new_tokens,
)
from mlx_qwen3_asr.tokenizer import canonicalize_language, parse_asr_output

from .batch import prefill_serial
from .speculative import make_block_verifier


def generate_continuous(model, jobs, batch_size=4, *, shared_prefix=512, token_budget=512,
                        on_complete=None, kv_policy="padded"):
    """Consume (prompt, GenerationConfig) lazily and return results in input order.

    Only `batch_size` KV streams are live. Prompt creation can include each next
    audio encoder invocation; no complete long recording must be resident. Every
    row keeps its own prompt length, position, token limit and termination state.
    By default the shared physical generation offset is fixed, with padding masked
    per row. Experimental compact policies write immediately after each row's
    prompt; growing128 expands the allocated cache only as positions require it.
    Experimental adaptive128/adaptive64 start from observed prompt lengths and
    grow in those bucket sizes, retaining the same maximum capacity and history.
    An optional callback receives (input_index, result) as each job finishes, which
    allows the caller to persist progress without waiting for the entire stream.
    """
    if not 1 <= batch_size <= 8 or shared_prefix < 1 or token_budget < 1:
        raise ValueError("Require batch_size=1..8 and positive prefix/token capacities")
    if kv_policy not in ("padded", "compact", "compact128", "growing128",
                         "adaptive128", "adaptive64"):
        raise ValueError("Unsupported continuous KV policy")
    adaptive = kv_policy in ("adaptive128", "adaptive64")
    quantum = 64 if kv_policy == "adaptive64" else 128
    stream = iter(jobs)
    maximum_capacity = ((shared_prefix + token_budget + 255) // 256) * 256
    capacity = (((shared_prefix + 127) // 128) * 128
                if kv_policy == "growing128" else maximum_capacity)
    if adaptive:
        capacity = 0
    keys = values = None
    lengths = [1] * batch_size
    counts = [0] * batch_size
    slots = [None] * batch_size
    outputs = []
    exhausted = False
    stats = {"decode_calls": 0, "active_row_steps": 0, "allocated_row_steps": 0,
             "prefills": 0, "refills": 0, "shared_prefix": shared_prefix,
             "cache_capacity": capacity, "peak_active_rows": 0,
             "kv_policy": kv_policy, "attention_width_calls": {},
             "maximum_cache_capacity": maximum_capacity,
             "cache_capacity_history": [] if adaptive else [capacity], "prompt_lengths": []}

    def ensure_capacity(last_position):
        nonlocal capacity, keys, values
        if last_position < capacity:
            return
        from .kv_window import attention_width

        new_capacity = attention_width(last_position, maximum_capacity, quantum)
        if keys is not None:
            padding = [(0, 0), (0, 0), (0, new_capacity - capacity), (0, 0)]
            keys = [mx.pad(array, padding) for array in keys]
            values = [mx.pad(array, padding) for array in values]
        capacity = new_capacity
        stats["cache_capacity"] = capacity
        stats["cache_capacity_history"].append(capacity)

    def next_job():
        nonlocal exhausted
        if exhausted:
            return None
        try:
            prompt, config = next(stream)
        except StopIteration:
            exhausted = True
            return None
        if config.temperature != 0 or not 1 <= config.max_new_tokens <= token_budget:
            raise ValueError("Require greedy generation and token cap within pool capacity")
        length = prompt[0].shape[1]
        if not 1 <= length <= shared_prefix:
            raise ValueError(f"Prompt length {length} exceeds fixed prefix capacity {shared_prefix}")
        if adaptive:
            # Refill can introduce a prompt larger than the currently allocated
            # pool. Grow every live row before packing/replacing only this row.
            ensure_capacity(length)
        token, new_keys, new_values = prefill_serial(model, [prompt], capacity)
        identity = len(outputs)
        outputs.append(None)
        stats["prefills"] += 1
        stats["prompt_lengths"].append(length)
        return {"id": identity, "config": config, "tokens": []}, length, token, new_keys, new_values

    initial = [next_job() for _ in range(batch_size)]
    if initial[0] is None:
        return [], stats
    prototype = initial[0]
    if adaptive:
        # Sequential initial prefills may have used different capacities. Keep
        # their valid KV entries and pad each to the final initial pool capacity.
        for prepared in initial:
            if prepared is not None:
                for arrays in prepared[3:]:
                    for layer, array in enumerate(arrays):
                        arrays[layer] = mx.pad(array, [(0, 0), (0, 0),
                                                      (0, capacity - array.shape[2]), (0, 0)])
        del prepared, arrays, array, layer
    first_tokens, key_rows, value_rows = [], [], []
    for row, job in enumerate(initial):
        if job is None:
            first_tokens.append(mx.zeros_like(prototype[2]))
            key_rows.append([mx.zeros_like(value) for value in prototype[3]])
            value_rows.append([mx.zeros_like(value) for value in prototype[4]])
        else:
            slots[row], lengths[row], token, new_keys, new_values = job
            first_tokens.append(token)
            key_rows.append(new_keys)
            value_rows.append(new_values)
    keys = [mx.concatenate([values[layer] for values in key_rows], axis=0)
            for layer in range(len(prototype[3]))]
    values = [mx.concatenate([items[layer] for items in value_rows], axis=0)
              for layer in range(len(prototype[4]))]
    token = mx.concatenate(first_tokens, axis=0)
    mx.eval(token, keys, values)
    del initial, prototype, first_tokens, key_rows, value_rows, job, new_keys, new_values
    methods = getattr(model, "_continuous_steps", None)
    if methods is None:
        methods = {}
        model._continuous_steps = methods
    if kv_policy == "padded" and batch_size not in methods:
        methods[batch_size] = make_block_verifier(model, "scatter")
    common = mx.array(shared_prefix, dtype=mx.int32)
    while any(slot is not None for slot in slots):
        active = sum(slot is not None for slot in slots)
        stats["peak_active_rows"] = max(stats["peak_active_rows"], active)
        stats["active_row_steps"] += active
        stats["allocated_row_steps"] += batch_size
        width = capacity
        if kv_policy == "padded":
            next_token, keys, values = methods[batch_size](
                token, common + mx.array(counts, dtype=mx.int32),
                mx.array(lengths, dtype=mx.int32), common, keys, values)
        else:
            from .kv_window import attention_width, make_compact_step

            positions = [length + count if slot is not None else 0
                         for length, count, slot in zip(lengths, counts, slots, strict=True)]
            if kv_policy == "growing128" or adaptive:
                ensure_capacity(max(positions))
            width = capacity
            if kv_policy == "compact128":
                width = attention_width(max(positions), capacity)
            key = (batch_size, "compact", width)
            if key not in methods:
                methods[key] = make_compact_step(model, width)
            next_token, keys, values = methods[key](
                token, mx.array(positions, dtype=mx.int32), keys, values)
        stats["attention_width_calls"][str(width)] = stats["attention_width_calls"].get(str(width), 0) + 1
        mx.async_eval(next_token, keys, values)
        stats["decode_calls"] += 1
        completed = []
        for row, value in enumerate(token.reshape(-1).tolist()):
            slot = slots[row]
            if slot is None:
                continue
            config = slot["config"]
            reason = None
            if value in config.eos_token_ids:
                reason = "eos"
            else:
                slot["tokens"].append(value)
                if len(slot["tokens"]) >= config.max_new_tokens:
                    reason = "length"
                elif detect_repetition(slot["tokens"]):
                    reason = "repetition"
            counts[row] += 1
            if reason is not None:
                output = slot["tokens"]
                outputs[slot["id"]] = GenerationResult(output, reason, len(output), config.max_new_tokens)
                if on_complete is not None:
                    on_complete(slot["id"], outputs[slot["id"]])
                slots[row] = None
                completed.append(row)
        for row in completed:
            job = next_job()
            if job is None:
                # Keep unused rows at an in-bounds position; never emit them.
                counts[row] = 0
                continue
            slots[row], lengths[row], new_token, new_keys, new_values = job
            counts[row] = 0
            offset = mx.array([row], dtype=mx.int32)
            keys = [mx.slice_update(old, new, offset, axes=[0])
                    for old, new in zip(keys, new_keys, strict=True)]
            values = [mx.slice_update(old, new, offset, axes=[0])
                      for old, new in zip(values, new_values, strict=True)]
            next_token = mx.slice_update(next_token, new_token, offset, axes=[0])
            mx.async_eval(next_token, keys, values)
            stats["refills"] += 1
            del job, new_token, new_keys, new_values
        token = next_token
    mx.synchronize()
    stats["active_fraction"] = stats["active_row_steps"] / stats["allocated_row_steps"]
    return outputs, stats


def transcribe_continuous_chunks(session, chunks, language, batch_size, on_complete, *,
                                 kv_policy="padded"):
    """Consume waveform/offset pairs and emit indexed rows in completion order.

    Row elapsed time is overlapping chunk latency, not an additive allocation of
    total runtime. Only a fixed number of audio/KV streams remain live at once.
    """
    hint = canonicalize_language(language) if language else None
    metadata = []

    def jobs():
        for index, (wave, offset) in enumerate(chunks):
            if not 0 < len(wave) <= 30 * 16000:
                raise ValueError("Continuous chunks require 0 < audio <= 30 seconds")
            duration = len(wave) / 16000
            metadata.append({"index": index, "start": offset, "end": offset + duration,
                             "started": time.perf_counter()})
            mel, lengths = compute_features(wave)
            features, _ = session.model.audio_tower(mel.astype(mx.float16), lengths)
            ids = mx.array([session.tokenizer.build_prompt_tokens(features.shape[1], language=hint)])
            positions = mx.broadcast_to(mx.arange(ids.shape[1])[None, None, :], (1, 3, ids.shape[1]))
            yield ((ids, features, positions), GenerationConfig(max_new_tokens=resolve_max_new_tokens(
                None, audio_duration_sec=duration)))

    def completed(index, generation):
        row = metadata[index]
        metadata[index] = None
        row["elapsed_s"] = time.perf_counter() - row.pop("started")
        detected, text = parse_asr_output(session.tokenizer.decode(generation.tokens),
                                          user_language=hint)
        row.update(text=text, language=canonicalize_language(detected) or detected,
                   elapsed_scope="concurrent_chunk_latency",
                   finish_reason=generation.finish_reason, truncated=generation.truncated,
                   generated_tokens=generation.generated_tokens,
                   max_new_tokens=generation.max_new_tokens)
        on_complete(row)

    _, stats = generate_continuous(session.model, jobs(), batch_size, on_complete=completed,
                                  kv_policy=kv_policy)
    return stats
