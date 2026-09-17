"""Experimental exact autoregressive steps between host/refill boundaries."""
from __future__ import annotations

import mlx.core as mx
from mlx_qwen3_asr.generate import GenerationResult, detect_repetition

from .batch import prefill_serial
from .kv_window import CompactRowCache, attention_width


def make_decode_block(model, width, steps):
    """Unroll dependent greedy predictions; this is not draft/speculative decoding.

    Every new token is chosen by the full unchanged model. Numerical equivalence
    of compiled graphs must still be verified. The caller bounds every cache write.
    """
    if width < 1 or not 1 <= steps <= 4:
        raise ValueError("Require positive width and1..4steps")

    def block(token, positions, keys, values):
        consumed = []
        for index in range(steps):
            consumed.append(token)
            current = positions + index
            cache = CompactRowCache(keys, values, current, width)
            mask = mx.arange(width)[None, None, None, :] <= current[:, None, None, None]
            rope = mx.broadcast_to(current[:, None, None], (token.shape[0], 3, 1))
            hidden = model.model(inputs_embeds=model.model.embed_tokens(token),
                                 position_ids=rope, attention_mask=mask, cache=cache)
            token = mx.argmax(model.lm_head(hidden), axis=-1)
            keys, values = cache.keys, cache.values
        return mx.concatenate(consumed, axis=1), token, keys, values

    return mx.compile(block)


def generate_continuous_blocks(model, jobs, batch_size=4, *, block_size=2,
                               shared_prefix=512, token_budget=512, on_complete=None,
                               kv_policy="growing128"):
    """Keep full-history growing128 KV, checking output/refilling after each block.

    A block stops at the next cache-capacity boundary or known token budget. EOS
    and repetition are checked in chronological order, with no emission after a
    row's first terminal token. Surplus computed continuation is discarded. Refill
    waits until this block's host boundary; that latency/work tradeoff is measured.
    """
    if (not 1 <= batch_size <= 8 or not 1 <= block_size <= 4
            or shared_prefix < 1 or token_budget < 1 or kv_policy != "growing128"):
        raise ValueError("Require batch1..8, block1..4, positive bounds and growing128")
    maximum = ((shared_prefix + token_budget + 255) // 256) * 256
    capacity = ((shared_prefix + 127) // 128) * 128
    stream = iter(jobs)
    slots, lengths, counts = [None] * batch_size, [1] * batch_size, [0] * batch_size
    outputs, exhausted = [], False
    stats = {"decode_calls": 0, "host_rounds": 0, "block_size": block_size, "actual_block_sizes": {},
             "active_row_steps": 0, "allocated_row_steps": 0, "deferred_finish_row_steps": 0,
             "prefills": 0, "refills": 0, "peak_active_rows": 0, "kv_policy": kv_policy,
             "shared_prefix": shared_prefix, "cache_capacity": capacity, "maximum_cache_capacity": maximum,
             "cache_capacity_history": [capacity], "attention_width_calls": {}, "prompt_lengths": []}

    def next_job():
        nonlocal exhausted
        if exhausted:
            return None
        try:
            prompt, config = next(stream)
        except StopIteration:
            exhausted = True
            return None
        length = prompt[0].shape[1]
        if config.temperature != 0 or not 1 <= config.max_new_tokens <= token_budget:
            raise ValueError("Require greedy generation and token cap within pool capacity")
        if not 1 <= length <= shared_prefix:
            raise ValueError("Prompt length exceeds shared prefix capacity")
        token, keys, values = prefill_serial(model, [prompt], capacity)
        slot = {"id": len(outputs), "config": config, "tokens": []}
        outputs.append(None)
        stats["prefills"] += 1
        stats["prompt_lengths"].append(length)
        return slot, length, token, keys, values

    initial = [next_job() for _ in range(batch_size)]
    if initial[0] is None:
        return [], stats
    prototype = initial[0]
    token_rows, key_rows, value_rows = [], [], []
    for row, job in enumerate(initial):
        if job is None:
            token_rows.append(mx.zeros_like(prototype[2]))
            key_rows.append([mx.zeros_like(array) for array in prototype[3]])
            value_rows.append([mx.zeros_like(array) for array in prototype[4]])
        else:
            slots[row], lengths[row], token, keys, values = job
            token_rows.append(token)
            key_rows.append(keys)
            value_rows.append(values)
    keys = [mx.concatenate([row[layer] for row in key_rows], axis=0) for layer in range(len(prototype[3]))]
    values = [mx.concatenate([row[layer] for row in value_rows], axis=0) for layer in range(len(prototype[4]))]
    token = mx.concatenate(token_rows, axis=0)
    mx.eval(token, keys, values)
    del initial, prototype, token_rows, key_rows, value_rows, job
    methods = getattr(model, "_autoregressive_blocks", None)
    if methods is None:
        methods = {}
        model._autoregressive_blocks = methods
    while any(slot is not None for slot in slots):
        active = sum(slot is not None for slot in slots)
        stats["peak_active_rows"] = max(stats["peak_active_rows"], active)
        positions = [length + count if slot is not None else 0
                     for length, count, slot in zip(lengths, counts, slots, strict=True)]
        last = max(positions)
        if last >= capacity:
            new_capacity = attention_width(last, maximum)
            padding = [(0, 0), (0, 0), (0, new_capacity - capacity), (0, 0)]
            keys, values = [mx.pad(a, padding) for a in keys], [mx.pad(a, padding) for a in values]
            capacity = new_capacity
            stats["cache_capacity"] = capacity
            stats["cache_capacity_history"].append(capacity)
        remaining = min(slot["config"].max_new_tokens - len(slot["tokens"])
                        for slot in slots if slot is not None)
        steps = min(block_size, capacity - last, remaining)
        key = (batch_size, capacity, steps)
        if key not in methods:
            methods[key] = make_decode_block(model, capacity, steps)
        emitted, next_token, keys, values = methods[key](
            token, mx.array(positions, dtype=mx.int32), keys, values)
        mx.async_eval(emitted, next_token, keys, values)
        stats["host_rounds"] += 1
        stats["decode_calls"] += steps
        stats["allocated_row_steps"] += batch_size * steps
        stats["actual_block_sizes"][str(steps)] = stats["actual_block_sizes"].get(str(steps), 0) + 1
        stats["attention_width_calls"][str(capacity)] = stats["attention_width_calls"].get(str(capacity), 0) + steps
        host_tokens = emitted.tolist()
        completed = []
        for index in range(steps):
            for row, row_tokens in enumerate(host_tokens):
                slot = slots[row]
                if slot is None:
                    continue
                value, reason = row_tokens[index], None
                config = slot["config"]
                stats["active_row_steps"] += 1
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
                    result = GenerationResult(output, reason, len(output), config.max_new_tokens)
                    outputs[slot["id"]] = result
                    if on_complete is not None:
                        on_complete(slot["id"], result)
                    slots[row] = None
                    completed.append(row)
                    stats["deferred_finish_row_steps"] += steps - index - 1
        for row in completed:
            job = next_job()
            if job is None:
                counts[row] = 0
                continue
            slots[row], lengths[row], new_token, new_keys, new_values = job
            counts[row] = 0
            offset = mx.array([row], dtype=mx.int32)
            keys = [mx.slice_update(old, new, offset, axes=[0]) for old, new in zip(keys, new_keys, strict=True)]
            values = [mx.slice_update(old, new, offset, axes=[0]) for old, new in zip(values, new_values, strict=True)]
            next_token = mx.slice_update(next_token, new_token, offset, axes=[0])
            mx.async_eval(next_token, keys, values)
            stats["refills"] += 1
            del job, new_token, new_keys, new_values
        token = next_token
    mx.synchronize()
    stats["active_fraction"] = stats["active_row_steps"] / stats["allocated_row_steps"]
    return outputs, stats
