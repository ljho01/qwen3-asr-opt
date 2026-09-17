"""Experimental target-model block verification; not enabled by transcription presets."""
from __future__ import annotations

import mlx.core as mx
from mlx_qwen3_asr.generate import GenerationResult, detect_repetition

from .batch import prefill_serial


class RowCache:
    """Write each row's contiguous proposal block at its own rollback position.

    Rejected cache slots need not be cleared: the caller's causal mask excludes
    them, and the next block overwrites from that row's accepted prefix boundary.
    """

    def __init__(self, keys, values, starts, mode):
        self.keys, self.values = list(keys), list(values)
        self.starts, self.mode = starts, mode
        self.offset = 1  # Explicit attention mask is supplied by the verifier.

    def put(self, destination, block):
        if self.mode == "scatter":
            positions = self.starts[:, None] + mx.arange(block.shape[2])[None, :]
            indices = mx.broadcast_to(positions[:, None, :, None], block.shape)
            return mx.put_along_axis(destination, indices, block, axis=2)
        for row in range(block.shape[0]):
            start = mx.stack([mx.array(row, dtype=mx.int32), self.starts[row]])
            destination = mx.slice_update(destination, block[row:row + 1], start, axes=[0, 2])
        return destination

    def update(self, key, value, layer_idx):
        self.keys[layer_idx] = self.put(self.keys[layer_idx], key)
        self.values[layer_idx] = self.put(self.values[layer_idx], value)
        return self.keys[layer_idx], self.values[layer_idx]


def make_block_verifier(model, cache_mode="scatter"):
    """Predict after every input token in a causally masked proposal block.

    Prompts occupy [0, prefix_lengths[row]); generated tokens start at the shared
    physical slot `shared_prefix`. Per-row starts may differ after rejections.
    Tokens have shape (batch, block); starts and prefix_lengths have shape (batch,).
    Caller must keep every write within the cache capacity and starts >= shared_prefix.
    This is a verification primitive, not a claim of bitwise greedy equivalence:
    block matrix kernels may round differently from single-token kernels.
    """
    if cache_mode not in ("scatter", "rows"):
        raise ValueError("cache_mode must be scatter or rows")

    def verify(tokens, starts, prefix_lengths, shared_prefix, keys, values):
        physical = starts[:, None] + mx.arange(tokens.shape[1])[None, :]
        logical = prefix_lengths[:, None] + physical - shared_prefix
        positions = mx.broadcast_to(logical[:, None, :], (tokens.shape[0], 3, tokens.shape[1]))
        slots = mx.arange(keys[0].shape[2])[None, None, None, :]
        mask = ((slots < prefix_lengths[:, None, None, None])
                | ((slots >= shared_prefix) & (slots <= physical[:, None, :, None])))
        cache = RowCache(keys, values, starts, cache_mode)
        hidden = model.model(inputs_embeds=model.model.embed_tokens(tokens),
                             position_ids=positions, attention_mask=mask, cache=cache)
        return mx.argmax(model.lm_head(hidden), axis=-1), cache.keys, cache.values

    return mx.compile(verify)


def generate_speculative(target, draft, target_prompts, draft_prompts, configs,
                         proposals=3, cache_mode="scatter"):
    """Greedy draft/verify with independent row acceptance and rollback.

    The target always selects the emitted tokens. Numerical differences between
    target block and target single-token kernels must still be measured separately.
    Returns GenerationResults and work counters; this remains an opt-in experiment.
    """
    batch = len(configs)
    if not batch or len(target_prompts) != batch or len(draft_prompts) != batch:
        raise ValueError("Expected matching nonempty target/draft prompts and configs")
    if not 1 <= proposals <= 8:
        raise ValueError("proposals must be 1..8")
    if any(c.temperature != 0 or c.max_new_tokens < 1 for c in configs):
        raise ValueError("Speculation requires greedy generation and positive token budgets")
    if target.config.text_config.vocab_size != draft.config.text_config.vocab_size:
        raise ValueError("Target and draft must share token IDs")
    max_tokens = max(c.max_new_tokens for c in configs)

    def prepare(model, prompts):
        lengths = mx.array([p[0].shape[1] for p in prompts])
        common = max(p[0].shape[1] for p in prompts)
        capacity = ((common + max_tokens + proposals + 2 + 255) // 256) * 256
        token, keys, values = prefill_serial(model, prompts, capacity)
        methods = getattr(model, "_speculative_verifiers", None)
        if methods is None:
            methods = {}
            model._speculative_verifiers = methods
        if cache_mode not in methods:
            methods[cache_mode] = make_block_verifier(model, cache_mode)
        return lengths, mx.array(common), token, keys, values, methods[cache_mode]

    t_lengths, t_common, current, t_keys, t_values, target_step = prepare(target, target_prompts)
    d_lengths, d_common, _, d_keys, d_values, draft_step = prepare(draft, draft_prompts)
    tokens, finishes = [[] for _ in configs], [None] * batch
    consumed = [0] * batch
    last_accepted = None
    stats = {"target_block_calls": 0, "draft_block_calls": 0,
             "proposed_tokens": 0, "accepted_proposals": 0, "rounds": 0}

    def emit(row, token):
        config = configs[row]
        if token in config.eos_token_ids:
            finishes[row] = "eos"
        else:
            tokens[row].append(token)
            if len(tokens[row]) >= config.max_new_tokens:
                finishes[row] = "length"
            elif detect_repetition(tokens[row]):
                finishes[row] = "repetition"

    while True:
        current_values = current.reshape(-1).tolist()
        for row, token in enumerate(current_values):
            if finishes[row] is None:
                emit(row, token)
        if all(reason is not None for reason in finishes):
            break
        active = [reason is None for reason in finishes]
        starts = mx.array(consumed, dtype=mx.int32)
        proposed = []
        draft_token = current
        for index in range(proposals):
            # The draft may lack the last accepted token when every proposal was
            # accepted. Replaying that token together with the target correction
            # repairs every row in one block, without a separate catch-up pass.
            if index == 0 and last_accepted is not None:
                block = mx.concatenate([last_accepted, draft_token], axis=1)
                position = d_common + starts - 1
            else:
                block = draft_token
                position = d_common + starts + index
            predictions, d_keys, d_values = draft_step(
                block, position, d_lengths, d_common, d_keys, d_values)
            draft_token = predictions[:, -1:]
            mx.async_eval(draft_token, d_keys, d_values)
            proposed.append(draft_token)
            stats["draft_block_calls"] += 1
        proposal_tokens = mx.concatenate(proposed, axis=1)
        target_inputs = mx.concatenate([current, proposal_tokens], axis=1)
        predictions, t_keys, t_values = target_step(
            target_inputs, t_common + starts, t_lengths, t_common, t_keys, t_values)
        mx.eval(predictions, t_keys, t_values, d_keys, d_values)
        predicted_values, proposed_values = predictions.tolist(), proposal_tokens.tolist()
        next_values, accepted_values = [], []
        stats["target_block_calls"] += 1
        stats["rounds"] += 1
        stats["proposed_tokens"] += sum(active) * proposals
        for row in range(batch):
            accepted = 0
            last = current_values[row]
            if active[row]:
                for index, candidate in enumerate(proposed_values[row]):
                    if candidate != predicted_values[row][index]:
                        break
                    accepted += 1
                    emit(row, candidate)
                    last = candidate
                    if finishes[row] is not None:
                        break
                stats["accepted_proposals"] += accepted
                consumed[row] += 1 + accepted
            # Finished rows remain allocated, with bounded positions, until all
            # rows finish. Their synthetic continuation is never emitted.
            next_values.append(predicted_values[row][accepted])
            accepted_values.append(last)
        current = mx.array(next_values, dtype=mx.int32)[:, None]
        last_accepted = mx.array(accepted_values, dtype=mx.int32)[:, None]
    mx.synchronize()
    stats["acceptance_fraction"] = (stats["accepted_proposals"] / stats["proposed_tokens"]
                                     if stats["proposed_tokens"] else None)
    return ([GenerationResult(t, reason, len(t), c.max_new_tokens)
             for t, reason, c in zip(tokens, finishes, configs, strict=True)], stats)
