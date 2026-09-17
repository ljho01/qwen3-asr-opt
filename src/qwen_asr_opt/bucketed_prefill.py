"""Experimental right-padded prefix compilation with bounded graph reuse."""
from __future__ import annotations

from collections import OrderedDict

import mlx.core as mx


class BucketedPrefill:
    """Serial prefill with shared compiled shapes and dynamic valid lengths.

    Embedding/audio validation runs on original inputs before padding. Causal
    attention isolates every valid query from the padded suffix. Gather the last
    valid hidden state and zero the suffix KV before returning it to the caller.
    Model weights must remain immutable throughout this helper's lifetime.
    This class does not install itself or change any default execution path.
    """

    def __init__(self, model, *, quantum=32, max_graphs=16):
        if type(quantum) is not int or quantum < 1 or type(max_graphs) is not int or max_graphs < 1:
            raise ValueError("Require positive integer quantum and max_graphs")
        self.model = model
        self.quantum = quantum
        self.max_graphs = max_graphs
        self.graphs = OrderedDict()
        self.calls = self.graphs_created = self.cache_hits = 0
        self.valid_tokens = self.padded_tokens = 0
        self.bucket_calls = {}

    def _function(self, embeds, positions):
        key = (embeds.shape, embeds.dtype, positions.shape, positions.dtype)
        if key in self.graphs:
            self.cache_hits += 1
            self.graphs.move_to_end(key)
            return self.graphs[key]
        model = self.model

        def function(embedded, position_ids, valid_length):
            cache = model.create_cache()
            hidden = model.model(inputs_embeds=embedded, position_ids=position_ids, cache=cache)
            last = mx.take(hidden, valid_length - 1, axis=1)
            token = mx.argmax(model.lm_head(last), axis=-1)
            valid = mx.arange(embedded.shape[1])[None, None, :, None] < valid_length.reshape(1, 1, 1, 1)
            keys = [mx.where(valid, value, mx.zeros((), value.dtype)) for value in cache.keys]
            values = [mx.where(valid, value, mx.zeros((), value.dtype)) for value in cache.values]
            return token, keys, values

        function = mx.compile(function)
        if len(self.graphs) == self.max_graphs:
            self.graphs.popitem(last=False)
        self.graphs[key] = function
        self.graphs_created += 1
        return function

    def __call__(self, model, prompts, capacity):
        if model is not self.model:
            raise ValueError("Prefill helper belongs to a different model")
        if not prompts or type(capacity) is not int or capacity < 1:
            raise ValueError("Require nonempty prompts and positive integer capacity")
        for ids, features, positions in prompts:
            if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= capacity:
                raise ValueError("Each prompt must have shape (1, positive length) fitting capacity")
            if features.ndim != 3 or features.shape[0] != 1:
                raise ValueError("Audio features must have shape (1, audio length, width)")
            if positions.shape != (1, 3, ids.shape[1]):
                raise ValueError("Position shape must match the prompt")
        first_tokens, key_rows, value_rows = [], [], []
        for ids, features, positions in prompts:
            length = ids.shape[1]
            bucket = min(capacity, ((length + self.quantum - 1) // self.quantum) * self.quantum)
            embeds = model._embed_tokens(ids, validate_input_ids=True)
            embeds = model._inject_audio_features(embeds, features, ids == model.audio_token_id)
            embeds = mx.pad(embeds, [(0, 0), (0, bucket - length), (0, 0)])
            positions = mx.pad(positions, [(0, 0), (0, 0), (0, bucket - length)])
            token, keys, values = self._function(embeds, positions)(
                embeds, positions, mx.array([length], dtype=mx.int32))
            padding = [(0, 0), (0, 0), (0, capacity - bucket), (0, 0)]
            first_tokens.append(token)
            key_rows.append([mx.pad(value, padding) for value in keys])
            value_rows.append([mx.pad(value, padding) for value in values])
            self.calls += 1
            self.valid_tokens += length
            self.padded_tokens += bucket
            self.bucket_calls[str(bucket)] = self.bucket_calls.get(str(bucket), 0) + 1
        keys = [mx.concatenate([row[layer] for row in key_rows], axis=0)
                for layer in range(len(key_rows[0]))]
        values = [mx.concatenate([row[layer] for row in value_rows], axis=0)
                  for layer in range(len(value_rows[0]))]
        token = mx.concatenate(first_tokens, axis=0)
        mx.eval(token, keys, values)
        return token, keys, values

    def statistics(self):
        return {"quantum": self.quantum, "calls": self.calls, "graphs_created": self.graphs_created,
                "cache_hits": self.cache_hits, "retained_graphs": len(self.graphs),
                "max_graphs": self.max_graphs, "valid_tokens": self.valid_tokens,
                "padded_tokens": self.padded_tokens, "bucket_calls": dict(self.bucket_calls)}
