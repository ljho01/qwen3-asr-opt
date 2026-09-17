"""Experimental fresh-prefix prefill with a bounded exact-shape compile cache."""
from __future__ import annotations

from collections import OrderedDict

import mlx.core as mx


def prefix_outputs(model, embeds, positions):
    """Use the upstream fresh dynamic cache: first K/V tensors need no zero buffer."""
    cache = model.create_cache()
    hidden = model.model(inputs_embeds=embeds, position_ids=positions, cache=cache)
    return model.lm_head(hidden[:, -1:, :]), cache.keys, cache.values


class PrefixPrefill:
    """Callable matching batch.prefill_serial for an immutable inference model.

    Strict upstream embedding/audio validation remains outside the compiled graph.
    Exact shapes are used because upstream masks/reshapes depend on Python shapes.
    Each retained function sees only one shape/dtype, with at most max_graphs live.
    Create a new instance after any model weight/module/dtype change. This helper
    is experimental and never installs itself into the model or default runtime.
    """

    def __init__(self, model, *, compiled=False, max_graphs=16):
        if not isinstance(max_graphs, int) or max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        self.model = model
        self.compiled = compiled
        self.max_graphs = max_graphs
        self.graphs = OrderedDict()
        self.calls = self.graphs_created = self.cache_hits = 0

    def _function(self, embeds, positions):
        key = (embeds.shape, embeds.dtype, positions.shape, positions.dtype)
        if self.compiled and key in self.graphs:
            self.cache_hits += 1
            self.graphs.move_to_end(key)
            return self.graphs[key]

        model = self.model

        def function(embedded, position_ids):
            logits, keys, values = prefix_outputs(model, embedded, position_ids)
            return mx.argmax(logits, axis=-1), keys, values

        if not self.compiled:
            return function
        # A separate shape-specific function allows LRU eviction of the whole
        # callable, rather than an unbounded internal shape cache on one function.
        function = mx.compile(function)
        if len(self.graphs) == self.max_graphs:
            self.graphs.popitem(last=False)
        self.graphs[key] = function
        self.graphs_created += 1
        return function

    def __call__(self, model, prompts, capacity):
        if model is not self.model:
            raise ValueError("Prefill helper belongs to a different model")
        if not prompts:
            raise ValueError("Require nonempty prompts")
        for ids, features, positions in prompts:
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
                raise ValueError("Each prompt must have shape (1, positive length)")
            if features.ndim != 3 or features.shape[0] != 1:
                raise ValueError("Audio features must have shape (1, audio length, width)")
            if positions.shape != (1, 3, ids.shape[1]):
                raise ValueError("Position shape must match the prompt")
            if capacity < ids.shape[1]:
                raise ValueError("Cache capacity must cover the prompt")
        first_tokens, key_rows, value_rows = [], [], []
        for ids, features, positions in prompts:
            embeds = model._embed_tokens(ids, validate_input_ids=True)
            embeds = model._inject_audio_features(embeds, features, ids == model.audio_token_id)
            token, keys, values = self._function(embeds, positions)(embeds, positions)
            mx.eval(token, keys, values)
            self.calls += 1
            first_tokens.append(token)
            padding = [(0, 0), (0, 0), (0, capacity - ids.shape[1]), (0, 0)]
            key_rows.append([mx.pad(value, padding) for value in keys])
            value_rows.append([mx.pad(value, padding) for value in values])
        keys = [mx.concatenate([row[layer] for row in key_rows], axis=0)
                for layer in range(len(key_rows[0]))]
        values = [mx.concatenate([row[layer] for row in value_rows], axis=0)
                  for layer in range(len(value_rows[0]))]
        token = mx.concatenate(first_tokens, axis=0)
        mx.eval(token, keys, values)
        return token, keys, values

    def statistics(self):
        return {"compiled": self.compiled, "calls": self.calls, "graphs_created": self.graphs_created,
                "cache_hits": self.cache_hits, "retained_graphs": len(self.graphs),
                "max_graphs": self.max_graphs}
