"""Experimental exact-shape compilation of the unchanged upstream audio tower."""
from __future__ import annotations

from collections import OrderedDict

import mlx.core as mx
from mlx import nn


class NativeCompiledEncoder(nn.Module):
    """Wrap an immutable, already loaded AudioEncoder with a bounded callable LRU.

    The original module, weights and upstream tail/window semantics are retained.
    Compilation can change rounding; this is not an exact-output guarantee.
    Create a new wrapper after ANY weight, dtype, config or module mutation.
    The LRU bounds retained Python callables, not Metal/driver compiler caches.
    This helper is experimental, single-threaded, and has no default/CLI hook.
    """

    def __init__(self, original, *, max_graphs=32):
        super().__init__()
        if type(max_graphs) is not int or max_graphs < 1:
            raise ValueError("max_graphs must be a positive integer")
        self.original = original
        self.config = original.config
        self.max_graphs = max_graphs
        # Cache keys are shape/dtype tuples, not module parameter paths.
        object.__setattr__(self, "graphs", OrderedDict())
        self.calls = self.graphs_created = self.cache_hits = self.evictions = 0

    def get_output_lengths(self, lengths):
        return self.original.get_output_lengths(lengths)

    def _function(self, mel):
        key = (mel.shape, mel.dtype)
        if key in self.graphs:
            self.graphs.move_to_end(key)
            self.cache_hits += 1
            return self.graphs[key]
        original = self.original
        chunk_size = self.config.n_window * 2
        window_size = self.config.n_window_infer

        def encode(value):
            return original._encode_single(value, chunk_size, window_size)

        function = mx.compile(encode, shapeless=False)
        if len(self.graphs) == self.max_graphs:
            self.graphs.popitem(last=False)
            self.evictions += 1
        self.graphs[key] = function
        self.graphs_created += 1
        return function

    def __call__(self, features, lengths):
        if (features.ndim != 3 or features.shape[0] < 1
                or features.shape[1] != self.config.num_mel_bins
                or features.dtype not in (mx.float16, mx.float32, mx.bfloat16)):
            raise ValueError("Expected nonempty floating (batch, mel bins, frames)")
        if lengths.shape != (features.shape[0],) or not mx.issubdtype(lengths.dtype, mx.integer):
            raise ValueError("Expected one integer frame count per sample")
        # Resolve/validate all lengths before tracing; evaluation is outside compile.
        frames = lengths.tolist()
        if any(length < 1 or length > features.shape[2] for length in frames):
            raise ValueError("Frame counts must be positive and within input bounds")
        outputs = []
        for index, length in enumerate(frames):
            mel = features[index, :, :length]
            outputs.append(self._function(mel)(mel))
            self.calls += 1
        output_lengths = [value.shape[0] for value in outputs]
        maximum = max(output_lengths)
        padded = [mx.pad(value, [(0, maximum - value.shape[0]), (0, 0)])
                  if value.shape[0] < maximum else value for value in outputs]
        return mx.stack(padded), mx.array(output_lengths)

    def statistics(self):
        return {"calls": self.calls, "graphs_created": self.graphs_created,
                "cache_hits": self.cache_hits, "evictions": self.evictions,
                "retained_graphs": len(self.graphs), "max_graphs": self.max_graphs}
