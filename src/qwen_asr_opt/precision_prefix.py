"""Experimental q8 acoustic/text prefill with MLP6 autoregressive decoding."""
from __future__ import annotations

from dataclasses import asdict

import mlx.core as mx
from mlx.utils import tree_flatten

from .batch import prefill_serial
from .mlp_precision import is_mlp


class HighPrecisionPrefix:
    """Share identical protected tensors, retaining both precisions only for MLPs.

    Construct before either model has compiled batch/continuous steps. No global
    hooks are installed here. Dedicated experiment workers choose where to call
    this helper. q8 supplies the first greedy token and all prompt K/V; subsequent
    steps use the supplied MLP6 decoder. This is approximate generation, not a
    claim of mathematical equivalence to q8 or to all-MLP6 inference.
    """

    def __init__(self, prefix_model, decode_model):
        if prefix_model is decode_model or asdict(prefix_model.config) != asdict(decode_model.config):
            raise ValueError("Require distinct models with identical configuration")
        if any(hasattr(model, attr) for model in (prefix_model, decode_model)
               for attr in ("_optimized_batch_step", "_continuous_steps")):
            raise ValueError("Construct before compiling generation steps")
        for prefix_layer, decode_layer in zip(prefix_model.model.layers, decode_model.model.layers, strict=True):
            for name in ("gate_proj", "up_proj", "down_proj"):
                a, b = getattr(prefix_layer.mlp, name), getattr(decode_layer.mlp, name)
                if a.bits != 8 or b.bits != 6 or a.group_size != 64 or b.group_size != 64:
                    raise ValueError("Require q8/MLP6 affine group64 projections")
        prefix = dict(tree_flatten(prefix_model.parameters()))
        decode = dict(tree_flatten(decode_model.parameters()))
        if prefix.keys() != decode.keys():
            raise ValueError("Model parameter paths differ")
        shared, mlp_arrays = {}, {}
        for name, value in decode.items():
            if is_mlp(name.rsplit(".", 1)[0]):
                mlp_arrays[id(value)] = value
                continue
            source = prefix[name]
            if value.shape != source.shape or value.dtype != source.dtype or not mx.array_equal(value, source).item():
                raise ValueError(f"Protected tensor differs: {name}")
            shared[name] = source
        # All compatibility checks finish before the first model mutation.
        decode_model.load_weights(list(shared.items()), strict=False)
        after = dict(tree_flatten(decode_model.parameters()))
        if not all(after[name] is value for name, value in shared.items()):
            raise RuntimeError("Protected tensor storage was not shared")
        self.prefix_model, self.decode_model = prefix_model, decode_model
        self.shared_entries = len(shared)
        self.additional_mlp_payload_bytes = sum(a.nbytes for a in mlp_arrays.values())
        self.calls = 0

    def __call__(self, model, prompts, capacity):
        if model is not self.decode_model:
            raise ValueError("Helper belongs to a different decode model")
        result = prefill_serial(self.prefix_model, prompts, capacity)
        self.calls += len(prompts)
        return result

    def statistics(self):
        return {"prefix_calls": self.calls, "protected_tensor_entries_shared": self.shared_entries,
                "additional_mlp_payload_bytes": self.additional_mlp_payload_bytes,
                "prefix_precision": "q8", "decode_mlp_bits": 6,
                "note": "Payload excludes graph/KV/allocator/RSS overhead; shared entries include tied aliases"}
