"""Experimental lower-precision output head with fixed-column residual correction.

This is approximate, not certified q8 argmax refinement. The original q8 input
embedding remains unchanged. No CLI/preset loads this module automatically.
Install only into a fresh model before any decoder function is compiled.
"""
from __future__ import annotations

import mlx.core as mx
from mlx import nn


class ApproximateHead(nn.Module):
    def __init__(self, weight, scales, biases, columns, correction, *, bits, group_size=64):
        super().__init__()
        if bits not in (4, 6) or group_size != 64:
            raise ValueError("Probe supports affine4/6bit with group64")
        if columns.ndim != 1 or columns.dtype != mx.int32 or correction.shape != (weight.shape[0], columns.size):
            raise ValueError("Residual columns and matrix have incompatible shapes")
        if correction.dtype != mx.float32:
            raise ValueError("Residual correction must be float32")
        self.weight, self.scales, self.biases = weight, scales, biases
        self.columns, self.correction = columns, correction
        self.bits, self.group_size = bits, group_size

    def __call__(self, x):
        scores = mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                    transpose=True, group_size=self.group_size, bits=self.bits, mode="affine")
        if self.columns.size:
            selected = mx.take(x, self.columns, axis=-1).astype(mx.float32)
            scores = scores.astype(mx.float32) + selected @ self.correction.T
        return scores

    @classmethod
    def from_q8(cls, original, *, bits, columns=()):
        if original.bits != 8 or original.group_size != 64 or original.mode != "affine" or original.get("bias") is not None:
            raise ValueError("Require bias-free affineq8/group64 source")
        columns = list(columns)
        width = original.scales.shape[1] * original.group_size
        if len(set(columns)) != len(columns) or any(type(i) is not int or not 0 <= i < width for i in columns):
            raise ValueError("Correction columns must be unique in-bounds integers")
        dense8 = mx.dequantize(original.weight, original.scales, original.biases, group_size=64, bits=8)
        parameters = mx.quantize(dense8, group_size=64, bits=bits, mode="affine")
        indices = mx.array(columns, dtype=mx.int32)
        if columns:
            dense_low = mx.dequantize(*parameters, group_size=64, bits=bits, mode="affine")
            correction = mx.take(dense8, indices, axis=1).astype(mx.float32) - mx.take(dense_low, indices, axis=1).astype(mx.float32)
        else:
            correction = mx.zeros((dense8.shape[0], 0), dtype=mx.float32)
        module = cls(*parameters, indices, correction, bits=bits)
        mx.eval(module.parameters())
        module.freeze()
        return module


def install_fresh_head(model, head):
    """Refuse replacing a head captured by an existing compiled decoder graph."""
    cached = ("_continuous_steps", "_optimized_batch_step", "_optimized_step", "_speculative_verifiers")
    if any(hasattr(model, name) for name in cached):
        raise ValueError("Install the experimental head only before decoder compilation")
    model.lm_head = head
