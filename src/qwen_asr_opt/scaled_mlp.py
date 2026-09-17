"""Experimental activation-scaled six-bit MLP linears; no automatic installation.

AWQ-inspired per-channel weight/input rescaling, not the complete AWQ algorithm:
no clipping, fused scale folding, training, or promise of recognition equivalence.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ROLES = ("gate_proj", "up_proj", "down_proj")


def activation_scales(samples, alpha):
    """Mean-absolute statistics in FP64; normalized positive scales stored FP16."""
    values = np.asarray(samples)
    if (values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all()
            or alpha not in ALPHAS):
        raise ValueError("Require finite nonempty samples and a fixed-grid alpha")
    mean = np.mean(np.abs(values.astype(np.float64)), axis=0)
    scales = np.maximum(mean, 1e-8) ** alpha
    scales = np.clip(scales, 1e-4, 1e4)
    scales /= np.sqrt(scales.max()*scales.min())
    # The fixed clamps ensure representable scales/inverses in this inference dtype.
    scales = np.clip(scales, 1/256, 256).astype(np.float16)
    return scales


class ScaledMLPLinear(nn.Module):
    """Q6(W8 * s) @ (x / s), with both scaled weights/activations rounded FP16.

    Only affineq8/group64 bias-free sources are accepted by from_q8. This module
    stores replacement parameters only, not a reference to the original weights.
    Install before any compilation; changing weights requires fresh decoder caches.
    """

    def __init__(self, weight, scales, biases, input_scale):
        super().__init__()
        if (weight.ndim != 2 or scales.ndim != 2 or biases.shape != scales.shape
                or input_scale.shape != (scales.shape[1]*64,)
                or weight.shape != (scales.shape[0], input_scale.size*6//32)
                or weight.dtype != mx.uint32 or scales.dtype != mx.float16
                or biases.dtype != mx.float16 or input_scale.dtype != mx.float16):
            raise ValueError("Require correctly shaped q6/group64 FP16 affine parameters")
        if not bool(mx.all(mx.isfinite(input_scale) & (input_scale > 0)).item()):
            raise ValueError("Input scales must be finite and positive")
        self.weight, self.scales, self.biases = weight, scales, biases
        self.input_scale = input_scale

    def __call__(self, x):
        return mx.quantized_matmul(x / self.input_scale.astype(x.dtype),
                                   self.weight, self.scales, self.biases,
                                   transpose=True, group_size=64, bits=6, mode="affine")

    @classmethod
    def from_q8(cls, source, scales):
        if (source.bits != 8 or source.group_size != 64 or source.mode != "affine"
                or source.get("bias") is not None):
            raise ValueError("Require a bias-free affineq8/group64 source")
        values = np.asarray(scales)
        width = source.scales.shape[1]*64
        if (values.shape != (width,) or values.dtype != np.float16
                or not np.isfinite(values).all() or np.any(values <= 0)):
            raise ValueError("Require one finite positive FP16 scale per input channel")
        scale = mx.array(values)
        dense = mx.dequantize(source.weight, source.scales, source.biases,
                              group_size=64, bits=8, mode="affine")
        transformed = (dense.astype(mx.float32)*scale.astype(mx.float32)).astype(mx.float16)
        parameters = mx.quantize(transformed, group_size=64, bits=6, mode="affine")
        result = cls(*parameters, scale)
        mx.eval(result.parameters())
        result.freeze()
        return result


def install_scaled_mlp(model, tensors):
    """Atomically validate all replacements, then install into a fresh model."""
    cached = ("_continuous_steps", "_optimized_batch_step", "_optimized_step", "_speculative_verifiers")
    if any(hasattr(model, name) for name in cached):
        raise ValueError("Install only before decoder compilation")
    replacements = []
    expected_keys = set()
    for index, layer in enumerate(model.model.layers):
        for role in ROLES:
            prefix = f"{index}.{role}."
            names = ("weight", "scales", "biases", "input_scale")
            expected_keys.update(prefix+name for name in names)
            replacement = ScaledMLPLinear(**{name: tensors[prefix+name] for name in names})
            old = getattr(layer.mlp, role)
            if replacement.scales.shape != old.scales.shape:
                raise ValueError("Replacement dimensions must match the original MLP")
            replacements.append((layer.mlp, role, replacement))
    if set(tensors) != expected_keys:
        raise ValueError("Require exactly all MLP replacement tensors")
    for module, role, replacement in replacements:
        setattr(module, role, replacement)
